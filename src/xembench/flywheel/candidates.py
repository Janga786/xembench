"""Failure-driven candidate generation: eval failures -> demo requests.

Converts an episode-level eval CSV (the format every xembench analysis tool
consumes) into CandidateDemoRequest records — concrete, prioritized "collect
this next" proposals. The mapping from failure label to proposal is a fixed,
documented rule table (below), not a learned model; rich labels beyond
success/timeout/unknown_failure only exist after Phase B instrumentation, so
every rule also has a coarse-label fallback.

One candidate = one requested episode. Failure groups emit up to
``max_per_group`` candidates (deterministic seeds, cycled variations), so
episode budgets in selection are exact.

CLI: ``python -m xembench.flywheel.generate_candidates``.
"""

import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from xembench.analysis.episodes import load_episodes, toy_banner
from xembench.flywheel.schemas import (
    CandidateDemoRequest,
    DatasetIndexRecord,
    derive_flywheel_split,
    stable_hash,
    write_jsonl,
)
from xembench.utils.config import sanitize_path

# Held-out and cross-embodiment failures are the benchmark's headline axes;
# their fixes are worth more per episode than train-split fixes.
SPLIT_WEIGHTS = {
    "cross_embodiment": 2.0,
    "heldout_paraphrase": 1.5,
    "heldout_color": 1.4,
    "train_seen": 1.0,
}

# Relative collection cost per episode (unitless; Linux wall-clock heuristic):
# Panda demos come from mplib motion planning, G1 demos need a trained PPO
# expert first, diagnostics are short scripted episodes.
EMBODIMENT_COST = {"panda": 1.0, "panda_stick": 1.0, "g1": 3.0}
DIAGNOSTIC_COST = 0.5

MAX_PER_GROUP_DEFAULT = 10
CANDIDATE_SEED_BASE = 500_000


@dataclass(frozen=True)
class _Rule:
    vary: tuple[str, ...]
    hold: tuple[str, ...] = ()
    policy_source: str = "sim_expert"
    design_sensitive: bool = False
    diagnostic: bool = False
    reason: str = ""


# Failure label -> what to collect. Grounded in the taxonomy
# (xembench.analysis.taxonomy); unknown labels fall back to BALANCED.
RULES: dict[str, _Rule] = {
    "wrong_object_selected": _Rule(
        vary=("paraphrase", "target_object", "target_color"),
        reason="grounding failure: same task, varied paraphrases and "
               "target objects/colors"),
    "language_grounding_failure": _Rule(
        vary=("paraphrase", "target_object", "target_color"),
        reason="grounding failure: same task, varied paraphrases and "
               "target objects/colors"),
    "reach_failure": _Rule(
        vary=("initial_pose",), hold=("target_object", "target_color"),
        reason="reach failure: same object/color condition, varied initial "
               "poses"),
    "grasp_failure": _Rule(
        vary=("initial_pose",), hold=("target_object", "target_color"),
        reason="grasp failure: same object/color condition, varied initial "
               "poses"),
    "object_dropped": _Rule(
        vary=("initial_pose",), hold=("target_object", "target_color"),
        reason="drop after grasp: same object/color condition, varied "
               "initial poses"),
    "timeout": _Rule(
        vary=("initial_pose",), design_sensitive=True,
        reason="timeout: easier start states (and consider a longer "
               "horizon — DESIGN-SENSITIVE, horizon changes alter the "
               "benchmark)"),
    "placement_failure": _Rule(
        vary=("goal_pose",), hold=("target_object", "target_color"),
        reason="placement failure: demos focused on the final placement "
               "condition"),
    "embodiment_reachability_limit": _Rule(
        vary=("target_pose_within_reach",),
        reason="reachability limit: same task with targets constrained to "
               "the workspace — EMBODIMENT-SPECIFIC data"),
    "adapter_action_mismatch": _Rule(
        vary=("none",), policy_source="diagnostic", diagnostic=True,
        reason="adapter mismatch: diagnostic episodes to debug the action "
               "mapping, NOT training demos"),
    "simulator_error": _Rule(
        vary=("none",), policy_source="diagnostic", diagnostic=True,
        reason="simulator error: diagnostic episodes to reproduce, NOT "
               "training demos"),
}

BALANCED = _Rule(
    vary=("paraphrase", "target_color", "initial_pose"),
    reason="unlabeled/unknown failure: balanced collection across the "
           "failed task/split conditions")


def _rule_for(label: str) -> _Rule:
    return RULES.get(label, BALANCED)


def _estimated_cost(embodiment: str, rule: _Rule) -> float:
    if rule.diagnostic:
        return DIAGNOSTIC_COST
    return EMBODIMENT_COST.get(embodiment, 1.0)


def _task_counts(index_records: list[DatasetIndexRecord] | None,
                 ) -> tuple[Counter, float]:
    if not index_records:
        return Counter(), 0.0
    counts = Counter(r.task for r in index_records)
    return counts, statistics.median(counts.values())


def generate_candidates(df: pd.DataFrame,
                        index_records: list[DatasetIndexRecord] | None = None,
                        *, max_per_group: int = MAX_PER_GROUP_DEFAULT,
                        ) -> list[CandidateDemoRequest]:
    """Candidates from the failures in an eval-episode DataFrame.

    Failure groups are (eval_task, eval_embodiment, split, failure_type);
    each emits min(n_failures, max_per_group) single-episode candidates with
    deterministic seeds. A dataset index, when provided, boosts the expected
    value of tasks that are underrepresented in existing demos.
    """
    df = df.copy()
    # append_episodes writes "" for optional v2 columns; read_csv gives NaN.
    df["template_family"] = df["template_family"].fillna("").astype(str)
    df["split"] = [derive_flywheel_split(t, et, ic, cc)
                   for t, et, ic, cc in zip(df["task"], df["eval_task"],
                                            df["instruction_cond"],
                                            df["color_cond"])]
    task_counts, median_count = _task_counts(index_records)

    candidates: list[CandidateDemoRequest] = []
    grouped = df.groupby(["eval_task", "eval_embodiment", "split",
                          "failure_type"], sort=True)
    for (task, embodiment, split, label), g in grouped:
        if label == "success":
            continue
        n_failures = len(g)
        cell = df[(df["eval_task"] == task) & (df["split"] == split)
                  & (df["eval_embodiment"] == embodiment)]
        failure_rate = n_failures / max(len(cell), 1)
        rule = _rule_for(label)
        weight = SPLIT_WEIGHTS.get(split, 1.0)
        boost = (0.25 if task_counts and
                 task_counts.get(task, 0) < median_count else 0.0)
        # Representative conditions from the actual failed episodes.
        color = str(g["color_cond"].mode().iat[0])
        family = str(g["template_family"].mode().iat[0]) \
            if g["template_family"].astype(bool).any() else ""

        n_emit = min(n_failures, max_per_group)
        for i in range(n_emit):
            identity = f"{task}|{embodiment}|{split}|{label}|{i}"
            constraints = {
                "vary": list(rule.vary),
                "hold": list(rule.hold),
                "variant": i,
            }
            if rule.design_sensitive:
                constraints["design_sensitive"] = True
            if rule.diagnostic:
                constraints["diagnostic"] = True
            if split == "cross_embodiment":
                constraints["cross_embodiment"] = True
            candidates.append(CandidateDemoRequest(
                candidate_id=f"cand_{stable_hash(identity)}",
                task=task,
                embodiment=embodiment,
                split=split,
                instruction_family=family,
                color_condition=color,
                target_failure_label=label,
                requested_policy_source=rule.policy_source,
                estimated_cost=_estimated_cost(embodiment, rule),
                expected_value=round(failure_rate * weight + boost, 4),
                priority=round(n_failures * weight, 4),
                reason=rule.reason,
                constraints=constraints,
                seed=CANDIDATE_SEED_BASE + len(candidates),
                status="proposed",
            ))
    return candidates


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def candidates_dataframe(candidates: list[CandidateDemoRequest],
                         ) -> pd.DataFrame:
    rows = []
    for c in candidates:
        row = dict(c.__dict__)
        row["constraints"] = ";".join(
            f"{k}={v}" for k, v in sorted(c.constraints.items()))
        rows.append(row)
    return pd.DataFrame(rows)


def candidates_markdown(candidates: list[CandidateDemoRequest],
                        source: str, synthetic: bool) -> str:
    banner = ("> **SYNTHETIC TOY DATA — NOT REAL BENCHMARK RESULTS.** "
              "These candidates were derived from synthetic eval data.\n\n"
              if synthetic else "")
    lines = [
        "# Candidate demo requests", "",
        banner + f"Source: `{source}` — {len(candidates)} candidates. "
        "Candidates are heuristic proposals (rule table in "
        "`xembench.flywheel.candidates`), not measured value.", "",
        "| task | embodiment | split | target failure | n | priority | "
        "est. cost/ep | policy source |",
        "|---|---|---|---|---|---|---|---|",
    ]
    groups: dict[tuple, list[CandidateDemoRequest]] = {}
    for c in candidates:
        groups.setdefault(
            (c.task, c.embodiment, c.split, c.target_failure_label), []
        ).append(c)
    for (task, emb, split, label), cs in sorted(groups.items()):
        lines.append(
            f"| {task} | {emb} | {split} | {label} | {len(cs)} | "
            f"{max(c.priority for c in cs):.2f} | "
            f"{cs[0].estimated_cost:.1f} | "
            f"{cs[0].requested_policy_source} |")
    lines += ["", "## Reasons", ""]
    for (task, emb, split, label), cs in sorted(groups.items()):
        lines.append(f"- **{task} / {split} / {label}** — {cs[0].reason}")
    return "\n".join(lines) + "\n"


def write_candidates(candidates: list[CandidateDemoRequest],
                     out_jsonl: str | Path, *, source: str,
                     synthetic: bool) -> list[Path]:
    out = Path(out_jsonl)
    written = [write_jsonl(candidates, out)]
    csv_path = out.with_suffix(".csv")
    candidates_dataframe(candidates).to_csv(csv_path, index=False)
    written.append(csv_path)
    md_path = out.with_suffix(".md")
    md_path.write_text(candidates_markdown(candidates, source, synthetic))
    written.append(md_path)
    return written


def load_candidates(path: str | Path) -> list[CandidateDemoRequest]:
    from xembench.flywheel.schemas import read_jsonl
    return read_jsonl(CandidateDemoRequest, path)


def eval_csv_is_synthetic(csv_path: str | Path) -> bool:
    return bool(toy_banner(load_episodes(csv_path)))
