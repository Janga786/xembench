"""Dataset validation: deterministic quality checks over a flywheel index.

Extends — never duplicates — the existing HDF5 validator: file-level checks
are delegated to ``xembench.data.schema.validate_demo_file`` and its findings
are mapped into DataQualityIssue records; this module adds the metadata-level
checks the H5 validator cannot know about (splits, leakage, provenance,
duplicates, synthetic labeling) plus a cheap action-bounds pass.

Scoring (documented rule, tested):

    critical present    -> quality_score 0.0, validation_status "failed"
    any error           -> -0.3 each,        validation_status "failed"
    warnings only       -> -0.1 each,        "passed_with_warnings"
    info                -> free
    clean               -> 1.0, "passed"

Deliberately no ML-based outlier detection — simple deterministic heuristics
first; anything cleverer belongs in a later phase with real data to tune on.

CLI: ``python -m xembench.flywheel.validate_dataset``.
"""

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from xembench.data.schema import validate_demo_file
from xembench.flywheel.schemas import (
    DataQualityIssue,
    DatasetIndexRecord,
    DemoEpisodeMetadata,
    stable_hash,
    to_json_dict,
    utc_now_iso,
)
from xembench.utils.config import load_yaml, repo_root, sanitize_path

# Episode-length heuristics (steps of pd_joint_delta_pos control).
MIN_REASONABLE_STEPS = 3
MAX_REASONABLE_STEPS = 1000

# ManiSkill normalizes pd_joint_delta_pos actions to [-1, 1]; a small slack
# absorbs float32 round-off without hiding genuinely unbounded actions.
ACTION_BOUND = 1.0 + 1e-4

SCORE_PENALTY = {"info": 0.0, "warning": 0.1, "error": 0.3}


@dataclass
class ValidationResult:
    issues: list[DataQualityIssue] = field(default_factory=list)
    records: list[DatasetIndexRecord] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def _task_registry() -> dict:
    """tasks.yaml task -> embodiment map; {} if the registry is unreadable."""
    try:
        cfg = load_yaml(repo_root() / "configs" / "tasks.yaml")
        return {name: spec.get("embodiment", "")
                for name, spec in (cfg.get("tasks") or {}).items()}
    except Exception:  # noqa: BLE001 — validation must not die on a bad yaml
        return {}


class _IssueCollector:
    def __init__(self) -> None:
        self.issues: list[DataQualityIssue] = []
        self._n = 0

    def add(self, severity: str, category: str, message: str, *,
            episode_id: str = "", field_name: str = "",
            suggested_fix: str = "") -> None:
        self._n += 1
        self.issues.append(DataQualityIssue(
            issue_id=f"iss_{self._n:05d}_{stable_hash(message, 6)}",
            severity=severity, category=category, message=message,
            episode_id=episode_id, field=field_name,
            suggested_fix=suggested_fix))


def _map_h5_error(msg: str) -> tuple[str, str]:
    """(severity, category) for a validate_demo_file finding."""
    if "non-finite" in msg:
        return "critical", "nan_value"
    if "rgb" in msg and ("missing" in msg or "shape" in msg):
        return "error", "missing_images" if "missing" in msg \
            else "observation_shape_mismatch"
    if "proprio" in msg and "shape" in msg:
        return "error", "observation_shape_mismatch"
    if "action" in msg and ("shape" in msg or "length" in msg):
        return "error", "action_shape_mismatch"
    if "missing" in msg:
        return "error", "missing_field"
    return "error", "unknown"


def _deep_check_h5(path: Path, collector: _IssueCollector) -> None:
    """Delegate to the existing validator, then bounds-check actions."""
    for msg in validate_demo_file(path):
        severity, category = _map_h5_error(msg)
        collector.add(severity, category,
                      f"{sanitize_path(path)}: {msg}",
                      suggested_fix="re-collect or drop the episode; see "
                      "xembench.data.schema for the expected layout")
    import h5py
    import numpy as np

    with h5py.File(path, "r") as f:
        for key in sorted(k for k in f.keys() if k.startswith("ep_")):
            ep = f[key]
            if "action_canon" not in ep:
                continue
            acts = np.asarray(ep["action_canon"])
            if acts.size and np.isfinite(acts).all() \
                    and np.abs(acts).max() > ACTION_BOUND:
                collector.add(
                    "warning", "action_out_of_bounds",
                    f"{sanitize_path(path)}/{key}: |action_canon| max "
                    f"{float(np.abs(acts).max()):.3f} > 1.0 "
                    "(pd_joint_delta_pos is normalized to [-1, 1])",
                    field_name="action_canon",
                    suggested_fix="check the adapter/expert clipping")


def _check_episode(ep: DemoEpisodeMetadata, collector: _IssueCollector,
                   task_embodiments: dict) -> None:
    eid = ep.episode_id or "(missing id)"

    # Schema problems, routed to the closest issue category.
    for problem in ep.validate():
        if "provenance" in problem:
            category = "missing_provenance"
        elif "native_action_dim" in problem or "embodiment" in problem:
            category = "embodiment_mismatch"
        elif "canonical_action_dim" in problem:
            category = "action_shape_mismatch"
        elif "success flag and failure_label" in problem:
            category = "failed_demo_labeled_success"
        elif "seed" in problem:
            category = "invalid_seed"
        elif "steps" in problem:
            category = "short_episode"
        else:
            category = "missing_field"
        collector.add("error", category, f"{eid}: {problem}",
                      episode_id=eid)

    # Synthetic mislabeling is the one non-negotiable: critical.
    if ep.provenance is not None:
        looks_synthetic = (ep.provenance.source_type == "synthetic_toy"
                           or "toy" in ep.provenance.generator_name.lower())
        if looks_synthetic and not ep.provenance.is_synthetic:
            collector.add(
                "critical", "missing_provenance",
                f"{eid}: toy-generated episode not marked is_synthetic — "
                "synthetic data can never be labeled real",
                episode_id=eid, field_name="provenance.is_synthetic",
                suggested_fix="regenerate with the toy generator (it stamps "
                "provenance) instead of hand-editing metadata")

    # Length heuristics beyond the schema's steps > 0.
    if 0 < ep.steps < MIN_REASONABLE_STEPS:
        collector.add("warning", "short_episode",
                      f"{eid}: {ep.steps} steps is suspiciously short",
                      episode_id=eid, field_name="steps")
    elif ep.steps > MAX_REASONABLE_STEPS:
        collector.add("warning", "long_episode",
                      f"{eid}: {ep.steps} steps is suspiciously long",
                      episode_id=eid, field_name="steps")

    # Instruction sanity.
    if ep.instruction.startswith("(instruction not recorded"):
        collector.add("warning", "suspicious_instruction",
                      f"{eid}: instruction text missing (v1 eval CSV?)",
                      episode_id=eid, field_name="instruction",
                      suggested_fix="re-run evaluation with the v2 episode "
                      "schema so instruction text is recorded")

    # Held-out split metadata must exist when the split says held-out.
    if ep.split == "heldout_paraphrase" and not ep.paraphrase_group:
        collector.add("warning", "missing_field",
                      f"{eid}: split heldout_paraphrase without a "
                      "paraphrase_group",
                      episode_id=eid, field_name="paraphrase_group")
    if ep.split == "heldout_color" and not ep.color_condition:
        collector.add("warning", "missing_field",
                      f"{eid}: split heldout_color without a color_condition",
                      episode_id=eid, field_name="color_condition")

    # Task registry vs claimed embodiment.
    expected = task_embodiments.get(ep.task)
    if expected and ep.embodiment != expected:
        collector.add(
            "error", "embodiment_mismatch",
            f"{eid}: task {ep.task} is registered for {expected}, episode "
            f"claims {ep.embodiment}",
            episode_id=eid, field_name="embodiment",
            suggested_fix="fix the loader that built this index entry")

    # Optional artifact paths: validate when present.
    if ep.video_path and not Path(ep.video_path).expanduser().exists() \
            and not (repo_root() / ep.video_path).exists():
        collector.add("info", "missing_field",
                      f"{eid}: video_path {ep.video_path!r} not found",
                      episode_id=eid, field_name="video_path")


def _check_dataset_level(episodes: list[DemoEpisodeMetadata],
                         collector: _IssueCollector) -> None:
    # Duplicate episode IDs.
    counts = Counter(ep.episode_id for ep in episodes)
    for eid, n in sorted(counts.items()):
        if n > 1:
            collector.add("error", "duplicate_episode",
                          f"episode_id {eid} appears {n} times",
                          episode_id=eid,
                          suggested_fix="episode IDs are content hashes — "
                          "duplicates usually mean the same input was "
                          "indexed twice")

    # Train/eval language leakage: identical instruction text in both the
    # train split and a held-out paraphrase split.
    by_split: dict[str, set[str]] = defaultdict(set)
    for ep in episodes:
        by_split[ep.split].add(stable_hash(ep.instruction))
    leaked = by_split.get("train_seen", set()) \
        & by_split.get("heldout_paraphrase", set())
    if leaked:
        collector.add(
            "error", "language_leakage",
            f"{len(leaked)} instruction(s) appear in BOTH train_seen and "
            "heldout_paraphrase — the held-out condition is compromised",
            suggested_fix="check the paraphrase tables "
            "(tests/test_paraphrase.py enforces disjointness at the "
            "language layer)")


def validate_index(records: list[DatasetIndexRecord],
                   episodes: list[DemoEpisodeMetadata] | None = None,
                   *, deep_h5: bool = False,
                   generated_at: str | None = None) -> ValidationResult:
    """Run all checks; returns issues + records with scores/statuses applied.

    ``episodes`` (the .episodes.jsonl sidecar) enables the metadata checks;
    without it only index-level checks run. ``deep_h5`` additionally opens
    every referenced HDF5 file (cheap on Mac for small files, skip for bulk).
    """
    collector = _IssueCollector()
    task_embodiments = _task_registry()

    if episodes:
        for ep in episodes:
            _check_episode(ep, collector, task_embodiments)
        _check_dataset_level(episodes, collector)
        if deep_h5:
            seen: set[str] = set()
            for ep in episodes:
                if not ep.hdf5_path or ep.hdf5_path in seen:
                    continue
                seen.add(ep.hdf5_path)
                h5 = Path(ep.hdf5_path)
                if not h5.is_absolute():
                    h5 = repo_root() / ep.hdf5_path
                if h5.exists():
                    _deep_check_h5(h5, collector)
                else:
                    collector.add("warning", "missing_field",
                                  f"hdf5_path {ep.hdf5_path!r} not found",
                                  episode_id=ep.episode_id,
                                  field_name="hdf5_path")
    else:
        for r in records:
            if not r.provenance_id:
                collector.add("error", "missing_provenance",
                              f"{r.episode_id}: index record has no "
                              "provenance_id",
                              episode_id=r.episode_id)
            if r.num_steps == 0:
                collector.add("error", "short_episode",
                              f"{r.episode_id}: zero-length episode",
                              episode_id=r.episode_id, field_name="num_steps")

    # Apply per-episode scores/statuses to the index records.
    per_episode: dict[str, list[DataQualityIssue]] = defaultdict(list)
    for issue in collector.issues:
        if issue.episode_id:
            per_episode[issue.episode_id].append(issue)

    scored = []
    for r in records:
        issues = per_episode.get(r.episode_id, [])
        severities = Counter(i.severity for i in issues)
        if severities["critical"]:
            score, status = 0.0, "failed"
        else:
            score = max(0.0, 1.0 - sum(SCORE_PENALTY[i.severity]
                                       for i in issues))
            if severities["error"]:
                status = "failed"
            elif severities["warning"]:
                status = "passed_with_warnings"
            else:
                status = "passed"
        scored.append(DatasetIndexRecord(
            **{**r.__dict__, "quality_score": round(score, 3),
               "validation_status": status}))

    status_counts = Counter(r.validation_status for r in scored)
    summary = {
        "generated_at": generated_at or utc_now_iso(),
        "n_records": len(scored),
        "n_episodes_checked": len(episodes) if episodes else 0,
        "deep_h5": deep_h5,
        "issues_by_severity": dict(Counter(i.severity
                                           for i in collector.issues)),
        "issues_by_category": dict(Counter(i.category
                                           for i in collector.issues)),
        "records_by_status": dict(status_counts),
        "mean_quality_score": round(
            sum(r.quality_score for r in scored) / len(scored), 4)
        if scored else None,
        "synthetic": bool(episodes) and any(
            ep.provenance and ep.provenance.is_synthetic for ep in episodes),
    }
    return ValidationResult(issues=collector.issues, records=scored,
                            summary=summary)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def report_markdown(result: ValidationResult, source: str) -> str:
    s = result.summary
    banner = ("> **SYNTHETIC TOY DATA — NOT REAL BENCHMARK RESULTS.** "
              "This validation report covers synthetic episodes only.\n\n"
              if s.get("synthetic") else "")
    lines = [
        "# Dataset validation report", "",
        banner + f"Source: `{source}` — {s['n_records']} index records, "
        f"{len(result.issues)} issues.", "",
        "## Summary", "",
        f"- records by status: "
        + (", ".join(f"{k}={v}" for k, v in
                     sorted(s["records_by_status"].items())) or "none"),
        f"- issues by severity: "
        + (", ".join(f"{k}={v}" for k, v in
                     sorted(s["issues_by_severity"].items())) or "none"),
        f"- mean quality score: {s['mean_quality_score']}",
        f"- deep HDF5 checks: {'on' if s['deep_h5'] else 'off'}", "",
        "Scoring rule: critical -> score 0 + failed; each error -0.3 "
        "(failed); each warning -0.1 (passed_with_warnings); info free.", "",
        "## Issues by category", "",
    ]
    if s["issues_by_category"]:
        lines += ["| category | count |", "|---|---|"]
        lines += [f"| {cat} | {n} |" for cat, n in
                  sorted(s["issues_by_category"].items(),
                         key=lambda kv: -kv[1])]
    else:
        lines.append("_No issues found._")
    if result.issues:
        lines += ["", "## Findings", ""]
        for issue in result.issues:
            fix = f" Fix: {issue.suggested_fix}" if issue.suggested_fix else ""
            lines.append(f"- **{issue.severity}** [{issue.category}] "
                         f"{issue.message}.{fix}")
    return "\n".join(lines) + "\n"


def write_validation_report(result: ValidationResult, out_md: str | Path,
                            source: str) -> tuple[Path, Path]:
    """Write <out>.md + sibling <out>.json; returns both paths."""
    out_md = Path(out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(report_markdown(result, source))
    out_json = out_md.with_suffix(".json")
    payload = {"summary": result.summary,
               "issues": [to_json_dict(i) for i in result.issues]}
    out_json.write_text(json.dumps(payload, indent=1) + "\n")
    return out_md, out_json
