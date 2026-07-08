"""Turn a SelectionPlan into an executable-later Linux command plan.

    python -m xembench.flywheel.experiment_plan \
        --plan artifacts/flywheel/plans/round1_selection_plan.json \
        --candidates artifacts/flywheel/candidates/round1_candidates.jsonl \
        --out-dir artifacts/flywheel/plans --round-id round1

Writes ``<round>_collection_plan.md`` (human-readable, every command tagged
Mac-safe/Linux-only and collection/training/evaluation/reporting) and
``<round>_linux_commands.sh`` (guarded so it refuses to run on macOS).
NOTHING here executes anything — this module is the Mac-side bridge to the
Linux session, mirroring docs/phase_b_commands.md conventions.
"""

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from xembench.flywheel._cli import guard_overwrite
from xembench.flywheel.candidates import load_candidates
from xembench.flywheel.schemas import (
    CandidateDemoRequest,
    SelectionPlan,
    load_json,
)
from xembench.utils.config import resolve_path, sanitize_path


@dataclass
class PlannedCommand:
    command: str
    category: str        # collection | training | evaluation | reporting | validation
    mac_safe: bool
    note: str = ""

    @property
    def tags(self) -> str:
        where = "Mac-safe" if self.mac_safe else "Linux-only"
        return f"[{where} · {self.category}]"


def _collection_command(task: str, embodiment: str, source: str,
                        n_episodes: int, label: str) -> PlannedCommand:
    if source == "diagnostic":
        return PlannedCommand(
            command=(f"uv run python -m xembench.scripts.collect_demos "
                     f"--set task={task} --set policy=random "
                     f"--set episodes={n_episodes} "
                     f"--set success_only=false "
                     f"--set source=diagnostic_{label}"),
            category="collection", mac_safe=True,
            note=f"diagnostic episodes for {label} — debugging data, "
                 "never training demos")
    if embodiment in ("panda", "panda_stick"):
        return PlannedCommand(
            command=(f"uv run python -m xembench.experts.motion_plan_arm "
                     f"--config configs/presets/panda_demo_collection.yaml "
                     f"--set task={task} --set episodes={n_episodes} "
                     f"--set source=flywheel_{label}"),
            category="collection", mac_safe=False,
            note="mplib motion planning (Linux-only); shakedown at 5 "
                 "episodes first — docs/phase_b_commands.md step 5")
    return PlannedCommand(
        command=(f"uv run python -m xembench.scripts.collect_demos "
                 f"--config configs/presets/g1_expert_collection.yaml "
                 f"--set task={task} --set episodes={n_episodes} "
                 f"--set source=flywheel_{label}"),
        category="collection", mac_safe=False,
        note="requires a trained G1 PPO expert ≥70% first "
             "(docs/phase_b_commands.md step 9 — the schedule-risk gate)")


def planned_commands(plan: SelectionPlan,
                     candidates: list[CandidateDemoRequest],
                     round_id: str) -> list[PlannedCommand]:
    by_id = {c.candidate_id: c for c in candidates}
    missing = [i for i in plan.selected_ids if i not in by_id]
    if missing:
        raise ValueError(
            f"selection plan references {len(missing)} candidate id(s) not "
            f"in the candidates file (first: {missing[0]}) — plan and "
            "candidates must come from the same round")
    chosen = [by_id[i] for i in plan.selected_ids]

    groups: dict[tuple, int] = defaultdict(int)
    for c in chosen:
        groups[(c.task, c.embodiment, c.requested_policy_source,
                c.target_failure_label)] += 1

    cmds: list[PlannedCommand] = []
    for (task, emb, source, label), n in sorted(groups.items()):
        cmds.append(_collection_command(task, emb, source, n, label))

    # Post-collection validation (flywheel index + quality report).
    idx = f"artifacts/flywheel/indexes/{round_id}_collected_index.jsonl"
    cmds.append(PlannedCommand(
        command=("uv run python -m xembench.flywheel.build_dataset_index "
                 f"--demo-h5 'data/demos/**/flywheel_*.h5' --out {idx}"),
        category="validation", mac_safe=True,
        note="glob is expanded by the shell script below"))
    cmds.append(PlannedCommand(
        command=(f"uv run python -m xembench.flywheel.validate_dataset "
                 f"--index {idx} --deep-h5 "
                 f"--out artifacts/flywheel/validation/"
                 f"{round_id}_validation_report.md"),
        category="validation", mac_safe=True))

    # Training / evaluation / packaging placeholders (real presets exist;
    # exact demo lists get filled in on the Linux session).
    tasks = sorted({c.task for c in chosen
                    if c.requested_policy_source != "diagnostic"})
    for task in tasks:
        cmds.append(PlannedCommand(
            command=(f"uv run python -m xembench.scripts.train_bc "
                     f"--config configs/presets/bc_train_full.yaml "
                     f"--set task={task} "
                     f"--set out_dir=data/checkpoints/"
                     f"bc_{task}_{round_id}  "
                     f"# PLACEHOLDER: add --set demos='[...]' with the "
                     f"round's validated files"),
            category="training", mac_safe=False))
    cmds.append(PlannedCommand(
        command=("uv run python -m xembench.scripts.evaluate "
                 "--config configs/presets/full_eval_matrix.yaml "
                 f"--set csv=data/results/{round_id}_eval.csv "
                 f"--set episodes_csv=data/results/"
                 f"{round_id}_eval_episodes.csv  "
                 "# PLACEHOLDER: point policies at the round's checkpoints"),
        category="evaluation", mac_safe=False))
    cmds.append(PlannedCommand(
        command=(f"uv run python -m xembench.flywheel.compare_rounds "
                 f"--before data/results/round0_eval_episodes.csv "
                 f"--after data/results/{round_id}_eval_episodes.csv "
                 f"--demos-added {len(chosen)} "
                 f"--out artifacts/flywheel/comparisons/"
                 f"round0_vs_{round_id}.md"),
        category="reporting", mac_safe=True))
    cmds.append(PlannedCommand(
        command=(f"uv run python -m xembench.artifacts.build_pack "
                 f"--eval-csv data/results/{round_id}_eval_episodes.csv "
                 f"--out artifacts/releases/{round_id}_pack "
                 f"--release-name {round_id}_pack "
                 f"--configs configs/tasks.yaml"),
        category="reporting", mac_safe=True))
    return cmds


def plan_markdown(plan: SelectionPlan, cmds: list[PlannedCommand],
                  round_id: str, synthetic: bool) -> str:
    banner = ("> **SYNTHETIC TOY DATA — NOT REAL BENCHMARK RESULTS.** "
              "This plan was generated from a synthetic selection plan; "
              "the commands are real, the data behind them is not.\n\n"
              if synthetic else "")
    lines = [
        f"# Collection plan — {round_id}", "",
        banner + f"From selection plan `{plan.plan_id}` "
        f"(strategy **{plan.strategy}**, {len(plan.selected_ids)} episodes "
        "budgeted). Commands below are ORDERED; nothing on this list has "
        "been executed. Linux-only commands must not run on macOS.", "",
    ]
    for i, cmd in enumerate(cmds, 1):
        lines += [f"### {i}. {cmd.tags}", "", "```bash", cmd.command, "```"]
        if cmd.note:
            lines += ["", f"_{cmd.note}_"]
        lines.append("")
    return "\n".join(lines) + "\n"


def plan_shell_script(cmds: list[PlannedCommand], round_id: str,
                      synthetic: bool) -> str:
    lines = [
        "#!/usr/bin/env bash",
        f"# {round_id} flywheel commands — GENERATED PLAN, review before "
        "running.",
        "# Linux-only: refuses to run on macOS (mplib/CUDA steps below).",
    ]
    if synthetic:
        lines.append("# SYNTHETIC TOY DATA — NOT REAL BENCHMARK RESULTS: "
                     "this plan was generated from toy inputs.")
    lines += [
        "set -euo pipefail",
        'if [[ "$(uname)" == "Darwin" ]]; then',
        '  echo "This plan contains Linux-only steps (mplib, CUDA);'
        ' run it on the Linux machine." >&2',
        "  exit 1",
        "fi",
        "",
    ]
    for cmd in cmds:
        lines.append(f"# {cmd.tags}" + (f" {cmd.note}" if cmd.note else ""))
        if "PLACEHOLDER" in cmd.command:
            lines.append(f"# {cmd.command}")
            lines.append("#   ^ uncomment after filling the placeholder in")
        else:
            lines.append(cmd.command)
        lines.append("")
    return "\n".join(lines) + "\n"


def write_experiment_plan(plan: SelectionPlan,
                          candidates: list[CandidateDemoRequest],
                          out_dir: str | Path, round_id: str, *,
                          synthetic: bool,
                          overwrite: bool = False) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = guard_overwrite(out_dir / f"{round_id}_collection_plan.md",
                              overwrite, "collection plan")
    sh_path = guard_overwrite(out_dir / f"{round_id}_linux_commands.sh",
                              overwrite, "command script")
    cmds = planned_commands(plan, candidates, round_id)
    md_path.write_text(plan_markdown(plan, cmds, round_id, synthetic))
    sh_path.write_text(plan_shell_script(cmds, round_id, synthetic))
    sh_path.chmod(0o755)
    return md_path, sh_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", required=True,
                        help="selection plan JSON from select_candidates")
    parser.add_argument("--candidates", required=True,
                        help="candidates JSONL from generate_candidates")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--round-id", required=True, help="e.g. round1")
    parser.add_argument("--synthetic", action="store_true",
                        help="stamp the plan as derived from toy data")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    plan = load_json(SelectionPlan, resolve_path(args.plan))
    candidates = load_candidates(resolve_path(args.candidates))
    md, sh = write_experiment_plan(
        plan, candidates, resolve_path(args.out_dir), args.round_id,
        synthetic=args.synthetic, overwrite=args.overwrite)
    print(f"wrote {sanitize_path(md)} and {sanitize_path(sh)}")
    print("Nothing was executed — review the plan, then run the script on "
          "the Linux machine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
