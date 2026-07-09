"""Generate a complete, deterministic SYNTHETIC toy flywheel round.

    python -m xembench.flywheel.generate_toy_flywheel \
        --out artifacts/flywheel/toy_round [--overwrite]

Exercises the whole Mac-side loop on labeled toy data (see
xembench.flywheel.toy_data for the constructed story): baseline eval ->
dataset index -> validation -> dataset card -> candidates -> selection
(failure_targeted + random control) -> Linux command plan -> round-1 eval
CSVs for every variant -> round comparison -> model-card placeholder ->
round manifest. Every file is stamped SYNTHETIC; the output directory is
byte-for-byte reproducible at a fixed git commit.
"""

import argparse
import sys
from pathlib import Path

from xembench.analysis import failure_report
from xembench.artifacts.manifest import git_state
from xembench.flywheel._cli import guard_overwrite
from xembench.flywheel.candidates import generate_candidates, write_candidates
from xembench.flywheel.cards import (
    build_dataset_card,
    build_model_card_placeholder,
    card_markdown,
    model_card_markdown,
)
from xembench.flywheel.compare_rounds import write_comparison
from xembench.flywheel.dataset_index import (
    build_index,
    episodes_from_eval_csv,
    write_index,
)
from xembench.flywheel.experiment_plan import write_experiment_plan
from xembench.flywheel.schemas import FlywheelRoundManifest, save_json
from xembench.flywheel.selection import plan_markdown, select
from xembench.flywheel.toy_data import (
    TOY_TIMESTAMP,
    VARIANT_DELTAS,
    toy_demo_episodes,
    write_toy_eval_csv,
)
from xembench.flywheel.validation import validate_index, write_validation_report
from xembench.utils.config import resolve_path, sanitize_path

ROUND1_VARIANTS = [v for v in VARIANT_DELTAS if v != "bc_baseline"]
TARGETED = "bc_after_failure_targeted_data"
TOY_BUDGET = 50


def generate_toy_round(out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Baseline (round 0) eval CSV + failure report. All source labels
    # below are basenames: every file lives in this directory, and the
    # outputs stay byte-identical wherever the round is generated.
    round0_csv = write_toy_eval_csv("bc_baseline",
                                    out / "round0_eval_episodes.csv")
    from xembench.analysis.episodes import load_episodes
    df = load_episodes(round0_csv)
    (out / "round0_failure_report.md").write_text(
        failure_report.to_markdown(df, round0_csv.name))

    # 2. Dataset index: toy training demos + baseline eval rollouts.
    episodes = toy_demo_episodes() + episodes_from_eval_csv(
        round0_csv, synthetic=True, machine="mac",
        created_at=TOY_TIMESTAMP, source_label=round0_csv.name)
    records = build_index(episodes, dataset_id="toy_round0")
    index_path = out / "round0_dataset_index.jsonl"

    # 3. Validate, then write the index WITH scores applied.
    result = validate_index(records, episodes,
                            generated_at=TOY_TIMESTAMP)
    write_index(result.records, episodes, index_path)
    write_validation_report(result, out / "round0_validation_report.md",
                            index_path.name)

    # 4. Dataset card.
    card = build_dataset_card(result.records, episodes, result.summary,
                              name="toy_round0", version="0.1",
                              generated_at=TOY_TIMESTAMP)
    (out / "round0_dataset_card.md").write_text(card_markdown(card))
    save_json(card, out / "round0_dataset_card.json")

    # 5. Candidates from baseline failures (index boosts coverage).
    candidates = generate_candidates(df, result.records)
    cand_path = out / "round1_candidates.jsonl"
    write_candidates(candidates, cand_path,
                     source=round0_csv.name, synthetic=True)

    # 6. Selection: failure_targeted (the plan of record) + random control.
    plans = {}
    for strategy, fname in [("failure_targeted", "round1_selection_plan"),
                            ("random", "round1_selection_plan_random")]:
        plan = select(candidates, strategy=strategy,
                      budget_episodes=TOY_BUDGET, seed=0,
                      input_eval_csv=round0_csv.name,
                      input_dataset_index=index_path.name,
                      notes="SYNTHETIC toy round")
        plan.created_at = TOY_TIMESTAMP  # byte-reproducible toy outputs
        save_json(plan, out / f"{fname}.json")
        (out / f"{fname}.md").write_text(
            plan_markdown(plan, candidates, synthetic=True))
        plans[strategy] = plan

    # 7. Linux command plan (generated, never executed).
    write_experiment_plan(plans["failure_targeted"], candidates, out,
                          "round1", synthetic=True, overwrite=True)

    # 8. Round-1 eval CSVs for every variant of the toy story.
    for variant in ROUND1_VARIANTS:
        write_toy_eval_csv(variant, out / f"round1_eval_{variant}.csv")

    # 9. Comparison: baseline vs failure-targeted (the headline toy delta).
    after_csv = out / f"round1_eval_{TARGETED}.csv"
    write_comparison(round0_csv, after_csv, out / "round0_vs_round1.md",
                     demos_added=TOY_BUDGET,
                     before_label=round0_csv.name,
                     after_label=after_csv.name)

    # 10. Model-card placeholder (no metrics by construction).
    mc = build_model_card_placeholder(
        model_name="bc_after_failure_targeted_data (toy placeholder)",
        training_dataset="toy_round0 + round1 failure-targeted demos",
        policy_type="bc")
    (out / "model_card_placeholder.md").write_text(model_card_markdown(mc))
    save_json(mc, out / "model_card_placeholder.json")

    # 11. Round manifest ties the files together for the pack builder.
    manifest = FlywheelRoundManifest(
        round_id="toy_round1",
        previous_round_id="toy_round0",
        stage="evaluated",
        dataset_index_path=index_path.name,   # basenames: files live beside
        eval_csv_path=round0_csv.name,        # this manifest by construction
        failure_report_path="round0_failure_report.md",
        candidate_request_path=cand_path.name,
        selection_plan_path="round1_selection_plan.json",
        artifact_pack_path="",
        created_at=TOY_TIMESTAMP,
        git_commit=git_state()[0],
        is_synthetic=True,
        notes="SYNTHETIC TOY DATA — NOT REAL BENCHMARK RESULTS. Generated "
              "by xembench.flywheel.generate_toy_flywheel.")
    save_json(manifest, out / "round_manifest.json")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="artifacts/flywheel/toy_round")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    out = guard_overwrite(resolve_path(args.out), args.overwrite,
                          "toy round directory")
    generate_toy_round(out)
    n = len(list(Path(out).iterdir()))
    print(f"wrote SYNTHETIC toy flywheel round ({n} files) -> "
          f"{sanitize_path(out)}")
    print("Reminder: toy data — every number is fake by construction.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
