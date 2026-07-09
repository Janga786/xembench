"""Select candidates under a budget with a named strategy.

    python -m xembench.flywheel.select_candidates \
        --candidates artifacts/flywheel/candidates/round1_candidates.jsonl \
        --strategy failure_targeted --budget-episodes 50 \
        --out artifacts/flywheel/plans/round1_selection_plan.json

Writes the SelectionPlan JSON (+ .md sibling). Strategies:
random, stratified, failure_targeted, diversity_greedy,
cost_aware_failure_targeted (docs/physical_ai_data_flywheel.md). All are
deterministic given --seed.
"""

import argparse
import sys
from pathlib import Path

from xembench.flywheel._cli import guard_overwrite
from xembench.flywheel.candidates import load_candidates
from xembench.flywheel.schemas import SELECTION_STRATEGIES, save_json
from xembench.flywheel.selection import plan_markdown, select
from xembench.utils.config import resolve_path, sanitize_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", required=True,
                        help="candidates JSONL from generate_candidates")
    parser.add_argument("--strategy", required=True,
                        choices=list(SELECTION_STRATEGIES))
    parser.add_argument("--budget-episodes", type=int, required=True)
    parser.add_argument("--budget-cost", type=float, default=None,
                        help="optional cost ceiling "
                             "(cost_aware_failure_targeted)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True,
                        help="plan JSON path (a .md sibling is written)")
    parser.add_argument("--input-eval-csv", default="",
                        help="recorded in the plan for provenance")
    parser.add_argument("--dataset-index", default="",
                        help="recorded in the plan for provenance")
    parser.add_argument("--synthetic", action="store_true",
                        help="stamp the plan markdown as toy-derived")
    parser.add_argument("--notes", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    out = guard_overwrite(resolve_path(args.out), args.overwrite,
                          "selection plan")
    candidates = load_candidates(resolve_path(args.candidates))
    plan = select(
        candidates, strategy=args.strategy,
        budget_episodes=args.budget_episodes, budget_cost=args.budget_cost,
        seed=args.seed,
        input_eval_csv=sanitize_path(args.input_eval_csv)
        if args.input_eval_csv else "",
        input_dataset_index=sanitize_path(args.dataset_index)
        if args.dataset_index else "",
        notes=args.notes)
    save_json(plan, out)
    md = Path(out).with_suffix(".md")
    md.write_text(plan_markdown(plan, candidates, args.synthetic))
    print(f"wrote {sanitize_path(out)} and {sanitize_path(md)}")
    print(f"selected {len(plan.selected_ids)}/{plan.candidates_total} "
          f"candidates ({plan.strategy}, budget {plan.budget_episodes})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
