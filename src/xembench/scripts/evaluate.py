"""Run the evaluation matrix and write CSV + printed tables.

    uv run python -m xembench.scripts.evaluate --config configs/eval_matrix.yaml
    # smoke: --set tasks=[pick_cube_arm] --set episodes_native=5 ...
"""

import sys

from xembench.utils.config import common_parser, load_config, repo_root
from xembench.utils.seeding import seed_everything


def main() -> int:
    parser = common_parser(__doc__)
    args = parser.parse_args()
    cfg = load_config(args, default_path=repo_root() / "configs" / "eval_matrix.yaml")
    seed_everything(cfg["seed"])

    from xembench.analysis.episodes import append_episodes
    from xembench.eval.matrix import run_matrix
    from xembench.eval.report import append_rows, print_table

    results = run_matrix(cfg)
    if not results["cells"]:
        print("no policies evaluated (checkpoints missing?)")
        return 1
    df = append_rows(results["cells"],
                     repo_root() / cfg.get("csv", "data/results/eval.csv"))
    episodes_csv = cfg.get("episodes_csv")
    if episodes_csv:
        path = repo_root() / episodes_csv
        append_episodes(results["episodes"], path)
        print(f"wrote {len(results['episodes'])} episode records -> {path}")
    print()
    print_table(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
