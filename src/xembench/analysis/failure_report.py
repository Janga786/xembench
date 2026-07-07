"""Break down failure types across the axes that matter for the benchmark.

    python -m xembench.analysis.failure_report \
        --input data/results/eval_episodes.csv --out reports/failures.md

Produces failure-count tables by task, embodiment, policy, instruction/color
split, and train->eval embodiment direction.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from xembench.analysis.episodes import load_episodes, toy_banner

BREAKDOWNS: list[tuple[str, list[str]]] = [
    ("By evaluation task", ["eval_task"]),
    ("By evaluation embodiment", ["eval_embodiment"]),
    ("By policy", ["policy_type"]),
    ("By instruction / color split", ["instruction_cond", "color_cond"]),
    ("By train -> eval embodiment", ["train_embodiment", "eval_embodiment"]),
]


def failure_table(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Rows = group, columns = failure_type counts (success included as a
    column so the denominator is visible)."""
    table = (
        df.groupby(keys + ["failure_type"]).size().unstack(fill_value=0)
    )
    table["total"] = table.sum(axis=1)
    return table.reset_index()


def to_markdown(df: pd.DataFrame, source: str) -> str:
    lines = [
        "# Failure breakdown",
        "",
        toy_banner(df) + f"Source: `{source}` — {len(df)} episodes, "
        f"{int((df['success'] == 0).sum())} failures.",
        "",
        "Labels beyond success/timeout/unknown_failure/simulator_error "
        "require Phase B instrumentation (see docs/metrics.md); their "
        "absence in real runs is expected for now.",
    ]
    for title, keys in BREAKDOWNS:
        lines += ["", f"## {title}", "",
                  failure_table(df, keys).to_markdown(index=False)]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    df = load_episodes(args.input)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(to_markdown(df, args.input))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
