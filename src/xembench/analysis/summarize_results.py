"""Summarize an episode-level eval CSV into a markdown report.

    python -m xembench.analysis.summarize_results \
        --input data/results/eval_episodes.csv --out reports/summary.md

Groups episodes by evaluation cell, reports success rates with Wilson CIs,
efficiency over successful episodes, and per-cell failure mixes.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from xembench.analysis.episodes import is_transfer, load_episodes, toy_banner
from xembench.analysis.stats import wilson_ci

CELL_KEYS = ["policy_type", "task", "eval_task", "eval_embodiment",
             "instruction_cond", "color_cond"]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(CELL_KEYS, sort=True):
        n, k = len(g), int(g["success"].sum())
        lo, hi = wilson_ci(k, n)
        succ = g[g["success"] == 1]
        failures = g[g["success"] == 0]["failure_type"].value_counts()
        rows.append(dict(
            zip(CELL_KEYS, keys),
            n_episodes=n,
            success_rate=k / n,
            ci_low=lo,
            ci_high=hi,
            avg_steps_to_success=succ["steps"].mean() if len(succ) else float("nan"),
            avg_steps_frac=(succ["steps"] / succ["max_steps"]).mean()
            if len(succ) else float("nan"),
            top_failure="; ".join(f"{t}:{c}" for t, c in failures.head(3).items())
            or "-",
        ))
    return pd.DataFrame(rows)


def to_markdown(summary: pd.DataFrame, df: pd.DataFrame, source: str) -> str:
    show = summary.copy()
    show["success"] = show.apply(
        lambda r: f"{r.success_rate:.0%} [{r.ci_low:.0%},{r.ci_high:.0%}]", axis=1)
    show["steps"] = show["avg_steps_to_success"].map(
        lambda v: "-" if pd.isna(v) else f"{v:.0f}")
    cols = CELL_KEYS + ["n_episodes", "success", "steps", "top_failure"]
    lines = [
        "# Evaluation summary",
        "",
        toy_banner(df) + f"Source: `{source}` — {len(df)} episodes, "
        f"{int(is_transfer(df).sum())} of them cross-embodiment transfer.",
        "",
        show[cols].to_markdown(index=False),
        "",
        "Columns: `task` = training task, `eval_task` = where evaluated "
        "(differing values = transfer cell); success shows a 95% Wilson CI; "
        "`steps` averages successful episodes only. "
        "See docs/metrics.md for formulas.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="episode-level eval CSV")
    parser.add_argument("--out", required=True, help="markdown output path")
    args = parser.parse_args(argv)

    df = load_episodes(args.input)
    report = to_markdown(summarize(df), df, args.input)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(f"wrote {out} ({len(df)} episodes summarized)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
