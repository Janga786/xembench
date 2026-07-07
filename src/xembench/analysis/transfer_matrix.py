"""Compute the cross-embodiment transfer matrix (the headline result).

    python -m xembench.analysis.transfer_matrix \
        --input data/results/eval_episodes.csv --out reports/transfer_matrix.csv

For each (policy, train task) with transfer episodes, pairs the native cell
against the transfer cell under matched instruction/color conditions and
reports:

    drop      = SR_native - SR_transfer            (percentage points)
    retention = SR_transfer / SR_native            (NaN when SR_native == 0)

Also prints/writes a markdown table next to the CSV.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from xembench.analysis.episodes import is_transfer, load_episodes, toy_banner
from xembench.analysis.stats import wilson_ci

MATCH_KEYS = ["policy_type", "task", "instruction_cond", "color_cond"]


def _rate(g: pd.DataFrame) -> tuple[float, int, float, float]:
    n, k = len(g), int(g["success"].sum())
    lo, hi = wilson_ci(k, n)
    return k / n, n, lo, hi


def compute_transfer(df: pd.DataFrame) -> pd.DataFrame:
    native = df[~is_transfer(df)]
    transfer = df[is_transfer(df)]
    rows = []
    for keys, tg in transfer.groupby(MATCH_KEYS, sort=True):
        cond = dict(zip(MATCH_KEYS, keys))
        ng = native
        for col, val in cond.items():
            ng = ng[ng[col] == val]
        if len(ng) == 0:
            # transfer cell without a matched native cell — surface it
            # rather than silently dropping (an eval-config smell)
            sr_n, n_n, nlo, nhi = float("nan"), 0, float("nan"), float("nan")
        else:
            sr_n, n_n, nlo, nhi = _rate(ng)
        sr_t, n_t, tlo, thi = _rate(tg)
        rows.append(dict(
            **cond,
            eval_task=tg["eval_task"].iloc[0],
            train_embodiment=tg["train_embodiment"].iloc[0],
            eval_embodiment=tg["eval_embodiment"].iloc[0],
            n_native=n_n, sr_native=sr_n, native_ci_low=nlo, native_ci_high=nhi,
            n_transfer=n_t, sr_transfer=sr_t, transfer_ci_low=tlo,
            transfer_ci_high=thi,
            drop=sr_n - sr_t,
            retention=(sr_t / sr_n) if sr_n and sr_n == sr_n else float("nan"),
        ))
    return pd.DataFrame(rows)


def to_markdown(matrix: pd.DataFrame) -> str:
    if len(matrix) == 0:
        return ("# Transfer matrix\n\nNo transfer episodes in the input — "
                "run evaluation with `transfer: true`.\n")
    show = matrix.copy()
    for col in ["sr_native", "sr_transfer", "drop", "retention"]:
        show[col] = show[col].map(
            lambda v: "-" if pd.isna(v) else f"{v:.0%}")
    cols = ["policy_type", "task", "eval_task", "train_embodiment",
            "eval_embodiment", "instruction_cond", "color_cond",
            "n_native", "sr_native", "n_transfer", "sr_transfer",
            "drop", "retention"]
    return ("# Transfer matrix\n\n`drop = SR_native - SR_transfer`; "
            "`retention = SR_transfer / SR_native`.\n\n"
            + show[cols].to_markdown(index=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True, help=".csv output path")
    args = parser.parse_args(argv)

    df = load_episodes(args.input)
    matrix = compute_transfer(df)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(out, index=False)
    md = toy_banner(df) + to_markdown(matrix)
    md_path = out.with_suffix(".md")
    md_path.write_text(md)
    print(f"wrote {out} and {md_path} ({len(matrix)} transfer pairings)")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
