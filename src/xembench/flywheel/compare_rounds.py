"""Compare two flywheel rounds: per-cell success-rate deltas.

    python -m xembench.flywheel.compare_rounds \
        --before outputs/round0_eval.csv --after outputs/round1_eval.csv \
        --out artifacts/flywheel/comparisons/round0_vs_round1.md

Writes a markdown report (+ JSON sibling) of success-rate changes per eval
cell and per flywheel split. With --demos-added N it also reports the
flywheel's core efficiency metric, improvement per additional demonstration
(mean SR delta / N). Toy-stamped inputs are banner-labeled automatically.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from xembench.analysis.episodes import load_episodes, toy_banner
from xembench.flywheel._cli import guard_overwrite
from xembench.flywheel.schemas import derive_flywheel_split
from xembench.utils.config import resolve_path, sanitize_path

CELL_KEYS = ["eval_task", "eval_embodiment", "split", "instruction_cond",
             "color_cond"]


def _with_split(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["split"] = [derive_flywheel_split(t, et, ic, cc)
                   for t, et, ic, cc in zip(df["task"], df["eval_task"],
                                            df["instruction_cond"],
                                            df["color_cond"])]
    return df


def _cell_rates(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(CELL_KEYS)["success"]
    out = g.agg(n="count", sr="mean").reset_index()
    out["sr"] = out["sr"].round(4)
    return out


def compare(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """Per-cell SR table with before/after/delta (outer join on cells)."""
    b = _cell_rates(_with_split(before)).rename(
        columns={"n": "n_before", "sr": "sr_before"})
    a = _cell_rates(_with_split(after)).rename(
        columns={"n": "n_after", "sr": "sr_after"})
    m = b.merge(a, on=CELL_KEYS, how="outer")
    m["delta"] = (m["sr_after"] - m["sr_before"]).round(4)
    return m.sort_values(CELL_KEYS).reset_index(drop=True)


def split_summary(cells: pd.DataFrame) -> pd.DataFrame:
    """Episode-weighted per-split before/after summary."""
    rows = []
    for split, g in cells.dropna(subset=["delta"]).groupby("split"):
        wb = (g["sr_before"] * g["n_before"]).sum() / g["n_before"].sum()
        wa = (g["sr_after"] * g["n_after"]).sum() / g["n_after"].sum()
        rows.append({"split": split, "sr_before": round(float(wb), 4),
                     "sr_after": round(float(wa), 4),
                     "delta": round(float(wa - wb), 4),
                     "n_cells": len(g)})
    return pd.DataFrame(rows)


def comparison_payload(before: pd.DataFrame, after: pd.DataFrame,
                       demos_added: int | None) -> dict:
    cells = compare(before, after)
    splits = split_summary(cells)
    matched = cells.dropna(subset=["delta"])
    overall_delta = (
        float((matched["delta"] * matched["n_after"]).sum()
              / matched["n_after"].sum()) if len(matched) else None)
    payload = {
        "synthetic": bool(toy_banner(before)) or bool(toy_banner(after)),
        "n_cells": int(len(cells)),
        "n_cells_matched": int(len(matched)),
        "overall_delta": round(overall_delta, 4)
        if overall_delta is not None else None,
        "demos_added": demos_added,
        "improvement_per_demo": round(overall_delta / demos_added, 6)
        if overall_delta is not None and demos_added else None,
        "by_split": splits.to_dict(orient="records"),
        "by_cell": cells.to_dict(orient="records"),
    }
    return payload


def comparison_markdown(payload: dict, before_src: str, after_src: str,
                        ) -> str:
    banner = ("> **SYNTHETIC TOY DATA — NOT REAL BENCHMARK RESULTS.** "
              "This comparison was computed from synthetic eval CSVs.\n\n"
              if payload["synthetic"] else "")
    lines = [
        "# Round comparison", "",
        banner + f"Before: `{before_src}` — After: `{after_src}` — "
        f"{payload['n_cells_matched']}/{payload['n_cells']} cells matched.",
        "",
        f"**Overall SR delta (episode-weighted): "
        f"{payload['overall_delta']:+.4f}**"
        if payload["overall_delta"] is not None
        else "**No matched cells — nothing to compare.**",
    ]
    if payload["improvement_per_demo"] is not None:
        lines += ["",
                  f"Improvement per additional demonstration: "
                  f"**{payload['improvement_per_demo']:+.6f}** "
                  f"({payload['demos_added']} demos added). This is the "
                  "flywheel's core efficiency metric "
                  "(docs/flywheel_experiments.md)."]
    lines += ["", "## By flywheel split", "",
              "| split | SR before | SR after | delta | cells |",
              "|---|---|---|---|---|"]
    for r in payload["by_split"]:
        lines.append(f"| {r['split']} | {r['sr_before']:.3f} | "
                     f"{r['sr_after']:.3f} | {r['delta']:+.3f} | "
                     f"{r['n_cells']} |")
    lines += ["", "## By cell", "",
              "| eval task | embodiment | split | instr | color | "
              "SR before | SR after | delta |",
              "|---|---|---|---|---|---|---|---|"]
    for r in payload["by_cell"]:
        def _f(v):
            return "n/a" if v != v or v is None else f"{v:.3f}"
        lines.append(
            f"| {r['eval_task']} | {r['eval_embodiment']} | {r['split']} | "
            f"{r['instruction_cond']} | {r['color_cond']} | "
            f"{_f(r['sr_before'])} | {_f(r['sr_after'])} | "
            f"{_f(r['delta'])} |")
    return "\n".join(lines) + "\n"


def write_comparison(before_csv: str | Path, after_csv: str | Path,
                     out_md: str | Path,
                     demos_added: int | None = None, *,
                     before_label: str | None = None,
                     after_label: str | None = None) -> tuple[Path, Path]:
    """Labels override the displayed source paths (e.g. basenames when
    everything lives in one round directory)."""
    before = load_episodes(before_csv)
    after = load_episodes(after_csv)
    payload = comparison_payload(before, after, demos_added)
    out_md = Path(out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(comparison_markdown(
        payload, before_label or sanitize_path(before_csv),
        after_label or sanitize_path(after_csv)))
    out_json = out_md.with_suffix(".json")
    out_json.write_text(json.dumps(payload, indent=1) + "\n")
    return out_md, out_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--before", required=True,
                        help="episode-level eval CSV of the earlier round")
    parser.add_argument("--after", required=True,
                        help="episode-level eval CSV of the later round")
    parser.add_argument("--out", required=True,
                        help="markdown output (a .json sibling is written)")
    parser.add_argument("--demos-added", type=int, default=None,
                        help="demos added between rounds -> reports "
                             "improvement per demonstration")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    out = guard_overwrite(resolve_path(args.out), args.overwrite,
                          "comparison report")
    md, js = write_comparison(resolve_path(args.before),
                              resolve_path(args.after), out,
                              args.demos_added)
    print(f"wrote {sanitize_path(md)} and {sanitize_path(js)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
