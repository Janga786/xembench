#!/usr/bin/env python
"""Generate all report assets from an episode-level eval CSV.

    uv run python scripts/make_report_assets.py                       # real results
    uv run python scripts/make_report_assets.py --toy                 # synthetic demo
    uv run python scripts/make_report_assets.py --input <csv> --out-dir reports

Produces in --out-dir:
    summary.md, failures.md, transfer_matrix.csv + .md
    assets/success_by_cell.png        (skipped if matplotlib unavailable)

With --toy, outputs are prefixed `example_` and generated from labeled
synthetic data (outputs/toy/) — safe to commit as documentation examples.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xembench.analysis import failure_report, summarize_results, transfer_matrix
from xembench.analysis.episodes import load_episodes, toy_banner
from xembench.analysis.summarize_results import summarize
from xembench.utils.config import resolve_path


def plot_success_by_cell(df, out_png: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable — skipping plot (markdown/CSV still written)")
        return False

    s = summarize(df).sort_values("success_rate")
    labels = [
        f"{r.task}→{r.eval_task}\n{r.instruction_cond}/{r.color_cond}"
        if r.task != r.eval_task
        else f"{r.eval_task}\n{r.instruction_cond}/{r.color_cond}"
        for r in s.itertuples()
    ]
    err_low = s["success_rate"] - s["ci_low"]
    err_high = s["ci_high"] - s["success_rate"]

    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(s)), 4.5))
    ax.bar(range(len(s)), s["success_rate"], yerr=[err_low, err_high],
           capsize=3, color="#4878a8")
    ax.set_xticks(range(len(s)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("success rate")
    ax.set_ylim(0, 1)
    title = "Success by evaluation cell (95% Wilson CI)"
    if toy_banner(df):
        title += " — SYNTHETIC TOY DATA"
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/results/eval_episodes.csv")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--toy", action="store_true",
                        help="generate from synthetic data; prefix outputs 'example_'")
    args = parser.parse_args()

    if args.toy:
        from xembench.scripts.generate_toy_results import generate

        args.input = "outputs/toy/eval_episodes.csv"
        generate(Path(resolve_path(args.input)))

    inp = resolve_path(args.input)
    out_dir = Path(resolve_path(args.out_dir))
    prefix = "example_" if args.toy else ""

    df = load_episodes(inp)  # validates schema + labels before any writing
    summarize_results.main(["--input", inp, "--out", str(out_dir / f"{prefix}summary.md")])
    failure_report.main(["--input", inp, "--out", str(out_dir / f"{prefix}failures.md")])
    transfer_matrix.main(["--input", inp,
                          "--out", str(out_dir / f"{prefix}transfer_matrix.csv")])
    plot_success_by_cell(df, out_dir / "assets" / f"{prefix}success_by_cell.png")
    print(f"report assets in {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
