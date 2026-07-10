#!/usr/bin/env python
"""Pick-cube data-scaling figure: demos vs native success for k=1 and k=8.

    uv run python scripts/make_scaling_figure.py --k8-10k 0.28
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# measured points (25-episode probes, canonical/train; see night3 logs +
# data/results/gonogo_chunk.csv / night3_probe.csv)
K1 = {200: 0.0, 2000: 0.0}
K8 = {200: 0.0, 2000: 0.04}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k8-10k", type=float, required=True,
                    help="measured success at 10k demos, k=8")
    ap.add_argument("--out", default="reports/assets/pick_scaling.png")
    args = ap.parse_args()
    k8 = {**K8, 10000: args.k8_10k}

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=160)
    for label, pts, color, marker in [
        ("single-step BC (k=1)", K1, "#9ec5f4", "o"),
        ("chunked BC (k=8)", k8, "#2a78d6", "s"),
    ]:
        xs = sorted(pts)
        ax.plot(xs, [pts[x] * 100 for x in xs], marker=marker, color=color,
                linewidth=2, markersize=7, label=label)
        for x in xs:
            ax.annotate(f"{pts[x]*100:.0f}%", (x, pts[x] * 100),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=9, color="#52514e")
    ax.set_xscale("log")
    ax.set_xticks([200, 2000, 10000], ["200", "2,000", "10,000"])
    ax.set_xlabel("demonstrations (log scale)")
    ax.set_ylabel("native success rate (%)")
    ax.set_title("pick_cube: data scaling × action chunking (25-episode probes)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
