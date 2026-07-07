"""Report-friendly figures from episode CSVs.

Data transforms are pure functions (unit-tested, no matplotlib import);
rendering is isolated in render_all() behind a guarded import so the module
works on machines without matplotlib (figures are skipped, never faked).
Synthetic data stamps every title — a screenshot can't silently drop the
label the way a caption can.
"""

from pathlib import Path

import pandas as pd

from xembench.analysis.stats import wilson_ci
from xembench.analysis.transfer_matrix import compute_transfer

SYNTHETIC_TITLE_SUFFIX = " — SYNTHETIC TOY DATA"


# ---------------------------------------------------------------- transforms
def retention_data(df: pd.DataFrame) -> pd.DataFrame:
    """Transfer pairs with native/transfer SR, drop, retention (sorted)."""
    matrix = compute_transfer(df)
    if len(matrix) == 0:
        return matrix
    matrix["label"] = (matrix["task"] + "→" + matrix["eval_task"]
                       + "\n" + matrix["policy_type"])
    return matrix.sort_values("label").reset_index(drop=True)


def success_by_group(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Success rate + Wilson CI per group, deterministic order."""
    rows = []
    for vals, g in df.groupby(keys, sort=True):
        vals = vals if isinstance(vals, tuple) else (vals,)
        n, k = len(g), int(g["success"].sum())
        lo, hi = wilson_ci(k, n)
        rows.append(dict(zip(keys, vals), n=n, success_rate=k / n,
                         ci_low=lo, ci_high=hi))
    return pd.DataFrame(rows)


def failure_counts(df: pd.DataFrame) -> pd.Series:
    """Counts per failure label among FAILED episodes, descending."""
    failures = df[df["success"] == 0]["failure_type"].value_counts()
    return failures.sort_values(ascending=False)


def task_success_table(df: pd.DataFrame) -> pd.DataFrame:
    """eval_task x policy_type mean success rate (heatmap input)."""
    return (df.groupby(["eval_task", "policy_type"])["success"].mean()
            .unstack(fill_value=float("nan")).sort_index())


# ----------------------------------------------------------------- rendering
FIGURE_NAMES = ["transfer_retention", "success_by_embodiment",
                "failure_counts", "task_success_heatmap"]


def render_all(df: pd.DataFrame, out_dir: str | Path,
               synthetic: bool) -> list[Path]:
    """Write PNG+SVG per figure; returns written paths ([] if matplotlib is
    unavailable). Skips figures whose data is empty rather than erroring."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable — figures skipped (all other pack "
              "contents unaffected)")
        return []

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = SYNTHETIC_TITLE_SUFFIX if synthetic else ""
    written: list[Path] = []

    def save(fig, name: str) -> None:
        fig.tight_layout()
        for ext in ("png", "svg"):
            path = out_dir / f"{name}.{ext}"
            fig.savefig(path, dpi=150, metadata={"Date": None}
                        if ext == "svg" else None)
            written.append(path)
        plt.close(fig)

    # 1. transfer retention
    ret = retention_data(df)
    if len(ret):
        fig, ax = plt.subplots(figsize=(max(5, 1.6 * len(ret)), 4.2))
        ax.bar(range(len(ret)), ret["sr_native"], width=0.38, label="native",
               color="#4878a8")
        ax.bar([i + 0.4 for i in range(len(ret))], ret["sr_transfer"],
               width=0.38, label="transfer (zero-shot)", color="#c88a4a")
        ax.set_xticks([i + 0.2 for i in range(len(ret))])
        ax.set_xticklabels(ret["label"], fontsize=8)
        ax.set_ylabel("success rate")
        ax.set_ylim(0, 1)
        ax.legend(frameon=False)
        ax.set_title("Cross-embodiment transfer: native vs zero-shot" + suffix,
                     fontsize=10, wrap=True)
        save(fig, "transfer_retention")

    # 2. success by eval embodiment x policy
    grp = success_by_group(df, ["eval_embodiment", "policy_type"])
    if len(grp):
        labels = grp["eval_embodiment"] + "\n" + grp["policy_type"]
        fig, ax = plt.subplots(figsize=(max(4, 1.2 * len(grp)), 4))
        yerr = [grp["success_rate"] - grp["ci_low"],
                grp["ci_high"] - grp["success_rate"]]
        ax.bar(range(len(grp)), grp["success_rate"], yerr=yerr, capsize=3,
               color="#4878a8")
        ax.set_xticks(range(len(grp)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("success rate")
        ax.set_ylim(0, 1)
        ax.set_title("Success by evaluation embodiment / policy (95% CI)"
                     + suffix, fontsize=10, wrap=True)
        save(fig, "success_by_embodiment")

    # 3. failure counts
    fails = failure_counts(df)
    if len(fails):
        fig, ax = plt.subplots(figsize=(max(4, 0.9 * len(fails)), 4))
        ax.bar(range(len(fails)), fails.values, color="#a85454")
        ax.set_xticks(range(len(fails)))
        ax.set_xticklabels(fails.index, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("failed episodes")
        ax.set_title("Failure types (failed episodes only)" + suffix,
                     fontsize=10, wrap=True)
        save(fig, "failure_counts")

    # 4. task x policy heatmap
    table = task_success_table(df)
    if table.size:
        fig, ax = plt.subplots(
            figsize=(1.6 + 1.1 * len(table.columns), 1.2 + 0.5 * len(table)))
        im = ax.imshow(table.values, cmap="Blues", vmin=0, vmax=1,
                       aspect="auto")
        ax.set_xticks(range(len(table.columns)))
        ax.set_xticklabels(table.columns, fontsize=8)
        ax.set_yticks(range(len(table)))
        ax.set_yticklabels(table.index, fontsize=8)
        for i in range(len(table)):
            for j in range(len(table.columns)):
                v = table.values[i, j]
                if v == v:  # not NaN
                    ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                            fontsize=8,
                            color="white" if v > 0.6 else "#1a2a3a")
        fig.colorbar(im, ax=ax, shrink=0.8, label="success rate")
        ax.set_title("Success rate: evaluation task × policy" + suffix,
                     fontsize=10, wrap=True)
        save(fig, "task_success_heatmap")

    return written
