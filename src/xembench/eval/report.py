"""Result sink: append rows to data/results/eval.csv and print pivot tables."""

from pathlib import Path

import pandas as pd

CSV_COLUMNS = [
    "policy_type", "task", "train_embodiment", "eval_task", "eval_embodiment",
    "instruction_cond", "color_cond", "n_episodes", "n_success", "success_rate",
    "ci_low", "ci_high", "avg_steps_to_success", "avg_steps_frac",
    "seed", "checkpoint",
]


def append_rows(rows: list[dict], csv_path: str | Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)[CSV_COLUMNS]
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)
    return df


def print_table(df: pd.DataFrame) -> None:
    """The user-facing summary: one line per eval cell, grouped per policy."""
    show = df.copy()
    show["success"] = show.apply(
        lambda r: f"{r.success_rate:.0%} [{r.ci_low:.0%},{r.ci_high:.0%}]", axis=1
    )
    show["steps"] = show["avg_steps_to_success"].map(
        lambda v: "-" if pd.isna(v) else f"{v:.0f}"
    )
    cols = ["policy_type", "task", "eval_task", "eval_embodiment",
            "instruction_cond", "color_cond", "n_episodes", "success", "steps"]
    print(show[cols].to_markdown(index=False))

    # Headline: native vs transfer per (policy, task)
    native = show[show.task == show.eval_task]
    transfer = show[show.task != show.eval_task]
    if len(transfer):
        print("\n== Transfer summary (success-rate drop) ==")
        for _, t in transfer.iterrows():
            n = native[
                (native.task == t.task)
                & (native.policy_type == t.policy_type)
                & (native.instruction_cond == t.instruction_cond)
                & (native.color_cond == t.color_cond)
            ]
            if len(n):
                drop = n.iloc[0].success_rate - t.success_rate
                print(f"{t.policy_type} {t.task} -> {t.eval_task} "
                      f"[{t.instruction_cond}/{t.color_cond}]: "
                      f"{n.iloc[0].success_rate:.0%} -> {t.success_rate:.0%} "
                      f"(drop {drop:+.0%})")
