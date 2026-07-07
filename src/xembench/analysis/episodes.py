"""Per-episode evaluation record schema: the canonical analysis input.

One row per evaluated episode, written by the eval matrix alongside the
aggregate CSV. Everything in xembench.analysis consumes this format.

Schema v2 (2026-07-07): the v1 core columns are unchanged; OPTIONAL columns
were appended (instruction text, return, timing, artifact paths) so the
Phase B runner can record richer traces. Migration is automatic: v1 CSVs
load fine — missing optional columns are filled with defaults on load, and
`append_episodes` fills them on write. Never reorder or remove core columns;
add new ones to OPTIONAL_COLUMNS only.
"""

import math
from pathlib import Path

import pandas as pd

from xembench.analysis.taxonomy import FAILURE_LABELS

EPISODE_SCHEMA_VERSION = 2

# v1 core: identity columns shared with the aggregate CSV + per-episode facts.
CORE_COLUMNS = [
    "policy_type", "task", "train_embodiment", "eval_task", "eval_embodiment",
    "instruction_cond", "color_cond", "seed", "checkpoint",
    "episode_idx", "episode_seed", "success", "steps", "max_steps",
    "failure_type",
]

# v2 optional: default-filled when absent (old CSVs, minimal writers).
OPTIONAL_COLUMNS = {
    "instruction": "",        # exact (possibly paraphrased) text shown to the policy
    "template_family": "",    # instruction family from the task registry
    "return_sum": math.nan,   # episode reward sum (env reward mode dependent)
    "elapsed_s": math.nan,    # wall-clock seconds for the episode
    "video_path": "",         # recorded rollout, if any (repo-relative)
    "config_path": "",        # eval config snapshot, if any (repo-relative)
}

EPISODE_COLUMNS = CORE_COLUMNS + list(OPTIONAL_COLUMNS)


def validate_episode_record(record: dict) -> list[str]:
    """Pre-write validation for a single episode record; [] = valid.

    Use before append_episodes when writing records from new code paths —
    a malformed row caught here is one that never poisons an analysis run.
    """
    errors = []
    missing = [c for c in CORE_COLUMNS if c not in record]
    if missing:
        errors.append(f"missing core fields {missing}")
        return errors  # dependent checks would just cascade
    unknown = [k for k in record if k not in EPISODE_COLUMNS]
    if unknown:
        errors.append(f"unknown fields {unknown} (add to OPTIONAL_COLUMNS, "
                      "never ad hoc)")
    if record["failure_type"] not in FAILURE_LABELS:
        errors.append(f"unknown failure_type {record['failure_type']!r}")
    success = record["success"]
    if success not in (0, 1, True, False):
        errors.append(f"success must be 0/1/bool, got {success!r}")
    else:
        if bool(success) != (record["failure_type"] == "success"):
            errors.append("success flag and failure_type disagree")
    try:
        if not (0 < int(record["steps"]) <= int(record["max_steps"])):
            errors.append(
                f"steps {record['steps']} outside (0, max_steps="
                f"{record['max_steps']}]")
    except (TypeError, ValueError):
        errors.append("steps/max_steps must be integers")
    return errors


def append_episodes(rows: list[dict], csv_path: str | Path) -> pd.DataFrame:
    rows = [dict(OPTIONAL_COLUMNS) | r for r in rows]  # default-fill v2 fields
    df = pd.DataFrame(rows)[EPISODE_COLUMNS]
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)
    return df


def load_episodes(csv_path: str | Path) -> pd.DataFrame:
    """Load + validate an episode CSV; fails loudly with actionable errors."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"episode CSV not found: {path}. Produce one with "
            "`python -m xembench.scripts.evaluate` (episodes_csv config key) "
            "or `python -m xembench.scripts.generate_toy_results` for a "
            "synthetic example."
        )
    df = pd.read_csv(path)
    missing = [c for c in CORE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} is missing core columns {missing}; expected the episode "
            f"schema {CORE_COLUMNS} (see docs/metrics.md)."
        )
    for col, default in OPTIONAL_COLUMNS.items():  # v1 -> v2 migration on load
        if col not in df.columns:
            df[col] = default
    bad = set(df["failure_type"].unique()) - FAILURE_LABELS
    if bad:
        raise ValueError(
            f"{path} contains unknown failure_type values {sorted(bad)}; "
            "allowed labels live in xembench.analysis.taxonomy.FailureType."
        )
    df["success"] = df["success"].astype(int)
    return df


def is_transfer(df: pd.DataFrame) -> pd.Series:
    return df["task"] != df["eval_task"]


def toy_banner(df: pd.DataFrame) -> str:
    """Loud banner when the data is synthetic (generate_toy_results stamps
    the checkpoint column); empty string otherwise."""
    if df["checkpoint"].astype(str).str.contains("TOY_DATA").any():
        return ("> **SYNTHETIC TOY DATA** — generated by "
                "`xembench.scripts.generate_toy_results` to exercise the "
                "analysis pipeline. Every number below is fake.\n\n")
    return ""
