"""Analysis tooling: taxonomy strictness, schema validation, exact math."""

import numpy as np
import pandas as pd
import pytest

from xembench.analysis.episodes import (
    EPISODE_COLUMNS,
    append_episodes,
    load_episodes,
    toy_banner,
)
from xembench.analysis.failure_report import failure_table
from xembench.analysis.stats import wilson_ci
from xembench.analysis.summarize_results import summarize
from xembench.analysis.taxonomy import (
    AUTO_LABELS,
    FAILURE_LABELS,
    classify_episode,
    validate_failure_type,
)
from xembench.analysis.transfer_matrix import compute_transfer
from xembench.scripts.generate_toy_results import generate


# ---------------------------------------------------------------- taxonomy
def test_taxonomy_accepts_only_known_labels():
    for label in FAILURE_LABELS:
        assert validate_failure_type(label) == label
    with pytest.raises(ValueError, match="unknown failure_type"):
        validate_failure_type("robot_exploded")


def test_auto_labels_are_a_subset():
    assert AUTO_LABELS <= FAILURE_LABELS


def test_classify_episode():
    assert classify_episode(success=True, truncated=True) == "success"
    assert classify_episode(success=False, truncated=True) == "timeout"
    assert classify_episode(success=False, truncated=False) == "unknown_failure"


# ------------------------------------------------------------------- stats
def test_wilson_ci_edges():
    assert wilson_ci(0, 0) == (0.0, 1.0)
    lo, hi = wilson_ci(0, 50)
    assert lo == 0.0 and 0 < hi < 0.1
    lo, hi = wilson_ci(50, 50)
    assert 0.9 < lo < 1.0 and hi == 1.0
    lo, hi = wilson_ci(40, 50)
    assert lo < 0.8 < hi


# ---------------------------------------------------------------- episodes
def _toy_df(tmp_path) -> pd.DataFrame:
    csv = tmp_path / "eval_episodes.csv"
    n = generate(csv, seed=0)
    df = load_episodes(csv)
    assert len(df) == n
    return df


def test_loader_rejects_missing_columns(tmp_path):
    csv = tmp_path / "bad.csv"
    pd.DataFrame({"task": ["a"], "success": [1]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_episodes(csv)


def test_loader_rejects_unknown_failure_labels(tmp_path):
    row = dict.fromkeys(EPISODE_COLUMNS, 0)
    row["failure_type"] = "not_a_label"
    csv = tmp_path / "bad_label.csv"
    pd.DataFrame([row]).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="unknown failure_type"):
        load_episodes(csv)


def test_loader_missing_file_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="generate_toy_results"):
        load_episodes(tmp_path / "nope.csv")


def test_append_orders_columns(tmp_path):
    rows = [dict.fromkeys(EPISODE_COLUMNS, 0) | {"failure_type": "timeout"}]
    csv = tmp_path / "out.csv"
    append_episodes(rows, csv)
    assert list(pd.read_csv(csv).columns) == EPISODE_COLUMNS


# --------------------------------------------------------------- summaries
def test_summarize_exact_rates(tmp_path):
    df = _toy_df(tmp_path)
    s = summarize(df)
    native_canon = s[(s.task == "pick_cube_arm") & (s.eval_task == "pick_cube_arm")
                     & (s.instruction_cond == "canonical") & (s.color_cond == "train")]
    assert len(native_canon) == 1
    assert native_canon.iloc[0].success_rate == pytest.approx(0.80)
    assert native_canon.iloc[0].n_episodes == 40


def test_failure_table_counts_sum_to_totals(tmp_path):
    df = _toy_df(tmp_path)
    table = failure_table(df, ["eval_task"])
    label_cols = [c for c in table.columns if c in FAILURE_LABELS]
    assert (table[label_cols].sum(axis=1) == table["total"]).all()
    assert table["total"].sum() == len(df)


def test_toy_banner_detection(tmp_path):
    df = _toy_df(tmp_path)
    assert "SYNTHETIC" in toy_banner(df)
    real = df.copy()
    real["checkpoint"] = "data/checkpoints/bc_x/best.pt"
    assert toy_banner(real) == ""


# ---------------------------------------------------------- transfer drop
def test_transfer_drop_exact(tmp_path):
    """Toy data is constructed with known rates: native 0.80 -> transfer 0.30
    (panda->g1) and native 0.70 -> transfer 0.25 (g1->panda)."""
    matrix = compute_transfer(_toy_df(tmp_path))
    assert len(matrix) == 2

    p2g = matrix[matrix.task == "pick_cube_arm"].iloc[0]
    assert p2g["eval_task"] == "apple_bowl_g1"
    assert p2g["sr_native"] == pytest.approx(0.80)
    assert p2g["sr_transfer"] == pytest.approx(0.30)
    # NOTE: subscript access — `.drop` is a pandas Series method
    assert p2g["drop"] == pytest.approx(0.50)
    assert p2g["retention"] == pytest.approx(0.375)

    g2p = matrix[matrix.task == "apple_bowl_g1"].iloc[0]
    assert g2p["drop"] == pytest.approx(0.45)


def test_transfer_matrix_flags_unmatched_native_cell():
    rows = []
    base = dict.fromkeys(EPISODE_COLUMNS, 0)
    for i in range(10):  # transfer episodes with NO matching native cell
        rows.append(base | dict(
            policy_type="bc", task="pick_cube_arm", eval_task="apple_bowl_g1",
            train_embodiment="panda", eval_embodiment="g1",
            instruction_cond="canonical", color_cond="train",
            episode_idx=i, success=i % 2, failure_type="timeout"))
    matrix = compute_transfer(pd.DataFrame(rows))
    assert len(matrix) == 1
    assert matrix.iloc[0].n_native == 0
    assert np.isnan(matrix.iloc[0].sr_native)  # surfaced, not silently dropped
