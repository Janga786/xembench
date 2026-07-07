"""Episode schema v2: record validation and v1->v2 load migration."""

import math

import pandas as pd
import pytest

from xembench.analysis.episodes import (
    CORE_COLUMNS,
    EPISODE_COLUMNS,
    EPISODE_SCHEMA_VERSION,
    OPTIONAL_COLUMNS,
    append_episodes,
    load_episodes,
    validate_episode_record,
)


def _valid_record() -> dict:
    return dict(
        policy_type="bc", task="pick_cube_arm", train_embodiment="panda",
        eval_task="pick_cube_arm", eval_embodiment="panda",
        instruction_cond="canonical", color_cond="train", seed=0,
        checkpoint="ckpt.pt", episode_idx=0, episode_seed=100000,
        success=1, steps=23, max_steps=50, failure_type="success",
    )


def test_schema_version_and_layout():
    assert EPISODE_SCHEMA_VERSION == 2
    assert EPISODE_COLUMNS[: len(CORE_COLUMNS)] == CORE_COLUMNS  # core order stable
    assert set(OPTIONAL_COLUMNS) & set(CORE_COLUMNS) == set()


def test_valid_record_passes():
    assert validate_episode_record(_valid_record()) == []
    # optional fields allowed
    rec = _valid_record() | {"instruction": "pick up the red cube",
                             "elapsed_s": 1.5}
    assert validate_episode_record(rec) == []


@pytest.mark.parametrize("mutation,fragment", [
    ({"failure_type": "gremlins"}, "unknown failure_type"),
    ({"success": 2}, "success must be"),
    ({"success": 0}, "disagree"),                       # success=0 but label 'success'
    ({"steps": 0}, "outside"),
    ({"steps": 51}, "outside"),
    ({"steps": "many"}, "integers"),
    ({"bogus_field": 1}, "unknown fields"),
])
def test_invalid_records_rejected(mutation, fragment):
    rec = _valid_record() | mutation
    errors = validate_episode_record(rec)
    assert any(fragment in e for e in errors), errors


def test_missing_core_field_reported():
    rec = _valid_record()
    del rec["episode_seed"]
    errors = validate_episode_record(rec)
    assert any("missing core fields" in e for e in errors)


def test_v1_csv_loads_with_migration(tmp_path):
    """A pre-v2 CSV (core columns only) must load; optional columns appear
    with defaults."""
    csv = tmp_path / "v1.csv"
    pd.DataFrame([{c: _valid_record()[c] for c in CORE_COLUMNS}]).to_csv(
        csv, index=False)
    df = load_episodes(csv)
    for col in OPTIONAL_COLUMNS:
        assert col in df.columns
    assert df.loc[0, "instruction"] == "" or pd.isna(df.loc[0, "instruction"])
    assert math.isnan(df.loc[0, "elapsed_s"])


def test_append_default_fills_optional(tmp_path):
    csv = tmp_path / "out.csv"
    append_episodes([_valid_record()], csv)  # record has NO optional fields
    df = pd.read_csv(csv)
    assert list(df.columns) == EPISODE_COLUMNS
    assert math.isnan(df.loc[0, "return_sum"])
