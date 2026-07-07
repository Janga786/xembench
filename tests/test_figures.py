"""Figure DATA transforms (pure) + a render smoke test when matplotlib exists."""

import importlib.util

import pytest

from xembench.analysis.episodes import load_episodes
from xembench.artifacts.figures import (
    failure_counts,
    render_all,
    retention_data,
    success_by_group,
    task_success_table,
)
from xembench.scripts.generate_toy_results import generate


@pytest.fixture()
def toy_df(tmp_path):
    csv = tmp_path / "eval_episodes.csv"
    generate(csv, seed=0)
    return load_episodes(csv)


def test_retention_data_matches_transfer_math(toy_df):
    ret = retention_data(toy_df)
    assert len(ret) == 2
    row = ret[ret.task == "pick_cube_arm"].iloc[0]
    assert row["sr_native"] == pytest.approx(0.80)
    assert row["retention"] == pytest.approx(0.375)
    assert "→" in row["label"]


def test_success_by_group_rates(toy_df):
    grp = success_by_group(toy_df, ["eval_embodiment"])
    panda = grp[grp.eval_embodiment == "panda"].iloc[0]
    # panda cells: 0.80, 0.65, 0.55 native (40 each) + 0.25 transfer (40)
    assert panda["n"] == 160
    assert panda["success_rate"] == pytest.approx((32 + 26 + 22 + 10) / 160)
    assert panda["ci_low"] < panda["success_rate"] < panda["ci_high"]


def test_failure_counts_exclude_success(toy_df):
    fails = failure_counts(toy_df)
    assert "success" not in fails.index
    assert fails.sum() == int((toy_df["success"] == 0).sum())
    assert list(fails.values) == sorted(fails.values, reverse=True)


def test_task_success_table_shape(toy_df):
    table = task_success_table(toy_df)
    assert set(table.index) == {"pick_cube_arm", "apple_bowl_g1"}
    assert list(table.columns) == ["bc"]
    assert 0 <= table.values.min() and table.values.max() <= 1


@pytest.mark.skipif(importlib.util.find_spec("matplotlib") is None,
                    reason="matplotlib not installed (CI)")
def test_render_all_writes_png_and_svg(toy_df, tmp_path):
    written = render_all(toy_df, tmp_path / "figs", synthetic=True)
    names = {p.name for p in written}
    assert "transfer_retention.png" in names
    assert "transfer_retention.svg" in names
    assert all(p.stat().st_size > 0 for p in written)
    # synthetic stamp reaches the SVG text
    svg = (tmp_path / "figs" / "transfer_retention.svg").read_text()
    assert "SYNTHETIC" in svg
