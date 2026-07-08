"""Toy flywheel pipeline: determinism, the constructed story, loud labels."""

import filecmp

import pytest

from xembench.analysis.episodes import load_episodes, toy_banner
from xembench.flywheel.compare_rounds import comparison_payload
from xembench.flywheel.generate_toy_flywheel import generate_toy_round
from xembench.flywheel.toy_data import (
    TOY_BANNER,
    VARIANT_DELTAS,
    toy_demo_episodes,
    toy_eval_rows,
    write_toy_eval_csv,
)


def _sr(df, **conds):
    for col, val in conds.items():
        df = df[df[col] == val]
    return df["success"].mean()


@pytest.fixture(scope="module")
def variant_csvs(tmp_path_factory):
    root = tmp_path_factory.mktemp("variants")
    return {v: write_toy_eval_csv(v, root / f"{v}.csv")
            for v in VARIANT_DELTAS}


def test_toy_eval_rows_deterministic():
    assert toy_eval_rows("bc_baseline") == toy_eval_rows("bc_baseline")
    assert toy_demo_episodes() == toy_demo_episodes()


def test_unknown_variant_fails_loudly():
    with pytest.raises(ValueError, match="unknown toy variant"):
        toy_eval_rows("bc_after_quantum_data")


def test_toy_csvs_are_stamped_synthetic(variant_csvs):
    for csv in variant_csvs.values():
        df = load_episodes(csv)
        assert toy_banner(df), "toy CSV must carry the TOY_DATA stamp"
        assert (csv.parent / "TOY_DATA_README.txt").exists()
        assert TOY_BANNER in (csv.parent / "TOY_DATA_README.txt").read_text()


def test_toy_story_ordering(variant_csvs):
    """The constructed hypothesis: targeted > random > baseline > filtered-
    recovers > noisy-hurts, sharpest on held-out and cross splits."""
    sr = {v: load_episodes(p)["success"].mean()
          for v, p in variant_csvs.items()}
    assert sr["bc_after_failure_targeted_data"] > sr["bc_after_random_data"]
    assert sr["bc_after_random_data"] > sr["bc_baseline"]
    assert sr["bc_after_noisy_data"] < sr["bc_baseline"]
    assert sr["bc_after_noisy_data"] < sr["bc_after_filtered_data"] \
        < sr["bc_after_random_data"]


def test_targeted_gains_concentrate_on_weighted_splits(variant_csvs):
    base = load_episodes(variant_csvs["bc_baseline"])
    targ = load_episodes(
        variant_csvs["bc_after_failure_targeted_data"])
    heldout_gain = (
        _sr(targ, instruction_cond="heldout")
        - _sr(base, instruction_cond="heldout"))
    seen_gain = (
        _sr(targ, instruction_cond="canonical", color_cond="train")
        - _sr(base, instruction_cond="canonical", color_cond="train"))
    assert heldout_gain > seen_gain


def test_comparison_payload_shows_positive_targeted_delta(variant_csvs):
    payload = comparison_payload(
        load_episodes(variant_csvs["bc_baseline"]),
        load_episodes(variant_csvs["bc_after_failure_targeted_data"]),
        demos_added=50)
    assert payload["synthetic"] is True
    assert payload["overall_delta"] > 0
    assert payload["improvement_per_demo"] == pytest.approx(
        payload["overall_delta"] / 50, abs=1e-6)
    by_split = {r["split"]: r["delta"] for r in payload["by_split"]}
    assert by_split["heldout_paraphrase"] > by_split["train_seen"]


def test_full_toy_round_is_deterministic(tmp_path):
    a = generate_toy_round(tmp_path / "a")
    b = generate_toy_round(tmp_path / "b")
    names_a = sorted(p.name for p in a.iterdir())
    names_b = sorted(p.name for p in b.iterdir())
    assert names_a == names_b
    # Byte-for-byte identical except for machine-local absolute paths that
    # appear when the round lives outside the repo (never in make targets).
    diff = []
    for name in names_a:
        if not filecmp.cmp(a / name, b / name, shallow=False):
            diff.append(name)
    assert diff == [], f"non-deterministic toy outputs: {diff}"


def test_toy_round_files_carry_banners(tmp_path):
    out = generate_toy_round(tmp_path / "round")
    for name in ["round0_dataset_card.md", "round1_candidates.md",
                 "round1_selection_plan.md", "round0_vs_round1.md",
                 "round0_validation_report.md", "round1_collection_plan.md"]:
        text = (out / name).read_text()
        assert "SYNTHETIC" in text, f"{name} missing synthetic banner"
    manifest = (out / "round_manifest.json").read_text()
    assert '"is_synthetic": true' in manifest
    sh = (out / "round1_linux_commands.sh").read_text()
    assert "SYNTHETIC TOY DATA" in sh and 'uname' in sh
