"""Dataset index: loaders (eval CSV + demo HDF5), writing, filtering,
and the validation layer's checks/scoring."""

import json

import h5py
import numpy as np
import pytest

from xembench.constants import A_CANON, IMG_SIZE, P_CANON
from xembench.flywheel.dataset_index import (
    build_index,
    episodes_from_demo_h5,
    episodes_from_eval_csv,
    filter_records,
    load_index,
    load_index_episodes,
    write_index,
)
from xembench.flywheel.toy_data import toy_demo_episodes, write_toy_eval_csv
from xembench.flywheel.validation import validate_index
from xembench.scripts.generate_toy_results import generate as gen_legacy_toy


@pytest.fixture()
def toy_csv(tmp_path):
    csv = tmp_path / "eval_episodes.csv"
    write_toy_eval_csv("bc_baseline", csv)
    return csv


def make_demo_h5(path, *, T=5, native_dim=8, bad_actions=False,
                 big_actions=False):
    with h5py.File(path, "w") as f:
        f.attrs.update(dict(schema_version=1, task="pick_cube_arm",
                            env_id="PickCubeLang-v1", embodiment="panda",
                            camera="base_camera",
                            control_mode="pd_joint_delta_pos",
                            mani_skill_version="3.0.1", source="test"))
        ep = f.create_group("ep_00000")
        ep.create_dataset("rgb", data=np.zeros((T, IMG_SIZE, IMG_SIZE, 3),
                                               dtype=np.uint8))
        ep.create_dataset("proprio", data=np.zeros((T, P_CANON), np.float32))
        canon = np.zeros((T, A_CANON), np.float32)
        if bad_actions:
            canon[0, 0] = np.nan
        if big_actions:
            canon[0, 0] = 3.0
        ep.create_dataset("action_canon", data=canon)
        ep.create_dataset("action_native",
                          data=np.zeros((T, native_dim), np.float32))
        ep.attrs.update(dict(
            instruction="pick up the red cube and place it on the green "
                        "target",
            success=True, seed=7, episode_len=T,
            slots=json.dumps({"color": "red", "object": "cube"})))
    return path


# --- loaders ------------------------------------------------------------------

def test_episodes_from_eval_csv_maps_columns(toy_csv):
    eps = episodes_from_eval_csv(toy_csv)
    assert len(eps) == 260  # 8 cells x 30 + 1 cell x 20
    splits = {ep.split for ep in eps}
    assert splits == {"train_seen", "heldout_paraphrase", "heldout_color",
                      "cross_embodiment"}
    # toy stamp forces synthetic provenance even without the flag
    assert all(ep.provenance.source_type == "synthetic_toy" for ep in eps)
    assert all(ep.provenance.is_synthetic for ep in eps)
    assert all(ep.validate() == [] for ep in eps)


def test_episodes_from_real_style_csv_are_sim_policy(tmp_path):
    csv = tmp_path / "eval.csv"
    gen_legacy_toy(csv, seed=0)  # stamped TOY_DATA -> still synthetic
    eps = episodes_from_eval_csv(csv)
    assert all(ep.provenance.source_type == "synthetic_toy" for ep in eps)


def test_episodes_from_demo_h5(tmp_path):
    h5 = make_demo_h5(tmp_path / "demo.h5")
    eps = episodes_from_demo_h5(h5)
    assert len(eps) == 1
    ep = eps[0]
    assert ep.task == "pick_cube_arm" and ep.embodiment == "panda"
    assert ep.native_action_dim == 8 and ep.steps == 5
    assert ep.object_condition == "cube" and ep.color_condition == "red"
    assert ep.split == "train_seen"
    assert ep.hdf5_path  # deep checks can find the file again
    assert ep.validate() == []


def test_demo_h5_missing_file_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="collect demos first"):
        episodes_from_demo_h5(tmp_path / "nope.h5")


# --- index write/load/filter ---------------------------------------------------

def test_index_roundtrip_and_sidecars(toy_csv, tmp_path):
    eps = episodes_from_eval_csv(toy_csv)
    records = build_index(eps, "ds0")
    out = tmp_path / "index.jsonl"
    written = write_index(records, eps, out)
    assert [p.name for p in written] == [
        "index.jsonl", "index.episodes.jsonl", "index.summary.csv"]
    assert load_index(out) == records
    assert load_index_episodes(out) == eps


def test_filtering(toy_csv):
    eps = episodes_from_eval_csv(toy_csv)
    records = build_index(eps, "ds0")
    picks = filter_records(records, task="pick_cube_arm",
                           split="heldout_paraphrase", success=False)
    assert picks and all(r.task == "pick_cube_arm"
                         and r.split == "heldout_paraphrase"
                         and not r.success for r in picks)
    assert filter_records(records, source_type="synthetic_toy") == records
    assert filter_records(records, source_type="human_teleop") == []
    assert filter_records(records, failure_label="timeout")


# --- validation ----------------------------------------------------------------

def test_validation_clean_toy_demos_pass():
    eps = toy_demo_episodes()
    result = validate_index(build_index(eps, "d"), eps)
    statuses = {r.validation_status for r in result.records}
    assert statuses <= {"passed", "passed_with_warnings"}
    # the two deliberate plants surface as warnings
    cats = {i.category for i in result.issues}
    assert {"short_episode", "suspicious_instruction"} <= cats


def test_validation_catches_missing_provenance():
    eps = toy_demo_episodes()[:3]
    eps[0].provenance = None
    result = validate_index(build_index(eps, "d"), eps)
    assert any(i.category == "missing_provenance" for i in result.issues)
    scored = {r.episode_id: r for r in result.records}
    assert scored[eps[0].episode_id].validation_status == "failed"


def test_validation_catches_synthetic_mislabeled_as_real():
    eps = toy_demo_episodes()[:3]
    eps[0].provenance = type(eps[0].provenance)(
        **{**eps[0].provenance.__dict__, "is_synthetic": False})
    result = validate_index(build_index(eps, "d"), eps)
    crits = [i for i in result.issues if i.severity == "critical"]
    assert crits and "never be labeled real" in crits[0].message
    scored = {r.episode_id: r for r in result.records}
    assert scored[eps[0].episode_id].quality_score == 0.0
    assert scored[eps[0].episode_id].validation_status == "failed"


def test_validation_catches_action_dim_mismatch():
    eps = toy_demo_episodes()[:3]
    eps[0].native_action_dim = 25  # G1 dim on a panda episode
    eps[1].canonical_action_dim = 7
    result = validate_index(build_index(eps, "d"), eps)
    cats = {i.category for i in result.issues}
    assert "embodiment_mismatch" in cats
    assert "action_shape_mismatch" in cats


def test_validation_catches_duplicate_episode_ids():
    eps = toy_demo_episodes()[:3]
    eps[1].episode_id = eps[0].episode_id
    result = validate_index(build_index(eps, "d"), eps)
    assert any(i.category == "duplicate_episode" for i in result.issues)


def test_validation_catches_language_leakage():
    eps = toy_demo_episodes()[:4]
    leak = type(eps[0])(**{**eps[0].__dict__,
                           "episode_id": "leak_1",
                           "split": "heldout_paraphrase",
                           "paraphrase_group": "heldout"})
    result = validate_index(build_index(eps + [leak], "d"), eps + [leak])
    assert any(i.category == "language_leakage" for i in result.issues)


def test_validation_catches_task_embodiment_mismatch():
    eps = toy_demo_episodes()[:2]
    eps[0].embodiment = "g1"          # pick_cube_arm is a panda task
    eps[0].native_action_dim = 25     # keep the schema itself happy
    result = validate_index(build_index(eps, "d"), eps)
    assert any(i.category == "embodiment_mismatch"
               and "registered for panda" in i.message
               for i in result.issues)


def test_deep_h5_maps_existing_validator_and_bounds(tmp_path):
    bad = make_demo_h5(tmp_path / "bad.h5", bad_actions=True)
    big = make_demo_h5(tmp_path / "big.h5", big_actions=True)
    eps = episodes_from_demo_h5(bad) + episodes_from_demo_h5(big)
    result = validate_index(build_index(eps, "d"), eps, deep_h5=True)
    cats = {i.category for i in result.issues}
    assert "nan_value" in cats            # from validate_demo_file
    assert "action_out_of_bounds" in cats  # flywheel bounds pass
    nan_issues = [i for i in result.issues if i.category == "nan_value"]
    assert all(i.severity == "critical" for i in nan_issues)


def test_quality_scores_follow_documented_rule():
    eps = toy_demo_episodes()
    result = validate_index(build_index(eps, "d"), eps)
    scored = {r.episode_id: r for r in result.records}
    warned = scored[eps[0].episode_id]  # the short-episode plant
    assert warned.quality_score == pytest.approx(0.9)
    assert warned.validation_status == "passed_with_warnings"
    clean = scored[eps[5].episode_id]
    assert clean.quality_score == 1.0
    assert clean.validation_status == "passed"
