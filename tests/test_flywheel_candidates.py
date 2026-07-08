"""Candidate generation: failure->request rules, coarse-label fallback."""

import pytest

from xembench.analysis.episodes import load_episodes
from xembench.flywheel.candidates import (
    generate_candidates,
    load_candidates,
    write_candidates,
)
from xembench.flywheel.dataset_index import build_index, episodes_from_eval_csv
from xembench.flywheel.toy_data import write_toy_eval_csv


@pytest.fixture()
def toy_df(tmp_path):
    csv = tmp_path / "eval_episodes.csv"
    write_toy_eval_csv("bc_baseline", csv)
    return load_episodes(csv)


def test_candidates_generated_from_failures_only(toy_df):
    cands = generate_candidates(toy_df)
    assert cands
    assert all(c.target_failure_label != "success" for c in cands)
    assert all(c.validate() == [] for c in cands)
    n_failures = int((toy_df["success"] == 0).sum())
    assert len(cands) <= n_failures  # capped per group, never exceeds


def test_rule_table_grounding_vs_grasp(toy_df):
    cands = generate_candidates(toy_df)
    grounding = [c for c in cands
                 if c.target_failure_label == "wrong_object_selected"]
    grasp = [c for c in cands if c.target_failure_label == "grasp_failure"]
    assert grounding and grasp
    assert all("paraphrase" in c.constraints["vary"] for c in grounding)
    assert all(c.constraints["vary"] == ["initial_pose"] for c in grasp)
    assert all("target_color" in c.constraints["hold"] for c in grasp)


def test_timeout_marked_design_sensitive(toy_df):
    timeouts = [c for c in generate_candidates(toy_df)
                if c.target_failure_label == "timeout"]
    assert timeouts
    assert all(c.constraints.get("design_sensitive") for c in timeouts)


def test_coarse_labels_only_still_produce_candidates(toy_df):
    # Pre-Phase-B reality: the runner only assigns timeout/unknown_failure.
    df = toy_df.copy()
    df.loc[df["failure_type"] != "success", "failure_type"] = "timeout"
    df.loc[df.index[:10], "failure_type"] = "unknown_failure"
    df.loc[df.index[:10], "success"] = 0
    cands = generate_candidates(df)
    labels = {c.target_failure_label for c in cands}
    assert labels <= {"timeout", "unknown_failure"}
    unknown = [c for c in cands
               if c.target_failure_label == "unknown_failure"]
    assert unknown and all("balanced" in c.reason for c in unknown)


def test_cross_embodiment_failures_weighted_higher(toy_df):
    cands = generate_candidates(toy_df)
    by_split = {}
    for c in cands:
        by_split.setdefault(c.split, []).append(c.expected_value)
    # same base failure rates, but cross-embodiment carries a 2.0x weight
    assert max(by_split["cross_embodiment"]) > max(by_split["train_seen"])


def test_g1_candidates_cost_more_than_panda(toy_df):
    cands = generate_candidates(toy_df)
    g1 = [c.estimated_cost for c in cands if c.embodiment == "g1"]
    panda = [c.estimated_cost for c in cands if c.embodiment == "panda"]
    assert min(g1) > max(panda)


def test_index_boosts_underrepresented_tasks(toy_df, tmp_path):
    csv = tmp_path / "e.csv"
    write_toy_eval_csv("bc_baseline", csv)
    eps = [e for e in episodes_from_eval_csv(csv)
           if e.task != "stack_cube_arm"]  # stack has zero demos indexed
    records = build_index(eps, "d")
    with_idx = {c.candidate_id: c
                for c in generate_candidates(toy_df, records)}
    without = {c.candidate_id: c for c in generate_candidates(toy_df)}
    stack = [cid for cid, c in with_idx.items()
             if c.task == "stack_cube_arm"]
    assert stack
    assert all(with_idx[cid].expected_value > without[cid].expected_value
               for cid in stack)


def test_generation_deterministic(toy_df):
    assert generate_candidates(toy_df) == generate_candidates(toy_df)


def test_write_and_load_roundtrip(toy_df, tmp_path):
    cands = generate_candidates(toy_df)
    out = tmp_path / "cands.jsonl"
    written = write_candidates(cands, out, source="test", synthetic=True)
    assert [p.suffix for p in written] == [".jsonl", ".csv", ".md"]
    assert load_candidates(out) == cands
    md = (tmp_path / "cands.md").read_text()
    assert "SYNTHETIC TOY DATA — NOT REAL BENCHMARK RESULTS" in md
