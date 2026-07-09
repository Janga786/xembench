"""Flywheel schemas: validation, JSON round-trips, safety invariants."""

import pytest

from xembench.constants import A_CANON
from xembench.flywheel.schemas import (
    CandidateDemoRequest,
    DataQualityIssue,
    DatasetCard,
    DatasetIndexRecord,
    DemoEpisodeMetadata,
    DemoProvenance,
    FlywheelRoundManifest,
    ModelCardPlaceholder,
    SelectionPlan,
    derive_flywheel_split,
    from_json_dict,
    read_jsonl,
    require_valid,
    to_json_dict,
    write_jsonl,
)


def make_provenance(**over) -> DemoProvenance:
    base = dict(source_type="sim_expert", source_run_id="run0",
                created_at="2026-01-01T00:00:00+00:00", machine="mac")
    return DemoProvenance(**{**base, **over})


def make_episode(**over) -> DemoEpisodeMetadata:
    base = dict(
        episode_id="ep_001", task="pick_cube_arm", embodiment="panda",
        policy="motion_planning", split="train_seen",
        instruction="pick up the red cube and place it on the green target",
        seed=3, success=True, failure_label="success", steps=20,
        canonical_action_dim=A_CANON, native_action_dim=8,
        provenance=make_provenance())
    return DemoEpisodeMetadata(**{**base, **over})


# --- validity ---------------------------------------------------------------

def test_valid_objects_have_no_problems():
    assert make_provenance().validate() == []
    assert make_episode().validate() == []
    assert DataQualityIssue(issue_id="i1", severity="warning",
                            category="short_episode",
                            message="x").validate() == []


def test_episode_invalid_fields_all_reported():
    # seed=-1 is the valid "source seed unknown" sentinel (replayed demos);
    # anything below it is invalid
    ep = make_episode(embodiment="ur5", split="nope", steps=0, seed=-2,
                      failure_label="exploded")
    problems = " ".join(ep.validate())
    for frag in ["embodiment", "split", "steps", "seed", "failure_label"]:
        assert frag in problems


def test_unknown_seed_sentinel_is_valid():
    assert make_episode(seed=-1).validate() == []


def test_success_flag_must_match_failure_label():
    ep = make_episode(success=True, failure_label="timeout")
    assert any("disagree" in p for p in ep.validate())


def test_wrong_action_dims_rejected():
    assert any("canonical_action_dim" in p for p in
               make_episode(canonical_action_dim=20).validate())
    assert any("native_action_dim" in p for p in
               make_episode(native_action_dim=25).validate())  # panda != 25


def test_synthetic_toy_source_requires_synthetic_flag():
    prov = make_provenance(source_type="synthetic_toy", is_synthetic=False)
    assert any("never be labeled real" in p for p in prov.validate())


def test_missing_provenance_reported():
    assert any("provenance" in p for p in
               make_episode(provenance=None).validate())


def test_selection_plan_budget_and_duplicates():
    plan = SelectionPlan(plan_id="p", strategy="random", budget_episodes=2,
                         candidates_total=5, selected_ids=["a", "b", "b"],
                         created_at="2026-01-01T00:00:00+00:00")
    problems = " ".join(plan.validate())
    assert "selected > budget" in problems
    assert "duplicates" in problems


def test_dataset_card_synthetic_and_count_rules():
    card = DatasetCard(dataset_name="d", dataset_version="1",
                       intended_use="test", total_episodes=3,
                       successful_episodes=1, failed_episodes=1,
                       source_types=["synthetic_toy"], synthetic=False,
                       generated_at="2026-01-01T00:00:00+00:00")
    problems = " ".join(card.validate())
    assert "never be labeled real" in problems
    assert "!= total" in problems


def test_model_card_placeholder_refuses_metrics():
    mc = ModelCardPlaceholder(model_name="m", metrics={"sr": 0.9})
    assert any("must not carry metrics" in p for p in mc.validate())
    mc.metrics = {}
    assert mc.validate() == []


def test_round_manifest_stage_vocabulary():
    m = FlywheelRoundManifest(round_id="r0", stage="launched",
                              created_at="2026-01-01T00:00:00+00:00")
    assert any("stage" in p for p in m.validate())


def test_require_valid_raises_with_all_problems():
    with pytest.raises(ValueError, match="invalid DemoEpisodeMetadata"):
        require_valid(make_episode(steps=0, seed=-1))


# --- JSON round-trips --------------------------------------------------------

def test_episode_json_roundtrip_preserves_nested_provenance():
    ep = make_episode()
    clone = from_json_dict(DemoEpisodeMetadata, to_json_dict(ep))
    assert clone == ep
    assert isinstance(clone.provenance, DemoProvenance)


def test_from_json_dict_rejects_unknown_fields():
    data = to_json_dict(make_episode())
    data["surprise"] = 1
    with pytest.raises(ValueError, match="unknown fields"):
        from_json_dict(DemoEpisodeMetadata, data)


def test_jsonl_roundtrip(tmp_path):
    eps = [make_episode(episode_id=f"ep_{i}") for i in range(3)]
    path = tmp_path / "eps.jsonl"
    write_jsonl(eps, path)
    assert read_jsonl(DemoEpisodeMetadata, path) == eps


def test_write_jsonl_refuses_invalid_objects(tmp_path):
    with pytest.raises(ValueError, match="invalid"):
        write_jsonl([make_episode(steps=0)], tmp_path / "bad.jsonl")


def test_read_jsonl_missing_file_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="build it first"):
        read_jsonl(DemoEpisodeMetadata, tmp_path / "nope.jsonl")


def test_candidate_roundtrip_and_vocab():
    c = CandidateDemoRequest(
        candidate_id="cand_1", task="pick_cube_arm", embodiment="panda",
        split="heldout_paraphrase", target_failure_label="grasp_failure",
        reason="test", estimated_cost=1.0)
    assert c.validate() == []
    assert from_json_dict(CandidateDemoRequest, to_json_dict(c)) == c
    bad = CandidateDemoRequest(**{**c.__dict__, "status": "shipped",
                                  "estimated_cost": 0.0})
    problems = " ".join(bad.validate())
    assert "status" in problems and "estimated_cost" in problems


def test_index_record_score_bounds():
    r = DatasetIndexRecord(dataset_id="d", episode_id="e", task="t",
                           embodiment="panda", split="train_seen",
                           success=True, failure_label="success",
                           quality_score=1.5)
    assert any("outside [0, 1]" in p for p in r.validate())


# --- split derivation --------------------------------------------------------

def test_derive_flywheel_split_precedence():
    assert derive_flywheel_split("a", "b", "canonical", "train") \
        == "cross_embodiment"
    assert derive_flywheel_split("a", "a", "heldout", "unseen") \
        == "heldout_paraphrase"  # paraphrase beats color, documented
    assert derive_flywheel_split("a", "a", "canonical", "unseen") \
        == "heldout_color"
    assert derive_flywheel_split("a", "a", "canonical", "train") \
        == "train_seen"
