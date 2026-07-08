"""Selection strategies: determinism, budgets, targeting, diversity."""

import pytest

from xembench.analysis.episodes import load_episodes
from xembench.flywheel.candidates import generate_candidates
from xembench.flywheel.schemas import CandidateDemoRequest
from xembench.flywheel.selection import (
    _score,
    apply_selection,
    plan_markdown,
    select,
)
from xembench.flywheel.toy_data import write_toy_eval_csv

BUDGET = 30


@pytest.fixture(scope="module")
def pool(tmp_path_factory):
    csv = tmp_path_factory.mktemp("sel") / "eval.csv"
    write_toy_eval_csv("bc_baseline", csv)
    return generate_candidates(load_episodes(csv))


def _by_id(pool):
    return {c.candidate_id: c for c in pool}


@pytest.mark.parametrize("strategy", ["random", "stratified",
                                      "failure_targeted", "diversity_greedy",
                                      "cost_aware_failure_targeted"])
def test_budget_ids_and_uniqueness(pool, strategy):
    plan = select(pool, strategy=strategy, budget_episodes=BUDGET, seed=0)
    assert len(plan.selected_ids) <= BUDGET
    assert len(set(plan.selected_ids)) == len(plan.selected_ids)
    known = set(_by_id(pool))
    assert set(plan.selected_ids) <= known
    assert plan.validate() == []
    assert plan.strategy == strategy


def test_random_deterministic_with_seed(pool):
    a = select(pool, strategy="random", budget_episodes=BUDGET, seed=7)
    b = select(pool, strategy="random", budget_episodes=BUDGET, seed=7)
    c = select(pool, strategy="random", budget_episodes=BUDGET, seed=8)
    assert a.selected_ids == b.selected_ids
    assert a.selected_ids != c.selected_ids


def test_failure_targeted_beats_random_on_targeting(pool):
    by_id = _by_id(pool)

    def total_score(plan):
        return sum(_score(by_id[i]) for i in plan.selected_ids)

    targeted = select(pool, strategy="failure_targeted",
                      budget_episodes=BUDGET, seed=0)
    random_ = select(pool, strategy="random", budget_episodes=BUDGET, seed=0)
    assert total_score(targeted) > total_score(random_)


def test_failure_targeted_prefers_high_failure_groups(pool):
    by_id = _by_id(pool)
    plan = select(pool, strategy="failure_targeted", budget_episodes=10,
                  seed=0)
    chosen = [by_id[i] for i in plan.selected_ids]
    # top picks come from weighted splits (cross-embodiment / heldout)
    assert all(c.split != "train_seen" for c in chosen[:5])


def test_diversity_greedy_covers_more_groups_than_top_score():
    # Constructed pool: one dominant-score group + many small groups.
    def cand(i, task, split, label, priority):
        return CandidateDemoRequest(
            candidate_id=f"cand_{i:03d}", task=task, embodiment="panda",
            split=split, target_failure_label=label, reason="t",
            estimated_cost=1.0, priority=priority, expected_value=0.1)

    pool = [cand(i, "pick_cube_arm", "train_seen", "timeout", 100.0)
            for i in range(10)]
    pool += [cand(10 + i, t, s, lbl, 1.0)
             for i, (t, s, lbl) in enumerate(
                 (t, s, lbl)
                 for t in ("stack_cube_arm", "pull_tool_arm")
                 for s in ("heldout_paraphrase", "heldout_color")
                 for lbl in ("grasp_failure", "placement_failure"))]

    def groups(ids):
        by_id = {c.candidate_id: c for c in pool}
        return {(by_id[i].task, by_id[i].split,
                 by_id[i].target_failure_label) for i in ids}

    naive = sorted(pool, key=lambda c: (-_score(c), c.candidate_id))[:8]
    diverse = select(pool, strategy="diversity_greedy", budget_episodes=8,
                     seed=0)
    assert len(groups(diverse.selected_ids)) \
        > len(groups([c.candidate_id for c in naive]))


def test_stratified_covers_every_group_before_repeating(pool):
    by_id = _by_id(pool)
    groups = {(c.task, c.embodiment, c.split, c.target_failure_label)
              for c in pool}
    plan = select(pool, strategy="stratified",
                  budget_episodes=len(groups), seed=0)
    chosen_groups = {(by_id[i].task, by_id[i].embodiment, by_id[i].split,
                      by_id[i].target_failure_label)
                     for i in plan.selected_ids}
    assert chosen_groups == groups


def test_cost_aware_respects_cost_budget(pool):
    by_id = _by_id(pool)
    plan = select(pool, strategy="cost_aware_failure_targeted",
                  budget_episodes=100, budget_cost=20.0, seed=0)
    spent = sum(by_id[i].estimated_cost for i in plan.selected_ids)
    assert spent <= 20.0
    assert plan.budget_cost == 20.0


def test_unknown_strategy_and_empty_pool_fail_loudly(pool):
    with pytest.raises(ValueError, match="unknown strategy"):
        select(pool, strategy="quantum", budget_episodes=5)
    with pytest.raises(ValueError, match="no candidates"):
        select([], strategy="random", budget_episodes=5)


def test_apply_selection_statuses(pool):
    plan = select(pool, strategy="random", budget_episodes=5, seed=0)
    updated = apply_selection(pool, plan)
    statuses = {c.status for c in updated}
    assert statuses == {"selected", "deferred"}
    assert sum(c.status == "selected" for c in updated) == 5


def test_plan_markdown_banner(pool):
    plan = select(pool, strategy="failure_targeted", budget_episodes=5,
                  seed=0)
    md = plan_markdown(plan, pool, synthetic=True)
    assert "SYNTHETIC TOY DATA — NOT REAL BENCHMARK RESULTS" in md
    assert "failure_targeted" in md
