"""Data selection strategies: candidates + budget -> SelectionPlan.

Five deterministic strategies over CandidateDemoRequest lists. All are
simple greedy/sampling algorithms — no heavy solvers, no optimality claims;
they exist so the MVP experiment (docs/flywheel_experiments.md) can compare
selection POLICIES, not solver quality.

    random                       seeded uniform sample (the control arm)
    stratified                   round-robin over (task, embodiment, split,
                                 failure) groups — balanced coverage
    failure_targeted             greedy on priority + expected_value with a
                                 redundancy penalty per already-picked group
    diversity_greedy             approximately-submodular greedy coverage of
                                 (axis, value) pairs
    cost_aware_failure_targeted  failure_targeted score per unit cost, with
                                 a one-per-failing-group coverage floor and
                                 an optional cost budget

Ties everywhere break on candidate_id, so plans are reproducible bit-for-bit.

CLI: ``python -m xembench.flywheel.select_candidates``.
"""

from collections import Counter, defaultdict

import numpy as np

from xembench.flywheel.schemas import (
    CandidateDemoRequest,
    SelectionPlan,
    stable_hash,
    utc_now_iso,
)

# Objective-term weights (recorded verbatim in every plan for auditability).
OBJECTIVE_TERMS = {
    "priority": 1.0,             # failure count x split weight
    "expected_value": 1.0,       # failure rate x split weight (+ coverage boost)
    "redundancy_penalty": 0.5,   # per already-selected candidate of same group
    "diversity_gain": 1.0,       # new (axis, value) pairs covered
    "cost": 1.0,                 # divisor in the cost-aware strategy
}

# Axes the diversity strategy tries to cover.
COVERAGE_AXES = ("task", "embodiment", "split", "target_failure_label",
                 "color_condition", "instruction_family")


def _group_key(c: CandidateDemoRequest) -> tuple:
    return (c.task, c.embodiment, c.split, c.target_failure_label)


def _score(c: CandidateDemoRequest) -> float:
    return (OBJECTIVE_TERMS["priority"] * c.priority
            + OBJECTIVE_TERMS["expected_value"] * c.expected_value)


def _sorted_pool(candidates: list[CandidateDemoRequest],
                 ) -> list[CandidateDemoRequest]:
    return sorted(candidates, key=lambda c: c.candidate_id)


def _check_pool(candidates: list[CandidateDemoRequest]) -> None:
    if not candidates:
        raise ValueError(
            "no candidates to select from — run generate_candidates first")
    ids = [c.candidate_id for c in candidates]
    dupes = [i for i, n in Counter(ids).items() if n > 1]
    if dupes:
        raise ValueError(f"duplicate candidate_ids in pool: {dupes[:5]}")


# ---------------------------------------------------------------------------
# Strategies (each returns selected candidate_ids, order = pick order)
# ---------------------------------------------------------------------------

def _select_random(pool, budget, seed) -> list[str]:
    rng = np.random.default_rng(seed)
    take = min(budget, len(pool))
    idx = rng.choice(len(pool), size=take, replace=False)
    return [pool[i].candidate_id for i in sorted(idx)]


def _select_stratified(pool, budget, seed) -> list[str]:
    del seed  # deterministic without randomness
    groups: dict[tuple, list] = defaultdict(list)
    for c in pool:
        groups[_group_key(c)].append(c)
    ordered_groups = [groups[k] for k in sorted(groups)]
    selected: list[str] = []
    lane = 0
    while len(selected) < budget and any(ordered_groups):
        group = ordered_groups[lane % len(ordered_groups)]
        if group:
            selected.append(group.pop(0).candidate_id)
        if not any(ordered_groups):
            break
        lane += 1
    return selected


def _select_failure_targeted(pool, budget, seed) -> list[str]:
    del seed
    picked_per_group: Counter = Counter()
    remaining = list(pool)
    selected: list[str] = []
    while remaining and len(selected) < budget:
        best = max(
            remaining,
            key=lambda c: (_score(c)
                           - OBJECTIVE_TERMS["redundancy_penalty"]
                           * picked_per_group[_group_key(c)],
                           c.candidate_id))
        selected.append(best.candidate_id)
        picked_per_group[_group_key(best)] += 1
        remaining.remove(best)
    return selected


def _coverage_pairs(c: CandidateDemoRequest) -> set[tuple[str, str]]:
    return {(axis, str(getattr(c, axis))) for axis in COVERAGE_AXES
            if getattr(c, axis)}


def _select_diversity_greedy(pool, budget, seed) -> list[str]:
    del seed
    covered: set[tuple[str, str]] = set()
    remaining = list(pool)
    selected: list[str] = []
    while remaining and len(selected) < budget:
        best = max(
            remaining,
            key=lambda c: (OBJECTIVE_TERMS["diversity_gain"]
                           * len(_coverage_pairs(c) - covered)
                           + 0.01 * _score(c),  # small value tie-breaker
                           c.candidate_id))
        selected.append(best.candidate_id)
        covered |= _coverage_pairs(best)
        remaining.remove(best)
    return selected


def _select_cost_aware(pool, budget, seed, budget_cost) -> list[str]:
    del seed
    selected: list[str] = []
    spent = 0.0
    picked_per_group: Counter = Counter()

    def affordable(c) -> bool:
        return budget_cost is None or spent + c.estimated_cost <= budget_cost

    # Coverage floor: cheapest candidate of every failing group first.
    groups: dict[tuple, list] = defaultdict(list)
    for c in pool:
        groups[_group_key(c)].append(c)
    for key in sorted(groups):
        if len(selected) >= budget:
            break
        cheapest = min(groups[key],
                       key=lambda c: (c.estimated_cost, c.candidate_id))
        if affordable(cheapest):
            selected.append(cheapest.candidate_id)
            spent += cheapest.estimated_cost
            picked_per_group[key] += 1

    chosen = set(selected)
    remaining = [c for c in pool if c.candidate_id not in chosen]
    while remaining and len(selected) < budget:
        best = max(
            remaining,
            key=lambda c: ((_score(c)
                            - OBJECTIVE_TERMS["redundancy_penalty"]
                            * picked_per_group[_group_key(c)])
                           / (OBJECTIVE_TERMS["cost"] * c.estimated_cost),
                           c.candidate_id))
        remaining.remove(best)
        if not affordable(best):
            continue
        selected.append(best.candidate_id)
        spent += best.estimated_cost
        picked_per_group[_group_key(best)] += 1
    return selected


_STRATEGIES = {
    "random": _select_random,
    "stratified": _select_stratified,
    "failure_targeted": _select_failure_targeted,
    "diversity_greedy": _select_diversity_greedy,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def expected_coverage(candidates: list[CandidateDemoRequest],
                      selected_ids: list[str]) -> dict:
    chosen = [c for c in candidates if c.candidate_id in set(selected_ids)]
    return {axis: sorted({str(getattr(c, axis)) for c in chosen
                          if getattr(c, axis)})
            for axis in COVERAGE_AXES}


def select(candidates: list[CandidateDemoRequest], *, strategy: str,
           budget_episodes: int, budget_cost: float | None = None,
           seed: int = 0, input_eval_csv: str = "",
           input_dataset_index: str = "", notes: str = "") -> SelectionPlan:
    _check_pool(candidates)
    pool = _sorted_pool(candidates)
    if strategy == "cost_aware_failure_targeted":
        selected = _select_cost_aware(pool, budget_episodes, seed,
                                      budget_cost)
    elif strategy in _STRATEGIES:
        selected = _STRATEGIES[strategy](pool, budget_episodes, seed)
    else:
        known = ", ".join(sorted(_STRATEGIES) +
                          ["cost_aware_failure_targeted"])
        raise ValueError(f"unknown strategy {strategy!r}; one of: {known}")

    plan = SelectionPlan(
        plan_id="plan_" + stable_hash(
            f"{strategy}|{budget_episodes}|{budget_cost}|{seed}|"
            f"{input_eval_csv}|{input_dataset_index}|{len(pool)}"),
        strategy=strategy,
        budget_episodes=budget_episodes,
        budget_cost=budget_cost,
        candidates_total=len(pool),
        selected_ids=selected,
        objective_terms=dict(OBJECTIVE_TERMS),
        expected_coverage=expected_coverage(pool, selected),
        created_at=utc_now_iso(),
        input_eval_csv=input_eval_csv,
        input_dataset_index=input_dataset_index,
        notes=notes,
    )
    from xembench.flywheel.schemas import require_valid
    require_valid(plan)
    return plan


def apply_selection(candidates: list[CandidateDemoRequest],
                    plan: SelectionPlan) -> list[CandidateDemoRequest]:
    """Candidates with status flipped to selected/deferred per the plan."""
    chosen = set(plan.selected_ids)
    out = []
    for c in candidates:
        status = "selected" if c.candidate_id in chosen else "deferred"
        out.append(CandidateDemoRequest(**{**c.__dict__, "status": status}))
    return out


def plan_markdown(plan: SelectionPlan,
                  candidates: list[CandidateDemoRequest],
                  synthetic: bool) -> str:
    banner = ("> **SYNTHETIC TOY DATA — NOT REAL BENCHMARK RESULTS.** "
              "This plan was computed from synthetic candidates.\n\n"
              if synthetic else "")
    by_id = {c.candidate_id: c for c in candidates}
    chosen = [by_id[i] for i in plan.selected_ids if i in by_id]
    cost = sum(c.estimated_cost for c in chosen)
    group_counts = Counter(_group_key(c) for c in chosen)
    lines = [
        "# Selection plan", "",
        banner + f"Plan `{plan.plan_id}` — strategy **{plan.strategy}**, "
        f"{len(plan.selected_ids)}/{plan.candidates_total} candidates "
        f"selected under a budget of {plan.budget_episodes} episodes"
        + (f" / {plan.budget_cost:.1f} cost units" if plan.budget_cost
           else "") + f"; estimated total cost {cost:.1f} units.", "",
        "Objective terms: "
        + ", ".join(f"{k}={v}" for k, v in plan.objective_terms.items()),
        "",
        "| task | embodiment | split | target failure | episodes |",
        "|---|---|---|---|---|",
    ]
    for (task, emb, split, label), n in sorted(group_counts.items()):
        lines.append(f"| {task} | {emb} | {split} | {label} | {n} |")
    lines += ["", "## Expected coverage", ""]
    for axis, values in plan.expected_coverage.items():
        lines.append(f"- **{axis}**: {', '.join(values) or '_none_'}")
    return "\n".join(lines) + "\n"
