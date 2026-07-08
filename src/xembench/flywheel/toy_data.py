"""Deterministic SYNTHETIC toy data for the flywheel pipeline.

EVERY NUMBER PRODUCED HERE IS FAKE. The toy data exists to exercise and test
the flywheel loop (index -> validate -> candidates -> select -> compare ->
pack) on a machine with no GPU and no real results. It is deliberately
constructed so the pipeline has visible structure to detect:

    bc_baseline                     has failures on every split
    bc_after_random_data            improves a little, uniformly
    bc_after_failure_targeted_data  improves more, most on heldout/cross
    bc_after_noisy_data             gets WORSE (corrupted demos hurt)
    bc_after_filtered_data          recovers most of the noisy damage

Those deltas are hand-written constants, not simulations — they encode the
HYPOTHESIS the real Linux experiment will test, so tests can assert the
analysis stack surfaces exactly this structure when it exists. Toy eval CSVs
carry the TOY_DATA checkpoint stamp (the existing analysis banner machinery)
and toy provenance is source_type=synthetic_toy / is_synthetic=True, so
nothing downstream can relabel it as real.

CLI: ``python -m xembench.flywheel.generate_toy_flywheel``.
"""

from pathlib import Path

import numpy as np

from xembench.analysis.episodes import append_episodes
from xembench.flywheel.schemas import (
    DemoEpisodeMetadata,
    DemoProvenance,
    derive_flywheel_split,
)

TOY_BANNER = "SYNTHETIC TOY DATA — NOT REAL BENCHMARK RESULTS"

# Fixed timestamp so toy outputs are byte-for-byte reproducible.
TOY_TIMESTAMP = "2026-01-01T00:00:00+00:00"

TOY_CHECKPOINT_PREFIX = "TOY_DATA_FLYWHEEL_"

# (task, eval_task, train_emb, eval_emb, instruction_cond, color_cond,
#  baseline_sr, n_episodes)
BASE_CELLS = [
    ("pick_cube_arm", "pick_cube_arm", "panda", "panda", "canonical", "train", 0.70, 30),
    ("pick_cube_arm", "pick_cube_arm", "panda", "panda", "heldout", "train", 0.50, 30),
    ("pick_cube_arm", "pick_cube_arm", "panda", "panda", "canonical", "unseen", 0.45, 30),
    ("stack_cube_arm", "stack_cube_arm", "panda", "panda", "canonical", "train", 0.55, 30),
    ("stack_cube_arm", "stack_cube_arm", "panda", "panda", "heldout", "train", 0.40, 30),
    ("pull_tool_arm", "pull_tool_arm", "panda", "panda", "canonical", "fixed", 0.60, 20),
    ("apple_bowl_g1", "apple_bowl_g1", "g1", "g1", "canonical", "train", 0.60, 30),
    ("pick_cube_arm", "apple_bowl_g1", "panda", "g1", "canonical", "train", 0.20, 30),
    ("apple_bowl_g1", "pick_cube_arm", "g1", "panda", "canonical", "train", 0.15, 30),
]

# Success-rate delta per flywheel split, per policy variant (the toy story).
VARIANT_DELTAS = {
    "bc_baseline": {},
    "bc_after_random_data": {
        "train_seen": 0.05, "heldout_paraphrase": 0.04,
        "heldout_color": 0.04, "cross_embodiment": 0.03},
    "bc_after_failure_targeted_data": {
        "train_seen": 0.08, "heldout_paraphrase": 0.15,
        "heldout_color": 0.14, "cross_embodiment": 0.12},
    "bc_after_noisy_data": {
        "train_seen": -0.10, "heldout_paraphrase": -0.08,
        "heldout_color": -0.08, "cross_embodiment": -0.05},
    "bc_after_filtered_data": {
        "train_seen": 0.04, "heldout_paraphrase": 0.02,
        "heldout_color": 0.02, "cross_embodiment": 0.02},
}

FAILURE_POOL = ["timeout", "grasp_failure", "wrong_object_selected",
                "placement_failure", "unknown_failure"]

# One fixed instruction per (task, instruction_cond, color_cond) so the
# language-leakage check has real text to compare and held-out phrasings
# never collide with train phrasings.
_INSTRUCTIONS = {
    ("pick_cube_arm", "canonical", "train"):
        "pick up the red cube and place it on the green target",
    ("pick_cube_arm", "heldout", "train"):
        "hoist the red brick and deposit it on the green target",
    ("pick_cube_arm", "canonical", "unseen"):
        "pick up the yellow cube and place it on the green target",
    ("stack_cube_arm", "canonical", "train"):
        "stack the red cube on top of the blue cube",
    ("stack_cube_arm", "heldout", "train"):
        "pile the red brick onto the blue brick",
    ("pull_tool_arm", "canonical", "fixed"):
        "use the tool to pull the blue cube closer",
    ("apple_bowl_g1", "canonical", "train"):
        "pick up the red apple and place it in the bowl",
}

_FAMILIES = {"pick_cube_arm": "pick_place", "stack_cube_arm": "stack",
             "pull_tool_arm": "tool_pull", "apple_bowl_g1": "pick_place"}

TOY_DEMO_TASKS = [("pick_cube_arm", "panda"), ("stack_cube_arm", "panda"),
                  ("pull_tool_arm", "panda"), ("apple_bowl_g1", "g1")]


def toy_eval_rows(variant: str, seed: int = 0) -> list[dict]:
    """Episode rows (analysis schema v2) for one policy variant."""
    if variant not in VARIANT_DELTAS:
        known = ", ".join(VARIANT_DELTAS)
        raise ValueError(f"unknown toy variant {variant!r}; one of: {known}")
    rng = np.random.default_rng(seed)
    deltas = VARIANT_DELTAS[variant]
    rows = []
    for task, eval_task, temb, eemb, instr, color, base_sr, n in BASE_CELLS:
        split = derive_flywheel_split(task, eval_task, instr, color)
        sr = min(0.95, max(0.0, base_sr + deltas.get(split, 0.0)))
        n_success = round(sr * n)  # exact rates -> tests assert equality
        max_steps = 50 if eemb == "panda" else 100
        instruction = _INSTRUCTIONS.get(
            (eval_task, instr, color),
            _INSTRUCTIONS.get((eval_task, "canonical", "train"), ""))
        for i in range(n):
            success = i < n_success
            rows.append(dict(
                policy_type=variant, task=task, train_embodiment=temb,
                eval_task=eval_task, eval_embodiment=eemb,
                instruction_cond=instr, color_cond=color, seed=0,
                checkpoint=TOY_CHECKPOINT_PREFIX + variant,
                episode_idx=i, episode_seed=200_000 + len(rows),
                success=int(success),
                steps=int(rng.integers(10, max_steps // 2)) if success
                else max_steps,
                max_steps=max_steps,
                failure_type="success" if success
                else FAILURE_POOL[len(rows) % len(FAILURE_POOL)],
                instruction=instruction,
                template_family=_FAMILIES[eval_task],
            ))
    return rows


def write_toy_eval_csv(variant: str, out_csv: str | Path,
                       seed: int = 0) -> Path:
    out = Path(out_csv)
    if out.exists():
        out.unlink()  # regenerate deterministically, never append
    append_episodes(toy_eval_rows(variant, seed), out)
    readme = out.parent / "TOY_DATA_README.txt"
    readme.write_text(
        f"{TOY_BANNER}\n\n"
        "Files in this directory were generated by\n"
        "xembench.flywheel.toy_data to exercise the flywheel pipeline.\n"
        "They are not experimental results.\n")
    return out


def toy_demo_episodes(n_per_task: int = 12, seed: int = 1,
                      ) -> list[DemoEpisodeMetadata]:
    """Toy demo metadata (no HDF5 payload) for index/validation testing.

    Deliberately includes two quality plants — one suspiciously short
    episode and one with a missing instruction — so `make
    flywheel-validate-toy` produces a report with visible (non-fatal)
    findings.
    """
    rng = np.random.default_rng(seed)
    prov = DemoProvenance(
        source_type="synthetic_toy",
        source_run_id="toy_round0_demos",
        source_path="(generated in memory)",
        generator_name="xembench.flywheel.toy_data",
        generator_version="1",
        created_at=TOY_TIMESTAMP,
        machine="mac",
        notes=TOY_BANNER,
        is_synthetic=True,
    )
    from xembench.flywheel.schemas import NATIVE_ACTION_DIMS
    episodes = []
    for task, embodiment in TOY_DEMO_TASKS:
        instruction = _INSTRUCTIONS.get(
            (task, "canonical", "train"),
            _INSTRUCTIONS.get((task, "canonical", "fixed"), ""))
        for i in range(n_per_task):
            failed = i == n_per_task - 1  # one failed demo per task
            steps = int(rng.integers(15, 45))
            episodes.append(DemoEpisodeMetadata(
                episode_id=f"toy_{task}_{i:03d}",
                task=task,
                embodiment=embodiment,
                policy="toy_scripted",
                split="train_seen",
                instruction=instruction,
                instruction_template_id=_FAMILIES[task],
                paraphrase_group="canonical",
                object_condition="cube" if embodiment == "panda" else "apple",
                color_condition="red",
                seed=1000 + i,
                success=not failed,
                failure_label="success" if not failed else "unknown_failure",
                steps=steps,
                native_action_dim=NATIVE_ACTION_DIMS[embodiment],
                provenance=prov,
            ))
    # Quality plants (still schema-valid; they trip validation heuristics).
    episodes[0].steps = 2  # suspiciously short
    episodes[1].instruction = "(instruction not recorded in this CSV)"
    return episodes
