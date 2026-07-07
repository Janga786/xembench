"""The canonical interface metadata is the benchmark's contract — pin it."""

import torch

from xembench.constants import (
    A_CANON,
    CANON_ACTION_DIM_NAMES,
    CANON_ACTION_SLOTS,
    TRAIN_COLORS,
    UNSEEN_COLORS,
)
from xembench.embodiment import get_adapter


def test_exactly_21_named_dims():
    assert A_CANON == 21
    assert len(CANON_ACTION_DIM_NAMES) == 21
    assert len(set(CANON_ACTION_DIM_NAMES)) == 21  # unique


def test_slots_partition_all_dims_disjointly():
    covered = []
    for sl in CANON_ACTION_SLOTS.values():
        covered.extend(range(sl.start, sl.stop))
    assert sorted(covered) == list(range(A_CANON))  # complete AND disjoint


def test_dim_names_match_their_slot():
    for slot, sl in CANON_ACTION_SLOTS.items():
        for i, dim in enumerate(range(sl.start, sl.stop)):
            assert CANON_ACTION_DIM_NAMES[dim] == f"{slot}_{i}"


def test_adapter_outputs_finite_at_extremes():
    """Adapters must never manufacture NaN/inf, including at action bounds."""
    extremes = torch.cat([
        torch.full((1, A_CANON), -1.0),
        torch.full((1, A_CANON), 1.0),
        torch.zeros(1, A_CANON),
        torch.rand(8, A_CANON) * 2 - 1,
    ])
    for name in ["panda", "panda_stick", "g1"]:
        adapter = get_adapter(name)
        native = adapter.to_native_action(extremes)
        back = adapter.to_canonical_action(native)
        assert torch.isfinite(native).all(), f"{name}: non-finite native action"
        assert torch.isfinite(back).all(), f"{name}: non-finite canonical action"
        qdim = adapter.qpos_dim
        obs = {"agent": {"qpos": torch.randn(4, qdim) * 3,
                         "qvel": torch.randn(4, qdim) * 3}}
        proprio, mask = adapter.extract_proprio(obs)
        assert torch.isfinite(proprio).all() and torch.isfinite(mask).all()


def test_train_and_unseen_colors_disjoint():
    assert set(TRAIN_COLORS) & set(UNSEEN_COLORS) == set()
    assert len(TRAIN_COLORS) == len(UNSEEN_COLORS)  # same clutter level
