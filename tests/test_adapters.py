"""Adapter round-trip and mapping invariants (no simulator needed)."""

import pytest
import torch

from xembench.constants import A_CANON, CANON_ACTION_SLOTS, P_CANON
from xembench.embodiment import get_adapter

ADAPTERS = ["panda", "panda_stick", "g1"]


@pytest.mark.parametrize("name", ADAPTERS)
def test_roundtrip_is_masked_identity(name):
    adapter = get_adapter(name)
    torch.manual_seed(0)
    x = torch.rand(16, A_CANON) * 2 - 1
    back = adapter.to_canonical_action(adapter.to_native_action(x))
    assert torch.allclose(back, x * adapter.action_mask, atol=1e-6)


@pytest.mark.parametrize("name,expected", [("panda", 8), ("panda_stick", 7), ("g1", 13)])
def test_action_mask_counts(name, expected):
    assert int(get_adapter(name).action_mask.sum()) == expected


def test_native_action_shapes():
    x = torch.zeros(4, A_CANON)
    assert get_adapter("panda").to_native_action(x).shape == (4, 8)
    assert get_adapter("panda_stick").to_native_action(x).shape == (4, 7)
    assert get_adapter("g1").to_native_action(x).shape == (4, 25)


def test_g1_right_arm_slots_map_to_right_joints():
    from xembench.constants import G1_CONTROLLER_JOINT_ORDER, G1_RIGHT_ARM_JOINTS

    adapter = get_adapter("g1")
    slot0 = CANON_ACTION_SLOTS["right_arm"].start
    for i, joint in enumerate(G1_RIGHT_ARM_JOINTS):
        native = dict(adapter.joint_map)[slot0 + i]
        assert G1_CONTROLLER_JOINT_ORDER[native] == joint


def test_g1_gripper_synergy_broadcasts_to_all_hand_joints():
    adapter = get_adapter("g1")
    x = torch.zeros(1, A_CANON)
    x[0, CANON_ACTION_SLOTS["right_gripper"].start] = -1.0  # close right hand
    native = adapter.to_native_action(x)
    from xembench.constants import G1_CONTROLLER_JOINT_ORDER, G1_RIGHT_HAND_JOINTS

    idx = [G1_CONTROLLER_JOINT_ORDER.index(j) for j in G1_RIGHT_HAND_JOINTS]
    assert (native[0, idx] != 0).all()
    others = [i for i in range(25) if i not in idx]
    assert (native[0, others] == 0).all()


def test_arm_embodiments_share_id_g1_differs():
    assert get_adapter("panda").embodiment_id == get_adapter("panda_stick").embodiment_id
    assert get_adapter("panda").embodiment_id != get_adapter("g1").embodiment_id


@pytest.mark.parametrize("name,qdim", [("panda", 9), ("panda_stick", 7), ("g1", 25)])
def test_extract_proprio_from_fake_obs(name, qdim):
    adapter = get_adapter(name)
    obs = {"agent": {"qpos": torch.rand(5, qdim), "qvel": torch.rand(5, qdim)}}
    proprio, mask = adapter.extract_proprio(obs)
    assert proprio.shape == (5, P_CANON) and mask.shape == (P_CANON,)
    # everything outside the mask must be zero
    assert (proprio * (1 - mask)).abs().max() == 0
    # qpos and qvel halves marked symmetrically
    assert torch.equal(mask[:A_CANON], mask[A_CANON : 2 * A_CANON])
    assert (mask[2 * A_CANON :] == 0).all()
