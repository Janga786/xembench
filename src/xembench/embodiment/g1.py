"""Unitree G1 simplified-upper-body adapter.

Native pd_joint_delta_pos action: 25 dims, one 'body' controller over torso,
both 5-DOF arms, and both 7-joint hands (order pinned in constants and
verified live: qpos order == controller order, an identity map).

Canonical mapping:
- right/left arm joints (proximal->distal) fill canonical arm slots 0..4 —
  canonical arm slots 5-6 exist only on 7-DOF arms like the Panda.
- each 7-joint hand collapses to its canonical gripper scalar (synergy).
- torso_joint -> canonical torso slot 0. The G1 has no head joints.
"""

from xembench.constants import (
    CANON_ACTION_SLOTS,
    EMBODIMENT_IDS,
    G1_CONTROLLER_JOINT_ORDER,
    G1_HAND_OPEN_SIGNS,
    G1_LEFT_ARM_JOINTS,
    G1_LEFT_HAND_JOINTS,
    G1_NATIVE_ACTION_DIM,
    G1_RIGHT_ARM_JOINTS,
    G1_RIGHT_HAND_JOINTS,
    G1_TORSO_JOINTS,
)
from xembench.embodiment.base import EmbodimentAdapter

_IDX = {name: i for i, name in enumerate(G1_CONTROLLER_JOINT_ORDER)}


def _arm_pairs(slot: slice, joints: list[str]) -> list[tuple[int, int]]:
    return [(slot.start + i, _IDX[j]) for i, j in enumerate(joints)]


def _hand_synergy(slot: slice, joints: list[str]) -> tuple[int, list[tuple[int, float]]]:
    return (slot.start, [(_IDX[j], G1_HAND_OPEN_SIGNS[j]) for j in joints])


class G1Adapter(EmbodimentAdapter):
    name = "g1"
    embodiment_id = EMBODIMENT_IDS["g1"]
    native_action_dim = G1_NATIVE_ACTION_DIM
    qpos_dim = G1_NATIVE_ACTION_DIM
    joint_map = (
        _arm_pairs(CANON_ACTION_SLOTS["right_arm"], G1_RIGHT_ARM_JOINTS)
        + _arm_pairs(CANON_ACTION_SLOTS["left_arm"], G1_LEFT_ARM_JOINTS)
        + _arm_pairs(CANON_ACTION_SLOTS["torso"], G1_TORSO_JOINTS)
    )
    synergy_map = [
        _hand_synergy(CANON_ACTION_SLOTS["right_gripper"], G1_RIGHT_HAND_JOINTS),
        _hand_synergy(CANON_ACTION_SLOTS["left_gripper"], G1_LEFT_HAND_JOINTS),
    ]
    qpos_joint_names = G1_CONTROLLER_JOINT_ORDER

    def validate_env(self, env) -> None:
        super().validate_env(env)
        joints = [
            j.name
            for j in env.unwrapped.agent.controller.controllers["body"].joints
        ]
        assert joints == G1_CONTROLLER_JOINT_ORDER, (
            f"G1 controller joint order changed:\n{joints}"
        )
