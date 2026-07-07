"""Franka Panda adapters.

Native pd_joint_delta_pos action: [7 arm joint deltas, 1 gripper] (the two
finger joints are a mimic pair driven by one action dim). PushT's panda_stick
variant has no gripper: 7 dims, 7 qpos.
"""

from xembench.constants import (
    A_CANON,
    CANON_ACTION_SLOTS,
    EMBODIMENT_IDS,
    PANDA_ARM_JOINTS,
    PANDA_GRIPPER_JOINTS,
    PANDA_NATIVE_ACTION_DIM,
    PANDA_STICK_NATIVE_ACTION_DIM,
)
from xembench.embodiment.base import EmbodimentAdapter

_RIGHT_ARM0 = CANON_ACTION_SLOTS["right_arm"].start
_RIGHT_GRIP = CANON_ACTION_SLOTS["right_gripper"].start


class PandaAdapter(EmbodimentAdapter):
    name = "panda"
    embodiment_id = EMBODIMENT_IDS["panda"]
    native_action_dim = PANDA_NATIVE_ACTION_DIM
    qpos_dim = 9
    # canonical right_arm[0:7] <-> native action dims 0..6 (same indices in qpos)
    joint_map = [(_RIGHT_ARM0 + i, i) for i in range(7)]
    # gripper: native action dim 7; qpos has TWO finger joints at 7, 8
    synergy_map = [(_RIGHT_GRIP, [(7, 1.0)])]
    qpos_joint_names = PANDA_ARM_JOINTS + PANDA_GRIPPER_JOINTS

    def extract_proprio(self, obs):
        # qpos/qvel have 9 dims (2 fingers) while the action has 8; the base
        # implementation would read qpos[7] only, so average both fingers.
        qpos, qvel = obs["agent"]["qpos"], obs["agent"]["qvel"]
        out, mask = super().extract_proprio(obs)
        out[:, _RIGHT_GRIP] = qpos[:, 7:9].mean(dim=1)
        out[:, A_CANON + _RIGHT_GRIP] = qvel[:, 7:9].mean(dim=1)
        return out, mask


class PandaStickAdapter(EmbodimentAdapter):
    name = "panda_stick"
    embodiment_id = EMBODIMENT_IDS["panda"]  # same embodiment family
    native_action_dim = PANDA_STICK_NATIVE_ACTION_DIM
    qpos_dim = 7
    joint_map = [(_RIGHT_ARM0 + i, i) for i in range(7)]
    synergy_map = []
    qpos_joint_names = PANDA_ARM_JOINTS
