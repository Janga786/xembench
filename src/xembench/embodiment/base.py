"""EmbodimentAdapter: canonical <-> native translation for one robot.

An adapter is defined entirely by two small maps built at construction:

- ``joint_map``: 1:1 canonical-slot -> native-action-index pairs (arm/torso
  joints). The same indices address qpos/qvel because qpos ordering equals
  controller ordering on both benchmark robots (verified by setup_check).
- ``synergy_map``: canonical gripper slot -> [(native_index, sign), ...].
  One scalar drives every mapped joint (broadcast on the way in, signed mean
  on the way out), which is how the G1's 7-joint dexterous hand collapses to
  the same "+1 open / -1 close" scalar as the Panda's parallel gripper.

Round-trip invariant (unit-tested): to_canonical(to_native(x)) == x * action_mask.

Zero-shot cross-embodiment transfer = run the SAME policy network and swap
only the adapter.
"""

from abc import ABC

import torch

from xembench.constants import A_CANON, P_CANON


class EmbodimentAdapter(ABC):
    name: str
    embodiment_id: int
    native_action_dim: int
    qpos_dim: int
    # canonical action-layout index -> native index (1:1 joints)
    joint_map: list[tuple[int, int]]
    # canonical action-layout index -> [(native index, sign), ...] (grippers)
    synergy_map: list[tuple[int, list[tuple[int, float]]]]
    # expected qpos joint names, asserted against live envs
    qpos_joint_names: list[str]

    def __init__(self):
        mask = torch.zeros(A_CANON)
        for c, _ in self.joint_map:
            mask[c] = 1.0
        for c, _ in self.synergy_map:
            mask[c] = 1.0
        self.action_mask = mask
        # proprio layout mirrors the action layout: [canon(21) qpos,
        # canon(21) qvel, padding]; mask marks written dims.
        pmask = torch.zeros(P_CANON)
        used = [c for c, _ in self.joint_map] + [c for c, _ in self.synergy_map]
        for c in used:
            pmask[c] = 1.0
            pmask[A_CANON + c] = 1.0
        self.proprio_mask = pmask

    # -- actions --------------------------------------------------------------
    def to_native_action(self, canon: torch.Tensor) -> torch.Tensor:
        """[B, A_CANON] -> [B, native_action_dim]"""
        native = canon.new_zeros(canon.shape[0], self.native_action_dim)
        for c, n in self.joint_map:
            native[:, n] = canon[:, c]
        for c, joints in self.synergy_map:
            for n, sign in joints:
                native[:, n] = canon[:, c] * sign
        return native

    def to_canonical_action(self, native: torch.Tensor) -> torch.Tensor:
        """[B, native_action_dim] -> [B, A_CANON] (unmapped dims stay 0)"""
        canon = native.new_zeros(native.shape[0], A_CANON)
        for c, n in self.joint_map:
            canon[:, c] = native[:, n]
        for c, joints in self.synergy_map:
            vals = torch.stack([native[:, n] * sign for n, sign in joints], dim=1)
            canon[:, c] = vals.mean(dim=1)
        return canon

    # -- proprio --------------------------------------------------------------
    def extract_proprio(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """ManiSkill obs dict -> (proprio [B, P_CANON], mask [P_CANON])."""
        qpos, qvel = obs["agent"]["qpos"], obs["agent"]["qvel"]
        B = qpos.shape[0]
        out = qpos.new_zeros(B, P_CANON)
        for c, n in self.joint_map:
            out[:, c] = qpos[:, n]
            out[:, A_CANON + c] = qvel[:, n]
        for c, joints in self.synergy_map:
            idx = [n for n, _ in joints]
            out[:, c] = qpos[:, idx].mean(dim=1)
            out[:, A_CANON + c] = qvel[:, idx].mean(dim=1)
        return out, self.proprio_mask.to(out.device)

    # -- validation against a live env ----------------------------------------
    def validate_env(self, env) -> None:
        act_dim = env.action_space.shape[-1]
        assert act_dim == self.native_action_dim, (
            f"{self.name}: env action dim {act_dim} != adapter {self.native_action_dim}"
        )
        names = [j.name for j in env.unwrapped.agent.robot.active_joints]
        assert names == self.qpos_joint_names, (
            f"{self.name}: qpos joint order changed:\n got {names}\n"
            f" expected {self.qpos_joint_names}"
        )
