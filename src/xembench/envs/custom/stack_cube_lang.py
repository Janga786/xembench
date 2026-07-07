"""StackCubeLang-v1: StackCube with three colored cubes and a language-chosen
(source, base) pair.

Stock StackCube-v1 always stacks red on green. Here each episode samples an
ordered pair of distinct cubes from the active color set; the instruction is
"stack the {color} cube on top of the {color2} cube" and success is judged on
that pair, so the instruction is load-bearing.
"""

import sapien
import torch

from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
from mani_skill.envs.utils import randomization
from mani_skill.utils import common
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

from xembench.constants import COLOR_RGBA, COLOR_SETS


@register_env("StackCubeLang-v1", max_episode_steps=50)
class StackCubeLangEnv(StackCubeEnv):
    """Language-grounded stacking: 'stack the {color} cube on top of the
    {color2} cube'."""

    def __init__(self, *args, color_set: str = "train", **kwargs):
        self.color_names = list(COLOR_SETS[color_set])
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        self.cube_half_size = common.to_tensor([0.02] * 3, device=self.device)
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cubes = [
            actors.build_cube(
                self.scene,
                half_size=0.02,
                color=COLOR_RGBA[name],
                name=f"cube_{name}",
                initial_pose=sapien.Pose(p=[0, 0.4 * (i + 1), 0.1]),
            )
            for i, name in enumerate(self.color_names)
        ]
        self.src_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.dst_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xy = torch.rand((b, 2)) * 0.2 - 0.1
            sampler = randomization.UniformPlacementSampler(
                bounds=[[-0.1, -0.2], [0.1, 0.2]], batch_size=b, device=self.device
            )
            radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.001
            for cube in self.cubes:
                xyz = torch.zeros((b, 3))
                xyz[:, :2] = xy + sampler.sample(radius, 100, verbose=False)
                xyz[:, 2] = 0.02
                qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
                cube.set_pose(Pose.create_from_pq(p=xyz, q=qs))

            # Ordered (source, base) pair of distinct cubes per env.
            n = len(self.cubes)
            src = torch.randint(0, n, (b,))
            shift = torch.randint(1, n, (b,))
            dst = (src + shift) % n
            self.src_idx[env_idx] = src
            self.dst_idx[env_idx] = dst

    # -- language protocol ----------------------------------------------------
    def language_slots(self) -> list[dict]:
        src, dst = self.src_idx.tolist(), self.dst_idx.tolist()
        return [
            {"color": self.color_names[s], "object": "cube", "color2": self.color_names[d]}
            for s, d in zip(src, dst)
        ]

    # -- per-env (source, base) helpers ---------------------------------------
    def _gather_pos(self, idx: torch.Tensor) -> torch.Tensor:  # [B, 3]
        pos = torch.stack([c.pose.p for c in self.cubes], dim=1)
        return torch.gather(pos, 1, idx.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)

    def _gather_scalar(self, per_cube: list[torch.Tensor], idx: torch.Tensor):
        stacked = torch.stack(per_cube, dim=1)  # [B, n]
        return torch.gather(stacked, 1, idx.view(-1, 1)).squeeze(1)

    def evaluate(self):
        pos_src = self._gather_pos(self.src_idx)
        pos_dst = self._gather_pos(self.dst_idx)
        offset = pos_src - pos_dst
        xy_flag = (
            torch.linalg.norm(offset[..., :2], axis=1)
            <= torch.linalg.norm(self.cube_half_size[:2]) + 0.005
        )
        z_flag = torch.abs(offset[..., 2] - self.cube_half_size[..., 2] * 2) <= 0.005
        is_src_on_dst = torch.logical_and(xy_flag, z_flag)
        is_src_static = self._gather_scalar(
            [c.is_static(lin_thresh=1e-2, ang_thresh=0.5) for c in self.cubes],
            self.src_idx,
        )
        is_src_grasped = self._gather_scalar(
            [self.agent.is_grasping(c) for c in self.cubes], self.src_idx
        )
        success = is_src_on_dst * is_src_static * (~is_src_grasped)
        return {
            "is_cubeA_grasped": is_src_grasped,
            "is_cubeA_on_cubeB": is_src_on_dst,
            "is_cubeA_static": is_src_static,
            "success": success.bool(),
        }

    def _get_obs_extra(self, info: dict):
        n = len(self.cubes)
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            src_onehot=torch.nn.functional.one_hot(self.src_idx, n).float(),
            dst_onehot=torch.nn.functional.one_hot(self.dst_idx, n).float(),
        )
        if "state" in self.obs_mode:
            pos_src = self._gather_pos(self.src_idx)
            pos_dst = self._gather_pos(self.dst_idx)
            obs.update(
                all_cube_pos=torch.stack([c.pose.p for c in self.cubes], dim=1).flatten(1),
                src_pose=pos_src,
                dst_pose=pos_dst,
                tcp_to_src_pos=pos_src - self.agent.tcp.pose.p,
                src_to_dst_pos=pos_dst - pos_src,
            )
        return obs

    def compute_dense_reward(self, obs, action: torch.Tensor, info: dict):
        # Stock StackCube shaping, re-aimed at the sampled (source, base) pair.
        pos_src = self._gather_pos(self.src_idx)
        pos_dst = self._gather_pos(self.dst_idx)

        src_to_tcp_dist = torch.linalg.norm(self.agent.tcp.pose.p - pos_src, axis=1)
        reward = 2 * (1 - torch.tanh(5 * src_to_tcp_dist))

        goal_xyz = torch.hstack(
            [pos_dst[:, 0:2], (pos_dst[:, 2] + self.cube_half_size[2] * 2)[:, None]]
        )
        src_to_goal_dist = torch.linalg.norm(goal_xyz - pos_src, axis=1)
        place_reward = 1 - torch.tanh(5.0 * src_to_goal_dist)
        reward[info["is_cubeA_grasped"]] = (4 + place_reward)[info["is_cubeA_grasped"]]

        gripper_width = (self.agent.robot.get_qlimits()[0, -1, 1] * 2).to(self.device)
        is_grasped = info["is_cubeA_grasped"]
        ungrasp_reward = (
            torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        )
        ungrasp_reward[~is_grasped] = 1.0
        lv = self._gather_scalar(
            [torch.linalg.norm(c.linear_velocity, axis=1) for c in self.cubes],
            self.src_idx,
        )
        av = self._gather_scalar(
            [torch.linalg.norm(c.angular_velocity, axis=1) for c in self.cubes],
            self.src_idx,
        )
        static_reward = 1 - torch.tanh(lv * 10 + av)
        reward[info["is_cubeA_on_cubeB"]] = (
            6 + (ungrasp_reward + static_reward) / 2.0
        )[info["is_cubeA_on_cubeB"]]

        reward[info["success"]] = 8
        return reward

    def compute_normalized_dense_reward(self, obs, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 8
