"""PickCubeLang-v1: PickCube with multiple colored cubes and a language target.

Stock PickCube-v1 has a single red cube, which makes language decorative. Here
the table holds one cube per color of the active color set; each episode picks
a target cube uniformly at random and the instruction names its color. Success
is judged on the *named* cube only, so a policy must actually ground the
instruction to solve the task above chance.

The green goal sphere is kept visible to sensor cameras (stock hides it) —
a pixel policy needs to see where "the green target" is.

``color_set="train"`` (red/green/blue) for training envs, ``"unseen"``
(yellow/purple/cyan) for the color-generalization eval. Cube colors are baked
at scene-build time (GPU-sim compatible); only the target choice and the
instruction vary per episode.
"""

import sapien
import torch

from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.envs.utils import randomization
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

from xembench.constants import COLOR_RGBA, COLOR_SETS


@register_env("PickCubeLang-v1", max_episode_steps=50)
class PickCubeLangEnv(PickCubeEnv):
    """Language-grounded pick-and-place: 'pick up the {color} cube and place
    it on the green target'."""

    def __init__(self, *args, color_set: str = "train", **kwargs):
        self.color_names = list(COLOR_SETS[color_set])
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cubes = [
            actors.build_cube(
                self.scene,
                half_size=self.cube_half_size,
                color=COLOR_RGBA[name],
                name=f"cube_{name}",
                initial_pose=sapien.Pose(p=[0, 0.4 * (i + 1), self.cube_half_size]),
            )
            for i, name in enumerate(self.color_names)
        ]
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        # NOT appended to self._hidden_objects: the goal must be visible in RGB.
        self.target_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Non-overlapping cube placement in the stock spawn region.
            sampler = randomization.UniformPlacementSampler(
                bounds=[
                    [
                        self.cube_spawn_center[0] - self.cube_spawn_half_size,
                        self.cube_spawn_center[1] - self.cube_spawn_half_size,
                    ],
                    [
                        self.cube_spawn_center[0] + self.cube_spawn_half_size,
                        self.cube_spawn_center[1] + self.cube_spawn_half_size,
                    ],
                ],
                batch_size=b,
                device=self.device,
            )
            radius = torch.linalg.norm(torch.tensor([self.cube_half_size] * 2)) + 0.001
            for cube in self.cubes:
                xyz = torch.zeros((b, 3))
                xyz[:, :2] = sampler.sample(radius, 100, verbose=False)
                xyz[:, 2] = self.cube_half_size
                qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
                cube.set_pose(Pose.create_from_pq(xyz, qs))

            self.target_idx[env_idx] = torch.randint(0, len(self.cubes), (b,))

            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = (
                torch.rand((b, 2)) * self.cube_spawn_half_size * 2
                - self.cube_spawn_half_size
            )
            goal_xyz[:, 0] += self.cube_spawn_center[0]
            goal_xyz[:, 1] += self.cube_spawn_center[1]
            goal_xyz[:, 2] = torch.rand((b,)) * self.max_goal_height + self.cube_half_size
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

    # -- language protocol (consumed by InstructionWrapper) ------------------
    def language_slots(self) -> list[dict]:
        idx = self.target_idx.tolist()
        return [
            {
                "color": self.color_names[i],
                "object": "cube",
                "relation": "on",
                "receptacle": "green target",
            }
            for i in idx
        ]

    # -- motion-planning compat: ManiSkill's Panda solver manipulates
    # `env.cube`; resolve it to the CURRENT episode's language target so the
    # stock solver produces color-grounded demos. num_envs=1 collection only.
    @property
    def cube(self):
        assert self.num_envs == 1, "cube property is for single-env MP collection"
        return self.cubes[int(self.target_idx[0])]

    # -- helpers over the per-env target cube --------------------------------
    def _all_cube_pos(self) -> torch.Tensor:  # [B, n_cubes, 3]
        return torch.stack([c.pose.p for c in self.cubes], dim=1)

    def _target_pos(self) -> torch.Tensor:  # [B, 3]
        pos = self._all_cube_pos()
        idx = self.target_idx.view(-1, 1, 1).expand(-1, 1, 3)
        return torch.gather(pos, 1, idx).squeeze(1)

    def _target_grasped(self) -> torch.Tensor:  # [B] bool
        grasped = torch.stack(
            [self.agent.is_grasping(c) for c in self.cubes], dim=1
        )  # [B, n]
        return torch.gather(grasped, 1, self.target_idx.view(-1, 1)).squeeze(1)

    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(self.goal_site.pose.p - self._target_pos(), axis=1)
            <= self.goal_thresh
        )
        is_robot_static = self.agent.is_static(0.2)
        return {
            "success": is_obj_placed & is_robot_static,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": self._target_grasped(),
        }

    def _get_obs_extra(self, info: dict):
        # target_onehot is the state-space stand-in for the instruction: the
        # (language-free) PPO expert reads it to know which cube to pick.
        target_onehot = torch.nn.functional.one_hot(
            self.target_idx, num_classes=len(self.cubes)
        ).float()
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp_pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
            target_onehot=target_onehot,
        )
        if "state" in self.obs_mode:
            target_pos = self._target_pos()
            obs.update(
                all_cube_pos=self._all_cube_pos().flatten(1),
                target_pos=target_pos,
                tcp_to_target_pos=target_pos - self.agent.tcp_pose.p,
                target_to_goal_pos=self.goal_site.pose.p - target_pos,
            )
        return obs

    def compute_dense_reward(self, obs, action: torch.Tensor, info: dict):
        # Stock PickCube shaping, re-aimed at the per-env target cube.
        target_pos = self._target_pos()
        tcp_to_obj_dist = torch.linalg.norm(target_pos - self.agent.tcp_pose.p, axis=1)
        reward = 1 - torch.tanh(5 * tcp_to_obj_dist)

        is_grasped = info["is_grasped"]
        reward += is_grasped

        obj_to_goal_dist = torch.linalg.norm(self.goal_site.pose.p - target_pos, axis=1)
        reward += (1 - torch.tanh(5 * obj_to_goal_dist)) * is_grasped

        qvel = self.agent.robot.get_qvel()[..., :-2]
        static_reward = 1 - torch.tanh(5 * torch.linalg.norm(qvel, axis=1))
        reward += static_reward * info["is_obj_placed"]

        reward[info["success"]] = 5
        return reward

    def compute_normalized_dense_reward(self, obs, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5
