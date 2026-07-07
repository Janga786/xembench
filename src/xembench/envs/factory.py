"""Single entrypoint for creating benchmark environments."""

import gymnasium as gym
import torch

import xembench  # noqa: F401  (macOS Vulkan ICD setup before sapien loads)
import mani_skill.envs  # noqa: F401  (registers stock ManiSkill envs)
import xembench.envs.custom  # noqa: F401  (registers *Lang-v1 envs)

from xembench.constants import CONTROL_MODE


def make_env(
    env_id: str,
    num_envs: int = 1,
    obs_mode: str = "rgb",
    control_mode: str = CONTROL_MODE,
    sim_backend: str = "auto",
    reward_mode: str | None = None,
    render_mode: str | None = None,
    **env_kwargs,
) -> gym.Env:
    """Create a ManiSkill env with benchmark defaults.

    sim_backend="auto" picks GPU-parallel simulation when CUDA is available
    (Linux box) and CPU simulation otherwise (Mac dev). Returns the raw
    ManiSkill env; RL training loops wrap it in ManiSkillVectorEnv themselves.
    """
    if sim_backend == "auto":
        sim_backend = "gpu" if torch.cuda.is_available() else "cpu"
    return gym.make(
        env_id,
        num_envs=num_envs,
        obs_mode=obs_mode,
        control_mode=control_mode,
        sim_backend=sim_backend,
        reward_mode=reward_mode,
        render_mode=render_mode,
        **env_kwargs,
    )
