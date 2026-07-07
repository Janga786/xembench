"""A0 gate: create every benchmark env, validate pinned constants, save frames.

Run on any machine after install:

    uv run python -m xembench.scripts.setup_check

For each task in configs/tasks.yaml this creates the env (CPU sim on Mac, GPU
on the Linux box), asserts that native action dims and the G1 controller joint
order still match xembench.constants (a ManiSkill upgrade that reorders
controllers must fail HERE, not silently corrupt adapters), steps with random
actions, and saves one sensor RGB frame to data/frames/ for eyeballing.
"""

import sys

import imageio.v3 as iio
import numpy as np
import torch

from xembench import constants
from xembench.envs import make_env
from xembench.utils.config import common_parser, load_config, repo_root
from xembench.utils.seeding import seed_everything

EXPECTED_ACTION_DIMS = {
    "panda": constants.PANDA_NATIVE_ACTION_DIM,
    "panda_stick": constants.PANDA_STICK_NATIVE_ACTION_DIM,
    "g1": constants.G1_NATIVE_ACTION_DIM,
}


def check_task(name: str, task: dict, n_steps: int = 10) -> list[str]:
    errors = []
    env = make_env(task["env_id"], num_envs=1, obs_mode="rgb")
    u = env.unwrapped
    obs, _ = env.reset(seed=0)

    # --- action space matches pinned constants ---
    act_dim = env.action_space.shape[-1]
    expected = EXPECTED_ACTION_DIMS[task["embodiment"]]
    if act_dim != expected:
        errors.append(f"{name}: action dim {act_dim} != pinned {expected}")

    # --- G1 controller joint order matches pinned constants ---
    if task["embodiment"] == "g1":
        joints = [j.name for j in u.agent.controller.controllers["body"].joints]
        if joints != constants.G1_CONTROLLER_JOINT_ORDER:
            errors.append(f"{name}: G1 controller joint order changed:\n{joints}")

    # --- expected camera present ---
    cams = list(obs["sensor_data"].keys())
    if task["camera"] not in cams:
        errors.append(f"{name}: camera {task['camera']!r} not in {cams}")

    # --- obs structure needed by wrappers/experts ---
    for key in ("agent", "extra", "sensor_data"):
        if key not in obs:
            errors.append(f"{name}: obs missing top-level key {key!r}")

    # --- episode length registered as configured (ManiSkill's own registry;
    # the gym spec does not carry it) ---
    from mani_skill.utils.registration import REGISTERED_ENVS

    spec_steps = REGISTERED_ENVS[task["env_id"]].max_episode_steps
    if spec_steps != task["max_steps"]:
        errors.append(f"{name}: max_episode_steps {spec_steps} != {task['max_steps']}")

    # --- random rollout ---
    for _ in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

    # --- save a frame for eyeballing ---
    frame_dir = repo_root() / "data" / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    rgb = obs["sensor_data"][task["camera"]]["rgb"]
    if torch.is_tensor(rgb):
        rgb = rgb.cpu().numpy()
    iio.imwrite(frame_dir / f"{task['env_id']}.png", rgb[0].astype(np.uint8))

    env.close()
    status = "OK " if not errors else "FAIL"
    print(f"[{status}] {name:18s} {task['env_id']:32s} act={act_dim:2d} "
          f"cams={cams} steps={spec_steps}")
    return errors


def main() -> int:
    parser = common_parser(__doc__)
    args = parser.parse_args()
    cfg = load_config(args, default_path=repo_root() / "configs" / "tasks.yaml")
    seed_everything(cfg["seed"])

    all_errors = []
    for name, task in cfg["tasks"].items():
        try:
            all_errors += check_task(name, task)
        except Exception as e:  # noqa: BLE001 - report and continue checking
            all_errors.append(f"{name}: {type(e).__name__}: {e}")
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")

    if all_errors:
        print("\nsetup_check FAILED:")
        for e in all_errors:
            print(" -", e)
        return 1
    print(f"\nsetup_check passed: {len(cfg['tasks'])} envs OK. "
          f"Frames in data/frames/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
