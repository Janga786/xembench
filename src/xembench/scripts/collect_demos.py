"""Collect demonstrations by rolling a policy in a live env -> xembench schema.

    uv run python -m xembench.scripts.collect_demos --config configs/collect.yaml \
        --set task=apple_bowl_g1 --set policy=ppo:data/checkpoints/expert_g1.pt \
        --set episodes=200

Policies:
  random           uniform actions (pipeline validation only; success_only=false)
  ppo:<ckpt.pt>    PPO expert acting on the flat state vector (G1 demo source)

The env runs obs_mode='state_dict+rgb' so the expert reads the exact
obs_mode='state' flattening while the writer records RGB + proprio. Episodes
are success-filtered by default. Motion-planning collection for the arm Lang
envs is a separate entrypoint (experts/motion_plan_arm.py, Linux-only mplib).
"""

import sys

import numpy as np
import torch

from xembench.constants import CONTROL_MODE
from xembench.data.schema import demo_path
from xembench.data.writer import DemoWriter
from xembench.embodiment import get_adapter
from xembench.envs import make_env
from xembench.envs.wrappers import UnifiedObsWrapper
from xembench.utils.config import common_parser, load_config, load_yaml, repo_root
from xembench.utils.seeding import seed_everything


def make_policy(spec: str, env, adapter):
    if spec == "random":
        space = env.action_space

        def random_policy(obs):
            return torch.as_tensor(space.sample()).reshape(1, -1)

        return random_policy
    if spec.startswith("ppo:"):
        from xembench.experts.ppo_expert import load_expert_policy

        return load_expert_policy(spec.split(":", 1)[1], env, adapter)
    raise ValueError(f"unknown policy spec: {spec!r}")


def collect(cfg: dict) -> int:
    tasks = load_yaml(repo_root() / "configs" / "tasks.yaml")["tasks"]
    task = tasks[cfg["task"]]
    adapter = get_adapter(task["embodiment"])

    env = make_env(task["env_id"], num_envs=1, obs_mode="state_dict+rgb")
    adapter.validate_env(env)
    wrapped = UnifiedObsWrapper(
        env, adapter, task["camera"], task["family"],
        static_slots=task.get("slots"), include_flat_state=True,
        language_seed=cfg["seed"],
    )
    policy = make_policy(cfg["policy"], env, adapter)

    from xembench.utils.config import resolve_path

    out = resolve_path(cfg.get("out") or demo_path(
        repo_root() / "data", task["embodiment"], task["env_id"], cfg["source"]
    ))
    n_target = cfg["episodes"]
    success_only = cfg.get("success_only", True)
    max_steps = task["max_steps"]

    n_written, attempt = 0, 0
    with DemoWriter(
        out, task=cfg["task"], env_id=task["env_id"], embodiment=task["embodiment"],
        camera=task["camera"], control_mode=CONTROL_MODE,
        native_action_dim=adapter.native_action_dim,
    ) as writer:
        while n_written < n_target and attempt < cfg.get("max_attempts", n_target * 20):
            seed = cfg["seed"] * 1_000_000 + attempt
            attempt += 1
            obs, _ = wrapped.reset(seed=seed)
            instruction = obs["instruction"][0]
            slots = wrapped._episode_slots()[0]
            frames, proprios, natives = [], [], []
            success = False
            for _ in range(max_steps):
                action = policy(obs)  # native space, [1, K]
                frames.append(obs["rgb"][0].permute(1, 2, 0).cpu().numpy())
                proprios.append(obs["proprio"][0].cpu().numpy())
                natives.append(np.asarray(action, dtype=np.float32).reshape(-1))
                obs, reward, terminated, truncated, info = wrapped.step(action)
                success = bool(torch.as_tensor(info["success"]).reshape(-1)[0])
                if bool(torch.as_tensor(terminated).reshape(-1)[0]) or bool(
                    torch.as_tensor(truncated).reshape(-1)[0]
                ):
                    break
            if success_only and not success:
                continue
            native = torch.from_numpy(np.stack(natives))
            writer.add_episode(
                rgb=np.stack(frames).astype(np.uint8),
                proprio=np.stack(proprios),
                action_canon=adapter.to_canonical_action(native).numpy(),
                action_native=native.numpy(),
                instruction=instruction,
                success=success,
                seed=seed,
                slots=slots,
            )
            n_written += 1
            if n_written % 10 == 0 or n_written == n_target:
                print(f"  {n_written}/{n_target} episodes "
                      f"({attempt} attempts, {n_written/attempt:.0%} kept)")

    wrapped.close()
    print(f"wrote {n_written} episodes -> {out}")
    return 0 if n_written == n_target else 1


def main() -> int:
    parser = common_parser(__doc__)
    args = parser.parse_args()
    cfg = load_config(args, default_path=repo_root() / "configs" / "collect.yaml")
    seed_everything(cfg["seed"])
    return collect(cfg)


if __name__ == "__main__":
    sys.exit(main())
