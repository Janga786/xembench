#!/usr/bin/env python
"""Empirical success rate of a state-PPO expert checkpoint (deterministic
actions, same acting mode demo collection uses).

    uv run python scripts/eval_state_expert.py <ckpt.pt> [--task apple_bowl_g1]
        [--num-envs 256] [--steps 400]

Pools per-episode outcomes over N parallel envs for `steps` env-steps
(several episode horizons), so early-termination successes and horizon
failures are both counted. This is the honest number to quote for the B2
gate — the training log's rolling window lags and mixes policies.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from xembench.envs import make_env
from xembench.models.ac_policy import StateActorCritic
from xembench.utils.config import load_yaml, repo_root

# same headroom as g1_expert_train.yaml — G1 scenes overflow the PhysX
# default collision stack and silently drop contacts
GPU_MEM = dict(sim_config=dict(gpu_memory_config=dict(collision_stack_size=536870912)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ckpt")
    ap.add_argument("--task", default="apple_bowl_g1")
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--stochastic", action="store_true",
                    help="sample actions (training-time behavior) instead of "
                         "the deterministic mean used for demo collection")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = load_yaml(repo_root() / "configs" / "tasks.yaml")["tasks"][args.task]
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    model = StateActorCritic(ckpt["state_dim"], ckpt["action_dim"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    raw = make_env(task["env_id"], num_envs=args.num_envs, obs_mode="state",
                   **GPU_MEM)
    envs = ManiSkillVectorEnv(raw, auto_reset=True, ignore_terminations=False,
                              record_metrics=False)
    obs, _ = envs.reset(seed=args.seed)
    outcomes = []
    with torch.no_grad():
        for _ in range(args.steps):
            act = model.act(obs.to(device), deterministic=not args.stochastic)
            obs, _, _, _, infos = envs.step(act.cpu())
            if "final_info" in infos:
                fi, mask = infos["final_info"], infos.get("_final_info")
                succ = fi.get("success") if isinstance(fi, dict) else None
                if succ is not None:
                    mask = torch.as_tensor(mask, device=succ.device)
                    if mask.any():
                        outcomes.append(succ[mask].float())
    envs.close()
    n = sum(o.numel() for o in outcomes)
    rate = torch.cat(outcomes).mean().item() if n else float("nan")
    mode = "stochastic" if args.stochastic else "deterministic"
    print(f"{args.ckpt}: success {rate:.2%} over {n} episodes "
          f"({args.num_envs} envs x {args.steps} steps, {mode})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
