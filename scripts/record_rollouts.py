#!/usr/bin/env python
"""Record policy rollouts as MP4s — the recording hook demo_video_plan.md
flags as TODO. Re-runs exact `episode_seed`s so clips correspond to reported
numbers (pull seeds from data/results/eval_episodes.csv), or sweeps seeds
until it has captured the requested outcome.

    uv run python scripts/record_rollouts.py --task push_t_arm \
        --policy bc:data/checkpoints/bc_push_t_arm_seed0/best.pt \
        --episode-seeds 100000 100003 --split canonical

    uv run python scripts/record_rollouts.py --task apple_bowl_g1 \
        --policy expert:data/checkpoints/expert_apple_bowl_g1_seed0/try1_last_23pct.pt \
        --want success --max-attempts 40

Videos land in artifacts/videos/raw/ using the enforced naming convention
(task__embodiment__policy__split__seed__result.mp4); run
`uv run python -m xembench.analysis.video_index` afterwards.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import imageio.v2 as iio
import numpy as np
import torch

from xembench.analysis.taxonomy import classify_episode
from xembench.embodiment import get_adapter
from xembench.envs import make_env
from xembench.envs.wrappers import UnifiedObsWrapper
from xembench.eval.matrix import _load_policy
from xembench.eval.runner import bc_policy_fn
from xembench.models.ac_policy import StateActorCritic
from xembench.utils.config import load_yaml, repo_root
from xembench.utils.torch_utils import auto_device


def build(task: dict, policy_spec: str, instruction_cond: str,
          color_cond: str, device):
    """-> (wrapped env, adapter, act(obs)->native action, policy label)."""
    kind, ckpt = policy_spec.split(":", 1)
    env_kwargs = {}
    if "Lang" in task["env_id"] and color_cond == "unseen":
        env_kwargs["color_set"] = "unseen"
    obs_mode = "state_dict+rgb" if kind == "expert" else "rgb"
    env = make_env(task["env_id"], num_envs=1, obs_mode=obs_mode,
                   render_mode="rgb_array",
                   max_episode_steps=task["max_steps"], **env_kwargs)
    adapter = get_adapter(task["embodiment"])
    wrapped = UnifiedObsWrapper(
        env, adapter, task["camera"], task["family"],
        static_slots=task.get("slots"),
        instruction_mode="canonical" if instruction_cond == "canonical"
        else "heldout",
        paraphrase_p=1.0, include_flat_state=(kind == "expert"),
    )
    if kind == "bc":
        model = _load_policy(ckpt, "nature_cnn", device)
        policy = bc_policy_fn(model, device)
        return wrapped, adapter, lambda o: adapter.to_native_action(policy(o)), "bc"
    if kind == "expert":
        sd = torch.load(ckpt, map_location=device, weights_only=True)
        model = StateActorCritic(sd["state_dim"], sd["action_dim"])
        model.load_state_dict(sd["model"])
        model.to(device).eval()

        @torch.no_grad()
        def act(obs):
            return model.act(obs["flat_state"].to(device),
                             deterministic=True).cpu()
        return wrapped, adapter, act, "expert"
    raise ValueError(f"unknown policy kind {kind!r} (use bc:/expert:)")


def record_episode(wrapped, act, episode_seed: int, max_steps: int, fps: int,
                   out_dir: Path, name_parts: dict) -> tuple[bool, str]:
    obs, _ = wrapped.reset(seed=episode_seed)
    frames = [np.asarray(torch.as_tensor(wrapped.render()).reshape(
        *torch.as_tensor(wrapped.render()).shape[-3:]).cpu())]
    success, was_truncated = False, True
    for _ in range(max_steps):
        obs, _, term, trunc, info = wrapped.step(act(obs))
        f = torch.as_tensor(wrapped.render())
        frames.append(np.asarray(f.reshape(*f.shape[-3:]).cpu()))
        success = bool(torch.as_tensor(info["success"]).reshape(-1)[0])
        if bool(torch.as_tensor(term).reshape(-1)[0]) or bool(
                torch.as_tensor(trunc).reshape(-1)[0]):
            was_truncated = bool(torch.as_tensor(trunc).reshape(-1)[0])
            break
    label = classify_episode(success, was_truncated)
    result = "success" if success else f"failure_{label}"
    name = "__".join([name_parts["task"], name_parts["embodiment"],
                      name_parts["policy"], name_parts["split"],
                      str(episode_seed), result]) + ".mp4"
    out = out_dir / name
    iio.mimwrite(out, [f.astype(np.uint8) for f in frames], fps=fps,
                 macro_block_size=1)
    print(f"wrote {out}  ({len(frames)} frames, "
          f"instruction: {wrapped.instructions[0]!r})")
    return success, str(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True)
    ap.add_argument("--policy", required=True, help="bc:<ckpt> | expert:<ckpt>")
    ap.add_argument("--episode-seeds", type=int, nargs="*", default=None,
                    help="exact seeds (e.g. from eval_episodes.csv)")
    ap.add_argument("--want", choices=["success", "failure", "any"],
                    default="any", help="sweep seeds until captured")
    ap.add_argument("--count", type=int, default=1,
                    help="how many matching episodes to keep (sweep mode)")
    ap.add_argument("--max-attempts", type=int, default=30)
    ap.add_argument("--start-seed", type=int, default=100000)
    ap.add_argument("--split", default="canonical",
                    choices=["canonical", "heldout", "unseen_color",
                             "transfer"])
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--out-dir", default="artifacts/videos/raw")
    args = ap.parse_args()

    device = auto_device("auto")
    task = load_yaml(repo_root() / "configs" / "tasks.yaml")["tasks"][args.task]
    instruction_cond = "heldout" if args.split == "heldout" else "canonical"
    color_cond = "unseen" if args.split == "unseen_color" else "train"
    wrapped, adapter, act, label = build(task, args.policy, instruction_cond,
                                         color_cond, device)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = dict(task=args.task, embodiment=task["embodiment"], policy=label,
                 split=args.split)

    kept = 0
    if args.episode_seeds:
        for s in args.episode_seeds:
            record_episode(wrapped, act, s, task["max_steps"], args.fps,
                           out_dir, parts)
            kept += 1
    else:
        for i in range(args.max_attempts):
            s = args.start_seed + i
            succ, path = record_episode(wrapped, act, s, task["max_steps"],
                                        args.fps, out_dir, parts)
            match = (args.want == "any" or
                     (args.want == "success") == succ)
            if not match:
                Path(path).unlink()  # keep only requested outcomes
            else:
                kept += 1
                if kept >= args.count:
                    break
    wrapped.close()
    print(f"kept {kept} episode video(s) -> {out_dir}")
    return 0 if kept else 1


if __name__ == "__main__":
    sys.exit(main())
