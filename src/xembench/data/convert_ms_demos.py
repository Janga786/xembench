"""Convert replayed ManiSkill trajectory files into the xembench demo schema.

Pipeline for stock-env tasks with prebuilt HuggingFace demos (PushT,
PullCubeTool, and the A3 PickCube pipeline check):

    python -m mani_skill.utils.download_demo "PickCube-v1"
    python -m mani_skill.trajectory.replay_trajectory \
        --traj-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5 \
        -o rgb -c pd_joint_delta_pos --save-traj --count 10
    python -m xembench.data.convert_ms_demos --input <replayed.h5> --task <name>

The replayed file stores nested obs arrays with T+1 steps; we keep (s_t, a_t)
pairs, translate native actions to canonical via the task's adapter, and stamp
the canonical instruction rendered from the task's template family + slots.
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

from xembench.constants import CONTROL_MODE
from xembench.data.schema import demo_path
from xembench.data.writer import DemoWriter
from xembench.embodiment import get_adapter
from xembench.language.templates import render
from xembench.utils.config import load_yaml, repo_root


def convert(
    input_h5: str | Path,
    out_path: str | Path,
    *,
    task_name: str,
    task_cfg: dict,
    max_episodes: int | None = None,
    success_only: bool = True,
) -> int:
    adapter = get_adapter(task_cfg["embodiment"])
    camera = task_cfg["camera"]
    n_written = 0

    with h5py.File(input_h5, "r") as f, DemoWriter(
        out_path,
        task=task_name,
        env_id=task_cfg["env_id"],
        embodiment=task_cfg["embodiment"],
        camera=camera,
        control_mode=CONTROL_MODE,
        native_action_dim=adapter.native_action_dim,
    ) as writer:
        keys = sorted([k for k in f.keys() if k.startswith("traj_")],
                      key=lambda k: int(k.split("_")[1]))
        for key in keys:
            if max_episodes is not None and n_written >= max_episodes:
                break
            ep = f[key]
            actions = ep["actions"][:]  # [T, K] native
            T = actions.shape[0]
            success = bool(ep["success"][-1]) if "success" in ep else True
            if success_only and not success:
                continue

            rgb = ep["obs"]["sensor_data"][camera]["rgb"][:T]
            qpos = ep["obs"]["agent"]["qpos"][:T]
            qvel = ep["obs"]["agent"]["qvel"][:T]
            obs = {"agent": {"qpos": torch.from_numpy(qpos),
                             "qvel": torch.from_numpy(qvel)}}
            proprio, _ = adapter.extract_proprio(obs)
            action_native = torch.from_numpy(actions.astype(np.float32))
            action_canon = adapter.to_canonical_action(action_native)

            # Stock envs have fixed scenes -> canonical instruction from the
            # task's static slots. (Custom Lang envs never take this path;
            # their demos are collected live with per-episode instructions.)
            slots = task_cfg.get("slots") or {}
            instruction = render(task_cfg["family"], slots)

            writer.add_episode(
                rgb=np.asarray(rgb, dtype=np.uint8),
                proprio=proprio.numpy(),
                action_canon=action_canon.numpy(),
                action_native=action_native.numpy(),
                instruction=instruction,
                success=success,
                seed=-1,  # original collection seed lives in the ms json sidecar
                slots=slots,
            )
            n_written += 1
    return n_written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="replayed ManiSkill .h5")
    parser.add_argument("--task", required=True, help="task name in configs/tasks.yaml")
    parser.add_argument("--out", default=None, help="output .h5 (default: schema path)")
    parser.add_argument("--source", default="hf_replay", help="source tag in filename")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--tasks-config", default=None)
    parser.add_argument("--env-id-override", default=None,
                        help="env the input was recorded in, if not the task's env")
    parser.add_argument("--slots", default=None,
                        help='JSON template slots override, e.g. \'{"color": "red"}\'')
    args = parser.parse_args()

    tasks = load_yaml(args.tasks_config or repo_root() / "configs" / "tasks.yaml")["tasks"]
    task_cfg = dict(tasks[args.task])
    if args.env_id_override:
        task_cfg["env_id"] = args.env_id_override
    if args.slots:
        import json

        task_cfg["slots"] = json.loads(args.slots)
    out = args.out or demo_path(
        repo_root() / "data", task_cfg["embodiment"], task_cfg["env_id"], args.source
    )
    n = convert(args.input, out, task_name=args.task, task_cfg=task_cfg,
                max_episodes=args.max_episodes)
    print(f"wrote {n} episodes -> {out}")
    return 0 if n > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
