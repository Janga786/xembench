#!/usr/bin/env python
"""G1 hand-sign diagnostic: does canonical gripper +1/-1 move the hand joints
the way G1_HAND_OPEN_SIGNS assumes? (B2 caveat — docs/phase_b_linux_plan.md)

Creates the apple_bowl_g1 env, applies a pure canonical open (+1) command to
both grippers for N steps, then a pure close (-1), and reports per hand joint:
qpos before/after, movement direction, and position within joint limits.
Saves head-camera frames (initial / after-open / after-close) for the human
eyeball check — joint-limit direction alone cannot prove which pose is
semantically "open".

    uv run python scripts/g1_hand_sign_check.py \
        [--steps 10] [--out data/shakedown/g1_hand_sign]

Exit 0 = commands finite, joints respond, and every hand joint moved in the
direction its sign assumes for "open". Nonzero = a joint fought the assumed
sign (flip it in G1_HAND_OPEN_SIGNS) or the env misbehaved. Either way the
saved frames are the ground truth.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import imageio.v3 as iio
import numpy as np
import torch

from xembench.constants import (
    A_CANON,
    CANON_ACTION_SLOTS,
    G1_HAND_OPEN_SIGNS,
    G1_LEFT_HAND_JOINTS,
    G1_RIGHT_HAND_JOINTS,
)
from xembench.embodiment import get_adapter
from xembench.envs import make_env


def hand_qpos(env, adapter, joints):
    obs = env.unwrapped.get_obs()
    qpos = obs["agent"]["qpos"][0]
    idx = {n: i for i, n in enumerate(adapter.qpos_joint_names)}
    return {j: float(qpos[idx[j]]) for j in joints}


def save_frame(env, out_dir: Path, name: str) -> None:
    obs = env.unwrapped.get_obs()
    rgb = obs["sensor_data"]["head_camera"]["rgb"]
    if torch.is_tensor(rgb):
        rgb = rgb.cpu().numpy()
    iio.imwrite(out_dir / f"{name}.png", rgb[0].astype(np.uint8))


def drive_gripper(env, adapter, value: float, steps: int) -> None:
    canon = torch.zeros(1, A_CANON)
    canon[0, CANON_ACTION_SLOTS["right_gripper"].start] = value
    canon[0, CANON_ACTION_SLOTS["left_gripper"].start] = value
    native = adapter.to_native_action(canon)
    assert torch.isfinite(native).all(), "non-finite native action"
    for _ in range(steps):
        env.step(native)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--out", default="data/shakedown/g1_hand_sign")
    args = ap.parse_args()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    hand_joints = G1_RIGHT_HAND_JOINTS + G1_LEFT_HAND_JOINTS
    adapter = get_adapter("g1")
    env = make_env("UnitreeG1PlaceAppleInBowl-v1", num_envs=1, obs_mode="rgb")
    adapter.validate_env(env)
    env.reset(seed=0)

    limits = {
        j.name: (float(j.limits[0, 0]), float(j.limits[0, 1]))
        for j in env.unwrapped.agent.robot.active_joints
        if j.name in hand_joints
    }
    print("assumed open signs (canonical +1 = open):")
    for j in hand_joints:
        lo, hi = limits[j]
        print(f"  {j:22s} sign={G1_HAND_OPEN_SIGNS[j]:+.0f} limits=[{lo:+.3f}, {hi:+.3f}]")

    q0 = hand_qpos(env, adapter, hand_joints)
    save_frame(env, out_dir, "0_initial")
    drive_gripper(env, adapter, +1.0, args.steps)
    q_open = hand_qpos(env, adapter, hand_joints)
    save_frame(env, out_dir, "1_after_open_cmd")
    drive_gripper(env, adapter, -1.0, 2 * args.steps)
    q_close = hand_qpos(env, adapter, hand_joints)
    save_frame(env, out_dir, "2_after_close_cmd")
    env.close()

    print(f"\n{'joint':22s} {'q0':>8s} {'q_open':>8s} {'q_close':>8s}  verdict")
    bad = 0
    for j in hand_joints:
        sign = G1_HAND_OPEN_SIGNS[j]
        d_open = q_open[j] - q0[j]
        d_close = q_close[j] - q_open[j]
        # In "open-direction space" both motions should point toward +sign
        # (open moves qpos by +sign, close by -sign). One direction usually
        # saturates at a joint limit (leaving eps-scale PD residue there), so
        # judge the dominant motion and only require the minor one to not
        # contradict it at comparable magnitude.
        eps = 0.02
        open_space = (d_open, -d_close)
        dominant, minor = sorted(open_space, key=abs, reverse=True)
        moved = abs(dominant) > eps
        ok = moved and dominant * sign > 0 and (
            minor * sign > 0 or abs(minor) < max(eps, 0.25 * abs(dominant)))
        verdict = "ok" if ok else ("STUCK" if not moved else "SIGN MISMATCH")
        bad += 0 if ok else 1
        print(f"{j:22s} {q0[j]:+8.3f} {q_open[j]:+8.3f} {q_close[j]:+8.3f}  {verdict}")

    print(f"\nframes -> {out_dir}")
    print("mechanical sign check:", "PASS" if bad == 0 else f"{bad} joints FAILED")
    print("NOTE: semantic open-vs-close still needs a human look at the frames.")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
