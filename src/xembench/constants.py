"""Canonical (embodiment-agnostic) interface constants.

The benchmark trains ONE network on a canonical action/proprio layout; thin
per-embodiment adapters (xembench.embodiment) translate to/from each robot's
native spaces. Canonical dimensions carry the same *semantic meaning* on both
robots so zero-shot transfer is well-defined: slot k is "right arm joint k
delta" on every embodiment that has such a joint.

Control mode is pd_joint_delta_pos everywhere — the only mode both the Panda
and the Unitree G1 upper body support — and ManiSkill normalizes those actions
to [-1, 1] on both robots, so canonical actions are well-scaled across
embodiments by construction.

Native joint layouts below were pinned from live environments on 2026-07-07
(mani_skill 3.0.1, sapien 3.0.3) by ``python -m xembench.scripts.setup_check``.
Adapters assert against them at construction so a ManiSkill upgrade that
changes controller layouts fails loudly instead of silently mis-mapping.
"""

# ---------------------------------------------------------------------------
# Canonical action space: 21 dims, semantic slots (NOT naive padding).
# ---------------------------------------------------------------------------
A_CANON = 21

CANON_ACTION_SLOTS = {
    "right_arm": slice(0, 7),  # up to 7 joint deltas, proximal -> distal
    "right_gripper": slice(7, 8),  # grasp synergy: +1 open, -1 close
    "left_arm": slice(8, 15),
    "left_gripper": slice(15, 16),
    "torso": slice(16, 19),
    "head": slice(19, 21),
}


def _canon_dim_names() -> list[str]:
    names = [""] * A_CANON
    for slot, sl in CANON_ACTION_SLOTS.items():
        for i, dim in enumerate(range(sl.start, sl.stop)):
            names[dim] = f"{slot}_{i}"
    assert all(names), "CANON_ACTION_SLOTS must partition all A_CANON dims"
    return names


# Per-dimension names derived from the slots (kept programmatic so docs,
# tests, and analysis tooling can never drift from the layout above).
CANON_ACTION_DIM_NAMES = _canon_dim_names()

# ---------------------------------------------------------------------------
# Canonical proprio: qpos then qvel, each laid out mirroring the action slots
# (21 + 21 = 42 used dims), zero-padded to P_CANON with a validity mask.
# ---------------------------------------------------------------------------
P_CANON = 64

# Instruction tokens (whitespace tokenizer, vocab.json)
L_MAX = 16

# Observation image (square RGB); stock sensor cams already render 128x128.
IMG_SIZE = 128

EMBODIMENT_IDS = {"panda": 0, "g1": 1}
NUM_EMBODIMENTS = 2

CONTROL_MODE = "pd_joint_delta_pos"

# ---------------------------------------------------------------------------
# Franka Panda (robots: panda, panda_wristcam, panda_stick)
# Action: [7 arm joint deltas, 1 gripper (mimic over 2 fingers)] = 8
# panda_stick (PushT) has no gripper: 7 dims. qpos: 7 arm (+2 fingers).
# ---------------------------------------------------------------------------
PANDA_ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
PANDA_GRIPPER_JOINTS = ["panda_finger_joint1", "panda_finger_joint2"]
PANDA_NATIVE_ACTION_DIM = 8
PANDA_STICK_NATIVE_ACTION_DIM = 7

# ---------------------------------------------------------------------------
# Unitree G1 simplified upper body ("unitree_g1_simplified_upper_body*")
# Single 'body' pd_joint_delta_pos controller over all 25 DOF.
# Order below is the controller/action-dimension order from the live env.
# ---------------------------------------------------------------------------
G1_NATIVE_ACTION_DIM = 25
G1_CONTROLLER_JOINT_ORDER = [
    "torso_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_pitch_joint", "right_elbow_pitch_joint",
    "left_elbow_roll_joint", "right_elbow_roll_joint",
    "left_zero_joint", "left_three_joint", "left_five_joint",
    "right_zero_joint", "right_three_joint", "right_five_joint",
    "left_one_joint", "left_four_joint", "left_six_joint",
    "right_one_joint", "right_four_joint", "right_six_joint",
    "left_two_joint", "right_two_joint",
]

# Arm chains proximal -> distal (5 joints; canonical arm slots 5-6 unused).
G1_RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_pitch_joint",
    "right_elbow_roll_joint",
]
G1_LEFT_ARM_JOINTS = [j.replace("right_", "left_") for j in G1_RIGHT_ARM_JOINTS]

# Dexterous hand joints driven as ONE grasp synergy per hand (canonical
# gripper scalar broadcast over all hand joints, sign per joint).
G1_RIGHT_HAND_JOINTS = [
    f"right_{n}_joint" for n in ["zero", "one", "two", "three", "four", "five", "six"]
]
G1_LEFT_HAND_JOINTS = [j.replace("right_", "left_") for j in G1_RIGHT_HAND_JOINTS]
# +1 canonical = open. Signs verified empirically on Linux (B0 prep) with
# scripts/g1_hand_sign_check.py + rendered frames: the hands' URDF joints are
# MIRRORED (one-sided limits flip side to side), so "open" is joint-negative
# on the right hand and joint-positive on the left. With all-(+1) signs the
# right hand curled closed on an open command. Frames: the check script
# saves before/open/close renders for human re-verification.
G1_HAND_OPEN_SIGNS = {
    **{j: -1.0 for j in G1_RIGHT_HAND_JOINTS},
    **{j: 1.0 for j in G1_LEFT_HAND_JOINTS},
}

G1_TORSO_JOINTS = ["torso_joint"]

# ---------------------------------------------------------------------------
# Object colors: instructions ground on these. Custom Lang envs build one cube
# per color of the active set; the instruction names which one is the target.
# UNSEEN colors never appear in training data (generalization eval).
# ---------------------------------------------------------------------------
COLOR_RGBA = {
    "red": [1.0, 0.0, 0.0, 1.0],
    "green": [0.0, 0.8, 0.0, 1.0],
    "blue": [0.0, 0.0, 1.0, 1.0],
    "yellow": [1.0, 0.9, 0.0, 1.0],
    "purple": [0.6, 0.0, 0.8, 1.0],
    "cyan": [0.0, 0.9, 0.9, 1.0],
}
TRAIN_COLORS = ["red", "green", "blue"]
UNSEEN_COLORS = ["yellow", "purple", "cyan"]
COLOR_SETS = {"train": TRAIN_COLORS, "unseen": UNSEEN_COLORS}
