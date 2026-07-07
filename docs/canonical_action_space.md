# The canonical action space

This interface is the benchmark's core contribution. Cross-embodiment
transfer is only *measurable* if "run the same policy on a different robot"
is well-defined; the canonical action space is that definition. Everything
else in the repo (models, trainers, eval) speaks only this interface.

Source of truth: `src/xembench/constants.py` —
`A_CANON`, `CANON_ACTION_SLOTS`, `CANON_ACTION_DIM_NAMES` (derived, tested).

## Layout: 21 dims, semantic slots

| Dims | Slot | Names | Meaning |
|---|---|---|---|
| 0–6 | `right_arm` | `right_arm_0..6` | joint position deltas, proximal → distal |
| 7 | `right_gripper` | `right_gripper_0` | grasp synergy: **+1 open, −1 close** |
| 8–14 | `left_arm` | `left_arm_0..6` | joint deltas, proximal → distal |
| 15 | `left_gripper` | `left_gripper_0` | grasp synergy |
| 16–18 | `torso` | `torso_0..2` | torso joint deltas |
| 19–20 | `head` | `head_0..1` | head joint deltas (no current robot uses these) |

This is **semantic slotting, not naive padding**: dim *k* means "k-th
proximal joint of the right arm" on every embodiment that has such a joint.
A policy that learned "dim 3 bends the elbow toward the table" on the Panda
is exercising the same *kind* of control when the same dim drives the G1's
elbow pitch. That alignment is what makes zero-shot transfer a meaningful
(if hard) test rather than a category error.

## Ranges and control mode

Control mode is `pd_joint_delta_pos` on every robot — the only mode both the
Panda and the G1 upper body support. ManiSkill normalizes these actions to
**[-1, 1]** per dim on both robots, so canonical actions are well-scaled
across embodiments by construction; the policy head is a `tanh`. Note the
normalization maps to each robot's own per-joint delta limits, so equal
canonical values do *not* mean equal radians — this is a documented
limitation, not a bug (see below).

Each adapter also exposes `action_mask: [21]` marking the dims that
embodiment actually owns. BC loss, PPO log-prob/entropy, and sampled actions
are all masked to active dims.

## Panda mapping (`embodiment/panda.py`)

Native action: 8 dims (7 arm joint deltas + 1 gripper driving a two-finger
mimic pair).

- `right_arm_0..6` ↔ native dims 0–6 (`panda_joint1..7`).
- `right_gripper_0` ↔ native dim 7 (+1 open matches Panda's convention).
- Everything else masked (mask sum = 8).
- Proprio: the gripper slot reads the *mean* of the two finger joints.
- `panda_stick` (PushT's gripperless variant): same arm mapping, no gripper,
  mask sum = 7. It shares the Panda `embodiment_id` — it is the same family.

## Unitree G1 mapping (`embodiment/g1.py`)

Native action: 25 dims, one `pd_joint_delta_pos` controller over torso, two
5-DOF arms, and two 7-joint hands. The controller's joint order is pinned in
`constants.G1_CONTROLLER_JOINT_ORDER` and asserted against the live env
(qpos order equals controller order — verified identity map).

- `right_arm_0..4` ↔ shoulder pitch/roll/yaw, elbow pitch, elbow roll
  (proximal → distal). **Canonical arm dims 5–6 are unused on the G1** —
  they exist only on 7-DOF arms.
- `right_gripper_0` → broadcast to all 7 right-hand joints with per-joint
  signs (`G1_HAND_OPEN_SIGNS`); inverse mapping is the signed mean. One
  scalar grasp synergy makes the dexterous hand commensurable with a
  parallel jaw. Signs currently assume +1 = open for every joint —
  **unverified**; revisit in Phase B if grasping underperforms.
- `left_arm`, `left_gripper` symmetric; `torso_0` ↔ `torso_joint`.
- Mask sum = 13 (5+5 arm, 2 grippers, 1 torso).

Round-trip invariant (unit-tested for every adapter):
`to_canonical(to_native(x)) == x * action_mask`.

## Why hand-designed, and what that costs

A learned latent action interface would itself be a confound: transfer
failures could mean "the latent space is bad" rather than "the skill did not
transfer". A fixed, human-auditable interface keeps the experimental
variable clean — every failure is attributable to policy, perception, or
kinematics, not to interface learning.

Known limitations (also candidates for the paper's Limitations section):

1. **Kinematic aliasing.** "k-th proximal joint" aligns joint *roles* only
   approximately: Panda joint 4 and G1 elbow pitch have different axes,
   limits, and link lengths. Equal canonical actions produce different
   end-effector motion.
2. **Normalized deltas hide scale.** [-1, 1] maps to different per-joint
   radian limits per robot; the interface guarantees boundedness, not
   metric equivalence.
3. **Grasp synergy is lossy.** Collapsing a 7-joint hand to one scalar
   discards in-hand dexterity; tasks requiring finger gaiting would need a
   richer gripper slot.
4. **No end-effector-space option.** An EE-delta canonical space would be
   more embodiment-agnostic, but the G1 agent ships no IK controller in
   ManiSkill; joint-space was the honest common denominator. Adding an
   IK-based canonical variant is a well-scoped extension.
5. **Fixed capacity.** 21 dims cover the current robots; a mobile base or
   bimanual+head robot uses more slots but the layout already reserves them
   (torso ×3, head ×2 are mostly unused today).

## Adding a new robot adapter

1. Subclass `EmbodimentAdapter` (`embodiment/base.py`). Define:
   `name`, `embodiment_id` (reuse a family id or extend
   `constants.EMBODIMENT_IDS` and `NUM_EMBODIMENTS`), `native_action_dim`,
   `qpos_dim`, `qpos_joint_names`, and the two maps:
   - `joint_map`: `[(canonical_dim, native_index), ...]` for 1:1 joints,
     proximal → distal into the matching slot;
   - `synergy_map`: `[(canonical_dim, [(native_index, sign), ...])]` for
     gripper-like groups.
2. Pin the robot's controller joint order into `constants.py` and assert it
   in `validate_env` (see `G1Adapter`) so simulator upgrades fail loudly.
3. Register it in `embodiment/registry.py`; add the embodiment to tasks in
   `configs/tasks.yaml`.
4. The base class derives `action_mask`, proprio extraction, and round-trip
   behavior; `tests/test_adapters.py` is parametrized — add the new name to
   its `ADAPTERS` list and the invariants come for free.

This is the extension path for future Isaac Lab / LeRobot / real-hardware
embodiments: nothing outside the adapter and the registry should change.

## Open research questions

- Is joint-space or EE-space the better canonical substrate for transfer,
  and does the answer depend on viewpoint (egocentric vs third-person)?
- How much of the arm→G1 drop is attributable to the camera change vs the
  kinematic change? (The G1 envs expose both a head camera and a
  third-person camera, so a controlled ablation is already possible.)
- Does the proprio mask + embodiment one-hot suffice as an embodiment cue,
  or does explicit morphology conditioning (e.g., joint-graph encodings)
  close part of the gap?
- Where does the single-scalar grasp synergy break down as tasks demand
  more dexterity?
