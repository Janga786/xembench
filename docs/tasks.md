# Task suite

Six tasks across two embodiments, registered in `configs/tasks.yaml` (the
single source of truth for env id, embodiment, camera, instruction family,
template slots, episode budget, and demo source). Cross-embodiment transfer
pairs: **pick_cube_arm ↔ apple_bowl_g1** (primary — same instruction
family) and **stack_cube_arm ↔ transport_box_g1** (secondary — weaker
analogy, reported but not headlined).

Common to all tasks:

- Simulator: ManiSkill3 (SAPIEN), control mode `pd_joint_delta_pos`,
  actions normalized to [-1, 1].
- Observations via `UnifiedObsWrapper`: RGB `uint8 [3,128,128]` from the
  task camera, canonical proprio `[64]` + mask, instruction tokens `[16]`,
  embodiment id.
- Success is the env's own `evaluate()` — we never redefine stock success
  conditions; the custom Lang envs re-aim them at the language target.

## pick_cube_arm — `PickCubeLang-v1` (custom, ours)

| | |
|---|---|
| Embodiment | Panda (7-DOF arm + parallel gripper), third-person `base_camera` |
| Source | `src/xembench/envs/custom/pick_cube_lang.py`, subclass of stock PickCube |
| Instruction | `pick up the {color} cube and place it on the green target` |
| Scene | one cube per color of the active set + a **visible** floating green goal sphere (stock hides it; a pixel policy must see the target) |
| Success | named cube within 2.5 cm of goal AND robot static |
| Episode budget | 50 steps |
| Train split | colors {red, green, blue}, train-synonym paraphrases |
| Eval splits | heldout paraphrases; unseen colors {yellow, purple, cyan} via `color_set="unseen"` |
| Demo source | motion planning (mplib, Linux) — target-aware via the env's `cube` property |
| Limitations | goal marker is always green (goal color is not part of the language variation); 3 cubes fixed at build time (GPU-sim constraint) |

Language is load-bearing: with 3 cubes on the table, ignoring the
instruction caps expected success at ~33% even with perfect manipulation.

## stack_cube_arm — `StackCubeLang-v1` (custom, ours)

| | |
|---|---|
| Embodiment | Panda (`panda_wristcam` robot; we use `base_camera`) |
| Source | `src/xembench/envs/custom/stack_cube_lang.py`, subclass of stock StackCube |
| Instruction | `stack the {color} cube on top of the {color2} cube` (ordered pair, sampled per episode) |
| Success | source cube on base cube (xy within half-size, z offset one cube), static, not grasped |
| Episode budget | 50 steps |
| Train/eval splits | as pick_cube_arm; both colors randomized, order matters |
| Demo source | motion planning (Linux) via `cubeA`/`cubeB` target-aware properties |
| Limitations | ordered-pair sampling means 6 pairs from 3 colors; some pairs are visually harder (e.g. adjacent hues) |

## push_t_arm — `PushT-v1` (stock ManiSkill)

| | |
|---|---|
| Embodiment | `panda_stick` (no gripper; 7-dim native action) |
| Instruction | `push the t-shaped block onto the red target` (fixed scene → fixed instruction; paraphrases still vary phrasing) |
| Success | stock PushT criterion (tee overlaps target tee pose) |
| Episode budget | 100 steps |
| Splits | paraphrase split only — **no color/object variation** (scene is fixed) |
| Demo source | HuggingFace prebuilt demos, replayed to rgb + `pd_joint_delta_pos` |
| Limitations | language is descriptive, not discriminative, on this task; it measures instruction-robustness of a single-goal skill |

## pull_tool_arm — `PullCubeTool-v1` (stock ManiSkill)

| | |
|---|---|
| Embodiment | Panda |
| Instruction | `use the tool to pull the blue cube closer` |
| Success | stock criterion (cube pulled within reach using the L-tool) |
| Episode budget | 100 steps |
| Splits | paraphrase split only (fixed scene: red tool, blue cube) |
| Demo source | HF prebuilt demos replay; motion-planning solution also exists |
| Limitations | as PushT — language not discriminative; tests tool-use skill retention |

## apple_bowl_g1 — `UnitreeG1PlaceAppleInBowl-v1` (stock ManiSkill)

| | |
|---|---|
| Embodiment | Unitree G1 upper body (25-dim native action), egocentric `head_camera` (a third-person `base_camera` also exists — kept for the camera ablation) |
| Instruction | `pick up the red apple and place it in the bowl` — the **same pick_place family** as pick_cube_arm; that shared family is what makes the transfer pair meaningful |
| Success | apple in bowl AND hand withdrawn (stock `evaluate()`) |
| Episode budget | 100 steps |
| Splits | paraphrase split only (fixed apple/bowl scene) |
| Demo source | PPO expert on flat state + dense reward (no prebuilt demos exist for G1), rolled out in rgb mode, success-filtered |
| Limitations | fixed object colors → no unseen-color cell on the humanoid side; expert quality bounds demo quality (B2 gate: ≥70% expert success) |

## transport_box_g1 — `UnitreeG1TransportBox-v1` (stock ManiSkill)

| | |
|---|---|
| Embodiment | Unitree G1 upper body, `head_camera` |
| Instruction | `pick up the box and carry it to the other table` |
| Success | box on destination table AND released (stock `evaluate()`) |
| Episode budget | 100 steps |
| Splits | paraphrase split only |
| Demo source | PPO expert (dense reward is staged in the stock env) |
| Limitations | bimanual, large object — the hardest task; marked experimental if the expert gate fails (plan risk #1) |

## Adding a task

1. Pick or subclass an env (see the two `*Lang` subclasses for the pattern:
   override `_load_scene`/`_initialize_episode`, re-aim `evaluate()` and the
   dense reward at the language target, expose `language_slots()`).
2. Register it in `configs/tasks.yaml` with embodiment, camera, family,
   slots, `max_steps`, `demo_source`.
3. Add/extend an instruction template family in
   `src/xembench/language/templates.py`, rebuild the vocab
   (`uv run python -m xembench.language.tokenizer`), and check
   `tests/test_tokenizer.py` still proves full vocab coverage.
4. `uv run python -m xembench.scripts.setup_check` must pass — it validates
   action dims, cameras, and episode budgets against the registry.
