# B0 Linux Prep Report — RTX 5090 · 2026-07-07

How the repo went from the macOS-frozen `v0.3.1` to Phase-B-ready on the
Linux / RTX 5090 box: what was run, what broke, what was fixed, and what
remains risky. Work performed on branch `phase-b-linux-prep`;
result committed as `ad73a8b` and tagged **`v0.3.2-linux-ready`** (branch and
tag on origin). Companion docs:
[phase_b_linux_plan.md](../docs/phase_b_linux_plan.md) ·
[phase_b_commands.md](../docs/phase_b_commands.md).

**Verdict: READY FOR TRAINING.** Every gate green, both previously untested
Linux-only code paths (mplib demo collection, G1 hand control) shaken down,
and two real bugs found and fixed before they could cost training time.

---

## Machine / environment

| Component | Value |
|---|---|
| OS / kernel | Ubuntu 24.04.3 LTS · 6.17.0-35-generic |
| GPU | NVIDIA GeForce RTX 5090 (32.6 GB), driver **580.159.03** |
| Python | **3.10.20** via uv (re-pinned from 3.11 — see fixes) |
| torch | 2.11.0+cu128 · CUDA 12.8 · capability **sm_120** |
| ManiSkill / SAPIEN | 3.0.1 / 3.0.3 |
| mplib / numpy | 0.1.1 / 1.26.4 (pinned `<2` — see fixes) |
| Disk / RAM | 3.0 TB free / 30 GB |

## Gate results (all green)

| Gate | Result |
|---|---|
| `./scripts/setup_linux.sh` | OK after Python re-pin; 6/6 envs validated; 8,048 env-steps/s raw sim @ 256 envs |
| `uv run python scripts/check_env.py` | `environment OK`; mplib available |
| `make test` | **110 passed** (runbook said 75 — count updated) |
| `make test-sim` | **5 passed** |
| `make smoke` | `=== PHASE A SMOKE GREEN ===` |
| `make artifact-pack-toy` | pack built; SYNTHETIC-labeled; 8 figures; checksums + manifest; no home-path leaks |
| GPU PPO shakedown (32,768 steps, 256 envs) | ~9,200–9,900 sps, checkpoints/config/tb written |
| mplib motion-planning collection (1, then 5 eps) | **5/5 success**, `validate_demo_file` → VALID |
| G1 random-policy demo path (2 eps) | VALID |
| `scripts/g1_hand_sign_check.py` (new) | **14/14 PASS** after sign fix |

The final gate list (`check_env` → `make test` → `make test-sim` →
`make smoke` → `make artifact-pack-toy`) was re-run in one pass at the end,
after all fixes: exit 0 throughout.

## Bugs found and fixed

### 1. `motion_plan_arm` drove solvers in the wrong control mode (hard B1 blocker)

First-ever run of the module (it was untestable on macOS — mplib is
Linux-only) failed on every attempt with
`Received action of shape torch.Size([15]) but expected shape (1, 8)`.
ManiSkill's motion-planning solutions only speak `pd_joint_pos` (absolute
joint targets); in any other control mode their `open/close_gripper`
helpers emit 15-dim pos+vel actions. The env was being created in the
benchmark's `pd_joint_delta_pos`.

**Fix** (`src/xembench/experts/motion_plan_arm.py`): the solver env now runs
`pd_joint_pos` on CPU sim (the backend upstream solutions are written
against), and the recorder converts each absolute arm target into the
equivalent normalized `pd_joint_delta_pos` action
(`delta = (target − qpos) / 0.1`, clipped to ±1 with a warning counter), so
demo files stay in the benchmark control mode. The delta-controller config
(`use_delta`, symmetric bounds) is asserted at startup. Verified: 5/5
episodes, validator clean, actions finite in [−1, 1], color-grounded
instructions, ~1 s/episode — so B1's 200 eps/task is **minutes**, not the
1–2 h/task the runbook budgeted.

### 2. `G1_HAND_OPEN_SIGNS` closed the right hand on an "open" command (silent B2b corruption)

The frozen constants assumed +1 = open for all 14 hand joints, with a
documented "refine empirically in B2" caveat. The new diagnostic
`scripts/g1_hand_sign_check.py` (joint-limit sweep + rendered frames) showed
the G1 hand URDF is **mirrored side-to-side** — one-sided joint limits flip
between hands — so all-(+1) opened the left hand but curled the **right**
hand closed. Confirmed visually in 512-px renders.

**Fix** (`src/xembench/constants.py`): right hand = −1, left hand = +1.
Mechanical check now 14/14 PASS; before/open/close frames saved to
`data/shakedown/g1_hand_sign/` for human re-verification. Blast radius had
it shipped: PPO experts act natively (unaffected), but every canonical-space
G1 demo (B2b) and the arm→G1 transfer evaluation — the paper's headline
number — would have been silently wrong.

## Environment fixes

- **Python 3.11 → 3.10** (`.python-version`): the 3.11 pin could never
  resolve on Linux — toppra (mplib's dependency → ManiSkill motion
  planning) ships cp310-only wheels and no sdist. Never surfaced on macOS
  because mplib is skipped there.
- **`numpy>=1.24,<2`** (`pyproject.toml` + `uv.lock` re-resolve): toppra's
  Cython extension is built against the NumPy 1.x C-ABI; `import mplib`
  crashed under numpy 2.2.
- **`unexport PYTHONPATH`** (`Makefile`): this machine sources ROS Jazzy in
  `~/.bashrc`, which leaks `/opt/ros/jazzy/.../site-packages` into the uv
  venv; pytest then died auto-loading a broken ROS plugin (`lark` missing).
  Both failure modes were added to the plan doc's debugging table.

## Shakedown details

- **GPU sim (Phase 5)**: tiny PPO burn-in (`linux_gpu_smoke` preset capped
  at 32,768 steps) ran end-to-end in ~7 s at ~9,200–9,900 sps on the 5090 —
  no Blackwell/compile/runtime issues; outputs landed in
  `data/shakedown/linux_gpu_smoke_tiny/` (best.pt, last.pt, config.yaml, tb/).
- **mplib / Panda (Phase 6)**: planner initialized; 1-episode then
  5-episode runs to `data/shakedown/mplib_panda/motion_planning_shakedown.h5`;
  HDF5 validator clean; colors {blue: 2, green: 3}; one clipped-delta step
  across 5 episodes (now warned at end of run).
- **G1 env (Phase 7)**: both G1 tasks create/reset; adapter validated
  (25 native / 21 canonical dims, controller joint order matches pinned
  constants); random-policy collection wrote 2 VALID episodes; no NaNs.
- **Artifact pack (Phase 8)**: 240 synthetic episode rows → full pack with
  figures, checksums, manifest, project-page assets; grep confirmed no
  `/home/janga` or `/Users/` leaks.

## Files changed (commit `ad73a8b`, 10 files, +437/−374)

`.python-version` · `pyproject.toml` · `uv.lock` · `Makefile` ·
`src/xembench/constants.py` · `src/xembench/experts/motion_plan_arm.py` ·
`scripts/g1_hand_sign_check.py` (new) · `docs/phase_b_commands.md` ·
`docs/phase_b_linux_plan.md` · `.gitignore` (+`data/shakedown/`)

No datasets, checkpoints, videos, or secrets committed.

## Known remaining risks

1. **B2 G1 expert training** — still the schedule risk the plan names;
   untested here by design (hours-long).
2. **Demo length vs eval horizon** — MP demos run 66–74 steps but
   pick/stack tasks truncate at `max_steps: 50` during eval. Pre-existing
   design tension, not Linux-specific; watch B3 native eval.
3. **Hand-sign semantics** — mechanically verified + frame-consistent, but
   grasp-under-load is unobservable until B2; re-run
   `scripts/g1_hand_sign_check.py` if grasping stalls.
4. Minor: ~1 clipped delta step per 5 MP episodes (warned, harmless at this
   rate); `grpcio` 1.82.0 is yanked upstream (cosmetic).

## Post-prep events (same evening, run by Janga)

- Branch `phase-b-linux-prep` and tag `v0.3.2-linux-ready` pushed to origin.
- Hand-sign frames (`data/shakedown/g1_hand_sign/*.png`) opened for visual
  review.
- **Runbook step 4** (full `linux_gpu_smoke` burn-in, 500k steps) completed:
  61/61 updates, no crash, sustained ~8.1–9.5k sps (gate: >5k ✓); best
  success **50%** checkpointed to `data/checkpoints/linux_gpu_smoke/`.
  Honest read: the throughput half of the gate passed decisively; the
  "success rises above zero within a few updates" half was only marginally
  met — a single nonzero window at update 40.
- **Second-seed check** (`--seed 1`, output to
  `data/shakedown/linux_gpu_smoke_seed1/`, 1m05s): 61/61 updates, ~8.2k sps,
  one nonzero window at **update 37 (100%)**. So the signal reproduces —
  both seeds start producing successful episodes at ~300k steps. Reading
  `training/ppo.py` explains the log shape: `nan%` windows are 32-step
  rollout windows containing no episode boundary (episodes run 50 steps);
  successful episodes *terminate early*, so a step where only successes end
  contributes a 1.0 sample, while the all-256-env truncation step
  contributes ≈0 — and the per-update stat is an unweighted mean over those
  per-step means. The metric therefore under-reports late-emerging success;
  the learning itself is real and reproducible. Gate: **pass**. Caveat
  carried forward: the same mean-of-means stat feeds TensorBoard
  `charts/success_rate`, which is the B2 "≥70% on apple_bowl_g1" gate —
  consider weighting by episodes-per-step (small `_episode_stats` change)
  before trusting that readout near the threshold.

## Next commands (runbook steps 5 →)

```bash
cd ~/xembench
# 5. Tiny Panda demo collection to the runbook path (already shaken down):
uv run python -m xembench.experts.motion_plan_arm --config configs/presets/panda_demo_collection.yaml --set episodes=5
uv run python -c "from xembench.data.schema import validate_demo_file as v; print(v('data/demos/panda/PickCubeLang-v1/motion_planning.h5') or 'VALID')"
# 6–8. G1 random demo path, BC small, eval small (minutes each), then:
# 9. G1 experts — THE de-risk item (~1–3 h/task):
uv run python -m xembench.scripts.train_expert_ppo --config configs/presets/g1_expert_train.yaml --set task=apple_bowl_g1
```

Gates and fallbacks per step: `make phase-b-help`.
