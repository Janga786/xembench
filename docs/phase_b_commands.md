# Phase B — ordered command sequence (Linux / RTX 5090)

Run top to bottom; don't skip gates. Context and debugging:
[phase_b_linux_plan.md](phase_b_linux_plan.md). Runtime categories:
seconds / minutes / ~1 h / hours. "Mac-safe: yes" = can also run on the
MacBook (slowly); everything is repo-root relative.

---

### 1. Setup Linux — minutes · Linux-only · output: .venv, FPS benchmark log
```bash
./scripts/setup_linux.sh
```
Fails on: missing NVIDIA driver, non-sm_120 wheel resolution. Debug:
`nvidia-smi`, then `uv run python -c "import torch; print(torch.__version__, torch.version.cuda)"`.

### 2. Verify environment — seconds · Mac-safe: yes
```bash
uv run python scripts/check_env.py
make test && make test-sim
```
Expect: `environment OK`, 110 fast + 5 sim tests pass. mplib must report
available (Linux).

### 3. Full smoke — ~10 min · Mac-safe: yes · output: data/checkpoints/*smoke*, data/results/eval_smoke.csv
```bash
make smoke
```
Green banner `=== PHASE A SMOKE GREEN ===` proves the whole pipeline on
this machine before any real compute is spent.

### 4. GPU-parallel sim burn-in — 5–10 min · Linux-only · output: data/checkpoints/linux_gpu_smoke/
```bash
uv run python -m xembench.scripts.train_expert_ppo --config configs/presets/linux_gpu_smoke.yaml
```
Gate: no crash, sps > 5000 in the log, success curve rises above zero.

### 5. Tiny Panda demo collection (mplib shakedown) — minutes · LINUX-ONLY (mplib) · output: data/demos/panda/PickCubeLang-v1/motion_planning.h5
```bash
uv run python -m xembench.experts.motion_plan_arm --config configs/presets/panda_demo_collection.yaml --set episodes=5
uv run python -c "from xembench.data.schema import validate_demo_file as v; print(v('data/demos/panda/PickCubeLang-v1/motion_planning.h5') or 'VALID')"
```
Shaken down during B0 Linux prep (5/5 episodes, validator clean, ~1 s per
episode — 200 eps/task is minutes). The solver env runs pd_joint_pos on CPU
sim internally; recorded natives are the equivalent pd_joint_delta_pos
actions. Failure notes: import errors → mplib missing; solver exceptions on
>20% of attempts → inspect with 1 episode + printouts; a clipped-delta
warning at the end means the planner outran the ±0.1 rad/step delta bound.

### 6. Tiny G1 demo path check (random policy) — minutes · Mac-safe: yes · output: data/demos/g1/.../random_check.h5
```bash
uv run python -m xembench.scripts.collect_demos --set task=apple_bowl_g1 --set policy=random --set episodes=2 --set success_only=false --set source=random_check
```
Pipeline check only (random ≠ expert). Real G1 demos come after step 9.

### 7. BC small — minutes · Mac-safe: yes · output: data/checkpoints/bc_pick_cube_arm_seed0/
```bash
uv run python -m xembench.scripts.train_bc --config configs/presets/bc_train_small.yaml --set task=pick_cube_arm
```
Gate: train/val loss clearly decreasing on the 5-episode B1 shakedown data.

### 8. Eval small (incl. one transfer cell) — minutes · Mac-safe: yes · output: data/results/eval_smoke*.csv
```bash
uv run python -m xembench.scripts.evaluate --config configs/presets/phase_a_eval.yaml --set policies='[{type: bc, ckpt: "data/checkpoints/bc_pick_cube_arm_seed0/best.pt"}]'
```
Success rates are meaningless here; the gate is that both CSVs get rows and
the transfer summary prints.

### 9. G1 experts (THE de-risk item) — ~1–3 h per task · LINUX-ONLY · output: data/checkpoints/expert_apple_bowl_g1_seed0/
```bash
uv run python -m xembench.scripts.train_expert_ppo --config configs/presets/g1_expert_train.yaml --set task=apple_bowl_g1
uv run python -m xembench.scripts.train_expert_ppo --config configs/presets/g1_expert_train.yaml --set task=transport_box_g1
```
GATE: ≥70% success on apple_bowl_g1 (console + TensorBoard
`charts/success_rate`) before proceeding. Stalled → gamma 0.8 / lr 1e-4 /
normalized_dense / re-run `scripts/g1_hand_sign_check.py` (signs were
verified+fixed in B0 prep; see docs/phase_b_linux_plan.md).

### 10. Full demo collection — ~1–2 h per task · LINUX-ONLY · output: data/demos/**/*.h5
```bash
# arm tasks (motion planning):
for t in pick_cube_arm stack_cube_arm pull_tool_arm; do
  uv run python -m xembench.experts.motion_plan_arm --config configs/presets/panda_demo_collection.yaml --set task=$t
done
# PushT (HF replay; downloads once, replays 200):
uv run python -m xembench.scripts.download_demos PushT-v1
uv run python -m mani_skill.trajectory.replay_trajectory --traj-path ~/.maniskill/demos/PushT-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.h5 -o rgb --save-traj --count 200 -b physx_cpu
uv run python -m xembench.data.convert_ms_demos --input ~/.maniskill/demos/PushT-v1/rl/trajectory.rgb.pd_joint_delta_pos.physx_cpu.h5 --task push_t_arm --max-episodes 200
# G1 (expert rollouts):
uv run python -m xembench.scripts.collect_demos --config configs/presets/g1_expert_collection.yaml
uv run python -m xembench.scripts.collect_demos --config configs/presets/g1_expert_collection.yaml --set task=transport_box_g1 --set policy=ppo:data/checkpoints/expert_transport_box_g1_seed0/best.pt
```
Gate per file: `validate_demo_file` clean; episode counts = 200; arm color
distribution covers all train colors. (Exact PushT replay filename may
differ — `ls ~/.maniskill/demos/PushT-v1/`.)

### 11. Full BC training — <1 h per (task, seed) · Linux recommended · output: data/checkpoints/bc_{task}_seed{seed}/
```bash
for t in pick_cube_arm stack_cube_arm push_t_arm pull_tool_arm apple_bowl_g1 transport_box_g1; do
  for s in 0 1 2; do
    uv run python -m xembench.scripts.train_bc --config configs/presets/bc_train_full.yaml --set task=$t --seed $s
  done
done
```
(Seeds 1–2 can be limited to the transfer-pair tasks if time-boxed.)

### 12. PPO fine-tuning — 4–12 h per task · LINUX-ONLY · output: data/checkpoints/ppo_{task}_seed{seed}/
```bash
for t in pick_cube_arm stack_cube_arm apple_bowl_g1; do
  uv run python -m xembench.scripts.train_ppo_finetune --config configs/presets/ppo_finetune_full.yaml --set task=$t
done
```
Gate: success ≥ BC baseline; abort a run whose success collapses while KL
spikes (raise `bc_reg_coef`).

### 13. Full eval matrix — hours · Linux recommended · output: data/results/eval.csv + eval_episodes.csv
```bash
uv run python -m xembench.scripts.evaluate --config configs/presets/full_eval_matrix.yaml
```
Gate: every cell present; missing-checkpoint cells only for `optional`
policies. Interrupted → fresh CSV names or prune partial rows (plan doc).

### 14. Reports + artifact pack — seconds · Mac-safe: yes · output: reports/*, artifacts/releases/phase_b_v1/
```bash
uv run python scripts/make_report_assets.py --input data/results/eval_episodes.csv
uv run python -m xembench.artifacts.build_pack \
  --eval-csv data/results/eval_episodes.csv \
  --out artifacts/releases/phase_b_v1 --release-name phase_b_v1 \
  --configs configs/presets/full_eval_matrix.yaml --configs configs/tasks.yaml \
  --videos artifacts/videos
```
The pack bundles metrics, figures, manifest (git commit + env versions),
config snapshots, video index, and project-page assets — the report is
written FROM it (reports/templates/report_skeleton.md). Rehearsed on Mac
by `make artifact-pack-toy`.

### 15. Videos — ~1 h · Linux recommended · output: artifacts/videos/raw/*.mp4 + INDEX.md
Follow [demo_video_plan.md](demo_video_plan.md) (uses `episode_seed`s from
the episode CSV to re-run and record exact episodes; requires the small
recording hook flagged as TODO in that doc). Then:
```bash
uv run python -m xembench.analysis.video_index
```

---

Done = the gates in steps 9–13 all passed and
[phase_b_linux_plan.md — "What success looks like"](phase_b_linux_plan.md)
holds.
