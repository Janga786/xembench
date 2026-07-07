# Phase B — Linux / RTX 5090 plan

The narrative plan: what to run, in what order, what can go wrong, and what
"done" means. The exact ordered command list lives in
[phase_b_commands.md](phase_b_commands.md) (`make phase-b-help` prints it).

## Assumptions

- Linux x86_64, NVIDIA driver new enough for the RTX 5090 (Blackwell,
  sm_120) — driver ≥ 570 series.
- Python 3.11 via uv (`.python-version` pins it). `uv sync` resolves
  **torch cu128 wheels** automatically on Linux (index gating in
  `pyproject.toml`); mplib installs automatically (mani_skill's own
  platform marker).
- ~50 GB free disk: demos are small (~0.3 GB/task) but checkpoints,
  TensorBoard logs, and the ManiSkill asset cache (~/.maniskill) add up.
- Everything runs from the repo root; no root privileges needed beyond the
  NVIDIA driver / (optional) Docker + NVIDIA Container Toolkit.

## Setup and verification (B0)

`./scripts/setup_linux.sh` does all of it: uv sync, `nvidia-smi`, a torch
CUDA assertion that **requires sm_120**, a Vulkan check, `setup_check`
across all 6 envs, and a GPU-parallel sim FPS benchmark (PickCubeLang rgb,
num_envs=256). Then `uv run python scripts/check_env.py` for the readable
report, and `make test && make test-sim` to prove the suites pass on this
machine too.

**First training run**: `configs/presets/linux_gpu_smoke.yaml` — a 5–10 min
PPO burn-in at num_envs=256. Gate: no crash, >5k env-steps/s in the console
log, success curve above zero within a few updates.

**What NOT to do first**: do not start with the G1 experts or any full
training before the burn-in passes; do not run Docker before native works
(one variable at a time); do not `pip install` anything ad hoc — uv owns
the environment.

## Work order and gates

The order de-risks the schedule: the two unknowns (mplib collection, G1
expert training) come first.

| Step | What | Gate before proceeding |
|---|---|---|
| B1 | Motion-planning demos for the 3 arm tasks (`experts.motion_plan_arm`, **untested until now** — expect small fixes) + HF replay for PushT | 200 eps/task; solver success ≥~80% of attempts; color distribution printed and covers all train colors; `validate_demo_file` clean; 3 spot-check videos |
| B2 | G1 PPO experts (dense reward, state obs) — **the schedule risk** | ≥70% success on apple_bowl_g1. If stalled after ~30M steps: try gamma 0.8, lr 1e-4, `reward_mode: normalized_dense`, longer training; check the hand-sign caveat (`G1_HAND_OPEN_SIGNS`). Worst case: ship PlaceAppleInBowl only, mark TransportBox experimental |
| B2b | Collect 200 G1 rgb demos via the expert | success-filtered by construction; `validate_demo_file`; eyeball 3 videos |
| B3 | BC per (task, embodiment) — 6 runs, `bc_train_full` preset | val loss converged; native eval success meaningfully above random on a 20-episode spot check |
| B4 | PPO fine-tune per task, `ppo_finetune_full` preset | success ≥ BC baseline; watch `losses/approx_kl` and `losses/bc_reg` for collapse |
| B5 | Full eval matrix (`full_eval_matrix` preset, 3 seeds) + reports + videos | every grid cell present in `data/results/eval.csv`; episode CSV written; `make`-generated reports sane |
| B6 | Stretch: diffusion policy, camera ablation, distractors | optional |

## Where outputs appear

```
data/demos/{embodiment}/{env_id}/{source}.h5      demonstrations
data/checkpoints/<run>/{best.pt,last.pt,config.yaml,tb/}
data/results/eval.csv, eval_episodes.csv          benchmark outputs
reports/summary.md, failures.md, transfer_matrix.{csv,md}
artifacts/videos/raw/*.mp4                        (naming: see artifacts/videos/README.md)
```

Every run directory contains `config.yaml` + git SHA — cite those in the
report appendix.

## Resuming after interruption

- **Demo collection**: collectors write episodes incrementally but open the
  file with mode `"w"` — an interrupted run leaves a partial file.
  Re-run to regenerate, or collect into a new `--set source=<tag>_part2`
  file (the BC trainer accepts multiple demo files). Don't append to a
  half-written file.
- **BC**: no mid-run resume (runs are <1 h; just restart). `best.pt` from a
  killed run is still usable if val loss had converged.
- **PPO**: checkpoints save every improvement (`best.pt`) and at the end
  (`last.pt`); restarting starts a fresh run dir — seed it as a new seed
  rather than pretending continuity.
- **Eval matrix**: CSVs are append-mode. If a run dies midway, delete the
  partial rows (filter by checkpoint/timestamp) or write to a fresh CSV via
  `--set csv=... --set episodes_csv=...`, then re-run remaining cells with
  `--set tasks=[...]`.

## Debugging quick table

| Symptom | Likely cause / fix |
|---|---|
| torch sees no GPU | driver/wheel mismatch — `nvidia-smi`, then `uv run python -c "import torch; print(torch.version.cuda)"`; must be cu128 |
| `no kernel image for device` | non-cu128 torch got installed — `uv sync --reinstall-package torch` |
| Vulkan errors headless | install `libvulkan1 vulkan-tools`; NVIDIA ICD at `/usr/share/vulkan/icd.d/nvidia_icd.json`; in Docker add `NVIDIA_DRIVER_CAPABILITIES=all` |
| mplib import errors (B1) | it only installs on Linux — confirm `uv sync` ran on THIS machine; `uv run python -c "import mplib"` |
| MP solver fails often on Lang envs | expected somewhat (3-cube clutter); acceptance is ≥~80% solve rate — investigate below that |
| G1 expert flatlines | see B2 gate row above; log a rollout video to check the hands actually close |
| GPU sim OOM | lower `num_envs`; rgb PPO buffers dominate (`num_steps × num_envs × 3×128×128`) |
| relative-path file-not-found mid-run | SAPIEN chdir'd — route the path through `resolve_path()` (tests/test_paths.py) |

## What success looks like

Phase B is done when: `data/results/eval.csv` covers the full grid at 3
seeds; the transfer matrix reports drops for both directions of both pairs
with CIs; failure distributions are generated from real episode records;
the four videos from docs/demo_video_plan.md exist; and every number in the
report traces to a checkpoint + config + git SHA.
