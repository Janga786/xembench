# Phase A status (audited 2026-07-07, macOS)

> Update, same day: the Phase A.5 polish pass this snapshot preceded is now
> complete (v0.2) — docs set, config presets, analysis tooling with
> per-episode records, 75 fast + 5 sim tests, CI, and the Phase B runbook
> (docs/phase_b_commands.md). Phase B remains pending Linux.

Snapshot of what exists, what was verified, and how — recorded before the
Phase A.5 polish pass. Nothing in this file is a claimed research result;
success rates from smoke runs are meaningless by design (tiny data, tiny
training budgets) and exist only to prove the pipeline executes.

## Verified baseline

Commands run on macOS (Apple Silicon, CPU/MPS only, no CUDA):

| Command | Result |
|---|---|
| `uv run pytest -q` | 39 passed (fast, sim-free suite) |
| `uv run pytest -m sim -q` | 5 passed (creates real ManiSkill envs, CPU sim) |
| `make smoke` (Phase A exit gate, prior session) | green: `=== PHASE A SMOKE GREEN ===` |

`make smoke` chains: env creation checks for all 6 tasks → both pytest
suites → HF demo download/replay/convert slice (10 PickCube episodes) → BC
overfit sanity (loss 0.22 → ~0.03) → eval matrix smoke including one
arm→G1 zero-shot transfer cell → state-PPO and visual-PPO no-crash runs.

## What is implemented and smoke-verified

- 6 benchmark envs create, step, and render on macOS CPU sim
  (`xembench.scripts.setup_check`), including the two custom
  language-grounded envs `PickCubeLang-v1` / `StackCubeLang-v1`.
- Canonical 21-dim action interface + Panda / panda_stick / G1 adapters,
  with round-trip and live-env validation tests.
- Language stack: templates, leakage-proof train/heldout paraphrase split,
  committed tokenizer vocab, keyword parser.
- HDF5 demo schema with two validated ingestion paths (HuggingFace replay
  conversion; live collection wrapper).
- BC trainer (masked SmoothL1 on canonical actions) — overfit sanity passed
  on MPS.
- CleanRL-style PPO, two modes: flat-state dense-reward expert and
  sparse-reward visual fine-tune of a BC checkpoint. Both complete tiny
  no-crash runs on CPU.
- Evaluation matrix producing the aggregate results CSV + printed transfer
  table, including the adapter-swap transfer mechanism.

## What is NOT implemented / not run yet (Phase B, Linux RTX 5090)

- No demonstration datasets at scale (only 10-episode pipeline-check slices).
- No trained BC/PPO policies with meaningful success rates. **No benchmark
  numbers exist yet.**
- Motion-planning demo collection (`xembench.experts.motion_plan_arm`) is
  written but UNTESTED — `mplib` is Linux-only.
- G1 PPO experts not trained (need GPU-parallel sim).
- Diffusion policy is a design stub only (`models/diffusion_policy.py`).
- GPU-parallel simulation paths (`sim_backend="gpu"`, `num_envs >> 1`)
  unexercised.

## Known platform facts (hard-won, do not rediscover)

- macOS rendering needs MoltenVK: `VK_ICD_FILENAMES` / `VK_DRIVER_FILES` →
  `/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json`. Set automatically in
  `xembench/__init__.py` and exported by the Makefile. An *empty*
  `VK_DRIVER_FILES` breaks the loader entirely.
- MPS `non_blocking=True` host→device copies from numpy-backed tensors
  deliver garbage; `xembench.utils.torch_utils.to_device` restricts
  non_blocking to CUDA.
- SAPIEN changes the process working directory during shader compilation;
  every path that crosses an env-creation boundary must be absolute
  (`xembench.utils.config.resolve_path`).
- RTX 5090 (Blackwell, sm_120) needs `torch>=2.7` cu128 wheels — gated to
  Linux in `pyproject.toml` via a uv index marker.
- G1 supports only `pd_joint_delta_pos` / `pd_joint_pos`; the shared control
  mode across embodiments is `pd_joint_delta_pos`.
- `G1_HAND_OPEN_SIGNS` (constants.py) assumes +1 = open for all hand joints;
  unverified — check empirically during B2 if grasping underperforms.
