# xembench — X-Embodiment Language-Grounded Manipulation Benchmark

Measures how well a **shared language-conditioned manipulation policy** transfers
between two robot embodiments simulated in [ManiSkill3](https://github.com/haosulab/ManiSkill):

1. **Franka Panda** 7-DOF tabletop arm
2. **Unitree G1** humanoid upper body (head-camera egocentric view)

Both embodiments receive the same natural-language instructions on analogous
tabletop tasks. The headline metric is the **success-rate drop under zero-shot
cross-embodiment transfer**: train on one robot, evaluate on the other by
swapping only a thin embodiment adapter.

## Pipeline

```
demos (motion planning / PPO experts / HF replay)
  -> behavior cloning (image + proprio + instruction -> canonical action)
  -> PPO fine-tuning (sparse reward)
  -> evaluation matrix: {original, paraphrased} x {seen, unseen colors}
                        x {native, cross-embodiment transfer}
  -> printed table + CSV
```

## Two-machine workflow

- **macOS (dev)**: everything builds and smoke-tests here — CPU physics,
  `num_envs=1`, tiny demo counts, unit tests, BC overfit sanity. `make smoke`.
- **Linux + RTX 5090 (scale)**: GPU-parallel sim, motion-planning demo
  generation (`mplib` is Linux-only), PPO experts, full training + eval.
  `scripts/setup_linux.sh` installs cu128 PyTorch (Blackwell needs sm_120).

## Setup

```bash
# macOS
./scripts/setup_mac.sh          # uv sync + env smoke check

# Linux (RTX 5090)
./scripts/setup_linux.sh        # uv sync (pulls cu128 torch) + GPU sim check

# Optional: Docker on Linux
docker build -f docker/Dockerfile.cu128 -t xembench .
docker run --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all -it xembench
```

## Entry points

```bash
uv run python -m xembench.scripts.setup_check          # create all envs, pin dims
uv run python -m xembench.scripts.download_demos       # HF demo datasets (arm)
uv run python -m xembench.scripts.collect_demos --config configs/collect.yaml
uv run python -m xembench.scripts.train_expert_ppo --config configs/ppo_expert_g1.yaml
uv run python -m xembench.scripts.train_bc --config configs/bc.yaml
uv run python -m xembench.scripts.train_ppo_finetune --config configs/ppo_finetune.yaml
uv run python -m xembench.scripts.evaluate --config configs/eval_matrix.yaml
uv run python -m xembench.scripts.make_report
```

Every script accepts `--seed N` and `--set key=value` overrides; artifact
directories receive a config dump + git SHA for reproducibility.

## Tests

```bash
uv run pytest            # fast, sim-free suite
uv run pytest -m sim     # slow: creates real ManiSkill envs
make smoke               # full Phase-A chain (~15 min on an M-series Mac)
```
