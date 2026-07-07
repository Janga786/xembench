#!/usr/bin/env bash
# Linux + RTX 5090 setup (B0). Installs cu128 torch via the platform-gated
# index in pyproject.toml, verifies Vulkan + GPU sim, pins nothing new.
set -euo pipefail
cd "$(dirname "$0")/.."

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

echo "== GPU =="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || {
  echo "nvidia-smi failed — install the NVIDIA driver first"; exit 1; }

echo "== torch/CUDA =="
uv run python - <<'EOF'
import torch
assert torch.cuda.is_available(), "torch does not see CUDA — check driver / cu128 wheels"
cap = torch.cuda.get_device_capability()
print(f"torch {torch.__version__} cuda {torch.version.cuda} capability sm_{cap[0]}{cap[1]}")
assert cap >= (12, 0), f"expected Blackwell (sm_120+), got sm_{cap[0]}{cap[1]}"
EOF

echo "== Vulkan (rendering) =="
if command -v vulkaninfo >/dev/null; then
  vulkaninfo --summary 2>/dev/null | grep -m1 deviceName || true
else
  echo "vulkaninfo not found (apt install vulkan-tools) — SAPIEN may still work via bundled loader"
fi

echo "== env creation check =="
uv run python -m xembench.scripts.setup_check

echo "== GPU-parallel sim throughput =="
uv run python - <<'EOF'
import time, torch
from xembench.envs import make_env
env = make_env("PickCubeLang-v1", num_envs=256, obs_mode="rgb", sim_backend="gpu")
obs, _ = env.reset(seed=0)
t0 = time.time(); steps = 100
for _ in range(steps):
    obs, *_ = env.step(env.action_space.sample())
fps = 256 * steps / (time.time() - t0)
print(f"PickCubeLang rgb num_envs=256: {fps:,.0f} env-steps/s")
env.close()
EOF
echo "linux setup OK"
