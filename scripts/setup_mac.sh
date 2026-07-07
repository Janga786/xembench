#!/usr/bin/env bash
# macOS dev setup (temporary dev machine; Linux 5090 is the real target).
set -euo pipefail
cd "$(dirname "$0")/.."

command -v brew >/dev/null || { echo "Homebrew required: https://brew.sh"; exit 1; }
command -v uv >/dev/null || brew install uv
brew list molten-vk >/dev/null 2>&1 || brew install molten-vk  # Vulkan ICD for SAPIEN

uv sync

# The Makefile exports these too; export in your shell for bare mani_skill CLIs:
export VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json
export VK_DRIVER_FILES=$VK_ICD_FILENAMES

uv run python -m xembench.scripts.setup_check
echo "mac setup OK — run 'make smoke' for the full Phase-A gate"
