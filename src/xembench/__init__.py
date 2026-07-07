"""X-Embodiment Language-Grounded Manipulation Benchmark.

Measures how well a shared language-conditioned manipulation policy transfers
between two embodiments (Franka Panda arm, Unitree G1 humanoid upper body)
simulated in ManiSkill3.
"""

import os
import platform

__version__ = "0.1.0"

# macOS: SAPIEN bundles a Vulkan loader but no ICD. Point the loader at
# Homebrew's MoltenVK so rendering works. Must happen before sapien import,
# so it lives here at the package root (every entrypoint imports xembench).
if platform.system() == "Darwin" and not os.environ.get("VK_ICD_FILENAMES"):
    _icd = "/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json"
    if os.path.isfile(_icd):
        os.environ["VK_ICD_FILENAMES"] = _icd
        os.environ["VK_DRIVER_FILES"] = _icd
