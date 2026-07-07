"""Custom language-grounded env variants (registered on import)."""

import xembench  # noqa: F401  (macOS Vulkan ICD setup before sapien loads)

from xembench.envs.custom.pick_cube_lang import PickCubeLangEnv  # noqa: F401
from xembench.envs.custom.stack_cube_lang import StackCubeLangEnv  # noqa: F401
