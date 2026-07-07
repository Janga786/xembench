"""Environment construction: importing this package registers all custom envs."""

from xembench.envs import custom  # noqa: F401  (registers *Lang-v1 envs)
from xembench.envs.factory import make_env  # noqa: F401
