"""Adapter lookup by the embodiment name used in configs/tasks.yaml."""

from functools import lru_cache

from xembench.embodiment.base import EmbodimentAdapter
from xembench.embodiment.g1 import G1Adapter
from xembench.embodiment.panda import PandaAdapter, PandaStickAdapter

_ADAPTERS = {
    "panda": PandaAdapter,
    "panda_stick": PandaStickAdapter,
    "g1": G1Adapter,
}


@lru_cache(maxsize=None)
def get_adapter(name: str) -> EmbodimentAdapter:
    return _ADAPTERS[name]()
