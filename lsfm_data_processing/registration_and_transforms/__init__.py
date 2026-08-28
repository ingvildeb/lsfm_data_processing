"""Registration orchestration built on the canonical atlasspace runtime."""

from importlib import import_module
from types import ModuleType

__all__ = [
    "batch_registration",
    "sweep_registration",
]


def __getattr__(name: str) -> ModuleType:
    if name not in __all__:
        raise AttributeError(name)
    return import_module(f"{__name__}.{name}")
