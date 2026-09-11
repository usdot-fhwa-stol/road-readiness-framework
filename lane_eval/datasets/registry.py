from __future__ import annotations
from typing import Type

_REGISTRY: dict[str, Type] = {}


def register_dataset(name: str):
    """Class decorator — registers an adapter under the given name."""
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_dataset_adapter(name: str):
    if name not in _REGISTRY:
        raise KeyError(
            f"Dataset '{name}' not registered. Available: {list(_REGISTRY)}"
        )
    return _REGISTRY[name]


def build_dataset_adapter(name: str, **kwargs):
    return get_dataset_adapter(name)(**kwargs)
