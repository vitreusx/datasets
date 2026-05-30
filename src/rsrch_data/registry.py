"""Registry of available datasets."""

import importlib.util
import pkgutil
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import rsrch_data

_REGISTRY: dict[str, type] = {}
"""A registry of available datasets."""

T = TypeVar("T")


def register_dataset(name: str) -> Callable[[T], T]:
    """Register dataset class."""

    def decorator(cls: T) -> T:
        if name in _REGISTRY and cls is not _REGISTRY[name]:
            msg = (
                f"Dataset with name {name} already exists: "
                f"{_REGISTRY[name].__qualname__}"
            )
            raise ValueError(msg)
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_registry() -> dict[str, type]:
    """Get the registry of classes.

    Because `REGISTRY` is only updated when importing dataset modules, we
    import all modules in the package containing a reference to `register_dataset`
    function to discover all datasets.
    """
    for module_info in pkgutil.walk_packages(rsrch_data.__path__, prefix="rsrch_data."):
        spec = importlib.util.find_spec(module_info.name)
        if spec is not None and spec.origin is not None:
            with Path(spec.origin).open() as f:
                source_code = f.read()
            if register_dataset.__name__ in source_code:
                importlib.import_module(module_info.name)
    return _REGISTRY
