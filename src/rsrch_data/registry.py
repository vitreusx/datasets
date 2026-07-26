"""Registry of available datasets."""

import importlib.util
import pkgutil
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypeVar

import rsrch_data

_REGISTRY: dict[str, type] = {}
"""A registry of available datasets."""

T = TypeVar("T")


def _validate_dataset_shape(cls: type) -> None:
    """Enforce the `rsrch_data/CLAUDE.md` dataset contract's base class.

    `collections.abc.Sequence` has no `__subclasshook__` (unlike
    `Iterable`), so `isinstance`/`issubclass` checks against it only
    recognize classes that explicitly subclass it -- structural duck-typing
    off `__getitem__` alone isn't enough, unlike `Iterable`, which
    recognizes any class with `__iter__` regardless of its declared bases.
    So a random-access dataset must explicitly subclass `Sequence` for that
    to be reliable -- checked here, at registration time, rather than left
    as a documentation-only convention that silently stops mattering the
    moment some code (e.g. `rsrch.utils.data.DataLoader`) needs a real
    `isinstance`/`issubclass` check against it.
    """
    if hasattr(cls, "__getitem__"):
        if not issubclass(cls, Sequence):
            msg = (
                f"{cls.__qualname__} implements __getitem__ but doesn't "
                "subclass collections.abc.Sequence, which the dataset "
                "contract (see rsrch_data/CLAUDE.md) requires explicitly."
            )
            raise TypeError(msg)
    elif not hasattr(cls, "__iter__"):
        msg = (
            f"{cls.__qualname__} implements neither __getitem__ nor "
            "__iter__ -- every dataset must be either a Sequence (random "
            "access) or an Iterable (sequential-only); see "
            "rsrch_data/CLAUDE.md."
        )
        raise TypeError(msg)


def register_dataset(name: str) -> Callable[[T], T]:
    """Register dataset class."""

    def decorator(cls: T) -> T:
        _validate_dataset_shape(cls)
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
