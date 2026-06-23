# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture

Project structure:

```
├── scripts          # Scripts for e.g. downloading datasets
├── tools            # Tools for e.g. viewing datasets
├── src      
│   └── rsrch_data   # Main package
│       ├── types    # Type definitions for the dataset sample types
│       ├── utils    # Common utils
│       └── ...      # <Dataset modules>
```

### Dataset classes (`src/rsrch_data/`)

Each class follows this contract:

```python
from rsrch_data.registry import register_dataset


@register_dataset("<dataset name>")
class MyDataset(Sequence):
    """<brief info about the dataset>"""

    def __init__(self, data_root: str | Path, ...):
        """Initialize the dataset."""

    def __len__(self) -> int:
        """Get dataset length (# of samples)."""

    def __getitem__(self, index: int) -> Sample:
        """Get sample #index from dataset."""

    def meta(self) -> Metadata:
        """Get metadata about the dataset (optional)."""
```

> Note: The base class can be omitted, as long as the class is a sequence (i.e., implements `__len__` and `__getitem__`).

In some cases, when the dataset is iterable-only, the contract looks like:

```python
from rsrch_data.registry import register_dataset


@register_dataset("<dataset name>")
class MyDataset(Iterable):
    """<brief info about the dataset>"""

    def __init__(self, data_root: str | Path, ...):
        """Initialize the dataset."""

    def __len__(self) -> int: ... # optional
        """Get dataset length (# of samples)."""

    def __iter__(self) -> Iterator[Sample]:
        """Create dataset iterator."""

    def meta(self) -> Metadata:
        """Get metadata about the dataset (optional)."""
```

> Note: The base class can be omitted, as long as the class is an iterable (i.e., implements `__iter__`).

The `Sample` class is a `TypedDict` with specifies what the fields of the item are going
to be. For example:

```python
class Sample(TypedDict):
    image: Image.Image
    label: int
```

> Dataset classes should ALWAYS return dicts

The `Metadata` class is one of metadata classes defined in `rsrch_data.types`, depending
on the task at hand - image classification, segmentation, object detection etc.

The `Sample` class for the dataset should be a superset of the `Sample` class for the
given task type, found in `rsrch_data.types`, possibly including extra dataset-specific fields.

### TypedDict Subclassing

TypedDicts can be subclassed to extend or reuse base sample types:

```python
from rsrch_data.types.image_cls import Sample as BaseSample

# Direct import when fields are identical
from rsrch_data.types.image_cls import Sample

# Extend with additional fields
class ExtendedSample(Sample):
    extra_field: str
```

> Note: When a dataset needs optional fields (e.g., test split without labels), define
> a local TypedDict with `NotRequired` instead of subclassing.

If the metadata is static, a YAML file should be defined next to the Python file. Then, the
`meta` functon should look as follows:

```python
from ruamel.yaml import YAML

class MyDataset:
    ...

    @staticmethod
    def meta() -> Metadata:
        yaml = YAML(typ="safe", pure=True)
        with (Path(__file__).parent / "my_dataset.yml").open() as f:
            data = yaml.load(f)
        return Metadata(**data)
```

where the stem in `my_dataset.yml` should be replaced by an appropriate name.

### Download scripts (`scripts/`)

All scripts use `tyro` for CLI parsing. The standard form of such a script is:

```python
"""Download <dataset name> to a local directory."""
from pydantic import BaseModel
import tyro

class Args(BaseModel):
    """CLI args for the <dataset name> downloader."""

    data_root: str
    ... # other arguments

def main(args: Args) -> None:
    """Download <dataset name> to a local directory."""
    ...

if __name__ == "__main__":
    main(tyro.cli(Args))
```

## Linting

We use `ruff` for linting the codebase. By default we enable all rules, and disable
a number of them, which can be found in the `[tool.ruff]` section of `pyproject.toml`.