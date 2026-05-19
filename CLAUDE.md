# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture

The repo has following parts:

1. Dataset classes in `src/rsrch_data/`;
2. Download scripts in `scripts/`;
3. Dataset metadata and typing in `src/rsrch_data/types`;
4. Utility functions in `src/rsrch_data/utils`.

### Dataset classes (`src/rsrch_data/`)

Each class follows this contract:

```python
class MyDataset(Sequence):
    def __init__(self, data_root: str | Path, , ...): ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Sample: ...

    def meta(self) -> Metadata: ...
```

In some cases, when the dataset is only-iterable, the contract looks like:

```python
class MyDataset(Iterable):
    def __init__(self, data_root: str | Path, , ...): ...
    def __len__(self) -> int: ... # optional
    def __iter__(self) -> Iterator[Sample]: ...

    def meta(self) -> Metadata: ...
```

The `Sample` class is a `TypedDict` with specifies what the fields of the item are going
to be. For example:

```python
class Sample(TypedDict):
    image: Image.Image
    label: int
```

The `Metadata` class is one of metadata classes defined in `rsrch_data.types.*`, depending
on the task at hand - image classification, segmentation, object detection etc.

The `Sample` class for the dataset should be a superset of the `Sample` class for the
given task type, possibly including extra dataset-specific fields.

If the metadata is static, a YAML file is defined next to the Python file. Then, the
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

where `my_dataset.yml` should be replaced by appropriate name.

### Download scripts (`scripts/`)

All scripts use `tyro` for CLI parsing. The standard form of such a script is:

```python
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

`ruff` runs with `select = ["ALL"]`. Key non-obvious ignores (see `pyproject.toml`):

- `S301` — `pickle.load` is allowed (trusted local checkpoints)
- `D105` / `D107` — magic methods and `__init__` don't need docstrings
- `INP001` — implicit namespace packages are intentional
- `FBT001/FBT002` — boolean args are fine in dataset constructors; only scripts need the `*` separator

Every public module, class, and function needs a docstring. Exception messages must be assigned to a variable before raising (`TRY003/EM101`): `msg = "..."; raise ValueError(msg)`.
