"""CLI display helpers."""

from rich.console import Console
from rich.table import Table


def print_table(
    columns: list[str],
    rows: list[tuple],
    *,
    title: str | None = None,
    caption: str | None = None,
) -> None:
    """Print rows as a rich table, with an optional title/caption."""
    table = Table(title=title, caption=caption)
    for col in columns:
        table.add_column(col)
    for row in rows:
        table.add_row(*(str(v) for v in row))
    Console().print(table)
