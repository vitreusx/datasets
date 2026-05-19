"""Utility functions for data types."""


def is_contiguous(xs: list[int]) -> bool:
    """Check whether integer list is contiguous."""
    return max(xs) - min(xs) + 1 == len(xs)
