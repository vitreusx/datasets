"""Color utilities for palette generation and label visualization."""

from dataclasses import dataclass
from functools import cache

import numpy as np


def rgb2hex(rgb: tuple[int, int, int]) -> str:
    """Convert an RGB tuple to a hex color string."""
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"


def hex2rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert a hex color string to an RGB tuple."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return r, g, b


@cache
def create_base_palette() -> np.ndarray:
    """Create a general palette starting from ``#000000``.

    Selects each successive color by maximizing Euclidean distance in RGB space
    from the colors already in the palette.
    """
    from scipy.spatial.distance import cdist

    # Create grid of colors with spacing 32
    q = np.clip(np.arange(0, 257, 32), 0, 255)
    colors = np.stack(np.meshgrid(q, q, q), axis=-1).reshape(-1, 3)

    # Initialize palette as [#000000, #ffffff]
    palette = np.empty_like(colors)
    palette[0] = 0
    palette[1] = 255

    # Select consecutive colors by maximizing the distance to the current palette
    for i in range(1, len(colors)):
        dists = cdist(colors, palette[:i])
        palette[i] = colors[np.argmax(dists.min(-1))]

    palette = palette.astype(np.uint8)
    return palette


@dataclass
class Palette:
    """A color palette mapping integer labels to RGB colors."""

    colors: np.ndarray
    ignore_color: tuple[int, int, int] | None

    def label2rgb(
        self,
        labels: np.ndarray,
        ignore_index: int | None = None,
    ) -> np.ndarray:
        """Map an array of integer labels to an RGB image."""
        if ignore_index is None:
            return self.colors[labels]
        output = np.empty(
            shape=[*labels.shape, self.colors.shape[-1]],
            dtype=self.colors.dtype,
        )
        mask = labels != ignore_index
        if ignore_index == 0:
            # Reduce zero label
            output[mask] = self.colors[labels[mask] - 1]
        else:
            output[mask] = self.colors[labels[mask]]
        output[~mask] = self.ignore_color
        return output


def get_palette(
    num_classes: int,
    ignore_index: int | None = None,
) -> Palette:
    """Build a ``Palette`` for *num_classes* labels from the base palette."""
    palette = create_base_palette()
    if ignore_index is not None:
        ignore_color = palette[1]  # White
        palette = np.concatenate((palette[:1], palette[2:]), axis=0)
    else:
        ignore_color = 0

    if num_classes > len(palette):
        msg = "Palette size is too large."
        raise ValueError(msg)

    return Palette(
        colors=palette[:num_classes],
        ignore_color=ignore_color,
    )
