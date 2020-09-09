"""
Plotting utilities.
"""
# Standard library
from typing import Union

# Third-party
import matplotlib as mpl
import numpy as np
from matplotlib.colors import Colormap


def truncate_cmap(
    cmap: Union[str, Colormap], minval: float = 0.0, maxval: float = 1.0, n: int = 100
) -> Colormap:
    """Truncate a color map.

    Based on: https://stackoverflow.com/a/18926541/4419816

    """
    if isinstance(cmap, str):
        cmap = mpl.cm.get_cmap(cmap)
    return mpl.colors.LinearSegmentedColormap.from_list(
        f"truncated({cmap.name}, {minval:.2f}, {maxval:.2f})",
        cmap(np.linspace(minval, maxval, n)),
    )


def concatenate_cmaps(
    cmap1: Union[str, Colormap], cmap2: Union[str, Colormap], n: int = 100
) -> Colormap:
    """Concatenate two color maps."""
    if isinstance(cmap1, str):
        cmap1 = mpl.cm.get_cmap(cmap1)
    if isinstance(cmap2, str):
        cmap2 = mpl.cm.get_cmap(cmap2)
    colors1 = cmap1(np.linspace(0.0, 1.0, n)).tolist()
    colors2 = cmap2(np.linspace(0.0, 1.0, n)).tolist()
    return mpl.colors.LinearSegmentedColormap.from_list(
        f"concatenated({cmap1.name}, {cmap2.name})", colors1 + colors2
    )
