"""Map axes projections."""
# Standard library
from dataclasses import dataclass

# Third-party
from cartopy.crs import Projection

# Local
from ..utils.summarize import summarizable


@summarizable
@dataclass
class MapAxesProjections:
    """Projections of a ``MapAxes``."""

    data: Projection
    map: Projection
    geo: Projection
    curr_proj: str = "data"
