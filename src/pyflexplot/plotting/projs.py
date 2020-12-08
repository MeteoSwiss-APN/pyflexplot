"""Map axes projections."""
# Standard library
import warnings
from typing import Any
from typing import Dict

# Third-party
import cartopy
from cartopy.crs import Projection


class MapAxesProjections:
    """Projections of a ``MapAxes`` for data on a regular lat/lon grid."""

    def __init__(self, proj_type: str, central_lon: float) -> None:
        """Create instance of ``MapAxesProjections``.

        Args:
            proj_type: type of projection.

            central_lon: Central longitude.

        """
        self.proj_type = proj_type
        self.central_lon = central_lon
        self.data: Projection = self._init_proj_data()
        self.map: Projection = self._init_proj_map()
        self.geo: Projection = self._init_proj_geo()

    # pylint: disable=R0201  # no-self-use
    def _init_proj_data(self) -> Projection:
        """Initialize projection of input data."""
        return self._init_proj_geo()

    def _init_proj_map(self) -> Projection:
        """Initialize projection of map plot."""
        if self.proj_type == "data":
            return self.data
        elif self.proj_type == "mercator":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return cartopy.crs.TransverseMercator(
                    central_longitude=self.central_lon, approx=True
                )
        else:
            choices = ["data", "mercator"]
            raise NotImplementedError(
                f"projection '{self.proj_type}'; choices: {choices}"
            )

    # pylint: disable=R0201  # no-self-use
    def _init_proj_geo(self) -> Projection:
        """Initialize geographical projection."""
        return cartopy.crs.PlateCarree()

    def summarize(self) -> Dict[str, Any]:
        return {
            "type": type(self).__name__,
            "proj_type": self.proj_type,
            "central_lon": self.central_lon,
        }


class RotPoleMapAxesProjections(MapAxesProjections):
    """Projections of a ``MapAxes`` for data on a rotated-pole grid."""

    def __init__(
        self, proj_type: str, central_lon: float, pollat: float, pollon: float
    ) -> None:
        """Create instance of ``RotPoleMapAxesProjections``.

        Args:
            proj_type: type of projection.

            central_lon: Central longitude.

            pollat: Latitude of rotated north pole.

            pollon: Longitude of rotated north pole.

        """
        self.pollat = pollat
        self.pollon = pollon
        super().__init__(proj_type, central_lon)

    def _init_proj_data(self) -> Projection:
        """Initialize projection of input data."""
        return cartopy.crs.RotatedPole(
            pole_latitude=self.pollat, pole_longitude=self.pollon
        )

    def summarize(self):
        return {
            **super().summarize(),
            "pollat": self.pollat,
            "pollon": self.pollon,
        }
