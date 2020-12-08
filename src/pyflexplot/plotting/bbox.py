"""Bounding box."""
# Standard library
from copy import copy

# Third-party
import numpy as np
from matplotlib.axes import Axes

# Local
from ..utils.logging import log
from .coord_trans import CoordinateTransformer
from .projs import MapAxesProjections


class MapAxesBoundingBox:
    """Bounding box of a ``MapAxes``."""

    # pylint: disable=R0913  # too-many-arguments
    def __init__(
        self,
        ax: Axes,
        projs: MapAxesProjections,
        coord_type: str,
        lon0: float,
        lon1: float,
        lat0: float,
        lat1: float,
    ) -> None:
        """Create an instance of ``MapAxesBoundingBox``.

        Args:
            ax: Axes object.

            projs: Map axes projections.

            coord_type: Coordinates type.

            lon0: Longitude of south-western corner.

            lon1: Longitude of north-eastern corner.

            lat0: Latitude of south-western corner.

            lat1: Latitude of north-eastern corner.

        """
        self.coord_types = ["data", "geo"]  # SR_TMP
        self.set(coord_type, lon0, lon1, lat0, lat1)
        self.trans = CoordinateTransformer(
            trans_axes=ax.transAxes,
            trans_data=ax.transData,
            proj_geo=projs.geo,
            proj_map=projs.map,
            proj_data=projs.data,
            invalid_ok=False,
        )

    def __repr__(self):
        return (
            f"{type(self).__name__}(\n  "
            + ",\n  ".join(
                [
                    f"coord_type='{self.coord_type}'",
                    f"lon0={self.lon0:.2f}",
                    f"lon1={self.lon1:.2f}",
                    f"lat0={self.lat0:.2f}",
                    f"lat1={self.lat1:.2f}",
                    f"trans={self.trans}",
                ]
            )
            + ",\n)"
        )

    @property
    def lon(self):
        return np.asarray(self)[:2]

    @property
    def lat(self):
        return np.asarray(self)[2:]

    @property
    def coord_type(self):
        return self._curr_coord_type

    @property
    def lon0(self):
        return self._curr_lon0

    @property
    def lon1(self):
        return self._curr_lon1

    @property
    def lat0(self):
        return self._curr_lat0

    @property
    def lat1(self):
        return self._curr_lat1

    # pylint: disable=R0913  # too-many-arguments
    def set(self, coord_type, lon0, lon1, lat0, lat1):
        if not all(np.isfinite(c) for c in (lon0, lon1, lat0, lat1)):
            raise ValueError(f"invalid coordinates: ({lon0}, {lon1}, {lat0}, {lat1}")
        self._curr_coord_type = coord_type
        self._curr_lon0 = lon0
        self._curr_lon1 = lon1
        self._curr_lat0 = lat0
        self._curr_lat1 = lat1

    def __iter__(self):
        """Iterate over the rotated corner coordinates."""
        yield self._curr_lon0
        yield self._curr_lon1
        yield self._curr_lat0
        yield self._curr_lat1

    def __len__(self):
        return len(list(iter(self)))

    def __getitem__(self, idx):
        return list(iter(self))[idx]

    def to_data(self):
        if self.coord_type == "geo":
            coords = np.concatenate(self.trans.geo_to_data(self.lon, self.lat))
        elif self.coord_type == "axes":
            return self.to_geo().to_data()
        else:
            raise self._error("to_data")
        self.set("data", *coords)
        return self

    def to_geo(self):
        if self.coord_type == "data":
            coords = np.concatenate(self.trans.data_to_geo(self.lon, self.lat))
        elif self.coord_type == "axes":
            coords = np.concatenate(self.trans.axes_to_geo(self.lon, self.lat))
        else:
            raise self._error("to_geo")
        self.set("geo", *coords)
        return self

    def to_axes(self):
        if self.coord_type == "geo":
            coords = np.concatenate(self.trans.geo_to_axes(self.lon, self.lat))
            self.set("axes", *coords)
            return self
        elif self.coord_type == "data":
            return self.to_geo().to_axes()
        raise self._error("to_axes")

    def _error(self, method) -> Exception:
        return NotImplementedError(
            f"{type(self).__name__}.{method} from '{self.coord_type}'"
        )

    # pylint: disable=R0914  # too-many-locals
    def zoom(self, fact, rel_offset):
        """Zoom into or out of the domain.

        Args:
            fact (float): Zoom factor, > 1.0 to zoom in, < 1.0 to zoom out.

            rel_offset (tuple[float, float], optional): Relative offset in x
                and y direction as a fraction of the respective domain extent.
                Defaults to (0.0, 0.0).

        Returns:
            ndarray[float, n=4]: Zoomed bounding box.

        """
        try:
            rel_x_offset, rel_y_offset = [float(i) for i in rel_offset]
        except Exception as e:
            raise ValueError(
                f"rel_offset expected to be a pair of floats, not {rel_offset}"
            ) from e

        # Restrict zoom to geographical latitude range [-90, 90]
        _, _, lat0_geo, lat1_geo = iter(copy(self).to_geo())
        lat0_geo_min = -90
        lat1_geo_max = 90
        d_lat_geo_max = min([lat0_geo - lat0_geo_min, lat1_geo_max - lat1_geo])
        d_lat_geo = lat1_geo - lat0_geo
        fact_min = d_lat_geo / (2 * d_lat_geo_max + d_lat_geo)
        if fact < fact_min:
            log(dbg=f"zoom factor {fact} adjusted to {fact_min} (too small for domain)")
            fact = fact_min

        lon0, lon1, lat0, lat1 = iter(self)
        dlon = lon1 - lon0
        dlat = lat1 - lat0
        clon = lon0 + (0.5 + rel_x_offset) * dlon
        clat = lat0 + (0.5 + rel_y_offset) * dlat
        dlon_zm = dlon / fact
        dlat_zm = dlat / fact

        coords = np.array(
            [
                clon - 0.5 * dlon_zm,
                clon + 0.5 * dlon_zm,
                clat - 0.5 * dlat_zm,
                clat + 0.5 * dlat_zm,
            ],
            float,
        )

        self.set(self.coord_type, *coords)
        return self
