"""Bounding box."""
from __future__ import annotations

# Standard library
import dataclasses as dc
from copy import copy
from typing import Iterator
from typing import Optional
from typing import Tuple

# Third-party
import numpy as np
from cartopy.crs import PlateCarree
from cartopy.crs import Projection
from cartopy.crs import RotatedPole
from matplotlib.axes import Axes

# Local
from ..utils.logging import log
from ..utils.summarize import summarizable
from .coord_trans import CoordinateTransformer


@summarizable
@dc.dataclass
class Projections:
    """Projections of a ``MapAxes``."""

    data: Projection
    map: Projection
    geo: Projection

    @classmethod
    def from_proj_data(cls, proj_data: Projection) -> Projections:
        """Derive a projections set from the data projection."""
        if isinstance(proj_data, RotatedPole):
            return cls.create_rotated(proj=proj_data)
        return cls.create_regular()

    @classmethod
    def create_regular(cls) -> Projections:
        # pylint: disable=E0110  # abstract-class-instatiated (PlateCarree)
        return cls(
            data=PlateCarree(central_longitude=0.0),
            map=PlateCarree(central_longitude=0.0),
            geo=PlateCarree(central_longitude=0.0),
        )

    @classmethod
    def create_rotated(
        cls,
        *,
        pollat: float = 90.0,
        pollon: float = 180.0,
        proj: Optional[RotatedPole] = None,
    ) -> Projections:
        # pylint: disable=E0110  # abstract-class-instatiated (PlateCarree, RotatedPole)
        if proj is not None:
            return cls(data=proj, map=proj, geo=PlateCarree(central_longitude=0.0))
        return cls(
            data=RotatedPole(pole_latitude=pollat, pole_longitude=pollon),
            map=RotatedPole(pole_latitude=pollat, pole_longitude=pollon),
            geo=PlateCarree(central_longitude=0.0),
        )


class ProjectedBoundingBox:
    """Bounding box of a ``MapAxes`` in a certain projection."""

    # pylint: disable=R0913  # too-many-arguments
    def __init__(
        self,
        ax: Axes,
        projs: Projections,
        *,
        curr_proj: str = "data",
        lon0: float = -180.0,
        lon1: float = 180.0,
        lat0: float = -90.0,
        lat1: float = 90.0,
    ) -> None:
        """Create an instance of ``ProjectedBoundingBox``.

        Args:
            ax: Axes object.

            projs: Map axes projections.

            curr_proj (optional): Current projection type.

            lon0 (optional): Longitude of south-western corner.

            lon1 (optional): Longitude of north-eastern corner.

            lat0 (optional): Latitude of south-western corner.

            lat1 (optional): Latitude of north-eastern corner.

        """
        self.set(curr_proj, lon0, lon1, lat0, lat1)
        self.trans = CoordinateTransformer(
            trans_axes=ax.transAxes,
            trans_data=ax.transData,
            proj_geo=projs.geo,
            proj_map=projs.map,
            proj_data=projs.data,
            invalid_ok=False,
        )

    @property
    def lon(self) -> np.ndarray:
        return np.asarray(self)[:2]

    @property
    def lat(self) -> np.ndarray:
        return np.asarray(self)[2:]

    @property
    def coord_type(self) -> str:
        return self._curr_coord_type

    @property
    def lon0(self) -> float:
        return self._curr_lon0

    @property
    def lon1(self) -> float:
        return self._curr_lon1

    @property
    def lat0(self) -> float:
        return self._curr_lat0

    @property
    def lat1(self) -> float:
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

    def to(self, proj: str) -> ProjectedBoundingBox:
        if proj == "axes":
            return self.to_axes()
        elif proj == "data":
            return self.to_data()
        elif proj == "geo":
            return self.to_geo()
        elif proj == "map":
            return self.to_map()
        else:
            raise self._error(f"to('{proj}'")

    def to_axes(self) -> ProjectedBoundingBox:
        if self.coord_type == "data":
            coords = np.concatenate(self.trans.data_to_axes(self.lon, self.lat))
        elif self.coord_type == "geo":
            coords = np.concatenate(self.trans.geo_to_axes(self.lon, self.lat))
        elif self.coord_type == "map":
            coords = np.concatenate(self.trans.map_to_axes(self.lon, self.lat))
        else:
            raise self._error("to_axes")
        self.set("axes", *coords)
        return self

    def to_data(self) -> ProjectedBoundingBox:
        if self.coord_type == "axes":
            coords = np.concatenate(self.trans.axes_to_data(self.lon, self.lat))
        elif self.coord_type == "geo":
            coords = np.concatenate(self.trans.geo_to_data(self.lon, self.lat))
        elif self.coord_type == "map":
            coords = np.concatenate(self.trans.data_to_map(self.lon, self.lat))
        else:
            raise self._error("to_data")
        self.set("data", *coords)
        return self

    def to_geo(self) -> ProjectedBoundingBox:
        if self.coord_type == "axes":
            coords = np.concatenate(self.trans.axes_to_geo(self.lon, self.lat))
        elif self.coord_type == "data":
            coords = np.concatenate(self.trans.data_to_geo(self.lon, self.lat))
        elif self.coord_type == "map":
            coords = np.concatenate(self.trans.map_to_geo(self.lon, self.lat))
        else:
            raise self._error("to_geo")
        self.set("geo", *coords)
        return self

    def to_map(self) -> ProjectedBoundingBox:
        if self.coord_type == "axes":
            coords = np.concatenate(self.trans.axes_to_map(self.lon, self.lat))
        elif self.coord_type == "data":
            coords = np.concatenate(self.trans.data_to_map(self.lon, self.lat))
        elif self.coord_type == "geo":
            coords = np.concatenate(self.trans.geo_to_map(self.lon, self.lat))
        else:
            raise self._error("to_map")
        self.set("map", *coords)
        return self

    # pylint: disable=R0914  # too-many-locals
    def zoom(self, fact, rel_offset) -> ProjectedBoundingBox:
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
        geo = copy(self).to_geo()
        lat0_geo = geo.lat0
        lat1_geo = geo.lat1
        lat0_geo_min = -90
        lat1_geo_max = 90
        d_lat_geo_max = min([lat0_geo - lat0_geo_min, lat1_geo_max - lat1_geo])
        d_lat_geo = lat1_geo - lat0_geo
        fact_min = d_lat_geo / (2 * d_lat_geo_max + d_lat_geo)
        if fact < fact_min:
            log(dbg=f"zoom factor {fact} adjusted to {fact_min} (too small for domain)")
            fact = fact_min

        lon0, lon1, lat0, lat1 = self._corners
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

    @property
    def _corners(self) -> Tuple[float, float, float, float]:
        return (
            self._curr_lon0,
            self._curr_lon1,
            self._curr_lat0,
            self._curr_lat1,
        )

    def _error(self, method) -> Exception:
        return NotImplementedError(f"{type(self).__name__}[{self.coord_type}].{method}")

    def __iter__(self) -> Iterator[float]:
        """Iterate over the rotated corner coordinates."""
        return iter(self._corners)

    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> float:
        return list(iter(self))[idx]

    def __repr__(self) -> str:
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
