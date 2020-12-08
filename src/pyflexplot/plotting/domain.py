"""Domains."""
# Standard library
from dataclasses import dataclass
from typing import Tuple

# Third-party
import numpy as np
from cartopy.crs import PlateCarree  # type: ignore
from cartopy.crs import RotatedPole
from matplotlib.axes import Axes

# Local
from ..input.data import Field
from ..utils.summarize import summarizable
from .bbox import MapAxesBoundingBox
from .projs import MapAxesProjections


@summarizable
@dataclass
class Domain:
    """Plot domain.

    Args:
        field: Field to be plotted on the domain.

        zoom_fact (optional): Zoom factor. Use values above/below 1.0 to zoom
            in/out.

        rel_offset (optional): Relative offset in x and y direction as a
            fraction of the respective domain extent.

    """

    field: Field
    zoom_fact: float = 1.0
    rel_offset: Tuple[float, float] = (0.0, 0.0)

    def get_bbox(
        self, ax: Axes, projs: MapAxesProjections, aspect: float
    ) -> "MapAxesBoundingBox":
        """Get bounding box of domain."""
        lllon, urlon, lllat, urlat = self._get_bbox_corners(aspect)
        bbox = MapAxesBoundingBox(ax, projs, "data", lllon, urlon, lllat, urlat)
        if self.zoom_fact != 1.0:
            bbox = bbox.to_axes().zoom(self.zoom_fact, self.rel_offset).to_data()
        return bbox

    # pylint: disable=W0613  # unused-argument (aspect)
    def _get_bbox_corners(self, aspect: float) -> Tuple[float, float, float, float]:
        """Return corners of domain: [lllon, lllat, urlon, urlat]."""
        lllat = self.field.lat[0]
        urlat = self.field.lat[-1]
        lllon = self.field.lon[0]
        urlon = self.field.lon[-1]
        return lllon, urlon, lllat, urlat


class CloudDomain(Domain):
    """Domain derived from spatial distribution of cloud over time."""

    # pylint: disable=R0914  # too-many-locals
    # pylint: disable=R0915  # too-many-statements
    def _get_bbox_corners(self, aspect: float) -> Tuple[float, float, float, float]:
        """Return corners of domain: [lllon, lllat, urlon, urlat]."""
        lat: np.ndarray = self.field.lat
        lon: np.ndarray = self.field.lon
        assert (lat.size, lon.size) == self.field.time_props.mask_nz.shape
        lat_min, lat_max = lat[0], lat[-1]
        lon_min, lon_max = lon[0], lon[-1]
        d_lat_max = lat_max - lat_min
        d_lon_max = lon_max - lon_min

        mask_lat = self.field.time_props.mask_nz.any(axis=1)
        mask_lon = self.field.time_props.mask_nz.any(axis=0)
        if not any(mask_lat):
            lllat = lat.min()
            urlat = lat.max()
        else:
            lllat = lat[mask_lat].min()
            urlat = lat[mask_lat].max()
        if not any(mask_lon):
            lllon = lon.min()
            urlon = lon.max()
        else:
            lllon = lon[mask_lon].min()
            urlon = lon[mask_lon].max()
        lllat = max([lllat, lat_min])
        urlat = min([urlat, lat_max])
        lllon = max([lllon, lon_min])
        urlon = min([urlon, lon_max])

        # Increase latitudinal size if minimum specified
        d_lat_min = self.field.var_setups.collect_equal("domain_size_lat")
        if d_lat_min is not None:
            d_lat = urlat - lllat
            if d_lat < d_lat_min:
                lllat -= 0.5 * min([d_lat_min - d_lat, d_lat_max - d_lat])
                urlat += 0.5 * min([d_lat_min - d_lat, d_lat_max - d_lat])

        # Increase latitudinal size if minimum specified
        d_lon_min = self.field.var_setups.collect_equal("domain_size_lon")
        if d_lon_min is not None:
            d_lon = urlon - lllon
            if d_lon < d_lon_min:
                lllon -= 0.5 * min([d_lon_min - d_lon, d_lon_max - d_lon])
                urlon += 0.5 * min([d_lon_min - d_lon, d_lon_max - d_lon])

        # Adjust aspect ratio to avoid distortion
        d_lat = urlat - lllat
        d_lon = urlon - lllon
        if d_lon < d_lat * aspect:
            lllon -= 0.5 * min([d_lat * aspect - d_lon, d_lon_max - d_lon])
            urlon += 0.5 * min([d_lat * aspect - d_lon, d_lon_max - d_lon])
        elif d_lat < d_lon / aspect:
            lllat -= 0.5 * min([d_lon / aspect - d_lat, d_lat_max - d_lat])
            urlat += 0.5 * min([d_lon / aspect - d_lat, d_lat_max - d_lat])

        # Adjust latitudinal range if necessary
        if urlat > lat_max:
            urlat -= urlat - lat_max
            lllat -= urlat - lat_max
        elif lllat < lat_min:
            urlat += lat_min - lllat
            lllat += lat_min - lllat

        return lllon, urlon, lllat, urlat


class ReleaseSiteDomain(Domain):
    """Domain relative to release point."""

    def _get_bbox_corners(self, aspect: float) -> Tuple[float, float, float, float]:
        """Return corners of domain: [lllon, lllat, urlon, urlat]."""
        assert self.field.mdata is not None  # mypy
        release_lat: float = self.field.mdata.release.lat
        release_lon: float = self.field.mdata.release.lon
        d_lat = self.field.var_setups.collect_equal("domain_size_lat")
        d_lon = self.field.var_setups.collect_equal("domain_size_lon")
        if d_lat is None and d_lon is None:
            raise Exception(
                "domain type 'release_site': setup params 'domain_size_(lat|lon)'"
                " are both None; one or both is required"
            )
        elif d_lat is None:
            d_lat = d_lon / aspect
        elif d_lon is None:
            d_lon = d_lat / aspect
        assert self.field.mdata is not None  # mypy
        if isinstance(self.field.proj, RotatedPole):
            c_lon, c_lat = self.field.proj.transform_point(
                release_lon, release_lat, PlateCarree()
            )
            lllat = c_lat - 0.5 * d_lat
            lllon = c_lon - 0.5 * d_lon
            urlat = c_lat + 0.5 * d_lat
            urlon = c_lon + 0.5 * d_lon
        else:
            lllat = release_lat - 0.5 * d_lat
            lllon = release_lon - 0.5 * d_lon
            urlat = release_lat + 0.5 * d_lat
            urlon = release_lon + 0.5 * d_lon
            lllon, lllat = self.field.proj.transform_point(lllon, lllat, PlateCarree())
            urlon, urlat = self.field.proj.transform_point(urlon, urlat, PlateCarree())
        return lllon, urlon, lllat, urlat
