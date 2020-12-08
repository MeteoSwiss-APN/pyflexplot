"""Domains."""
# Standard library
from typing import Any
from typing import Optional
from typing import Tuple

# Third-party
import numpy as np
from cartopy.crs import PlateCarree  # type: ignore
from cartopy.crs import Projection
from cartopy.crs import RotatedPole
from matplotlib.axes import Axes

# Local
from ..utils.summarize import summarizable
from .bbox import MapAxesBoundingBox
from .projs import MapAxesProjections


# @summarizable(attrs=["zoom_fact", "rel_offset"])
@summarizable(attrs=["lat", "lon", "zoom_fact", "rel_offset"])
class Domain:
    """Plot domain."""

    def __init__(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        zoom_fact: float = 1.0,
        rel_offset: Tuple[float, float] = (0.0, 0.0),
    ) -> None:
        """Create an instance of ``domain``.

        Args:
            lat: 1D latitude array.

            lon: 1D longitude array.

            field: Field to be plotted on the domain.

            zoom_fact (optional): Zoom factor. Use values above/below 1.0 to
                zoom in/out.

            rel_offset (optional): Relative offset in x and y direction as a
                fraction of the respective domain extent.

        """
        self.lat = lat
        self.lon = lon
        self.zoom_fact = zoom_fact
        self.rel_offset = rel_offset

    def get_bbox(
        self, ax: Axes, projs: MapAxesProjections, aspect: float
    ) -> MapAxesBoundingBox:
        """Get bounding box of domain."""
        lllon, urlon, lllat, urlat = self._get_bbox_corners(aspect)
        bbox = MapAxesBoundingBox(ax, projs, "data", lllon, urlon, lllat, urlat)
        if self.zoom_fact != 1.0:
            bbox = bbox.to_axes().zoom(self.zoom_fact, self.rel_offset).to_data()
        return bbox

    # pylint: disable=W0613  # unused-argument (aspect)
    def _get_bbox_corners(self, aspect: float) -> Tuple[float, float, float, float]:
        """Return corners of domain: [lllon, lllat, urlon, urlat]."""
        lllat = self.lat[0]
        urlat = self.lat[-1]
        lllon = self.lon[0]
        urlon = self.lon[-1]
        return lllon, urlon, lllat, urlat


@summarizable(attrs_add=["domain_size_lat", "domain_size_lon"])
class CloudDomain(Domain):
    """Domain derived from spatial distribution of cloud over time."""

    def __init__(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        domain_size_lat: float,
        domain_size_lon: float,
        mask_nz: np.ndarray,
        **kwargs: Any,
    ) -> None:
        """Create an instance of ``CloudDomain``.

        Args:
            lat: 1D latitude array.

            lon: 1D longitude array.

            domain_size_lat: Latitudinal extent of the domain in degrees.

            domain_size_lon: Longitudinal extent of the domain in degrees.

            mask_nz: Mask with dimensions (lat, lon) of non-zero field values.

            **kwargs: Keyword arguments passed to ``Domain``.

        """
        super().__init__(lat, lon, **kwargs)
        self.domain_size_lat = domain_size_lat
        self.domain_size_lon = domain_size_lon
        self.mask_nz = mask_nz
        if self.mask_nz.shape != (self.lat.size, self.lon.size):
            raise ValueError(
                "shape of mask_nz inconsistent with lat/lon"
                f": {self.mask_nz.shape} != ({self.lat.size}, {self.lon.size}"
            )

    # pylint: disable=R0914  # too-many-locals
    # pylint: disable=R0915  # too-many-statements
    def _get_bbox_corners(self, aspect: float) -> Tuple[float, float, float, float]:
        """Return corners of domain: [lllon, lllat, urlon, urlat]."""
        lat_min, lat_max = self.lat[0], self.lat[-1]
        lon_min, lon_max = self.lon[0], self.lon[-1]
        d_lat_max = lat_max - lat_min
        d_lon_max = lon_max - lon_min

        mask_lat = self.mask_nz.any(axis=1)
        mask_lon = self.mask_nz.any(axis=0)
        if not any(mask_lat):
            lllat = self.lat.min()
            urlat = self.lat.max()
        else:
            lllat = self.lat[mask_lat].min()
            urlat = self.lat[mask_lat].max()
        if not any(mask_lon):
            lllon = self.lon.min()
            urlon = self.lon.max()
        else:
            lllon = self.lon[mask_lon].min()
            urlon = self.lon[mask_lon].max()
        lllat = max([lllat, lat_min])
        urlat = min([urlat, lat_max])
        lllon = max([lllon, lon_min])
        urlon = min([urlon, lon_max])

        # Increase latitudinal size if minimum specified
        d_lat_min = self.domain_size_lat
        if d_lat_min is not None:
            d_lat = urlat - lllat
            if d_lat < d_lat_min:
                lllat -= 0.5 * min([d_lat_min - d_lat, d_lat_max - d_lat])
                urlat += 0.5 * min([d_lat_min - d_lat, d_lat_max - d_lat])

        # Increase latitudinal size if minimum specified
        d_lon_min = self.domain_size_lon
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


@summarizable(
    attrs_add=["domain_size_lat", "domain_size_lon", "release_lat", "release_lon"]
)
class ReleaseSiteDomain(Domain):
    """Domain relative to release point."""

    def __init__(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        domain_size_lat: Optional[float],
        domain_size_lon: Optional[float],
        release_lat: float,
        release_lon: float,
        field_proj: Projection,
        **kwargs: Any,
    ) -> None:
        """Create an instance of ``CloudDomain``.

        Args:
            lat: 1D latitude array.

            lon: 1D longitude array.

            domain_size_lat: Latitudinal extent of the domain in degrees.

            domain_size_lon: Longitudinal extent of the domain in degrees.

            release_lat: Latitude of release point.

            release_lon: Longitude of release point.

            field_proj: Field projection.

            **kwargs: Keyword arguments passed to ``Domain``.

        """
        super().__init__(lat, lon, **kwargs)
        self.domain_size_lat = domain_size_lat
        self.domain_size_lon = domain_size_lon
        self.release_lat = release_lat
        self.release_lon = release_lon
        self.field_proj = field_proj

    def _get_bbox_corners(self, aspect: float) -> Tuple[float, float, float, float]:
        """Return corners of domain: [lllon, lllat, urlon, urlat]."""
        d_lat = self.domain_size_lat
        d_lon = self.domain_size_lon
        if d_lat is None and d_lon is None:
            raise Exception(
                "domain type 'release_site': setup params 'domain_size_(lat|lon)'"
                " are both None; one or both is required"
            )
        elif d_lat is None:
            assert d_lon is not None  # mypy
            d_lat = d_lon / aspect
        elif d_lon is None:
            d_lon = d_lat / aspect
        if isinstance(self.field_proj, RotatedPole):
            c_lon, c_lat = self.field_proj.transform_point(
                self.release_lon, self.release_lat, PlateCarree()
            )
            lllat = c_lat - 0.5 * d_lat
            lllon = c_lon - 0.5 * d_lon
            urlat = c_lat + 0.5 * d_lat
            urlon = c_lon + 0.5 * d_lon
        else:
            lllat = self.release_lat - 0.5 * d_lat
            lllon = self.release_lon - 0.5 * d_lon
            urlat = self.release_lat + 0.5 * d_lat
            urlon = self.release_lon + 0.5 * d_lon
            lllon, lllat = self.field_proj.transform_point(lllon, lllat, PlateCarree())
            urlon, urlat = self.field_proj.transform_point(urlon, urlat, PlateCarree())
        return lllon, urlon, lllat, urlat
