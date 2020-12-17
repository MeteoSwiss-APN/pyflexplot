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
from .proj_bbox import ProjectedBoundingBox
from .proj_bbox import Projections


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
        self, ax: Axes, projs: Projections, curr_proj: str = "data"
    ) -> ProjectedBoundingBox:
        """Get bounding box of domain."""
        lllon, urlon, lllat, urlat = self.get_bbox_corners()
        bbox = ProjectedBoundingBox(
            ax=ax,
            projs=projs,
            lon0=lllon,
            lon1=urlon,
            lat0=lllat,
            lat1=urlat,
        )
        if self.zoom_fact != 1.0:
            bbox.to_axes().zoom(self.zoom_fact, self.rel_offset)
        return bbox.to(curr_proj)

    def get_bbox_corners(self) -> Tuple[float, float, float, float]:
        """Return corners of domain: [lllon, lllat, urlon, urlat]."""
        lllat = self.lat[0]
        urlat = self.lat[-1]
        lllon = self.lon[0]
        urlon = self.lon[-1]
        return lllon, urlon, lllat, urlat


@summarizable(attrs_add=["aspect", "min_size_lat", "min_size_lon", "periodic_lon"])
class CloudDomain(Domain):
    """Domain derived from spatial distribution of cloud over time."""

    def __init__(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        *,
        aspect: Optional[float] = None,
        mask_nz: Optional[np.ndarray] = None,
        min_size_lat: float = 0.0,
        min_size_lon: float = 0.0,
        periodic_lon: bool = False,
        **kwargs: Any,
    ) -> None:
        """Create an instance of ``CloudDomain``.

        Args:
            lat: 1D latitude array.

            lon: 1D longitude array.

            aspect (optional): Target aspect ratio of the domain.

            mask_nz (optional): Mask with dimensions (lat, lon) of non-zero
                field values; defaults to an empty mask.

            min_size_lat (optional): Minimum latitudinal extent of the domain in
                degrees.

            min_size_lon (optional): Minimum longitudinal extent of the domain
                in degrees.

            periodic_lon (optional): Whether the domain is zonally periodic.

            **kwargs: Keyword arguments passed to ``Domain``.

        """
        super().__init__(lat, lon, **kwargs)

        if mask_nz is None:
            mask_nz = np.zeros([self.lat.size, self.lon.size], dtype=np.bool)

        self.aspect: Optional[float] = aspect
        self.mask_nz: np.ndarray = mask_nz
        self.min_size_lat: float = min_size_lat
        self.min_size_lon: float = min_size_lon
        self.periodic_lon: bool = periodic_lon

        if self.mask_nz.shape != (self.lat.size, self.lon.size):
            raise ValueError(
                "shape of mask_nz inconsistent with lat/lon"
                f": {self.mask_nz.shape} != ({self.lat.size}, {self.lon.size}"
            )

    # pylint: disable=R0912  # too-many-branches (>12)
    # pylint: disable=R0914  # too-many-locals (>15)
    # pylint: disable=R0915  # too-many-statements (>50)
    def get_bbox_corners(self) -> Tuple[float, float, float, float]:
        """Return corners of domain: [lllon, lllat, urlon, urlat]."""
        lat_min, lat_max = self.lat[0], self.lat[-1]
        lon_min, lon_max = self.lon[0], self.lon[-1]
        d_lat_max = lat_max - lat_min
        d_lon_max = lon_max - lon_min

        # Latitude
        mask_lat = self.mask_nz.any(axis=1)
        if not any(mask_lat):
            lllat = self.lat.min()
            urlat = self.lat.max()
        else:
            lllat = self.lat[mask_lat].min()
            urlat = self.lat[mask_lat].max()
        lllat = max([lllat, lat_min])
        urlat = min([urlat, lat_max])

        # Longitude
        mask_lon = self.mask_nz.any(axis=0)
        crossing_dateline = (
            self.periodic_lon and mask_lon[0] and mask_lon[-1] and not mask_lon.all()
        )
        if crossing_dateline:
            idx_lllon = min([np.where(~mask_lon)[0][-1] + 1, self.lon.size - 1])
            idx_urlon = max([np.where(~mask_lon)[0][0] - 1, 0])
            lllon = self.lon[idx_lllon]
            urlon = self.lon[idx_urlon]
            lllon = min([lllon, lon_max])
            urlon = max([urlon, lon_min])
        else:
            if not any(mask_lon):
                lllon = self.lon.min()
                urlon = self.lon.max()
            else:
                lllon = self.lon[mask_lon].min()
                urlon = self.lon[mask_lon].max()
            lllon = max([lllon, lon_min])
            urlon = min([urlon, lon_max])

        # Increase latitudinal size if minimum specified
        d_lat_min = self.min_size_lat
        if d_lat_min is not None:
            d_lat = urlat - lllat
            if d_lat < d_lat_min:
                lllat -= 0.5 * min([d_lat_min - d_lat, d_lat_max - d_lat])
                urlat += 0.5 * min([d_lat_min - d_lat, d_lat_max - d_lat])

        # Increase longitudinal size if minimum specified
        d_lon_min = self.min_size_lon
        if d_lon_min is not None:
            if crossing_dateline:
                d_lon = lllon - urlon
                if d_lon < d_lon_min:
                    lllon += 0.5 * min([d_lon_min - d_lon, d_lon_max - d_lon])
                    urlon -= 0.5 * min([d_lon_min - d_lon, d_lon_max - d_lon])
            else:
                d_lon = urlon - lllon
                if d_lon < d_lon_min:
                    lllon -= 0.5 * min([d_lon_min - d_lon, d_lon_max - d_lon])
                    urlon += 0.5 * min([d_lon_min - d_lon, d_lon_max - d_lon])

        if self.aspect:
            # Adjust self.aspect ratio to avoid distortion
            if crossing_dateline:
                d_lat = urlat - lllat
                d_lon = lllon - urlon
                if d_lon < d_lat * self.aspect:
                    urlon -= 0.5 * min([d_lat * self.aspect - d_lon, d_lon_max - d_lon])
                    lllon += 0.5 * min([d_lat * self.aspect - d_lon, d_lon_max - d_lon])
                elif d_lat < d_lon / self.aspect:
                    lllat -= 0.5 * min([d_lon / self.aspect - d_lat, d_lat_max - d_lat])
                    urlat += 0.5 * min([d_lon / self.aspect - d_lat, d_lat_max - d_lat])
            else:
                d_lat = urlat - lllat
                d_lon = urlon - lllon
                if d_lon < d_lat * self.aspect:
                    lllon -= 0.5 * min([d_lat * self.aspect - d_lon, d_lon_max - d_lon])
                    urlon += 0.5 * min([d_lat * self.aspect - d_lon, d_lon_max - d_lon])
                elif d_lat < d_lon / self.aspect:
                    lllat -= 0.5 * min([d_lon / self.aspect - d_lat, d_lat_max - d_lat])
                    urlat += 0.5 * min([d_lon / self.aspect - d_lat, d_lat_max - d_lat])

        # Adjust latitudinal range if necessary
        if urlat > lat_max:
            urlat -= urlat - lat_max
            lllat -= urlat - lat_max
        elif lllat < lat_min:
            urlat += lat_min - lllat
            lllat += lat_min - lllat

        return lllon, urlon, lllat, urlat


@summarizable(
    attrs_add=[
        "aspect",
        "field_proj",
        "min_size_lat",
        "min_size_lon",
        "release_lat",
        "release_lon",
    ]
)
class ReleaseSiteDomain(Domain):
    """Domain relative to release point."""

    def __init__(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        *,
        aspect: Optional[float] = None,
        field_proj: Projection = PlateCarree(),
        min_size_lat: float = 0.0,
        min_size_lon: float = 0.0,
        release_lat: Optional[float] = None,
        release_lon: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Create an instance of ``CloudDomain``.

        Args:
            lat: 1D latitude array.

            lon: 1D longitude array.

            aspect (optional): Target aspect ratio of the domain; must be non-
                zero unless both ``min_size_lat`` and ``min_size_lon`` are non-
                zero.

            field_proj (optional): Field projection.

            min_size_lat (optional): Minimum latitudinal extent of the domain
                in degrees; must be non-zero unless ``min_size_lon`` is non-
                zero.

            min_size_lon (optional): Minimum longitudinal extent of the domain
                in degrees; must be non-zero unless ``min_size_lat`` is non-
                zero.

            release_lat (optional): Latitude of release point; default to the
                center of the domain.

            release_lon (optional): Longitude of release point; default to the
                center of the domain.

            **kwargs: Keyword arguments passed to ``Domain``.

        """
        super().__init__(lat, lon, **kwargs)

        if release_lat is None:
            release_lat = self.lat.mean()
        if release_lon is None:
            release_lon = self.lon.mean()

        self.aspect: Optional[float] = aspect
        self.field_proj: Projection = field_proj
        self.min_size_lat: float = min_size_lat
        self.min_size_lon: float = min_size_lon
        self.release_lat: float = release_lat
        self.release_lon: float = release_lon

        if not min_size_lat and not min_size_lon:
            raise ValueError(
                "one or both of min_size_lat and min_size_lon must be non-zero"
            )
        elif (not min_size_lat or not min_size_lon) and not aspect:
            raise ValueError(
                "aspect must be non-zero unless both min_size_lat and min_size_lon are"
                " non-zero"
            )

    def get_bbox_corners(self) -> Tuple[float, float, float, float]:
        """Return corners of domain: [lllon, lllat, urlon, urlat]."""
        d_lat = self.min_size_lat
        d_lon = self.min_size_lon
        if d_lon and not d_lat:
            assert self.aspect  # proper check in __init__
            assert d_lon is not None  # mypy
            d_lat = d_lon / self.aspect
        elif d_lat and not d_lon:
            assert self.aspect  # proper check in __init__
            d_lon = d_lat / self.aspect
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
        return lllon, urlon, lllat, urlat
