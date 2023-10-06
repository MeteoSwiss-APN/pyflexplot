"""Domains."""
# Standard library
import dataclasses as dc
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

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


@summarizable
@dc.dataclass
class DomainConfig:
    """Configuration of a ``Domain`` instance.

    Attributes:
        rel_offset (optional): Relative offset in x and y direction as a
            fraction of the respective domain extent.

        zoom_fact (optional): Zoom factor. Use values above/below 1.0 to zoom
            in/out.

    """

    rel_offset: Tuple[float, float] = (0.0, 0.0)
    zoom_fact: float = 1.0


@summarizable(
    summarize=lambda self: {
        "type": type(self).__name__,
        "lat": {
            "dtype": str(self.lat.dtype),
            "shape": self.lat.shape,
            "min": self.lat.min(),
            "max": self.lat.max(),
            "start": self.lat[:10].tolist(),
            "end": self.lat[-10:].tolist(),
        },
        "lon": {
            "dtype": str(self.lon.dtype),
            "shape": self.lon.shape,
            "min": self.lon.min(),
            "max": self.lon.max(),
            "start": self.lon[:10].tolist(),
            "end": self.lon[-10:].tolist(),
        },
        "config": self.config,
    }
)
class Domain:
    """Plot domain."""

    cls_config = DomainConfig

    def __init__(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        config: Optional[Union[DomainConfig, Dict[str, Any]]] = None,
    ) -> None:
        """Create an instance of ``domain``.

        Args:
            lat: 1D latitude array.

            lon: 1D longitude array.

            config (optional): Configuration object or parameters.

        """
        if not isinstance(config, self.cls_config):
            config = self.cls_config(**(config or {}))

        self.lat = lat
        self.lon = lon
        self.config = config

    def get_bbox_extent(self) -> Tuple[float, float, float, float]:
        """Return domain corners ``(lllon, lllat, urlon, urlat)``."""
        lllat = self.lat[0]
        urlat = self.lat[-1]
        lllon = self.lon[0]
        urlon = self.lon[-1]
        return lllon, urlon, lllat, urlat

    def get_center(self) -> Tuple[float, float]:
        """Return the domain center as ``(clon, clat)``."""
        lllon, urlon, lllat, urlat = self.get_bbox_extent()
        clon = 0.5 * (lllon + urlon)
        if self.crosses_dateline():
            clon += 180.0
        clat = 0.5 * (lllat + urlat)
        return (clon, clat)

    def get_bbox_size(self) -> Tuple[float, float]:
        """Return the domain size as the distance between the bbox corners."""
        lllon, urlon, lllat, urlat = self.get_bbox_extent()
        dlon = urlon - lllon
        if self.crosses_dateline():
            dlon += 360
        assert 0 < dlon <= 360
        dlat = urlat - lllat
        assert 0 < dlat <= 180
        return (dlon, dlat)

    def crosses_dateline(self) -> bool:
        """Determine whether domain crosses dateline."""
        lllon, urlon, _, _ = self.get_bbox_extent()
        return bool(lllon > urlon)  # np.bool_ => bool (?)

    def get_bbox(
        self, ax: Axes, projs: Projections, curr_proj: str = "data"
    ) -> ProjectedBoundingBox:
        """Get bounding box of domain."""
        lllon, urlon, lllat, urlat = self.get_bbox_extent()
        bbox = ProjectedBoundingBox(
            ax=ax,
            projs=projs,
            lon0=lllon,
            lon1=urlon,
            lat0=lllat,
            lat1=urlat,
        )
        if self.config.zoom_fact != 1.0:
            bbox.to_axes().zoom(self.config.zoom_fact, self.config.rel_offset)
        return bbox.to(curr_proj)


@summarizable
@dc.dataclass
class CloudDomainConfig(DomainConfig):
    """Configuration of a ``CloudDomain`` instance.

    Attributes:
        rel_offset (optional): See docstring of ``DomainConfig``.

        zoom_fact (optional): See docstring of ``DomainConfig``.

        aspect (optional): Target aspect ratio of the domain.

        min_size_lat (optional): Minimum latitudinal extent of the domain in
            degrees.

        min_size_lon (optional): Minimum longitudinal extent of the domain in
            degrees.

        periodic_lon (optional): Whether the domain is zonally periodic.

        release_lat (optional): Latitude of release point, used if there is no
            cloud; default to the center of the domain.

        release_lon (optional): Longitude of release point, used if there is no
            cloud; default to the center of the domain.

    """

    aspect: Optional[float] = None
    min_size_lat: float = 0.0
    min_size_lon: float = 0.0
    periodic_lon: bool = False
    release_lat: Optional[float] = None
    release_lon: Optional[float] = None


@summarizable
class CloudDomain(Domain):
    """Domain derived from spatial distribution of cloud over time."""

    cls_config = CloudDomainConfig

    def __init__(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        config: Optional[Union[CloudDomainConfig, Dict[str, Any]]] = None,
        *,
        mask: Optional[np.ndarray] = None,
    ) -> None:
        """Create an instance of ``CloudDomain``.

        Args:
            lat: 1D latitude array.

            lon: 1D longitude array.

            config (optional): Configuration object or parameters.

            mask (optional): Mask with dimensions (lat, lon) of non-zero field
                values; defaults to an empty mask.

        """
        self.config: CloudDomainConfig
        super().__init__(lat, lon, config)
        self.cloud_bbox: Optional[GeoMaskBoundingBox] = (
            None
            if mask is None or not mask.any()
            else GeoMaskBoundingBox(
                mask,
                self.lat,
                self.lon,
                periodic_lon=self.config.periodic_lon,
            )
        )
        self._crosses_dateline: Optional[bool] = None

    def get_bbox_extent(self) -> Tuple[float, float, float, float]:
        """Return domain corners ``(lllon, lllat, urlon, urlat)``."""
        extent, self._crosses_dateline = self._get_bbox_extent()
        return extent

    def crosses_dateline(self) -> bool:
        """Determine whether domain crosses dateline."""
        if self._crosses_dateline is None:
            *_, self._crosses_dateline = self._get_bbox_extent()
        assert isinstance(self._crosses_dateline, bool)
        return self._crosses_dateline

    # pylint: disable=R0912  # too-many-branches (>12)
    # pylint: disable=R0914  # too-many-locals (>15)
    # pylint: disable=R0915  # too-many-statements (>50)
    def _get_bbox_extent(self) -> Tuple[Tuple[float, float, float, float], bool]:
        """Return ``((lllon, lllat, urlon, urlat), crosses_dateline)``."""
        lat_min: float = self.lat[0]
        lat_max: float = self.lat[-1]
        lon_min: float = self.lon[0]
        lon_max: float = self.lon[-1]
        d_lat_max: float = lat_max - lat_min
        d_lon_max: float = lon_max - lon_min
        d_lat_min: Optional[float] = self.config.min_size_lat
        d_lon_min: Optional[float] = self.config.min_size_lon

        if self.cloud_bbox is None:
            # In absence of cloud, default to release site domain
            domain = ReleaseSiteDomain(
                self.lat,
                self.lon,
                config={
                    "aspect": self.config.aspect,
                    "min_size_lat": self.config.min_size_lat,
                    "min_size_lon": self.config.min_size_lon,
                    "release_lat": self.config.release_lat,
                    "release_lon": self.config.release_lon,
                },
            )
            return domain.get_bbox_extent(), domain.crosses_dateline()

        lllon, urlon, lllat, urlat = self.cloud_bbox.get_extent()
        crosses_dateline = self.cloud_bbox.crosses_dateline
        assert isinstance(crosses_dateline, bool)

        lllat = max([lllat, lat_min])
        urlat = min([urlat, lat_max])
        lllon = max([lllon, lon_min])
        urlon = min([urlon, lon_max])

        # Increase latitudinal size if minimum specified
        if d_lat_min is not None:
            d_lat = urlat - lllat
            if d_lat < d_lat_min:
                dd_lat = min([d_lat_min - d_lat, d_lat_max - d_lat])
                lllat -= 0.5 * dd_lat
                urlat += 0.5 * dd_lat

        # Increase longitudinal size if minimum specified
        if d_lon_min is not None:
            d_lon = (urlon - lllon) % 360.0
            if d_lon < d_lon_min:
                dd_lon = min([d_lon_min - d_lon, d_lon_max - d_lon])
                lllon -= 0.5 * dd_lon
                urlon += 0.5 * dd_lon

        if self.config.aspect:
            # Adjust self.aspect ratio to avoid distortion
            aspect = self.config.aspect
            if crosses_dateline:
                d_lat = urlat - lllat
                d_lon = 360.0 - (lllon - urlon)
                if d_lon < d_lat * aspect:
                    dd_lon = min([d_lat * aspect - d_lon, d_lon_max - d_lon])
                    lllon -= 0.5 * dd_lon
                    urlon += 0.5 * dd_lon
                elif d_lat < d_lon / aspect:
                    dd_lat = min([d_lon / aspect - d_lat, d_lat_max - d_lat])
                    lllat -= 0.5 * dd_lat
                    urlat += 0.5 * dd_lat
            else:
                d_lat = urlat - lllat
                d_lon = urlon - lllon
                if d_lon < d_lat * aspect:
                    dd_lon = min([d_lat * aspect - d_lon, d_lon_max - d_lon])
                    lllon -= 0.5 * dd_lon
                    urlon += 0.5 * dd_lon
                elif d_lat < d_lon / aspect:
                    dd_lat = min([d_lon / aspect - d_lat, d_lat_max - d_lat])
                    lllat -= 0.5 * dd_lat
                    urlat += 0.5 * dd_lat

        # Adjust latitudinal range if necessary
        if urlat > lat_max:
            dd_lat = urlat - lat_max
            lllat -= dd_lat
            urlat -= dd_lat
        elif lllat < lat_min:
            dd_lat = lat_min - lllat
            urlat += dd_lat
            lllat += dd_lat

        return (lllon, urlon, lllat, urlat), crosses_dateline


# pylint: disable=R0902  # too-many-instance-attributes (>7)
class GeoMaskBoundingBox:
    """Bounding box of a geographical mask."""

    class EmptyMaskError(ValueError):
        """Cloud mask is empty."""

    def __init__(
        self,
        mask: np.ndarray,
        lat: np.ndarray,
        lon: np.ndarray,
        *,
        periodic_lon: bool = False,
    ) -> None:
        """Create a new instance."""
        shape = (lat.size, lon.size)
        if mask.shape != shape:
            raise ValueError(
                "inconsistent shapes of mask and lat/lon"
                f": {mask.shape} != ({shape[0]}, {shape[1]})"
            )
        if not mask.any():
            raise self.EmptyMaskError(shape)
        self.mask: np.ndarray = mask
        self.lat: np.ndarray = lat
        self.lon: np.ndarray = lon
        self.periodic_lon: bool = periodic_lon

        self.mask_lat: np.ndarray = self.mask.any(axis=1)
        self.mask_lon: np.ndarray = mask.any(axis=0)

        self.lllat: float
        self.urlat: float
        self.lllat, self.urlat = self._get_lat_extent()

        self.lllon: float
        self.urlon: float
        self.crosses_dateline: bool
        (self.lllon, self.urlon), self.crosses_dateline = self._get_lon_extent()

    def get_extent(self) -> Tuple[float, float, float, float]:
        """Get ``(lllon, urlon, lllat, urlat)``."""
        return (self.lllon, self.urlon, self.lllat, self.urlat)

    def get_lat_extent(self) -> Tuple[float, float]:
        """Get ``(lllat, urlat)``."""
        return (self.lllat, self.urlat)

    def get_lon_extent(self) -> Tuple[float, float]:
        """Get ``(lllon, urlon)``."""
        return (self.lllon, self.urlon)

    def _get_lat_extent(self) -> Tuple[float, float]:
        """Return ``(lllat, urlat)``."""
        if not any(self.mask_lat):
            lllat = self.lat.min()
            urlat = self.lat.max()
        else:
            lllat = self.lat[self.mask_lat].min()
            urlat = self.lat[self.mask_lat].max()
        return (lllat, urlat)

    def _get_lon_extent(self) -> Tuple[Tuple[float, float], bool]:
        """Return ``((lllon, urlon), crosses_dateline)``."""
        if self.mask_lon.all():
            lllon = self.lon.min()
            urlon = self.lon.max()
            crosses_dateline = False
        elif not self.periodic_lon:
            if not any(self.mask_lon):
                lllon = self.lon.min()
                urlon = self.lon.max()
            else:
                lllon = self.lon[self.mask_lon].min()
                urlon = self.lon[self.mask_lon].max()
            crosses_dateline = False
        else:
            gaps = find_gaps(self.mask_lon, periodic=True)
            largest_gap = next(iter(sorted(gaps, reverse=True)))
            _, idx_gap_start, idx_gap_end = largest_gap
            idx_lllon = idx_gap_end + 1 if idx_gap_end < self.mask_lon.size - 1 else 0
            idx_urlon = (
                idx_gap_start - 1 if idx_gap_start > 1 else self.mask_lon.size - 1
            )
            lllon = self.lon[idx_lllon]
            urlon = self.lon[idx_urlon]
            crosses_dateline = idx_lllon > idx_urlon
        return (lllon, urlon), crosses_dateline

    def __repr__(self) -> str:
        """Return a string representation."""
        mask = f"<{self.mask.shape}, true={self.mask.sum()}/{self.mask.size}>"
        lat = (
            f"<{self.lat.shape}, "
            f"[{self.lat[0]}, {self.lat[1]}, ..., {self.lat[-2]}, {self.lat[-1]}]>"
        )
        lon = (
            f"<{self.lon.shape}, "
            f"[{self.lon[0]}, {self.lon[1]}, ..., {self.lon[-2]}, {self.lon[-1]}]>"
        )
        return "\n".join(
            [
                f"{type(self).__name__}(",
                f"  mask={mask},",
                f"  lat={lat}",
                f"  lon={lon}",
                ")",
            ]
        )


@summarizable
@dc.dataclass
class ReleaseSiteDomainConfig(DomainConfig):
    """Configuration of a ``Domain`` instance.

    Attributes:
        rel_offset (optional): See docstring of ``DomainConfig``.

        zoom_fact (optional): See docstring of ``DomainConfig``.
            in/out.

        aspect (optional): Target aspect ratio of the domain; must be non-zero
            unless both ``min_size_lat`` and ``min_size_lon`` are non-zero.

        field_proj (optional): Field projection.

        min_size_lat (optional): Minimum latitudinal extent of the domain in
            degrees; must be non-zero unless ``min_size_lon`` is non-zero.

        min_size_lon (optional): Minimum longitudinal extent of the domain in
            degrees; must be non-zero unless ``min_size_lat`` is non-zero.

        release_lat (optional): Latitude of release point; default to the center
            of the domain.

        release_lon (optional): Longitude of release point; default to the
            center of the domain.

    """

    aspect: Optional[float] = None
    field_proj: Projection = dc.field(
        # pylint: disable=E0110  # abstract-class-instatiated (PlateCarree)
        default_factory=PlateCarree
    )
    min_size_lat: float = 0.0
    min_size_lon: float = 0.0
    release_lat: Optional[float] = None
    release_lon: Optional[float] = None


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

    cls_config = ReleaseSiteDomainConfig

    def __init__(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        config: Optional[Union[DomainConfig, Dict[str, Any]]] = None,
    ) -> None:
        """Create an instance of ``ReleaseSiteDomain``.

        Args:
            lat: 1D latitude array.

            lon: 1D longitude array.

            config (optional): Configuration object or parameters.

        """
        self.config: ReleaseSiteDomainConfig
        super().__init__(lat, lon, config)

        if self.config.release_lon is None:
            self.config.release_lon = self.lon.mean()

        if not self.config.min_size_lat and not self.config.min_size_lon:
            raise ValueError(
                "one or both of min_size_lat and min_size_lon must be non-zero"
            )
        elif (
            not self.config.min_size_lat or not self.config.min_size_lon
        ) and not self.config.aspect:
            raise ValueError(
                "aspect must be non-zero unless both min_size_lat and min_size_lon are"
                " non-zero"
            )

    def get_bbox_extent(self) -> Tuple[float, float, float, float]:
        """Return domain corners ``(lllon, lllat, urlon, urlat)``."""
        d_lat = self.config.min_size_lat
        d_lon = self.config.min_size_lon
        if d_lon and not d_lat:
            assert self.config.aspect  # proper check in __init__
            assert d_lon is not None  # mypy
            d_lat = d_lon / self.config.aspect
        elif d_lat and not d_lon:
            assert self.config.aspect  # proper check in __init__
            d_lon = d_lat / self.config.aspect
        if isinstance(self.config.field_proj, RotatedPole):
            c_lon, c_lat = self.config.field_proj.transform_point(
                # pylint: disable=E0110  # abstract-class-instatiated (PlateCarree)
                self.config.release_lon,
                self.get_release_lat(),
                PlateCarree(),
            )
            lllat = c_lat - 0.5 * d_lat
            lllon = c_lon - 0.5 * d_lon
            urlat = c_lat + 0.5 * d_lat
            urlon = c_lon + 0.5 * d_lon
        else:
            lllat = self.get_release_lat() - 0.5 * d_lat
            lllon = self.get_release_lon() - 0.5 * d_lon
            urlat = self.get_release_lat() + 0.5 * d_lat
            urlon = self.get_release_lon() + 0.5 * d_lon
        return lllon, urlon, lllat, urlat

    def get_release_lat(self) -> float:
        return self.config.release_lat or self.lat.mean()

    def get_release_lon(self) -> float:
        return self.config.release_lon or self.lon.mean()


def find_gaps(
    mask: Union[np.ndarray, Sequence[int]], periodic: bool = True
) -> List[Tuple[int, int, int]]:
    """Return a size, start and end of all gaps in a 1D mask."""
    mask = np.asarray(mask, bool)
    if not len(mask.shape) == 1:
        raise ValueError(f"mask1d must have one dimension, not {len(mask.shape)}")
    starts = []
    val_prev = mask[-1] if periodic else True
    for idx, val in enumerate(mask):
        if val_prev and not val:
            starts.append(idx)
        val_prev = val
    gaps: List[Tuple[int, int, int]] = []
    for start in starts:
        idx = start
        size = 0
        while not mask[idx]:
            size += 1
            if idx < mask.size - 1:
                idx += 1
            elif not periodic:
                idx += 1
                break
            else:
                idx = 0
        end = (idx if idx > 0 else mask.size) - 1
        gaps.append((size, start, end))
    return gaps
