"""Data structures."""
# Standard library
import dataclasses as dc
import warnings
from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

# Third-party
import numpy as np

# First-party
from srutils.str import join_multilines

# Local
from ..plotting.domain import CloudDomain
from ..plotting.domain import Domain
from ..plotting.domain import ReleaseSiteDomain
from ..plotting.proj_bbox import Projections
from ..setup import Setup
from ..setup import SetupGroup
from ..utils.exceptions import ArrayDimensionError
from ..utils.exceptions import FieldAllNaNError
from ..utils.exceptions import InconsistentArrayShapesError
from ..utils.formatting import format_ens_file_path
from ..utils.summarize import summarizable
from ..utils.summarize import summarize
from .meta_data import MetaData


def summarize_field(obj: Any) -> Dict[str, Any]:
    dct = {
        "type": type(obj).__name__,
        "fld": {
            "dtype": str(obj.fld.dtype),
            "shape": obj.fld.shape,
            "nanmin": np.nanmin(obj.fld),
            "nanmean": np.nanmean(obj.fld),
            "nanmedian": np.nanmedian(obj.fld),
            "nanmax": np.nanmax(obj.fld),
            "nanmin_nonzero": np.nanmin(np.where(obj.fld == 0, np.nan, obj.fld)),
            "nanmean_nonzero": np.nanmean(np.where(obj.fld == 0, np.nan, obj.fld)),
            "nanmedian_nonzero": np.nanmedian(np.where(obj.fld == 0, np.nan, obj.fld)),
            "nanmax_nonzero": np.nanmax(np.where(obj.fld == 0, np.nan, obj.fld)),
            "n_nan": np.count_nonzero(np.isnan(obj.fld)),
            "n_zero": np.count_nonzero(obj.fld == 0),
        },
        "lat": {
            "dtype": str(obj.lat.dtype),
            "shape": obj.lat.shape,
            "min": obj.lat.min(),
            "max": obj.lat.max(),
        },
        "lon": {
            "dtype": str(obj.lon.dtype),
            "shape": obj.lon.shape,
            "min": obj.lon.min(),
            "max": obj.lon.max(),
        },
        "mdata": obj.mdata,
        "time_props": obj.time_props,
        "var_setups": obj.var_setups,
        "projs": obj.projs,
    }
    return summarize(dct)


@summarizable(summarize=summarize_field)
# pylint: disable=R0902  # too-many-instance-attributes
class Field:
    """FLEXPART field on rotated-pole grid."""

    def __init__(
        self,
        fld: np.ndarray,
        lat: np.ndarray,
        lon: np.ndarray,
        *,
        var_setups: SetupGroup,
        time_props: "FieldTimeProperties",
        mdata: MetaData,
    ) -> None:
        """Create an instance of ``Field``.

        Args:
            fld: Field array (2D) with dimensions (lat, lon).

            lat: Latitude array (1D).

            lon: Longitude array (1D).

            mdata: Meta data for plot for labels etc.

            time_props: Properties of the field across all time steps.

            var_setups: Variables setups.

        """
        self.fld: np.ndarray = fld
        self.lat: np.ndarray = lat
        self.lon: np.ndarray = lon
        self.mdata: MetaData = mdata
        self.time_props: "FieldTimeProperties" = time_props
        self.var_setups: SetupGroup = var_setups
        try:
            self.check_consistency()
        except Exception as e:
            raise ValueError(f"{type(e).__name__}: {e}") from e

        self.projs: Projections = self._init_projs()

    def check_consistency(self):
        """Check consistency of field, dimensions, etc."""
        # Check dimensionalities
        for name, arr, ndim in [
            ("fld", self.fld, 2),
            ("lat", self.lat, 1),
            ("lon", self.lon, 1),
        ]:
            shape = arr.shape
            if len(shape) != ndim:
                raise ArrayDimensionError(f"{name}: len({shape}) != {ndim}")

        # Check consistency
        grid_shape = (self.lat.size, self.lon.size)
        if self.fld.shape[-2:] != grid_shape:
            raise InconsistentArrayShapesError(f"{self.fld.shape} != {grid_shape}")

    def locate_max(self) -> Tuple[float, float]:
        """Find location of field maximum in geographical coordinates."""
        if np.isnan(self.fld).all():
            raise FieldAllNaNError(self.fld.shape)
        assert len(self.fld.shape) == 2  # pylint
        # pylint: disable=W0632  # unbalanced-tuple-unpacking
        jmax, imax = np.unravel_index(np.nanargmax(self.fld), self.fld.shape)
        p_lon, p_lat = self.projs.geo.transform_point(
            self.lon[imax], self.lat[jmax], self.projs.data
        )
        return (p_lat, p_lon)

    def get_domain(self, aspect: float) -> Domain:
        """Initialize Domain object (projection and extent)."""
        lat = self.lat
        lon = self.lon
        model_name = self.var_setups.collect_equal("model.name")
        domain_type = self.var_setups.collect_equal("domain")
        domain_size_lat = self.var_setups.collect_equal("domain_size_lat")
        domain_size_lon = self.var_setups.collect_equal("domain_size_lon")
        assert self.mdata is not None  # mypy
        release_lat = self.mdata.release.lat
        release_lon = self.mdata.release.lon
        field_proj = self.projs.data
        mask_nz = self.time_props.mask_nz
        domain: Optional[Domain] = None
        if domain_type == "full":
            if model_name.startswith("COSMO"):
                domain = Domain(lat, lon, config={"zoom_fact": 1.01})
            else:
                domain = Domain(lat, lon)
        elif domain_type == "release_site":
            domain = ReleaseSiteDomain(
                lat,
                lon,
                config={
                    "aspect": aspect,
                    "field_proj": field_proj,
                    "min_size_lat": domain_size_lat,
                    "min_size_lon": domain_size_lon,
                    "release_lat": release_lat,
                    "release_lon": release_lon,
                },
            )
        elif domain_type == "alps":
            if model_name == "IFS-HRES-EU":
                domain = Domain(
                    lat, lon, config={"zoom_fact": 3.4, "rel_offset": (-0.165, -0.11)}
                )
        elif domain_type == "cloud":
            domain = CloudDomain(
                lat,
                lon,
                mask=mask_nz,
                config={
                    "zoom_fact": 0.9,
                    "aspect": aspect,
                    "min_size_lat": domain_size_lat,
                    "min_size_lon": domain_size_lon,
                    "periodic_lon": (model_name == "IFS-HRES"),
                    "release_lat": release_lat,
                    "release_lon": release_lon,
                },
            )
        elif domain_type == "ch":
            if model_name.startswith("COSMO-1"):
                domain = Domain(
                    lat, lon, config={"zoom_fact": 3.6, "rel_offset": (-0.02, 0.045)}
                )
            elif model_name.startswith("COSMO-2"):
                domain = Domain(
                    lat, lon, config={"zoom_fact": 3.23, "rel_offset": (0.037, 0.1065)}
                )
            elif model_name == "IFS-HRES-EU":
                domain = Domain(
                    lat, lon, config={"zoom_fact": 11.0, "rel_offset": (-0.18, -0.11)}
                )
        if domain is None:
            raise NotImplementedError(
                f"domain for model '{model_name}' and domain type '{domain_type}'"
            )
        return domain

    def __repr__(self):
        lines = [
            f"fld=array[shape={self.fld.shape}, dtype={self.fld.dtype}],",
            f"lat=array[shape={self.lat.shape}, dtype={self.lat.dtype}],",
            f"lon=array[shape={self.lon.shape}, dtype={self.lon.dtype}],",
            f"mdata={self.mdata},",
            f"time_stats={self.time_props},",
            f"var_setups={self.var_setups},",
            f"projs={self.projs},",
        ]
        body = join_multilines(lines, indent=2)
        return "\n".join([f"{type(self).__name__}(", body, ")"])

    def _init_projs(self) -> Projections:
        if self.mdata.simulation.grid_is_rotated:
            return Projections.create_rotated(
                pollat=self.mdata.simulation.grid_north_pole_lat,
                pollon=self.mdata.simulation.grid_north_pole_lon,
            )
        else:
            return Projections.create_regular(clon=self.mdata.release.lon)


@summarizable
@dc.dataclass
class FieldStats:
    min: float = np.nan
    max: float = np.nan
    mean: float = np.nan
    median: float = np.nan

    @classmethod
    def create(cls, arr: np.ndarray) -> "FieldStats":
        arr = np.where(np.isfinite(arr), arr, np.nan)
        # Avoid zero-size errors below
        if arr.size == 0:
            return cls()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning, message="All-NaN slice encountered"
            )
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning, message="Mean of empty slice"
            )
            return cls(
                min=np.nanmin(arr),
                max=np.nanmax(arr),
                mean=np.nanmean(arr),
                median=np.nanmedian(arr),
            )


@summarizable
class FieldTimeProperties:
    """Properties of a 2D field over time."""

    summarizable_attrs = ["stats", "stats_nz"]

    def __init__(self, arr: np.ndarray) -> None:
        """Create an instance of ``FieldTimeProperties``."""
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message="Invalid value encountered in greater",
            )
            arr_nz = np.where(arr > 0, arr, np.nan)
        self.stats = FieldStats.create(arr)
        self.stats_nz = FieldStats.create(arr_nz)
        self.mask: np.ndarray = (~np.isnan(arr)).sum(axis=0) > 0
        self.mask_nz: np.ndarray = (~np.isnan(arr_nz)).sum(axis=0) > 0


@dc.dataclass
class FieldGroupAttrs:
    """Attributes of a ``FieldGroup`` instance."""

    raw_path: str
    paths: Sequence[str]
    ens_member_ids: Optional[Sequence[int]]

    def format_path(self) -> str:
        return format_ens_file_path(self.raw_path, self.ens_member_ids)


class FieldGroup:
    """A group of related ``Field`` objects."""

    def __init__(
        self,
        fields: Sequence[Field],
        attrs=Union[FieldGroupAttrs, Dict[str, Any]],
    ) -> None:
        """Create an instance of ``FieldGroup``."""
        if not isinstance(attrs, FieldGroupAttrs):
            attrs = FieldGroupAttrs(**attrs)

        self.fields: List[Field] = list(fields)
        self.attrs: FieldGroupAttrs = attrs

        setups = SetupGroup([setup for field in fields for setup in field.var_setups])
        self.shared_setup: Setup = setups.compress()

    def __len__(self) -> int:
        return len(self.fields)

    def __iter__(self) -> Iterator[Field]:
        return iter(self.fields)
