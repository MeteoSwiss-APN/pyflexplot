"""Data structures."""
# Standard library
import warnings
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Dict
from typing import Mapping
from typing import Sequence
from typing import Tuple
from typing import Union

# Third-party
import cartopy
import numpy as np
from cartopy.crs import PlateCarree
from cartopy.crs import Projection

# First-party
from srutils.str import join_multilines

# Local
from ..plotting.proj_bbox import MapAxesProjections
from ..setup import SetupCollection
from ..utils.exceptions import ArrayDimensionError
from ..utils.exceptions import FieldAllNaNError
from ..utils.exceptions import InconsistentArrayShapesError
from ..utils.summarize import summarizable
from ..utils.summarize import summarize
from .meta_data import MetaData


def summarize_field(obj: Any) -> Dict[str, Any]:
    dct = {
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
        "proj": obj.proj,
    }
    return summarize(dct)


@summarizable(
    attrs_skip=["fld", "lat", "lon", "mdata"],
    post_summarize=lambda self, summary: {**summary, **summarize_field(self)},
)
@dataclass
# pylint: disable=R0902  # too-many-instance-attributes
class Field:
    """FLEXPART field on rotated-pole grid.

    Args:
        fld: Field array (2D) with dimensions (lat, lon).

        lat: Latitude array (1D).

        lon: Longitude array (1D).

        proj: Projection of input field.

        var_setups: Variables setups.

        time_props: Properties of the field across all time steps.

        nc_meta_data: Meta data from NetCDF input file.

        mdata: Meta data for plot for labels etc.

    """

    fld: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    proj: Projection
    var_setups: SetupCollection
    time_props: "FieldTimeProperties"
    nc_meta_data: Mapping[str, Any]
    mdata: MetaData

    def __post_init__(self):
        try:
            self.check_consistency()
        except Exception as e:
            raise ValueError(f"{type(e).__name__}: {e}") from e

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

    def __repr__(self):
        lines = [
            (
                f"fld=array[shape={self.fld.shape}, dtype={self.fld.dtype}],"
                f" lat=array[shape={self.lat.shape}, dtype={self.lat.dtype}],"
                f" lon=array[shape={self.lon.shape}, dtype={self.lon.dtype}],"
                f" proj={type(self.proj).__name__}(...),"
            ),
            f"var_setups={self.var_setups},",
            f"time_stats={self.time_props},",
            (
                f"nc_meta_data="
                f"dict[n={len(self.nc_meta_data)}, keys={tuple(self.nc_meta_data)}],"
            ),
        ]
        body = join_multilines(lines, indent=2)
        return "\n".join([f"{type(self).__name__}(", body, ")"])

    def locate_max(self) -> Tuple[float, float]:
        """Find location of field maximum in geographical coordinates."""
        if np.isnan(self.fld).all():
            raise FieldAllNaNError(self.fld.shape)
        assert len(self.fld.shape) == 2  # pylint
        # pylint: disable=W0632  # unbalanced-tuple-unpacking
        jmax, imax = np.unravel_index(np.nanargmax(self.fld), self.fld.shape)
        p_lon, p_lat = PlateCarree().transform_point(
            self.lon[imax], self.lat[jmax], self.proj
        )
        return (p_lat, p_lon)

    def get_projs(self) -> MapAxesProjections:
        proj_geo = cartopy.crs.PlateCarree()
        if isinstance(self.proj, cartopy.crs.RotatedPole):
            rotpol_attrs = self.nc_meta_data["variables"]["rotated_pole"]["ncattrs"]
            proj_data = cartopy.crs.RotatedPole(
                pole_latitude=rotpol_attrs["grid_north_pole_latitude"],
                pole_longitude=rotpol_attrs["grid_north_pole_longitude"],
            )
            proj_map = proj_data
        else:
            proj_data = cartopy.crs.PlateCarree(central_longitude=0.0)
            proj_map = cartopy.crs.PlateCarree(central_longitude=self.mdata.release.lon)
        return MapAxesProjections(data=proj_data, map=proj_map, geo=proj_geo)


@summarizable
@dataclass
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
    """Standard statistics of a 2D field over time."""

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


def ensemble_probability(
    arr: np.ndarray, thr: float = 0.0, thr_type: str = "lower"
) -> np.ndarray:
    """Ensemble-based probability of threshold exceedence at each point.

    Args:
        arr: Data array with ensemble members as the first dimension.

        thr (optional): Threshold value for data selection in each member.

        thr_type (optional): Threshold type (lower or upper).

    Returns:
        Field with the number of members with a cloud at each grid point.

    """
    if thr_type == "lower":
        mask = arr > thr
    elif thr_type == "upper":
        mask = arr < thr
    else:
        raise ValueError(
            f"invalid threshold type '{thr_type}' (neither 'lower' nor 'upper')"
        )
    n_mem = arr.shape[0]
    arr = np.count_nonzero(mask, axis=0).astype(np.float32) * 100 / n_mem
    return arr


class Cloud:
    """Particle cloud."""

    def __init__(
        self,
        mask: np.ndarray,
        ts: float = 1.0,
    ) -> None:
        """Create an instance of ``Cloud``.

        Args:
            mask: Cloud mask array with two or more dimensions (time plus one or
                more spatial dimensions).

            ts: Time step duration.

            thr (optional): Threshold value defining a cloud.

        """
        self.mask = np.asarray(mask, np.bool)
        self.ts = ts
        if len(self.mask.shape) < 2:
            raise ValueError(f"mask must be 2D or more, not {len(self.mask.shape)}D")

    def departure_time(self) -> np.ndarray:
        """Time until the last cloud has departed.

        Returns:
            Array with the same shape as ``mask`` containing:

                - -inf: Cloud-free until the end, regardless of what was before.
                - > 0: Time until the last cloud will have departed.
                - inf: A cloud is still present at the last time step.

        """
        arr = np.full(self.mask.shape, -np.inf, np.float32)

        # Set points with a cloud at the last time step to +INF at all steps
        arr[:, self.mask[-1]] = np.inf

        # Points without a cloud until the last time step
        m_clear_till_end = self.mask[::-1].cumsum(axis=0)[::-1] == 0

        # Points where the last cloud disappears at the next time step
        m_last_cloud = np.concatenate(
            [
                (m_clear_till_end[1:].astype(int) - m_clear_till_end[:-1]),
                np.zeros([1] + list(self.mask.shape[1:])),
            ],
            axis=0,
        ).astype(bool)

        # Points where a cloud will disappear before the last time step
        m_will_disappear = m_last_cloud[::-1].cumsum(axis=0)[::-1].astype(np.bool)

        # Set points where a cloud will disappear to the time until it's gone
        arr[:] = np.where(
            m_will_disappear,
            m_will_disappear[::-1].cumsum(axis=0)[::-1] * self.ts,
            arr,
        )

        return arr

    def arrival_time(self) -> np.ndarray:
        """Time until the first cloud has arrived.

        Returns:
            Array with the same shape as ``mask`` containing:

                - inf: Cloud-free until the end, regardless of what was before.
                - > 0: Time until the first cloud will have arrived.
                - < 0: Time since the before first cloud has arrived.
                - -inf: A cloud has been present since the first time step.

        """
        arr = np.full(self.mask.shape, np.inf, np.float32)

        # Points without a cloud since the first time step
        m_clear_since_start = self.mask.cumsum(axis=0) == 0

        # Points without a cloud until the last time step
        m_clear_till_end = self.mask[::-1].cumsum(axis=0)[::-1] == 0

        # Set points that have been cloudy since the start to -INF
        arr[self.mask[:1] & ~m_clear_till_end] = -np.inf

        # Points where the first cloud has appeard during the previous time step
        m_first_cloud = np.concatenate(
            [
                (m_clear_since_start[1:].astype(int) - m_clear_since_start[:-1]),
                np.zeros([1] + list(self.mask.shape[1:])),
            ],
            axis=0,
        ).astype(bool)

        # Points where first cloud will appear before the last time step
        m_will_appear = m_first_cloud[::-1].cumsum(axis=0)[::-1].astype(np.bool)

        # Set points where first cloud will appear to the time until it's there
        arr[:] = np.where(
            m_will_appear, m_will_appear[::-1].cumsum(axis=0)[::-1] * self.ts, arr
        )

        # Points where first cloud has appeared before the current time step
        m_has_appeared = (
            ~m_clear_since_start & ~m_will_appear & ~m_clear_till_end & ~self.mask[:1]
        )

        # Set points where first cloud has appeared to time since before it has
        arr[:] = np.where(m_has_appeared, -m_has_appeared.cumsum(axis=0) * self.ts, arr)

        return arr


# SR_TODO Eliminate EnsembleCloud once Cloud works
class EnsembleCloud(Cloud):
    """Particle cloud in an ensemble simulation."""

    def __init__(self, mask: np.ndarray, mem_min: int = 1, ts: float = 1.0) -> None:
        """Create in instance of ``EnsembleCloud``.

        Args:
            mask: Cloud mask array with at least three dimensions (ensemble
                members, time and one or more spatial dimensions).

            mem_min: Minimum number of members required per grid point to define
                the ensemble cloud.

            ts (optional): Time step duration.

        """
        mask = np.asarray(mask, np.bool)
        if len(mask.shape) < 3:
            raise ValueError(f"mask must be 3D or more, not {len(mask.shape)}D")
        mask = np.count_nonzero(mask, axis=0) >= mem_min
        super().__init__(mask=mask, ts=ts)


def merge_fields(
    flds: Sequence[np.ndarray], op: Union[Callable, Sequence[Callable]] = np.nansum
) -> np.ndarray:
    """Merge fields by applying a single operator or an operator chain.

    Args:
        flds: Fields to be merged.

        op (optional): Opterator(s) used to combine input fields. Must accept
            argument ``axis=0`` to only reduce along over the fields.

            If a single operator is passed, it is used to sequentially combine
            one field after the other, in the same order as the corresponding
            specifications (``var_setups``).

            If a list of operators has been passed, then it's length must be
            one smaller than that of ``var_setups``, such that each
            operator is used between two subsequent fields (again in the same
            order as the corresponding specifications).

    """
    if callable(op):
        return op(flds, axis=0)
    elif isinstance(op, Sequence):
        op_lst = op
        if not len(flds) == len(op_lst) + 1:
            raise ValueError("wrong number of fields", len(flds), len(op_lst) + 1)
        fld = flds[0]
        for i, fld_i in enumerate(flds[1:]):
            _op = op_lst[i]
            fld = _op([fld, fld_i], axis=0)
        return fld
    else:
        raise Exception("no operator(s) defined")
