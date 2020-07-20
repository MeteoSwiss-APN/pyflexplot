# -*- coding: utf-8 -*-
"""
Data structures.
"""
# Standard library
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Dict
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

# Third-party
import numpy as np

# First-party
from srutils.str import join_multilines

# Local
from .meta_data import MetaData
from .setup import SetupCollection
from .utils.exceptions import ArrayDimensionError
from .utils.exceptions import FieldAllNaNError
from .utils.exceptions import InconsistentArrayShapesError
from .utils.summarize import default_summarize
from .utils.summarize import summarizable


def summarize_field(obj: Any) -> Dict[str, Dict[str, Any]]:
    dct = {
        "fld": {
            "dtype": str(obj.fld.dtype),
            "shape": obj.fld.shape,
            "nanmin": np.nanmin(obj.fld),
            "nanmean": np.nanmean(obj.fld),
            "nanmedian": np.nanmedian(obj.fld),
            "nanmax": np.nanmax(obj.fld),
            "nanmin_nonzero": np.nanmin(obj.fld[obj.fld != 0]),
            "nanmean_nonzero": np.nanmean(obj.fld[obj.fld != 0]),
            "nanmedian_nonzero": np.nanmedian(obj.fld[obj.fld != 0]),
            "nanmax_nonzero": np.nanmax(obj.fld[obj.fld != 0]),
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
    }
    return default_summarize(dct)


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

        rotated_pole: Whether pole is rotated.

        var_setups: Variables setups.

        time_stats: Some statistics across all time steps.

        nc_meta_data: Meta data from NetCDF input file.

        mdata: Meta data for plot for labels etc.

    """

    fld: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    rotated_pole: bool
    var_setups: SetupCollection
    time_stats: Mapping[str, np.ndarray]
    nc_meta_data: Mapping[str, Any]
    mdata: Optional[MetaData]

    def __post_init__(self):
        try:
            self.check_consistency()
        except Exception as e:
            raise ValueError(f"{type(e).__name__}: {e}")

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
                f" rotated_pole={self.rotated_pole},"
            ),
            f"var_setups={self.var_setups},",
            (
                f"time_stats=dict[n={len(self.time_stats)},"
                f" keys={tuple(self.time_stats)}],"
            ),
            (
                f"nc_meta_data="
                f"dict[n={len(self.nc_meta_data)}, keys={tuple(self.nc_meta_data)}],"
            ),
        ]
        body = join_multilines(lines, indent=2)
        return "\n".join([f"{type(self).__name__}(", body, ")"])

    def locate_max(self) -> Tuple[float, float]:
        if np.isnan(self.fld).all():
            raise FieldAllNaNError(self.fld.shape)
        assert len(self.fld.shape) == 2  # pylint
        # pylint: disable=W0632  # unbalanced-tuple-unpacking
        jmax, imax = np.unravel_index(np.nanargmax(self.fld), self.fld.shape)
        return (
            self.lat[jmax],
            self.lon[imax],
        )


def ensemble_probability(arr: np.ndarray, thr: float, n_mem: int) -> np.ndarray:
    """Ensemble-based probability of threshold exceedence at each point.

    Args:
        arr: Data array with dimensions (member, ...).

        thr: Minimum threshold value defining a cloud.

        n_mem: Total number of members.

    Returns:
        Field with the number of members with a cloud at each grid point.

    """
    arr = np.count_nonzero(arr >= thr, axis=0).astype(np.float32) * 100 / n_mem
    return arr


@dataclass
class EnsembleCloud:
    """Partical cloud in an ensemble simulation.

    Args:
        arr: Data array with dimensions (members, time, space), where space
            represents at least one spatial dimension.

        time: Time dimension values.

        thr: Minimum threshold value defining a cloud in a single member.

    """

    arr: np.ndarray
    time: np.ndarray
    thr: float

    def __post_init__(self):
        self._n_time: int = self.arr.shape[1]
        self.m_cloud_prev: Optional[np.ndarray] = None

    def arrival_time(self, mem: int) -> np.ndarray:
        """Time until the cloud arrives.

        Args:
            mem: Minimum number of members defining the ensemble cloud.

        Returns:
            Field over time with the time until/since the first cloud arrival:
                - > 0: Time until the point encounters its first cloud.
                - 0: Time step when the point encounters its first cloud.
                - < 0: Time since the point has encountered its first cloud.
                - -inf: The point is already cloudy at the first time step.
                - nan: The point never encounters a cloud.

        """
        arr_bak = self.arr.copy()
        self.arr = self.arr[:, ::-1]
        arr = self.departure_time(mem)[::-1]
        self.arr = arr_bak
        return np.where(~np.isnan(arr), -arr + 1, arr)

    def departure_time(self, mem: int) -> np.ndarray:
        """Time until the cloud departs.

        Args:
            mem: Minimum number of members defining the ensemble cloud.

        Returns:
            Field over time with the time since/until the first cloud departure:
                - > 0: Time until its last cloud leaves the point.
                - 0: Time step when the point has just been left its last cloud.
                - < 0: Time since its last cloud has left the point.
                - inf: The point is still cloudy at the last time step.
                - nan: The point never encounters a cloud.

        """

        # Points that never encounter a cloud will retain NaN
        departure_time = self._init_result(np.nan)

        # Mark points where cloud has just departed with 0
        m_cloud = self._identify_ens_cloud(mem)
        m_departed = np.full(m_cloud.shape, False)
        m_departed[1:] = m_cloud[:-1] & ~m_cloud[1:]
        departure_time[m_departed] = 0

        # Iterate backward in time
        departure_time[-1][m_cloud[-1]] = np.inf
        for time_idx in range(self._n_time - 2, -1, -1):
            d_time = self.time[time_idx + 1] - self.time[time_idx]

            # Set points where cloud will vanish to the time until it does so
            m_fin_next = np.isfinite(departure_time[time_idx + 1])
            m_will_depart = np.where(m_fin_next, departure_time[time_idx + 1], -1) >= 0
            departure_time[time_idx][m_will_depart] = (
                departure_time[time_idx + 1][m_will_depart] + d_time
            )

            # Set points where cloud will never depart for good to INF
            m_wont_depart = np.isinf(departure_time[time_idx + 1])
            departure_time[time_idx][m_wont_depart] = np.inf

        # Iterate forward in time
        for time_idx in range(1, self._n_time):
            d_time = self.time[time_idx] - self.time[time_idx - 1]

            # Mark points where cloud has vanished with the time since then
            m_nan_prev = np.isnan(departure_time[time_idx - 1])
            m_has_departed = ~m_cloud[time_idx] & (
                np.where(m_nan_prev, np.inf, departure_time[time_idx - 1]) <= 0
            )
            departure_time[time_idx][m_has_departed] = (
                departure_time[time_idx - 1][m_has_departed] - d_time
            )

        return departure_time

    def occurrence_probability(self, win: int) -> np.ndarray:
        """Probability that a cloud occurs in a time window.

        Args:
            win: Time window for the cloud to occur.

        Returns:
            Field over time with the probability that a cloud occurs:
                - TODO

        """
        if int(win) != win:
            raise ValueError("win must be an integer", win)
        win = int(win)
        occurr_prob = self._init_result(np.nan)
        cloudy_members = self._count_cloudy_members()
        occurr_prob[:] = cloudy_members[:]
        n_ts = self._init_result(win + 1)
        for idx in range(1, win + 1):
            try:
                occurr_prob[:-idx] += cloudy_members[idx:]
                n_ts[-idx] = idx
            except IndexError:
                break
        occurr_prob[:] = occurr_prob * 100 / (n_ts * self._n_members())
        return occurr_prob

    def _n_members(self):
        """Number of ensemble members."""
        return self.arr.shape[0]

    def _init_result(self, val: float) -> np.ndarray:
        """Initalize results array."""
        shape = tuple([self.arr.shape[1]] + list(self.arr.shape[2:]))
        return np.full(shape, val)

    def _identify_ens_cloud(self, mem: Optional[int] = None) -> np.ndarray:
        """Identify ensemble cloud across at each grid point and time step."""
        return self._count_cloudy_members() >= mem

    def _count_cloudy_members(self) -> np.ndarray:
        """Count the members with a cloud at each grid point and time step."""
        return np.count_nonzero(self.arr >= self.thr, axis=0)


def merge_fields(
    flds: Sequence[np.ndarray], op: Union[Callable, Sequence[Callable]] = np.nansum,
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
            raise ValueError(
                "wrong number of fields", len(flds), len(op_lst) + 1,
            )
        fld = flds[0]
        for i, fld_i in enumerate(flds[1:]):
            _op = op_lst[i]
            fld = _op([fld, fld_i], axis=0)
        return fld
    else:
        raise Exception("no operator(s) defined")
