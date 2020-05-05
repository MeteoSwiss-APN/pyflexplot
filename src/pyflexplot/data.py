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
from typing import Union

# Third-party
import numpy as np

# Local
from .setup import InputSetupCollection
from .utils import summarizable


def summarize_field(obj: Any) -> Dict[str, Dict[str, Any]]:
    return {
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


@summarizable(
    attrs_skip=["fld", "lat", "lon"],
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

    """

    fld: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    rotated_pole: bool
    var_setups: InputSetupCollection
    time_stats: Mapping[str, np.ndarray]
    nc_meta_data: Mapping[str, Any]

    def __post_init__(self):
        self._check_args(self.fld, self.lat, self.lon)

    def _check_args(self, fld, lat, lon, *, ndim_fld=2):
        """Check consistency of field, dimensions, etc."""

        # Check dimensionalities
        for name, arr, ndim in [
            ("fld", fld, ndim_fld),
            ("lat", lat, 1),
            ("lon", lon, 1),
        ]:
            shape = arr.shape
            if len(shape) != ndim:
                raise ValueError(
                    f"{name}: expect {ndim} dimensions, got {len(shape)}: {shape}"
                )

        # Check consistency
        grid_shape = (lat.size, lon.size)
        if fld.shape[-2:] != grid_shape:
            raise ValueError(
                f"shape of fld inconsistent with (lat, lon): {fld.shape} != "
                r"{grid_shape}"
            )


def threshold_agreement(arr, thr, *, axis=None):
    """Count the members exceeding a threshold at each grid point.

    Args:
        arr (np.ndarray[float]): Data array.

        thr (float): Threshold to be exceeded.

        axis (int, optional): Index of ensemble member axis, along which the
            reduction is performed. Defaults to None.

    """
    # SR_TMP < TODO Remove once type hints added to arguments
    if arr is None:
        raise ValueError("arr is None")
    if thr is None:
        raise ValueError("thr is None")
    # SR_TMP >
    result = np.count_nonzero(arr > thr, axis=axis)
    return result


@dataclass
class EnsembleCloud:
    """Partical cloud in an ensemble simulation.

    Args:
        arr: Data array with dimensions (members, time, space), where space
            represents at least one spatial dimension.

        time: Time dimension values.

        thr: Threshold to be exceeded.

        n_mem_min: Minimum number of members required to agreement.

    """

    arr: np.ndarray
    time: np.ndarray
    thr: float
    n_mem_min: int

    def __post_init__(self):
        self._n_time: int = self.arr.shape[1]
        self.m_cloud_prev: Optional[np.ndarray] = None

    def arrival_time(self) -> np.ndarray:
        """Compute the time at each grid point until the cloud arrives."""
        arr = self.departure_time(self.arr[:, ::-1])[::-1]
        return np.where(~np.isnan(arr), -arr + 1, arr)

    def departure_time(self, arr: Optional[np.array] = None) -> np.ndarray:
        """Compute the time at each grid point until the cloud departs."""
        if arr is None:
            arr = self.arr

        # Initialize all points to NAN
        # Only points that never encounter a cloud will retain this value
        departure_time = np.full(
            tuple([arr.shape[1]] + [n for i, n in enumerate(arr.shape) if i > 1]),
            np.nan,
        )

        # Identify ensemble cloud across all time steps
        cloudy_members = np.count_nonzero(arr > self.thr, axis=0)
        m_cloud = cloudy_members >= self.n_mem_min

        # Mark points where cloud has just departed with 0
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
