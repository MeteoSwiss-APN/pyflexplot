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


def threshold_agreement(arr, thr, *, axis=None, thr_eq_ok=False):
    """Count the members exceeding a threshold at each grid point.

    Args:
        arr (np.ndarray[float]): Data array.

        thr (float): Threshold to be exceeded.

        axis (int, optional): Index of ensemble member axis, along which the
            reduction is performed. Defaults to None.

        thr_eq_ok (bool, optional): Whether values equal to the threshold are
            counted as exceedences. Defaults to False.

    """
    # SR_TMP < TODO Remove once type hints added to arguments
    if arr is None:
        raise ValueError("arr is None")
    if thr is None:
        raise ValueError("thr is None")
    # SR_TMP >
    compare = np.greater_equal if thr_eq_ok else np.greater
    result = np.count_nonzero(compare(arr, thr), axis=axis)
    return result


def cloud_arrival_time(
    arr: np.ndarray,
    thr: float,
    n_mem_min: int,
    *,
    mem_axis: Optional[int] = None,
    time_axis: Optional[int] = None,
    thr_eq_ok: bool = False,
) -> np.ndarray:
    """Count the time steps until a cloud arrives in enough members.

    Args:
        arr: Data array.

        thr: Threshold to be exceeded.

        n_mem_min: Minimum number of members required to agreement.

        mem_axis (optional): Index of ensemble member axis, along which the
            reduction is performed.

        time_axis (optional): Index of time axis. If None, the first
            non-member-axis is chosen.

        thr_eq_ok (optional): Whether values equal to the threshold are counted
            as exceedences.

    """
    # SR_TMP < TODO Remove once type hints added to arguments
    if arr is None:
        raise ValueError("arr is None")
    if thr is None:
        raise ValueError("thr is None")
    if n_mem_min is None:
        raise ValueError("n_mem_min is None")
    # SR_TMP >
    if time_axis is None:
        time_axis = 1 if mem_axis == 0 else 0
    compare = np.greater_equal if thr_eq_ok else np.greater
    time_idx_max = arr.shape[time_axis] - 1
    shape = tuple(
        [arr.shape[time_axis]]
        + [n for i, n in enumerate(arr.shape) if i not in (time_axis, mem_axis)]
    )
    result = np.full(shape, np.nan)
    for time_idx in range(time_idx_max, -1, -1):
        arr_i = np.take(arr, time_idx, time_axis)
        m_cloud = np.count_nonzero(compare(arr_i, thr), axis=mem_axis) >= n_mem_min
        result[time_idx][m_cloud] = 0
        if time_idx < time_idx_max:
            result[time_idx][~m_cloud] = result[time_idx + 1][~m_cloud] + 1
    return result


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
