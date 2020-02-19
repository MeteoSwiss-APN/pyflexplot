# -*- coding: utf-8 -*-
"""
Data structures.
"""
# Standard library
from typing import List

# Third-party
import numpy as np

# Local
from .utils import SummarizableClass


class Field(SummarizableClass):
    """FLEXPART field on rotated-pole grid."""

    def __init__(self, fld, rlat, rlon, attrs, field_specs, time_stats):
        """Create an instance of ``Field``.

        Args:
            fld (ndarray[float, float]): Field array (2D) with
                dimensions (rlat, rlon).

            rlat (ndarray[float]): Rotated latitude array (1D).

            rlon (ndarray[float]): Rotated longitude array (1D).

            attrs (AttrGroupCollection): Attributes collection.

            field_specs (FieldSpecs): Input field specifications.

            time_stats (dict): Some statistics across all time steps.

        """
        self._check_args(fld, rlat, rlon)
        self.fld = fld
        self.rlat = rlat
        self.rlon = rlon
        self.attrs = attrs
        self.field_specs = field_specs
        self.time_stats = time_stats
        self.scale_fact = 1.0

    def _check_args(self, fld, rlat, rlon, *, ndim_fld=2):
        """Check consistency of field, dimensions, etc."""

        # Check dimensionalities
        for name, arr, ndim in [
            ("fld", fld, ndim_fld),
            ("rlat", rlat, 1),
            ("rlon", rlon, 1),
        ]:
            shape = arr.shape
            if len(shape) != ndim:
                raise ValueError(
                    f"{name}: expect {ndim} dimensions, got {shape}: {shape}"
                )

        # Check consistency
        grid_shape = (rlat.size, rlon.size)
        if fld.shape[-2:] != grid_shape:
            raise ValueError(
                f"shape of fld inconsistent with (rlat, rlon): {fld.shape} != "
                r"{grid_shape}"
            )

    summarizable_attrs: List[str] = ["attrs", "field_specs", "time_stats"]

    def summarize(self, *args, **kwargs):
        data = super().summarize(*args, **kwargs)
        data["fld"] = {
            "dtype": str(self.fld.dtype),
            "shape": self.fld.shape,
            "nanmin": np.nanmin(self.fld),
            "nanmean": np.nanmean(self.fld),
            "nanmedian": np.nanmedian(self.fld),
            "nanmax": np.nanmax(self.fld),
            "nanmin_nonzero": np.nanmin(self.fld[self.fld != 0]),
            "nanmean_nonzero": np.nanmean(self.fld[self.fld != 0]),
            "nanmedian_nonzero": np.nanmedian(self.fld[self.fld != 0]),
            "nanmax_nonzero": np.nanmax(self.fld[self.fld != 0]),
            "n_nan": np.count_nonzero(np.isnan(self.fld)),
            "n_zero": np.count_nonzero(self.fld == 0),
        }
        data["rlat"] = {
            "dtype": str(self.rlat.dtype),
            "shape": self.rlat.shape,
            "min": self.rlat.min(),
            "max": self.rlat.max(),
        }
        data["rlon"] = {
            "dtype": str(self.rlon.dtype),
            "shape": self.rlon.shape,
            "min": self.rlon.min(),
            "max": self.rlon.max(),
        }
        return data

    def scale(self, fact):
        if fact is None:
            return
        self.scale_fact *= fact
        self.fld = self.fld * fact
        for key in self.time_stats:
            self.time_stats[key] *= fact


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
    arr, thr, n_mem_min, *, mem_axis=None, time_axis=None, thr_eq_ok=False,
):
    """Count the time steps until a cloud arrives in enough members.

    Args:
        arr (np.ndarray[float]): Data array.

        thr (float): Threshold to be exceeded.

        n_mem_min (int): Minimum number of members required to agreement.

        mem_axis (int, optional): Index of ensemble member axis, along which
            the reduction is performed. Defaults to None.

        time_axis (int, optional): Index of time axis. If None, the first non-
            member-axis is chosen. Defaults to None.

        thr_eq_ok (bool, optional): Whether values equal to the threshold are
            counted as exceedences. Defaults to False.

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
