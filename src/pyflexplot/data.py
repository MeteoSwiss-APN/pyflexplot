# -*- coding: utf-8 -*-
"""
Data structures.
"""
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

    summarizable_attrs = ["attrs", "field_specs", "time_stats"]

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


def threshold_agreement(arr, thr, *, axis=None, eq_ok=False, dtype=None):
    """Count the members exceeding a threshold at each grid point.

    Args:
        arr (np.ndarray[float]): Data array.

        thr (float): Threshold to be exceeded.

        axis (int, optional): Index of ensemble member axis, along which the
            reduction is performed. Defaults to None.

        eq_ok (bool, optional): Whether values equal to the threshold are
            counted as exceedences. Defaults to False.

        dtype (type, optional): Type of result. Defaults to None.

    """
    if arr is None:
        raise ValueError("arr is None")
    if thr is None:
        raise ValueError("thr is None")
    result = np.count_nonzero(arr >= thr if eq_ok else arr > thr, axis=axis)
    if dtype is not None:
        result = result.astype(dtype)
    return result
