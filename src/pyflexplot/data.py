# -*- coding: utf-8 -*-
"""
Data structures.
"""
import datetime
import logging as log
import numpy as np

from .utils_dev import ipython  #SR_DEV


class FlexField:
    """FLEXPART field on rotated-pole grid."""

    def __init__(self, fld, rlat, rlon, attrs, field_specs, time_stats):
        """Create an instance of ``FlexField``.

        Args:
            fld (ndarray[float, float]): Field array (2D) with
                dimensions (rlat, rlon).

            rlat (ndarray[float]): Rotated latitude array (1D).

            rlon (ndarray[float]): Rotated longitude array (1D).

            attrs (FlexAttrGroupCollection): Attributes collection.

            field_specs (FlexFieldSpecs): Input field specifications.

            time_stats (dict): Some statistics across all time steps.

        """
        self._check_args(fld, rlat, rlon)
        self.fld = fld
        self.rlat = rlat
        self.rlon = rlon
        self.attrs = attrs
        self.field_specs = field_specs
        self.time_stats = time_stats

    def _check_args(self, fld, rlat, rlon, *, ndim_fld=2):
        """Check consistency of field, dimensions, etc."""

        # Check dimensionalities
        for name, arr, ndim in [('fld', fld, ndim_fld), ('rlat', rlat, 1),
                                ('rlon', rlon, 1)]:
            shape = arr.shape
            if len(shape) != ndim:
                raise ValueError(
                    f"{name}: expect {ndim} dimensions, got {shape}: {shape}")

        # Check consistency
        grid_shape = (rlat.size, rlon.size)
        if fld.shape[-2:] != grid_shape:
            raise ValueError(
                f"shape of fld inconsistent with (rlat, rlon): "
                f"{fld.shape} != {grid_shape}")


def threshold_agreement(arr, thr, *, axis=None, eq_ok=False, dtype=None):
    """Count the members exceeding a threshold at each grid point.

    Args:
        arr (np.ndarray[float]): Data array.

        thr (float): Threshold to be exceeded.

        axis (int, optional): Index of ensemble member axis, along
            which the reduction is performed. Defaults to None.

        eq_ok (bool, optional): Whether values equal to the threshold
            are counted as exceedences. Defaults to False.

    """
    if arr is None:
        raise ValueError('arr is None')
    if thr is None:
        raise ValueError('thr is None')
    result = np.count_nonzero(arr >= thr if eq_ok else arr > thr, axis=axis)
    if dtype is not None:
        result = result.astype(dtype)
    return result
