# -*- coding: utf-8 -*-
"""
Data structures.
"""
import datetime
import logging as log
import numpy as np

from .utils_dev import ipython  #SR_DEV


class FlexFieldRotPole:
    """FLEXPART field on rotated-pole grid."""

    def __init__(self, fld, rlat, rlon, attrs, field_specs, time_stats):
        """Create an instance of ``FlexFieldRotPole``.

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


class FlexFieldEnsRotPole(FlexFieldRotPole):
    """FLEXPART field ensemble on rotated-pole grid."""

    def __init__(self, fld, *args, **kwargs):
        """Create an instance of ``FlexFieldEnsRotPole``.

        Args:
            fld (ndarray[float, float, float]): Field array (3D) with
                dimensions (member, rlat, rlon).

            *args: Positional parameters passed to ``FlexFieldRotPole``.

            *kwargs: Keyword parameters passed to ``FlexFieldRotPole``.

        """
        super().__init__(fld, *args, **kwargs)

    def _check_args(self, fld, rlat, rlon, *, ndim_fld=3):
        super()._check_args(fld, rlat, rlon, ndim_fld=ndim_fld)

    @classmethod
    def from_fields(cls, fields):
        """Create an field ensemble from multiple individual fields."""

        def collect_attrs(name):
            return [getattr(f, name) for f in fields]

        fld = cls._merge_fld(collect_attrs('fld'))
        rlat = cls._merge_rlat(collect_attrs('rlat'))
        rlon = cls._merge_rlon(collect_attrs('rlon'))
        attrs = cls._merge_attrs(collect_attrs('attrs'))
        field_specs = cls._merge_field_specs(collect_attrs('field_specs'))
        time_stats = cls._merge_time_stats(collect_attrs('time_stats'))

        return cls(fld, rlat, rlon, attrs, field_specs, time_stats)

    @classmethod
    def _merge_fld(cls, fld_lst):
        return np.array(fld_lst)

    @classmethod
    def _merge_rlat(cls, rlat_lst):
        rlat = rlat_lst[0]
        if not (rlat == np.array(rlat_lst)).all():
            raise ValueError(
                f"({len(rlat_lst)} rlat arrays not identical: {rlat_lst}")
        return rlat

    @classmethod
    def _merge_rlon(cls, rlon_lst):
        rlon = rlon_lst[0]
        if not (rlon == np.array(rlon_lst)).all():
            raise ValueError(
                f"{len(rlon_lst)} rlon arrays not identical: {rlon_lst}")
        return rlon

    @classmethod
    def _merge_attrs(cls, attrs_lst):
        attrs = attrs_lst[0]
        for attrs_i in attrs_lst[1:]:
            if attrs_i != attrs:
                raise ValueError(
                    f"{len(attrs_lst)} attrs not identical: {attrs_lst}")
        return attrs

    @classmethod
    def _merge_field_specs(cls, field_specs_lst):
        field_specs = field_specs_lst[0]
        for field_specs_i in field_specs_lst[1:]:
            if field_specs_i != field_specs:
                raise ValueError(
                    f"{len(field_specs_lst)} field_specs not identical: "
                    f"{field_specs_lst}")
        return field_specs

    @classmethod
    def _merge_time_stats(cls, time_stats_lst):

        # Check names
        var_names = time_stats_lst[0].keys()
        for var_names_i in (t.keys() for t in time_stats_lst[1:]):
            if var_names_i != var_names:
                raise ValueError(
                    f"{len(time_stats_lst)} differ in their variable names: "
                    f"{[sorted(t.keys()) for t in time_stats_lst]}")
        var_names = sorted(var_names)

        # Merge values
        time_stats = {}
        for var_name in var_names:
            vals = [t[var_name] for t in time_stats_lst]
            if var_name == 'max':
                val = max(vals)
            #SR_TMP< TODO Find solution! Need full fields since start!
            elif var_name in ['mean', 'mean_nz', 'median', 'median_nz']:
                log.warning(f"time_stats: cannot merge '{var_name}'")
                val = np.nan
            #SR_TMP>
            else:
                raise NotImplementedError(f"time stat '{var_name}'")
            time_stats[var_name] = val

        return time_stats
