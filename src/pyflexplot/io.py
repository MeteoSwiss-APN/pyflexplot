# -*- coding: utf-8 -*-
"""
Input/output.
"""
import datetime
import itertools
import logging as log
import netCDF4 as nc4
import numpy as np
import os
import re
import time

from collections import namedtuple
from copy import copy, deepcopy
from pprint import pformat
from pprint import pprint  #SR_DEV

from .attr import FlexAttrGroupCollection
from .data import FlexFieldRotPole
from .utils import check_array_indices
from .utils import pformat_dictlike
from .utils import nested_dict_set

from .utils_dev import ipython  #SR_DEV


def _nc_content():
    """Content of NetCDF file; dummy function to fold the comment!"""
    #
    # grid_conc_20190514000000.nc
    # ---------------------------
    #
    # Dimensions:
    #   <name>          <size>      <description>
    #   time            11          time
    #   rlon            1158        rotated longitude
    #   rlat            774         rotated latitude
    #   level           3           vertical level
    #   numspec         2           species
    #   numpoint        1           release point
    #   nageclass       1           particla age class
    #   nchar           45          max. string length
    #
    # Variables (0) -- Dimensions etc.:
    #   <name>          <type>  <dimensions>    <description>
    #   rotated_pole    char    (,)             rotated pole lon/lat
    #   time            int     (time,)         seconds since 201905140000
    #   rlon            float   (rlon,)         rotated longitudes
    #   rlat            float   (rlat,)         rotated latitudes
    #   level           float   (level,)        height in meters
    #
    # Variables (1) -- Release points:
    #   <name>          <type>  <dimensions>        <description>
    #   RELCOM          char    (numpoint, nchar)   release point name
    #   RELLNG1         float   (numpoint,)         release longitude lower left
    #   RELLNG2         float   (numpoint,)         release latitude lower left
    #   RELLAT1         float   (numpoint,)         release longitude upper right
    #   RELLAT2         float   (numpoint,)         release latitude upper right
    #   RELZZ1          float   (numpoint,)         release height bottom
    #   RELZZ2          float   (numpoint,)         release height top
    #   RELKINDZ        int     (numpoint,)         release kind
    #   RELSTART        int     (numpoint,)         release start rel. to sim. start
    #   RELEND          int     (numpoint,)         release end rel. to sim. start
    #   RELPART         int     (numpoint,)         number of release particles
    #   RELXMASS        float   (numspec, numpoint) total release particle mass
    #   LAGE            int     (nageclass)         age class
    #
    # Variables (2) -- Particle fields:
    #   <name>          <type>  <dimensions>                                    <description>
    #   spec001         float   (nageclass, numpoint, time, level, rlat, rlon)  concentration of CS-137
    #   WD_spec001      float   (nageclass, numpoint, time, rlat, rlon)         wet deposition of CS-137
    #   DD_spec001      float   (nageclass, numpoint, time, rlat, rlon)         dry deposition of CS-137
    #   spec002         float   (nageclass, numpoint, time, level, rlat, rlon)  concentration of I-131a
    #   WD_spec002      float   (nageclass, numpoint, time, rlat, rlon)         wet deposition of I-131a
    #   DD_spec002      float   (nageclass, numpoint, time, rlat, rlon)         dry deposition of I-131a
    #   fptot           float   (numpoint, rlat, rlon)                          total footprint
    #


def int_or_list(arg):
    try:
        iter(arg)
    except TypeError:
        return int(arg)
    else:
        return [int(a) for a in arg]


class FlexVarSpecs:
    """FLEXPART input variable specifications."""

    # Keys with respective type
    _keys_w_type = {
        'species_id': int_or_list,
        'integrate': bool,
        # Dimensions
        'time': int,
        'nageclass': int,
        'numpoint': int,
    }

    @classmethod
    def specs(cls, types=False):
        for k, v in sorted(cls._keys_w_type.items()):
            if types:
                yield k, v
            else:
                yield k

    def __init__(self, *, rlat=None, rlon=None, **kwargs):
        """Create an instance of ``FlexVarSpecs``.

        Args:
            rlat (tuple, optional): Rotated latitude slice parameters,
                passed to built-in ``slice``. Defaults to None.

            rlon (tuple, optional): Rotated longitude slice parameters,
                passed to built-in ``slice``. Defaults to None.

            **kwargs: Arguments as in ``FlexVarSpecs.specs(types=True)``.
                The keys correspond to the argument's names, and the
                values specify a type which the respective argument
                value must be compatible with.

        """

        def prepare_dim(dim):
            if dim is None:
                dim = (None,)
            elif isinstance(dim, slice):
                dim = (dim.start, dim.stop, dim.step)
            try:
                slice(*dim)
            except ValueError:
                raise
            return dim

        self.rlat = prepare_dim(rlat)
        self.rlon = prepare_dim(rlon)

        for key, type_ in self.specs(types=True):
            try:
                val = kwargs.pop(key)
            except KeyError:
                raise ValueError(f"missing argument '{key}'")
            try:
                setattr(self, key, type_(val))
            except TypeError:
                raise ValueError(
                    f"argument '{key}': type '{type(val).__name__}' "
                    f"incompatible with '{type_.__name__}'")
        if kwargs:
            raise ValueError(
                f"{len(kwargs)} unexpected arguments: {sorted(kwargs)}")

    @classmethod
    def multiple(cls, *args, **kwargs):
        """Create multiple instances of ``FlexVarSpecs``.

        Each of the arguments of ``__init__`` can be passed by the
        original name with one value (e.g., ``time=1``) or
        pluralized with multiple values (e.g., ``time=[1, 2]``).

        One ``FlexVarSpecs`` instance is created for each combination
        of all input arguments.

        """
        return cls._multiple_as_type(cls, *args, **kwargs)

    @classmethod
    def multiple_as_dict(cls, *args, **kwargs):
        return cls._multiple_as_type(dict, *args, **kwargs)

    @classmethod
    def _multiple_as_type(
            cls, type_, rlat=slice(None), rlon=slice(None), **kwargs):
        keys_singular = sorted(cls.specs())
        vals_plural = []
        for key_singular in keys_singular:
            key_plural = f'{key_singular}_lst'

            if key_plural in kwargs:
                # Passed as plural
                if key_singular in kwargs:
                    # Error: passed as both plural and sigular
                    raise ValueError(
                        f"argument conflict: '{key_singular}', '{key_plural}'")
                vals_plural.append([v for v in kwargs.pop(key_plural)])

            elif key_singular in kwargs:
                # Passed as sigular
                vals_plural.append([kwargs.pop(key_singular)])

            else:
                # Not passed at all
                raise ValueError(
                    f"missing argument: '{key_singular}' or '{key_plural}'")

        if kwargs:
            # Passed too many arguments
            raise ValueError(
                f"{len(kwargs)} unexpected arguments: {sorted(kwargs)}")

        # Create one specs per parameter combination
        specs_lst = []
        for vals in itertools.product(*vals_plural):
            kwargs_i = {k: v for k, v in zip(keys_singular, vals)}
            specs = type_(rlat=rlat, rlon=rlon, **kwargs_i)
            specs_lst.append(specs)

        return specs_lst

    def merge_with(self, others):
        attrs = {}
        for key, val0 in sorted(self):

            vals = [val0]
            for other in others:
                val = getattr(other, key)
                if val not in vals:
                    vals.append(val)

            if len(vals) == 1:
                attrs[key] = next(iter(vals))
            elif key == 'deposition' and set(vals) == set(['dry', 'wet']):
                attrs[key] = 'tot'
            else:
                attrs[key] = vals

        return self.__class__(**attrs)

    def __hash__(self):
        h, f = 0, 1
        for key, val in sorted(iter(self)):
            try:
                h += int(val)*f
            except (TypeError, ValueError):
                continue
            else:
                f *= 10
        return h

    def __lt__(self, other):
        return hash(self) < hash(other)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __repr__(self):
        return pformat_dictlike(self)

    def __str__(self):
        return pformat_dictlike(self)

    def __getitem__(self, key):
        if key.startswith('_'):
            raise ValueError(f"invalid key '{key}'")
        try:
            return self.__dict__[key]
        except KeyError as e:
            raise e from None

    def __iter__(self):
        for key, val in sorted(self.__dict__.items()):
            if not key.startswith('_'):
                yield key, val

    def var_name(self):
        """Derive variable name from specifications."""
        raise NotImplementedError(f'{self.__class__.__name__}.var_name')

    def dim_inds_by_name(self, *, rlat=slice(None), rlon=slice(None)):
        """Derive indices along NetCDF dimensions."""

        inds = {}

        inds['nageclass'] = self.nageclass
        inds['numpoint'] = self.numpoint

        if not self.integrate or self.time == slice(None):
            inds['time'] = self.time
        else:
            inds['time'] = slice(None, self.time + 1)

        inds['rlat'] = slice(*self.rlat)
        inds['rlon'] = slice(*self.rlon)

        return inds


class FlexVarSpecsConcentration(FlexVarSpecs):

    _keys_w_type = {
        **FlexVarSpecs._keys_w_type,
        'level': int_or_list,
    }

    def var_name(self):
        """Derive variable name from specifications."""

        def fmt(sid):
            return f'spec{sid:03d}'

        try:
            iter(self.species_id)
        except TypeError:
            return fmt(self.species_id)
        else:
            return [fmt(sid) for sid in self.species_id]

    def dim_inds_by_name(self, *args, **kwargs):
        """Derive indices along NetCDF dimensions."""
        inds = super().dim_inds_by_name(*args, **kwargs)
        inds['level'] = self.level
        return inds


class FlexVarSpecsDeposition(FlexVarSpecs):

    _keys_w_type = {
        **FlexVarSpecs._keys_w_type,
        'deposition': str,
    }

    def var_name(self):
        """Derive variable name from specifications."""
        prefix = {'wet': 'WD', 'dry': 'DD'}[self.deposition]
        return f'{prefix}_spec{self.species_id:03d}'


class FlexVarSpecsAffectedArea(FlexVarSpecsDeposition):

    pass


class FlexFieldSpecs:
    """FLEXPART field specifications."""

    cls_var_specs = FlexVarSpecs

    # Dimensions with optionally multiple values
    dims_opt_mult_vals = ['species_id']

    def __init__(
            self, var_specs_lst, op=np.nansum, *, var_attrs_replace=None,
            lang='en'):
        """Create an instance of ``FlexFieldSpecs``.

        Args:
            var_specs_lst (list[dict]): Specifications dicts of input
                variables, each of which is is used to create an
                instance of ``FlexVarSpecs`` as specified by the class
                attribute ``cls_var_specs``. Each ultimately yields a
                2D slice of an input variable.

            op (function or list[function], optional): Opterator(s) to
                combine the input fields obtained based on the input
                variable specifications. If multipe operators are
                passed, their number must one smaller than that of the
                specifications, and they are applied consecutively from
                left to right without regard to operator precedence.
                Must accept argument ``axis=0`` to only recude along
                over the fields. Defaults to np.nansum.

            var_attrs_replace (dict[str: dict], optional): Variable
                attributes to be replaced. Necessary if multiple
                specifications dicts are passed for all those
                attributes that differ between the resulting attributes
                collections. Defaults to '{}'.

            lang (str, optional): Language, e.g., 'de' for German.
                Defaults to 'en' (English).
        """

        self._prepare_var_specs_lst(var_specs_lst)

        # Create variable specifications objects
        self.var_specs_lst = self.create_var_specs(var_specs_lst)

        # Store operator(s)
        self.check_op(op)
        if callable(op):
            self.op = op
            self.op_lst = None
        else:
            self.op = None
            self.op_lst = op

        # Store variable attributes
        if var_attrs_replace is None:
            var_attrs_replace = {}
        self.var_attrs_replace = var_attrs_replace

    def _prepare_var_specs_lst(self, var_specs_lst):

        # Handle dimensions with optionally multiple values
        # Example: Sum over multiple species
        for key in self.dims_opt_mult_vals:
            for var_specs in copy(var_specs_lst):
                try:
                    iter(var_specs[key])
                except TypeError:
                    pass
                else:
                    vals = copy(var_specs[key])
                    var_specs[key] = vals.pop(0)
                    var_specs_lst_new = [deepcopy(var_specs) for _ in vals]
                    for var_specs_new, val in zip(var_specs_lst_new, vals):
                        var_specs_new[key] = val
                        var_specs_lst.append(var_specs_new)

    def create_var_specs(self, var_specs_dct_lst):
        """Create variable specifications objects from dicts."""
        try:
            iter(var_specs_dct_lst)
        except TypeError:
            raise ValueError(
                f"var_specs: type '{type(var_specs_dct_lst).__name__}' "
                f"not iterable") from None

        var_specs_lst = []
        for i, kwargs_specs in enumerate(var_specs_dct_lst):
            try:
                var_specs = self.cls_var_specs(**kwargs_specs)
            except Exception as e:
                raise ValueError(
                    f"var_specs[{i}]: cannot create instance of "
                    f"{self.cls_var_specs.__name__} from {kwargs_specs}: "
                    f"{e.__class__.__name__}({e})") from None
            else:
                var_specs_lst.append(var_specs)

        return var_specs_lst

    def check_op(self, op):
        """Check operator(s)."""
        try:
            n_ops = len(op)
        except TypeError:
            if not callable(op):
                raise ValueError(f"op: {type(op).__name__} not callable")
            return

        n_var_specs = len(self.var_specs_lst)
        if n_ops != n_var_specs - 1:
            raise ValueError(
                f"wrong number of operators passed in "
                f"{type(ops).__name__}: {n_ops} != {n_var_specs}")
        for op_i in op:
            if not callable(op_i):
                raise ValueError(f"op: {type(op_i).__name__} not callable")

    def __repr__(self):
        s = f"{self.__class__.__name__}(\n"

        # Variables specifications
        s += f"  var_specs: {len(self.var_specs_lst)}x\n"
        for var_specs in self.var_specs_lst:
            for line in str(var_specs).split('\n'):
                s += f"    {line}\n"

        # Operator(s)
        if self.op is not None:
            s += f"  op: {self.op.__name__}\n"
        else:
            s += f"  ops: {len(self.op_lst)}x\n"
            for op in self.op_lst:
                s += "f    {op.__name__}\n"

        # Variable attributes replacements
        s += f"  var_attrs_replace: {len(self.var_attrs_replace)}x\n"
        for key, val in sorted(self.var_attrs_replace.items()):
            s += f"    '{key}': {val}\n"

        s += f")"
        return s

    def __hash__(self):
        # yapf: disable
        return sum([
            sum([hash(vs) for vs in self.var_specs_lst]),
        ])
        # yapf: enable

    def __lt__(self, other):
        return hash(self) < hash(other)

    def __eq__(self, other):
        return hash(self) == hash(other)

    @classmethod
    def multiple(cls, vars_specs, lang='en'):
        var_specs_lst = cls.cls_var_specs.multiple_as_dict(**vars_specs)
        field_specs_lst = []
        for var_specs in var_specs_lst:
            try:
                field_specs = cls(var_specs, lang=lang)
            except Exception as e:
                raise Exception(
                    f"cannot initialize {cls.__name__} "
                    f"({type(e).__name__}: {e})"
                    f"\nvar_specs: {var_specs}")
            else:
                field_specs_lst.append(field_specs)
        return field_specs_lst

    def var_specs_merged(self):
        """Return merged variable specifications."""
        return self.var_specs_lst[0].merge_with(self.var_specs_lst[1:])

    def var_specs_shared(self, key):
        """Return a varible specification, if it is shared by all."""
        vals = [getattr(vs, key) for vs in self.var_specs_lst]
        all_equal = all(v == vals[0] for v in vals[1:])
        if not all_equal:
            raise ValueError(
                f"'{key}' differs among {len(self.var_specs_lst)} var stats: "
                f"{vals}")
        return next(iter(vals))


class FlexFieldSpecsConcentration(FlexFieldSpecs):

    cls_var_specs = FlexVarSpecsConcentration

    # Dimensions with optionally multiple values
    dims_opt_mult_vals = FlexFieldSpecs.dims_opt_mult_vals + ['level']

    def __init__(self, var_specs, **kwargs):
        """Create an instance of ``FlexFieldSpecsConcentration``.

        Args:
            var_specs (dict): Specifications dict of input variable
                used to create an instance of ``FlexVarSpecsConcentration``
                as specified by the class attribute ``cls_var_specs``.

            **kwargs: Keyword arguments passed to ``FlexFieldSpecs``.
        """
        if not isinstance(var_specs, dict):
            raise ValueError(
                f"var_specs must be 'dict', not '{type(var_specs).__name__}'")
        super().__init__([var_specs], **kwargs)


class FlexFieldSpecsDeposition(FlexFieldSpecs):

    cls_var_specs = FlexVarSpecsDeposition

    def __init__(self, var_specs, lang='en', **kwargs):
        """Create an instance of ``FlexFieldSpecsDeposition``.

        Args:
            var_specs (dict): Specifications dict of input variable
                used to create instance(s) of ``FlexVarSpecsDeposition``
                as specified by the class attribute ``cls_var_specs``.

            lang (str, optional): Language, e.g., 'de' for German.
                Defaults to 'en' (English).

            **kwargs: Keyword arguments passed to ``FlexFieldSpecs``.
        """
        var_specs_lst = [dict(var_specs)]

        # Deposition mode
        for var_specs in copy(var_specs_lst):

            if var_specs['deposition'] in ['wet', 'dry']:
                pass

            elif var_specs['deposition'] == 'tot':
                nested_dict_set(
                    kwargs,
                    ['var_attrs_replace', 'variable', 'long_name', 'value'],
                    FlexAttrsCollector.get_long_name(
                        var_specs,
                        type_=self.cls_var_specs,
                        lang=lang,
                    ),
                )
                var_specs_new = deepcopy(var_specs)
                var_specs['deposition'] = 'wet'
                var_specs_new['deposition'] = 'dry'
                var_specs_lst.append(var_specs_new)

            else:
                raise NotImplementedError(
                    f"deposition type '{var_specs['deposition']}'")

        super().__init__(var_specs_lst, **kwargs)


class FlexFieldSpecsAffectedArea(FlexFieldSpecsDeposition):

    cls_var_specs = FlexVarSpecsAffectedArea


class FlexFileRotPole:
    """NetCDF file containing FLEXPART data on rotated-pole grid."""

    def __init__(self, path, cmd_open=nc4.Dataset):
        """Create an instance of ``FlexFileRotPole``.

        Args:
            path (str): File path.

            cmd_open (function, optional): Function to open the input
                file. Must support context manager interface, i.e.,
                ``with cmd_open(path, mode) as f:``. Defaults to
                netCDF4.Dataset.

        """
        self.path = path
        self.cmd_open = cmd_open

        self.reset()

    def reset(self):
        self._fi = None
        self._var_specs_curr = None

    def read(self, fld_specs, lang='en'):
        """Read one or more fields from a file from disc.

        Args:
            fld_specs (FlexFieldSpecs or list[FlexFieldSpecs]):
                Specifications for one or more input fields.

        Returns:
            FlexFieldRotPole: Single data object; if ``fld_specs``
                constitutes a single ``FlexFieldSpecs`` instance.

            or

            list[FlexFieldRotPole]: One data object for each field;
                if ``fld_specs`` constitutes a list of ``FlexFieldSpecs``
                instances.

        """
        if isinstance(fld_specs, FlexFieldSpecs):
            multiple = False
            fld_specs_lst = [fld_specs]
        else:
            multiple = True
            fld_specs_lst = fld_specs
        del fld_specs
        n_fld_specs = len(fld_specs_lst)

        # Group field specifications objects such that all in one group
        # only differ in time; collect the respective time indices; and
        # merge the group into one field specifications instance with
        # time dimension 'slice(None)'. This allows for each such group
        # to first read and process all time steps (e.g., derive some
        # statistics across all time steps), and subsequently extract
        # the requested time steps using the separately stored indices.
        fld_specs_time_lst, time_inds_lst = (
            self.group_fld_specs_by_time(fld_specs_lst))
        n_fst = len(fld_specs_time_lst)

        log.debug(f"read {self.path}")
        flex_data_lst = []
        with self.cmd_open(self.path, 'r') as self._fi:

            log.debug(f"process {n_fst} field specs groups")
            for i_fst, (fld_specs_time,
                        time_inds) in enumerate(zip(fld_specs_time_lst,
                                                    time_inds_lst)):
                log.debug(f"{i_fst + 1}/{n_fst}: {fld_specs_time}")
                n_t = len(time_inds)

                fld_time = self._import_field(fld_specs_time)
                time_stats = self.collect_time_stats(fld_time)

                log.debug(f"extract {n_fst} time steps")
                for i_t, time_ind in enumerate(time_inds):
                    log.debug(f"{i_t + 1}/{n_t}")

                    # Create time-step-specific field specifications
                    fld_specs = deepcopy(fld_specs_time)
                    for var_specs in fld_specs.var_specs_lst:
                        var_specs.time = time_ind

                    # Extract field
                    fld = fld_time[time_ind]

                    # Collect attributes
                    log.debug("collect attributes")
                    attrs_lst = []
                    for var_specs in fld_specs.var_specs_lst:
                        attrs = FlexAttrsCollector(
                            self._fi,
                            var_specs,
                        ).run(lang=lang)
                        attrs_lst.append(attrs)
                    attrs = attrs_lst[0].merge_with(
                        attrs_lst[1:], **fld_specs.var_attrs_replace)

                    #SR_TMP<
                    log.debug("fix nc data")
                    if i_t == 0:
                        # Scale time_stats only once
                        self._fix_nc_data(fld, attrs, time_stats)
                    else:
                        self._fix_nc_data(fld, attrs)
                    #SR_TMP>

                    # Read grid variables
                    inds_rlat = slice(*fld_specs.var_specs_shared('rlat'))
                    inds_rlon = slice(*fld_specs.var_specs_shared('rlon'))
                    rlat = self._fi.variables['rlat'][inds_rlat]
                    rlon = self._fi.variables['rlon'][inds_rlon]

                    # Collect data
                    log.debug("create data object")
                    flex_data_lst.append(
                        FlexFieldRotPole(
                            rlat, rlon, fld, attrs, fld_specs, time_stats))

        self.reset()

        # Return result(s)
        if multiple:
            return flex_data_lst
        else:
            assert len(flex_data_lst) == 1
            return flex_data_lst[0]

    def group_fld_specs_by_time(self, fld_specs_lst):
        """Group specs that differ only in their time dimension."""

        fld_specs_time_inds_by_hash = {}
        for fld_specs in fld_specs_lst:
            fld_specs_time = deepcopy(fld_specs)

            # Extract time index and check its the same for all
            time_ind = None
            for var_specs in fld_specs_time.var_specs_lst:
                if time_ind is None:
                    time_ind = var_specs.time
                elif var_specs.time != time_ind:
                    raise Exception(
                        f"{var_specs.__class__.__name__} instances of "
                        f"{fld_spec_time.__class__.__name__} instance "
                        f"differ in 'time' ({var_specs.time} != {time_ind}):"
                        f"\n{fld_specs_time}")
                var_specs.time = slice(None)

            # Store time-neutral fld specs alongside resp. time inds
            key = hash(fld_specs_time)
            if key not in fld_specs_time_inds_by_hash:
                fld_specs_time_inds_by_hash[key] = (fld_specs_time, [])
            if time_ind in fld_specs_time_inds_by_hash[key][1]:
                raise Exception(
                    f"duplicate time index {time_ind} in {fld_specs}")
            fld_specs_time_inds_by_hash[key][1].append(time_ind)

        # Regroup time-neutral fld specs and time inds into lists
        fld_specs_time_lst, time_inds_lst = [], []
        for _, (fld_specs_time,
                time_inds) in sorted(fld_specs_time_inds_by_hash.items()):
            fld_specs_time_lst.append(fld_specs_time)
            time_inds_lst.append(time_inds)

        return fld_specs_time_lst, time_inds_lst

    def collect_time_stats(self, fld_time):
        stats = {
            'mean': np.nanmean(fld_time),
            'median': np.nanmedian(fld_time),
            'mean_nz': np.nanmean(fld_time[fld_time > 0]),
            'median_nz': np.nanmedian(fld_time[fld_time > 0]),
            'max': np.nanmax(fld_time),
        }
        return stats

    def _import_field(self, fld_specs):

        # Read fields and attributes from var specifications
        fld_lst = []
        for var_specs in fld_specs.var_specs_lst:

            # Field
            log.debug("read field")
            fld = self._read_var(var_specs)

            fld_lst.append(fld)

        # Merge fields
        fld = self._merge_fields(fld_lst, fld_specs)

        return fld

    def _read_var(self, var_specs):

        # Select variable in file
        var_name = var_specs.var_name()
        var = self._fi.variables[var_name]

        # Indices of field along NetCDF dimensions
        dim_inds_by_name = var_specs.dim_inds_by_name()

        # Assemble indices for slicing
        inds = [None]*len(var.dimensions)
        for dim_name, dim_ind in dim_inds_by_name.items():
            ind = var.dimensions.index(dim_name)
            inds[ind] = dim_ind
        if None in inds:
            raise Exception(
                f"variable '{var_name}': could not resolve all indices!"
                f"\ndim_inds   : {dim_inds_by_name}"
                f"\ndimensions : {var.dimensions}"
                f"\ninds       : {inds}")
        inds = inds
        log.debug(f"indices: {inds}")
        check_array_indices(var.shape, inds)

        # Read field
        log.debug(f"shape: {var.shape}")
        log.debug(f"indices: {inds}")
        fld = var[inds]

        if isinstance(var_specs, FlexVarSpecsConcentration):
            if var_specs.integrate:
                # Integrate over time
                fld = np.cumsum(fld, axis=0)
        elif isinstance(var_specs, FlexVarSpecsDeposition):
            if not var_specs.integrate:
                # Revert integration over time
                fld[1:] -= fld[:-1].copy()
        else:
            raise NotImplementedError(
                f"var_specs of type '{type(var_specs).__name__}'")

        return fld

    def _merge_fields(self, fld_lst, fld_specs):

        if fld_specs.op is not None:
            # Single operator
            return fld_specs.op(fld_lst, axis=0)

        # Operator chain
        fld = fld_lst[0]
        for i, fld_i in enumerate(fld_list[1:]):
            _op = fld_specs.op_lst[i]
            fld = _op([fld, fld_i], axis=0)
        return fld

    #SR_TMP<<<
    def _fix_nc_data(self, fld, attrs, time_stats=None):

        def scale_time_stats(fact):
            if time_stats is not None:
                for key, val in time_stats.items():
                    time_stats[key] = val*fact

        #SR_TMP< TODO more general solution to combined species
        names = [
            'Cs-137',
            'I-131a',
            ['Cs-137', 'I-131a'],
            ['I-131a', 'Cs-137'],
        ]
        #SR_TMP>
        #-if attrs.species.name in names:  #SR_ATTR
        if attrs.species.name.value in names:  #SR_ATTR

            #-if attrs.variable.unit == 'ng kg-1':  #SR_ATTR
            if attrs.variable.unit.value == 'ng kg-1':  #SR_ATTR
                #-attrs.variable.unit = 'Bq m-3'  #SR_ATTR
                attrs.variable.unit.value = 'Bq m-3'  #SR_ATTR
                fld[:] *= 1e-12
                scale_time_stats(1e-12)

            #-elif attrs.variable.unit == '1e-12 kg m-2':  #SR_ATTR
            elif attrs.variable.unit.value == '1e-12 kg m-2':  #SR_ATTR
                #-attrs.variable.unit = 'Bq m-2'  #SR_ATTR
                attrs.variable.unit.value = 'Bq m-2'  #SR_ATTR
                fld[:] *= 1e-12
                scale_time_stats(1e-12)

            else:
                raise NotImplementedError(
                    #-f"species '{attrs.species.name}': "  #SR_ATTR
                    #-f"unknown unit '{attrs.variable.unit}'")  #SR_ATTR
                    f"species '{attrs.species.name.value}': "  #SR_ATTR
                    f"unknown unit '{attrs.variable.unit.value}'")  #SR_ATTR
        else:
            #-raise NotImplementedError(f"species '{attrs.species.name}'")  #SR_ATTR
            raise NotImplementedError(
                f"species '{attrs.species.name.value}'")  #SR_ATTR


class FlexAttrsCollector:
    """Collect attributes for a field from an open NetCDF file."""

    def __init__(self, fi, var_specs):
        """Create an instance of ``FlexAttrsCollector``.

        Args:
            fi (netCDF4.Dataset): An open FLEXPART NetCDF file.

            var_specs (FlexVarSpecs): Input field specifications.

        """
        self.fi = fi
        self.var_specs = var_specs

        # Collect all global attributes
        self.ncattrs_global = {
            attr: self.fi.getncattr(attr) for attr in self.fi.ncattrs()
        }

        # Collect all variables attributes
        self.ncattrs_vars = {}
        for var in self.fi.variables.values():
            self.ncattrs_vars[var.name] = {
                attr: var.getncattr(attr) for attr in var.ncattrs()
            }

        # Select attributes of field variable
        self.field_var_name = self.var_specs.var_name()
        self.ncattrs_field = self.ncattrs_vars[self.field_var_name]

    def run(self, lang='en'):
        """Collect attributes."""

        self.lang = lang

        attrs_raw = {
            'grid': self.collect_grid_attrs(),
            'release': self.collect_release_attrs(),
            'variable': self.collect_variable_attrs(),
            'species': self.collect_species_attrs(),
            'simulation': self.collect_simulation_attrs(),
        }

        return FlexAttrGroupCollection(lang=lang, **attrs_raw)

    def collect_grid_attrs(self):
        """Collect grid attributes."""

        np_lat = self.ncattrs_vars['rotated_pole']['grid_north_pole_latitude']
        np_lon = self.ncattrs_vars['rotated_pole']['grid_north_pole_longitude']

        return {
            'north_pole_lat': np_lat,
            'north_pole_lon': np_lon,
        }

    def collect_release_attrs(self):
        """Collect release point attributes."""

        # Collect release point information
        _ind = self.var_specs.numpoint
        release_point = ReleasePoint.from_file(self.fi, _ind)

        lat = np.mean([release_point.lllat, release_point.urlat])
        lon = np.mean([release_point.lllon, release_point.urlon])
        site_name = release_point.name

        height = np.mean([release_point.zbot, release_point.ztop])
        height_unit = 'm AGL'  #SR_HC

        assert len(release_point.ms_parts) == 1
        mass = next(iter(release_point.ms_parts))
        mass_unit = 'Bq'  #SR_HC

        duration = release_point.end - release_point.start
        duration_unit = 's'  #SR_HC

        rate = mass/duration
        rate_unit = f'{mass_unit} {duration_unit}-1'

        return {
            'lat': lat,
            'lon': lon,
            'site_name': site_name,
            'height': height,
            'height_unit': height_unit,
            'rate': rate,
            'rate_unit': rate_unit,
            'mass': mass,
            'mass_unit': mass_unit,
        }

    def collect_variable_attrs(self):
        """Collect variable attributes."""

        # Variable name
        if isinstance(self.var_specs, FlexVarSpecsConcentration):
            long_name = self.get_long_name(self.var_specs, lang=self.lang)
            short_name = {
                'en': 'Concentration',  #SR_HC
                'de': 'Konzentration',  #SR_HC
            }[self.lang]
        elif isinstance(self.var_specs, FlexVarSpecsDeposition):
            long_name = self.get_long_name(self.var_specs, lang=self.lang)
            short_name = {
                'en': f'Deposition',  #SR_HC
                'de': f'Ablagerung',  #SR_HC
            }[self.lang]
        else:
            raise NotImplementedError(
                f"var_specs of type {type(var_specs).__name__}")

        try:
            _i = self.var_specs.level
        except AttributeError:
            #SR_TMP<
            level_unit = ''
            level_bot = -1
            level_top = -1
            #SR_TMP>
        else:
            level_unit = 'm AGL'  #SR_HC
            _var = self.fi.variables['level']
            level_bot = 0.0 if _i == 0 else float(_var[_i - 1])
            level_top = float(_var[_i])

        return {
            'long_name': long_name,
            'short_name': short_name,
            'unit': self.ncattrs_field['units'],
            'level_bot': level_bot,
            'level_bot_unit': level_unit,
            'level_top': level_top,
            'level_top_unit': level_unit,
        }

    #SR_HC<<<
    @staticmethod
    def get_long_name(var_specs, *, type_=None, lang='en'):
        """Return long variable name."""

        type_base = FlexVarSpecs
        if type_ is None:
            type_ = type(var_specs)
        if not issubclass(type_, type_base):
            raise ValueError(
                f"var_specs: invalid type {type_}: "
                f"not a subclass of {type_base.__name__}!")

        if type_ is FlexVarSpecsConcentration:
            return {
                'en': 'Activity Concentration',
                'de': r'Aktivit$\mathrm{\"a}$tskonzentration',
            }[lang]

        else:
            dep = dict(var_specs)['deposition']

            if lang == 'en':
                dep_type = {
                    'wet': 'Wet',
                    'dry': 'Dry',
                    'tot': 'Total',
                }[dep]

            elif lang == 'de':
                dep_type = {
                    'wet': 'Nass',
                    'dry': 'Trocken',
                    'tot': 'Total',
                }[dep]

            if type_ is FlexVarSpecsDeposition:
                return {
                    'en': f'{dep_type} Surface Deposition',
                    'de': f'{dep_type}e Bodenablagerung',
                }[lang]

            elif type_ is FlexVarSpecsAffectedArea:
                return {
                    'en': f'Affected Area ({dep_type})',
                    #'en': f'Affected Area ({dep_type} Deposition)',
                    'de': f'Beaufschlagtes Gebiet ({dep_type})',
                    #'de': f'Beaufschl. Gebiet ({dep_type}e Ablagerung)',
                }[lang]

        raise NotImplementedError(f"var_specs of type '{type_.__name__}'")

    def collect_species_attrs(self):
        """Collect species attributes."""

        substance = self._get_substance()

        # Get deposition and washout data
        if isinstance(self.var_specs, FlexVarSpecsConcentration):
            name_core = self.field_var_name
        elif isinstance(self.var_specs, FlexVarSpecsDeposition):
            name_core = self.field_var_name[3:]
        deposit_vel = self.ncattrs_vars[f'DD_{name_core}']['dryvel']
        washout_coeff = self.ncattrs_vars[f'WD_{name_core}']['weta']
        washout_exponent = self.ncattrs_vars[f'WD_{name_core}']['wetb']

        # Get half life information
        try:
            half_life, half_life_unit = {
                'Cs-137': (30.17, 'y'),  #SR_HC
                'I-131a': (8.02, 'd'),  #SR_HC
            }[substance]
        except KeyError:
            raise NotImplementedError(f"half_life of '{substance}'")

        deposit_vel_unit = 'm s-1'  #SR_HC
        sediment_vel_unit = 'm s-1'  #SR_HC
        washout_coeff_unit = 's-1'  #SR_HC

        return {
            'name': substance,
            'half_life': half_life,
            'half_life_unit': half_life_unit,
            'deposit_vel': deposit_vel,
            'deposit_vel_unit': deposit_vel_unit,
            'sediment_vel': 0.0,
            'sediment_vel_unit': sediment_vel_unit,
            'washout_coeff': washout_coeff,
            'washout_coeff_unit': washout_coeff_unit,
            'washout_exponent': washout_exponent,
        }

    def _get_substance(self):
        substance = self.ncattrs_field['long_name']
        if isinstance(self.var_specs, FlexVarSpecsDeposition):
            substance = substance.replace(
                f'_{self.var_specs.deposition}_deposition', '')  #SR_HC
        return substance

    def collect_simulation_attrs(self):
        """Collect simulation attributes."""

        # Start and end timesteps of simulation
        _ga = self.ncattrs_global
        ts_start = datetime.datetime(
            *time.strptime(_ga['ibdate'] + _ga['ibtime'], '%Y%m%d%H%M%S')[:6],
            tzinfo=datetime.timezone.utc,
        )
        ts_end = datetime.datetime(
            *time.strptime(_ga['iedate'] + _ga['ietime'], '%Y%m%d%H%M%S')[:6],
            tzinfo=datetime.timezone.utc,
        )

        model_name = 'COSMO-1'  #SR_HC

        ts_now, ts_integr_start = self._get_current_timestep_etc()

        return {
            'model_name': model_name,
            'start': ts_start,
            'end': ts_end,
            'now': ts_now,
            'integr_start': ts_integr_start,
        }

    def _get_current_timestep_etc(self, i=None):
        """Get the current timestep, or a specific one by index."""

        if i is None:
            # Default to timestep of current field
            i = self.var_specs.time

        if not isinstance(i, int):
            raise Exception(
                f"expect type 'int', not '{type(i).__name__}': {i}")

        var = self.fi.variables['time']

        # Obtain start from time unit
        rx = re.compile(
            r'seconds since '
            r'(?P<yyyy>[12][0-9][0-9][0-9])-'
            r'(?P<mm>[01][0-9])-'
            r'(?P<dd>[0-3][0-9]) '
            r'(?P<HH>[0-2][0-9]):'
            r'(?P<MM>[0-6][0-9])')
        match = rx.match(var.units)
        if not match:
            raise Exception(f"cannot extract start from units '{var.units}'")
        start = datetime.datetime(
            int(match['yyyy']),
            int(match['mm']),
            int(match['dd']),
            int(match['HH']),
            int(match['MM']),
            tzinfo=datetime.timezone.utc,
        )

        # Determine time since start
        delta_tot = datetime.timedelta(seconds=int(var[i]))

        # Determine current timestep
        now = start + delta_tot

        # Determine start timestep of integration period
        if self.var_specs.integrate:
            delta_integr = delta_tot
        else:
            delta_prev = delta_tot/(i + 1)
            delta_integr = delta_prev
        ts_integr_start = now - delta_integr

        return now, ts_integr_start


class ReleasePoint:
    """Release point information."""

    # Define the init arguments with target types
    # Done here to avoid duplication
    attr_types = {
        'name': str,
        'age_id': int,
        'kind': str,
        'lllat': float,
        'lllon': float,
        'urlat': float,
        'urlon': float,
        'zbot': float,
        'ztop': float,
        'start': int,
        'end': int,
        'n_parts': int,
        'ms_parts': list,
    }

    def __init__(self, **kwargs):
        """Create an instance of ``ReleasePoint``."""

        # Keep track of passed keyword arguments
        attr_keys_todo = [k for k in self.attr_types]

        # Process keyword arguments
        for key, val in kwargs.items():

            # Check existence of argument while fetching target type
            try:
                type_ = self.attr_types[key]
            except KeyError:
                raise ValueError(f"unexpected argument {key}='{val}'")

            # Cross valid argument off the todo list
            attr_keys_todo.remove(key)

            # Check type while converting to target type
            try:
                val = type_(val)
            except TypeError:
                raise ValueError(
                    f"argument {key}={val} of type '{type(val).__name__}' "
                    f"incompatible with expected type '{type_.__name__}'")

            # Set as attribute
            setattr(self, key, val)

        # Check that all arguments have been passed
        if attr_keys_todo:
            n = len(attr_keys_todo)
            raise ValueError(
                f"missing {n} argument{'' if n == 1 else 's'}: "
                f"{attr_keys_todo}")

    def __repr__(self):
        return pformat_dictlike(self)

    def __str__(self):
        return pformat_dictlike(self)

    def __iter__(self):
        for key in self.attr_types:
            yield key, getattr(self, key)

    @classmethod
    def from_file(cls, fi, i=None, var_name='RELCOM'):
        """Read information on single release point from open file.

        Args:
            fi (netCDF4.Dataset): Open NetCDF file handle.

            i (int, optional): Release point index. Mandatory if ``fi``
                contains multiple release points. Defaults to None.

            var_name (str, optional): Variable name of release point.
                Defaults to 'RELCOM'.

        Returns:
            ReleasePoint: Release point object.

        """
        var = fi.variables[var_name]

        # Check index against no. release point and set it if necessary
        n = var.shape[0]
        if n == 0:
            raise ValueError(
                f"file '{fi.name}': no release points ('{var_name}')")
        elif n == 1:
            if i is None:
                i = 0
        elif n > 1:
            if i is None:
                raise ValueError(
                    f"file '{fi.name}': i is None despite {n} release points")
        if i < 0 or i >= n:
            raise ValueError(
                f"file '{fi.name}': invalid index {i} for {n} release points")

        kwargs = {}

        # Name -- byte character array
        kwargs['name'] = (
            var[i][~var[i].mask].tostring().decode('utf-8').rstrip())

        # Other attributes
        key_pairs = [
            ('age_id', 'LAGE'),
            ('kind', 'RELKINDZ'),
            ('lllat', 'RELLAT1'),
            ('lllon', 'RELLNG1'),
            ('urlat', 'RELLAT2'),
            ('urlon', 'RELLNG2'),
            ('zbot', 'RELZZ1'),
            ('ztop', 'RELZZ2'),
            ('start', 'RELSTART'),
            ('end', 'RELEND'),
            ('n_parts', 'RELPART'),
            ('ms_parts', 'RELXMASS'),
        ]
        for key_out, key_in in key_pairs:
            kwargs[key_out] = fi.variables[key_in][i].tolist()

        return cls(**kwargs)

    @classmethod
    def from_file_multi(cls, fi, var_name='RELCOM'):
        """Read information on multiple release points from open file.

        Args:
            fi (netCDF4.Dataset): Open NetCDF file handle.

            var_name (str, optional): Variable name of release point.
                Defaults to 'RELCOM'.

        Returns:
            list[ReleasePoint]: List of release points objects.

        """
        n = fi.variables[var_name].shape[0]
        return [cls.from_file(fi, i, var_name) for i in range(n)]
