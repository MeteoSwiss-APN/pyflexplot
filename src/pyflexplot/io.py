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

from .data import FlexAttrsCollection
from .data import FlexFieldRotPole
from .utils import check_array_indices
from .utils import pformat_dictlike

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


class FlexVarSpecs:
    """FLEXPART input variable specifications."""

    # Keys with respective type
    _keys_w_type = {
        'species_id': int,
        'integrate': bool,
        # Dimensions
        'time': int,
        'nageclass': int,
        'numpoint': int,
    }

    @classmethod
    def specs(cls, types=False):
        for k, v in cls._keys_w_type.items():
            if types:
                yield k, v
            else:
                yield k

    def __init__(self, **kwargs):
        """Create an instance of ``FlexVarSpecs``.

        Args:
            **kwargs: Arguments as in ``FlexVarSpecs.specs(types=True)``.
                The keys correspond to the argument's names, and the
                values specify a type which the respective argument
                value must be compatible with.

        """
        for key, type_ in self.specs(types=True):
            try:
                val = kwargs.pop(key)
            except KeyError:
                raise ValueError(f"missing argument '{key}'") from None
            try:
                setattr(self, key, type_(val))
            except TypeError:
                raise ValueError(
                    f"argument '{key}': type '{type(val).__name__}' "
                    f"incompatible with '{type_.__name__}'") from None
        if kwargs:
            raise ValueError(
                f"{len(kwargs)} unexpected arguments: {sorted(kwargs)}")

    @classmethod
    def multiple(cls, **kwargs):
        """Create multiple instances of ``FlexVarSpecs``.

        Each of the arguments of ``__init__`` can be passed by the
        original name with one value (e.g., ``time=1``) or
        pluralized with multiple values (e.g., ``time=[1, 2]``).

        One ``FlexVarSpecs`` instance is created for each combination
        of all input arguments.

        """
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
            specs_lst.append(cls(**kwargs_i))

        return specs_lst

    def merge(self, others):
        attrs = {}
        for key, val0 in sorted(self):
            vals = set([val0] + [getattr(o, key) for o in others])
            if len(vals) == 1:
                attrs[key] = next(iter(vals))
            else:
                if key == 'deposition' and vals == set(['dry', 'wet']):
                    attrs[key] = 'tot'
                else:
                    raise NotImplementedError(
                        f"{self.__class__.__name__}.merge for '{key}'")
        return self.__class__(**attrs)

    def __hash__(self):
        return hash(tuple(sorted(iter(self))))

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
        for key, val in self.__dict__.items():
            if not key.startswith('_'):
                yield key, val

    def var_name(self):
        """Derive variable name from specifications."""
        raise NotImplementedError(f'{self.__class__.__name__}.var_name')

    def dim_inds_by_name(self):
        """Derive indices along NetCDF dimensions."""

        inds = {}

        inds['nageclass'] = self.nageclass
        inds['numpoint'] = self.numpoint

        if not self.integrate:
            inds['time'] = self.time
        else:
            inds['time'] = slice(None, self.time + 1)

        inds['rlat'] = slice(None)
        inds['rlon'] = slice(None)

        return inds


class FlexVarSpecsConcentration(FlexVarSpecs):

    _keys_w_type = {
        **FlexVarSpecs._keys_w_type,
        'level': int,
    }

    def var_name(self):
        """Derive variable name from specifications."""
        return f'spec{self.species_id:03d}'

    def dim_inds_by_name(self):
        """Derive indices along NetCDF dimensions."""
        inds = super().dim_inds_by_name()
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


class FlexFieldSpecs:
    """FLEXPART field specifications."""

    cls_var_specs = FlexVarSpecs

    def __init__(self, var_specs_lst, op=np.nansum, var_attrs_replace=None):
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

        """
        # Create the specs instances
        try:
            iter(var_specs_lst)
        except TypeError:
            raise ValueError(
                f"var_specs: type '{type(var_specs_lst).__name__}' "
                f"not iterable") from None
        else:
            self.var_specs_lst = []
            for i, kwargs_specs in enumerate(var_specs_lst):
                try:
                    var_specs = self.cls_var_specs(**kwargs_specs)
                except Exception as e:
                    raise ValueError(
                        f"var_specs[{i}]: cannot create instance of "
                        f"{self.cls_var_specs.__name__} from {kwargs_specs}: "
                        f"{e.__class__.__name__}({e})") from None
                else:
                    self.var_specs_lst.append(var_specs)
            del var_specs_lst
        n_var_specs = len(self.var_specs_lst)

        # Check and store the operator(s)
        try:
            n_ops = len(op)
        except TypeError:
            if not callable(op):
                raise ValueError(f"op: {type(op).__name__} not callable")
        else:
            if n_ops != n_var_specs - 1:
                raise ValueError(
                    f"wrong number of operators passed in "
                    f"{type(ops).__name__}: {n_ops} != {n_var_specs}")
            for op_i in op:
                if not callable(op_i):
                    raise ValueError(f"op: {type(op_i).__name__} not callable")
        if callable(op):
            self.op = op
            self.op_lst = None
        else:
            self.op = None
            self.op_lst = op

        self.op = op if callable(op) else None
        self.op_lst = op if not callable(op) else None

        if var_attrs_replace is None:
            var_attrs_replace = {}
        self.var_attrs_replace = var_attrs_replace

    def __hash__(self):
        # yapf: disable
        return hash((
            hash(tuple(sorted(self.var_specs_lst))),
        ))
        # yapf: enable

    def __lt__(self, other):
        return hash(self) < hash(other)

    def __eq__(self, other):
        return hash(self) == hash(other)

    @classmethod
    def multiple(cls, vars_specs):
        var_specs_lst = cls.cls_var_specs.multiple(**vars_specs)
        return [cls(dict(var_specs)) for var_specs in var_specs_lst]

    def var_specs_merged(self):
        """Return merged variable specifications."""
        return self.var_specs_lst[0].merge(self.var_specs_lst[1:])


class FlexFieldSpecsConcentration(FlexFieldSpecs):

    cls_var_specs = FlexVarSpecsConcentration

    def __init__(self, var_specs):
        """Create an instance of ``FlexFieldSpecsConcentration``.

        Args:
            var_specs (dict): Specifications dict of input variable
                used to create an instance of ``FlexVarSpecsConcentration``
                as specified by the class attribute ``cls_var_specs``.
        """
        if not isinstance(var_specs, dict):
            raise ValueError(
                f"var_specs must be 'dict', not '{type(var_specs).__name__}'")
        super().__init__([var_specs])


class FlexFieldSpecsDeposition(FlexFieldSpecs):

    cls_var_specs = FlexVarSpecsDeposition

    def __init__(self, var_specs):
        """Create an instance of ``FlexFieldSpecsDeposition``.

        Args:
            var_specs (dict): Specifications dict of input variable
                used to create instance(s) of ``FlexVarSpecsDeposition``
                as specified by the class attribute ``cls_var_specs``.

        """
        if not isinstance(var_specs, dict):
            raise ValueError(
                f"var_specs must be 'dict', not '{type(var_specs).__name__}'")
        try:
            deposit_mode = var_specs['deposition']
        except KeyError as e:
            raise ValueError(f"var_specs: missing key '{e}'") from None

        kwargs = {}
        if deposit_mode in ['wet', 'dry']:
            var_specs_lst = [var_specs]

        elif deposit_mode == 'tot':
            var_specs_lst = [
                {
                    **var_specs, 'deposition': 'dry'
                },
                {
                    **var_specs, 'deposition': 'wet'
                },
            ]
            kwargs['op'] = np.nansum
            kwargs['var_attrs_replace'] = {
                'variable': {
                    'long_name': 'Total Deposition',
                    #'short_name': '
                },
            }

        super().__init__(var_specs_lst, **kwargs)


class FlexFileRotPole:
    """NetCDF file containing FLEXPART data on rotated-pole grid."""

    def __init__(self, path):
        """Create an instance of ``FlexFileRotPole``.

        Args:
            path (str): File path.

        """
        self.path = path

        self.reset()

    def reset(self):
        self._fi = None
        self._var_specs_curr = None

    def read(self, field_specs):
        """Read one or more fields from a file from disc.

        Args:
            field_specs (FlexFieldSpecs or list[FlexFieldSpecs]):
                Specifications for one or more input fields.

        Returns:
            FlexFieldRotPole: Single data object; if ``field_specs``
                constitutes a single ``FlexFieldSpecs`` instance.

            or

            list[FlexFieldRotPole]: One data object for each field;
                if ``field_specs`` constitutes a list of
                ``FlexFieldSpecs`` instances.

        """
        if isinstance(field_specs, FlexFieldSpecs):
            multiple = False
            fld_specs_lst = [field_specs]
        else:
            multiple = True
            fld_specs_lst = field_specs
        del field_specs
        n_fld_specs = len(fld_specs_lst)

        log.debug(f"read {self.path}")
        flex_data_lst = []
        with nc4.Dataset(self.path, 'r') as self._fi:

            log.debug("read grid: rlat, rlon")
            rlat = self._fi.variables['rlat'][:]
            rlon = self._fi.variables['rlon'][:]

            log.debug(f"process {n_fld_specs} field specs")
            for i_fld_specs, fld_specs in enumerate(fld_specs_lst):
                log.debug(f"{i_fld_specs + 1}/{n_fld_specs}: {fld_specs}")
                fld, attrs, time_stats = self._import_field(fld_specs)

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

    def _import_field(self, fld_specs):

        # Read fields and attributes from var specifications
        fld_lst = []
        attrs_lst = []
        time_stats_lst = []
        for var_specs in fld_specs.var_specs_lst:
            fld, attrs, time_stats = self._read_flex_data(var_specs)
            fld_lst.append(fld)
            attrs_lst.append(attrs)
            time_stats_lst.append(time_stats)

        # Merge fields and attributes
        fld = self._merge_fields(fld_lst, fld_specs)
        attrs = attrs_lst[0].merge(
            attrs_lst[1:], **fld_specs.var_attrs_replace)
        time_stats = self._merge_time_stats(time_stats_lst, fld_specs)

        return fld, attrs, time_stats

    def _read_flex_data(self, var_specs):
        """Core routine reading one field from an open file."""

        # Field
        log.debug("read field")
        fld, time_stats = self._read_var(var_specs)

        # Attributes
        log.debug("collect attributes")
        attrs = FlexAttrsCollector(self._fi, var_specs).run()

        #SR_TMP<
        log.debug("fix nc data")
        self._fix_nc_data(fld, attrs, time_stats)
        #SR_TMP>

        return fld, attrs, time_stats

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
        log.debug(f"indices: {inds}")
        check_array_indices(var.shape, inds)

        # Read field
        log.debug(f"field variable shape: {var.shape}")
        fld = var[inds]

        # Extract some stats across all time steps
        ind_time = var.dimensions.index('time')
        inds_time = inds[:ind_time] + [slice(None)] + inds[ind_time + 1:]
        time_stats = {
            'mean': np.nanmean(var[inds_time]),
            'median': np.nanmedian(var[inds_time]),
            'max': np.nanmax(var[inds_time]),
        }

        if var_specs.integrate:
            assert len(fld.shape) == 3  #SR_TMP
            fld = np.nansum(fld, axis=0)

        return fld, time_stats

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

    def _merge_time_stats(self, time_stats_lst, fld_specs):
        if len(time_stats_lst) == 1:
            return copy(next(iter(time_stats_lst)))
        #SR_TMP< TODO proper implementation
        if fld_specs.op is not None:
            if fld_specs.op is np.nansum:
                #SR This implementation is not very good!
                #SR For instance, max values cannot be summed up!
                time_stats = {}
                for key in time_stats_lst[0].keys():
                    vals = [s[key] for s in time_stats_lst]
                    time_stats[key] = fld_specs.op(vals)
                return time_stats
            raise NotImplementedError('merge time stats for op != nansum')
        raise NotImplementedError('merge time stats for multiple operators')
        #SR_TMP>

    #SR_TMP<<<
    def _fix_nc_data(self, fld, attrs, time_stats):

        def scale_time_stats(fact):
            for key, val in time_stats.items():
                time_stats[key] = fact*val

        if attrs.species.name in ['Cs-137', 'I-131a']:

            if attrs.variable.unit == 'ng kg-1':
                attrs.variable.unit = 'Bq m-3'
                fld *= 1e-12
                scale_time_stats(1e-12)

            elif attrs.variable.unit == '1e-12 kg m-2':
                attrs.variable.unit = 'Bq m-2'
                fld *= 1e-12
                scale_time_stats(1e-12)

            else:
                raise NotImplementedError(
                    f"species '{attrs.species.name}': "
                    f"unknown unit '{attrs.variable.unit}'")
        else:
            raise NotImplementedError(f"species '{attrs.species.name}'")


class FlexAttrsCollector:
    """Collect attributes for a field from an open NetCDF file."""

    def __init__(self, fi, field_specs):
        """Create an instance of ``FlexAttrsCollector``.

        Args:
            fi (netCDF4.Dataset): An open FLEXPART NetCDF file.

            field_specs (FlexVarSpecs): Input field specifications.

        """
        self.fi = fi
        self.field_specs = field_specs

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
        self.field_var_name = self.field_specs.var_name()
        self.ncattrs_field = self.ncattrs_vars[self.field_var_name]

    def run(self):
        """Collect attributes."""

        attrs_raw = {
            'grid': self._collect_grid_attrs(),
            'release': self._collect_release_attrs(),
            'variable': self._collect_variable_attrs(),
            'species': self._collect_species_attrs(),
            'simulation': self._collect_simulation_attrs(),
        }

        #ipython(globals(), locals(), 'FlexFile._collect_attrs')

        return FlexAttrsCollection(**attrs_raw)

    def _collect_grid_attrs(self):
        """Collect grid attributes."""

        np_lat = self.ncattrs_vars['rotated_pole']['grid_north_pole_latitude']
        np_lon = self.ncattrs_vars['rotated_pole']['grid_north_pole_longitude']

        return {
            'north_pole_lat': np_lat,
            'north_pole_lon': np_lon,
        }

    def _collect_release_attrs(self):
        """Collect release point attributes."""

        # Collect release point information
        _ind = self.field_specs.numpoint
        release_point = ReleasePoint.from_file(self.fi, _ind)

        site_lat = np.mean([release_point.lllat, release_point.urlat])
        site_lon = np.mean([release_point.lllon, release_point.urlon])
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
            'site_lat': site_lat,
            'site_lon': site_lon,
            'site_name': site_name,
            'height': (height, height_unit),
            'rate': (rate, rate_unit),
            'mass': (mass, mass_unit),
        }

    def _collect_variable_attrs(self):
        """Collect variable attributes."""

        specs = self.field_specs

        # Variable name
        if isinstance(specs, FlexVarSpecsConcentration):
            long_name = 'Activity'  #SR_HC
            short_name = 'Activity'  #SR_HC
        elif isinstance(specs, FlexVarSpecsDeposition):
            _dep_type = specs.deposition.lower().capitalize()
            long_name = f'{_dep_type} Surface Deposition'  #SR_HC
            short_name = f'Deposition'  #SR_HC
        if specs.integrate:
            long_name = f'Integr. {long_name}'
            short_name = f'Integr. {short_name}'

        try:
            _i = self.field_specs.level
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
            'level_bot': (level_bot, level_unit),
            'level_top': (level_top, level_unit),
        }

    def _collect_species_attrs(self):
        """Collect species attributes."""

        substance = self._get_substance()

        # Get deposition and washout data
        if isinstance(self.field_specs, FlexVarSpecsConcentration):
            name_core = self.field_var_name
        elif isinstance(self.field_specs, FlexVarSpecsDeposition):
            name_core = self.field_var_name[3:]
        deposit_vel = self.ncattrs_vars[f'DD_{name_core}']['dryvel']
        washout_coeff = self.ncattrs_vars[f'WD_{name_core}']['weta']
        washout_exponent = self.ncattrs_vars[f'WD_{name_core}']['wetb']

        # Get half life information
        try:
            half_life, half_life_unit = {
                'Cs-137': (30.17, 'years'),  #SR_HC
                'I-131a': (8.02, 'days'),  #SR_HC
            }[substance]
        except KeyError:
            raise NotImplementedError(f"half_life of '{substance}'")

        deposit_vel_unit = 'm s-1'  #SR_HC
        sediment_vel_unit = 'm s-1'  #SR_HC
        washout_coeff_unit = 's-1'  #SR_HC

        return {
            'name': substance,
            'half_life': (half_life, half_life_unit),
            'deposit_vel': (deposit_vel, deposit_vel_unit),
            'sediment_vel': (0.0, sediment_vel_unit),
            'washout_coeff': (washout_coeff, washout_coeff_unit),
            'washout_exponent': washout_exponent,
        }

    def _get_substance(self):
        substance = self.ncattrs_field['long_name']
        if isinstance(self.field_specs, FlexVarSpecsDeposition):
            substance = substance.replace(
                f'_{self.field_specs.deposition}_deposition', '')  #SR_HC
        return substance

    def _collect_simulation_attrs(self):
        """Collect simulation attributes."""

        # Start and end timesteps of simulation
        ts_start = datetime.datetime(
            *time.strptime(
                self.ncattrs_global['ibdate'] + self.ncattrs_global['ibtime'],
                '%Y%m%d%H%M%S',
            )[:6],
            tzinfo=datetime.timezone.utc,
        )
        ts_end = datetime.datetime(
            *time.strptime(
                self.ncattrs_global['iedate'] + self.ncattrs_global['ietime'],
                '%Y%m%d%H%M%S',
            )[:6],
            tzinfo=datetime.timezone.utc,
        )

        model_name = 'COSMO-1'  #SR_HC

        return {
            'model_name': model_name,
            'start': ts_start,
            'end': ts_end,
            'now': self._get_timestep(),
        }

    def _get_timestep(self, i=None):
        """Get the current timestep, or a specific one by index."""

        if i is None:
            # Default to timestep of current field
            i = self.field_specs.time

        var = self.fi.variables['time']

        # Obtain start from time unit
        rx = re.compile(
            r'seconds since (?P<yyyy>[12][0-9][0-9][0-9])-(?P<mm>[01][0-9])-'
            r'(?P<dd>[0-3][0-9]) (?P<HH>[0-2][0-9]):(?P<MM>[0-6][0-9])')
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
        delta = datetime.timedelta(seconds=int(var[i]))

        # Determine current timestep
        now = start + delta

        return now


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
