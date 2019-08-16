# -*- coding: utf-8 -*-
"""
Input specifications.
"""
import datetime
import itertools
import logging as log
import numpy as np
import re
import time

from copy import copy, deepcopy
from pprint import pformat
from pprint import pprint  #SR_DEV

from .attr import FlexAttrGroupCollection
from .utils import pformat_dictlike
from .utils import nested_dict_set

from .utils_dev import ipython  #SR_DEV


def int_or_list(arg):
    try:
        iter(arg)
    except TypeError:
        return int(arg)
    else:
        return [int(a) for a in arg]


#======================================================================
# Variable Specifications
#======================================================================


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


#----------------------------------------------------------------------


class FlexVarSpecs_Concentration(FlexVarSpecs):

    _keys_w_type = {
        **FlexVarSpecs._keys_w_type,
        'level': int_or_list,
    }

    @classmethod
    def long_name(cls, lang, var_specs):
        return {
            'en': 'Activity Concentration',
            'de': r'Aktivit$\mathrm{\"a}$tskonzentration',
        }[lang]

    @classmethod
    def short_name(cls, lang, var_specs):
        return {
            'en': 'Concentration',
            'de': 'Konzentration',
        }[lang]

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


class FlexVarSpecs_Deposition(FlexVarSpecs):

    _keys_w_type = {
        **FlexVarSpecs._keys_w_type,
        'deposition': str,
    }

    @classmethod
    def deposition_type_long_name(cls, lang, var_specs):
        if lang == 'en':
            choices = {
                'wet': 'Wet',
                'dry': 'Dry',
                'tot': 'Total',
            }

        elif lang == 'de':
            choices = {
                'wet': 'Nass',
                'dry': 'Trocken',
                'tot': 'Total',
            }
        else:
            raise NotImplementedError(f"lang='{lang}'")
        return choices[var_specs.deposition]

    @classmethod
    def long_name(cls, lang, var_specs):
        dep_type = cls.deposition_type_long_name(lang, var_specs)
        return {
            'en': f'{dep_type} Surface Deposition',
            'de': f'{dep_type}e Bodenablagerung',
        }[lang]

    @classmethod
    def short_name(cls, lang, var_specs):
        return {
            'en': f'Deposition',
            'de': f'Ablagerung',
        }[lang]

    def var_name(self):
        """Derive variable name from specifications."""
        prefix = {'wet': 'WD', 'dry': 'DD'}[self.deposition]
        return f'{prefix}_spec{self.species_id:03d}'


class FlexVarSpecs_AffectedArea(FlexVarSpecs_Deposition):

    @classmethod
    def long_name(cls, lang, var_specs):
        dep_type = cls.deposition_type_long_name(lang, var_specs)
        return {
            'en': f'Affected Area ({dep_type})',
            'de': f'Beaufschlagtes Gebiet ({dep_type})',
        }[lang]


class FlexVarSpecs_EnsMean_Concentration(FlexVarSpecs_Concentration):

    @classmethod
    def long_name(cls, lang, var_specs):
        return {
            'en': 'Ensemble-Mean Activity Concentration',
            'de': r'Ensemble-Mittel der Aktivit$\mathrm{\"a}$tskonzentration',
        }[lang]


class FlexVarSpecs_EnsThrAgrmt_Concentration(FlexVarSpecs_Concentration):

    @classmethod
    def long_name(cls, lang, var_specs):
        s_de = (
            r'Ensemble-Grenzwert$\mathrm{\"u}$bereinstimmung '
            r'der Aktivit$\mathrm{\"a}$tskonzentration')
        return {
            'en': 'Ensemble Threshold Agreement of Activity Concentration',
            'de': s_de,
        }[lang]

    @classmethod
    def short_name(cls, lang, var_specs):
        return {
            #'en': 'No. Members',
            #'de': 'Anz. Members',
            'en': 'Members',
            'de': 'Members',
        }[lang]


class FlexVarSpecs_EnsMean_Deposition(FlexVarSpecs_Deposition):

    @classmethod
    def long_name(cls, lang, var_specs):
        dep_type = cls.deposition_type_long_name(lang, var_specs)
        return {
            'en': f'Ensemble-Mean {dep_type} Surface Deposition',
            'de': f'Ensemble-Mittel der {dep_type}en Bodenablagerung',
        }[lang]


class FlexVarSpecs_EnsMeanAffectedArea(FlexVarSpecs_AffectedArea):

    @classmethod
    def long_name(cls, lang, var_specs):
        dep_type = cls.deposition_type_long_name(lang, var_specs)
        return {
            'en': f'Ensemble-Mean Affected Area ({dep_type})',
            'de': f'Ensemble-Mittel des Beaufschlagtes Gebiets ({dep_type})',
        }[lang]


#----------------------------------------------------------------------

FlexVarSpecs.Concentration = FlexVarSpecs_Concentration
FlexVarSpecs.Deposition = FlexVarSpecs_Deposition
FlexVarSpecs.AffectedArea = FlexVarSpecs_AffectedArea
FlexVarSpecs.EnsMean_Concentration = FlexVarSpecs_EnsMean_Concentration
FlexVarSpecs.EnsMean_Deposition = FlexVarSpecs_EnsMean_Deposition
FlexVarSpecs.EnsMeanAffectedArea = FlexVarSpecs_EnsMeanAffectedArea
FlexVarSpecs.EnsThrAgrmt_Concentration = FlexVarSpecs_EnsThrAgrmt_Concentration

#======================================================================
# Field Specifications
#======================================================================


class FlexFieldSpecs:
    """FLEXPART field specifications."""

    cls_var_specs = FlexVarSpecs

    # Dimensions with optionally multiple values
    dims_opt_mult_vals = ['species_id']

    def __init__(
            self,
            var_specs_lst,
            *,
            op=np.nansum,
            var_attrs_replace=None,
            lang='en',
            **addtl_attrs):
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

        # Set additional attributes
        self.set_addtl_attrs(**addtl_attrs)

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

    def set_addtl_attrs(self, **attrs):
        """Set additional attributes."""
        for attr, val in sorted(attrs.items()):
            if hasattr(self, attr):
                raise ValueError(
                    f"attribute '{type(self).__name__}.{attr}' already exists")
            setattr(self, attr, val)

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
    def multiple(cls, vars_specs, *args, **kwargs):
        var_specs_lst = cls.cls_var_specs.multiple_as_dict(**vars_specs)
        field_specs_lst = []
        for var_specs in var_specs_lst:
            try:
                field_specs = cls(var_specs, *args, **kwargs)
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


#----------------------------------------------------------------------


class FlexFieldSpecs_Concentration(FlexFieldSpecs):
    cls_var_specs = FlexVarSpecs_Concentration

    # Dimensions with optionally multiple values
    dims_opt_mult_vals = FlexFieldSpecs.dims_opt_mult_vals + ['level']

    def __init__(self, var_specs, *args, **kwargs):
        """Create an instance of ``FlexFieldSpecs_Concentration``.

        Args:
            var_specs (dict): Specifications dict of input variable
                used to create an instance of ``FlexVarSpecs_Concentration``
                as specified by the class attribute ``cls_var_specs``.

            **kwargs: Keyword arguments passed to ``FlexFieldSpecs``.
        """
        if not isinstance(var_specs, dict):
            raise ValueError(
                f"var_specs must be 'dict', not '{type(var_specs).__name__}'")
        super().__init__([var_specs], *args, **kwargs)


class FlexFieldSpecs_Deposition(FlexFieldSpecs):
    cls_var_specs = FlexVarSpecs_Deposition

    def __init__(self, var_specs, *args, lang='en', **kwargs):
        """Create an instance of ``FlexFieldSpecs_Deposition``.

        Args:
            var_specs (dict): Specifications dict of input variable
                used to create instance(s) of ``FlexVarSpecs_Deposition``
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

        super().__init__(var_specs_lst, *args, **kwargs)


class FlexFieldSpecs_AffectedArea(FlexFieldSpecs_Deposition):
    cls_var_specs = FlexVarSpecs_AffectedArea


class FlexFieldSpecs_Ens:
    pass


class FlexFieldSpecs_EnsMean_Concentration(FlexFieldSpecs_Ens,
                                          FlexFieldSpecs_Concentration):
    cls_var_specs = FlexVarSpecs_EnsMean_Concentration


class FlexFieldSpecs_EnsMean_Deposition(FlexFieldSpecs_Ens,
                                       FlexFieldSpecs_Deposition):
    cls_var_specs = FlexVarSpecs_EnsMean_Deposition


class FlexFieldSpecs_EnsMeanAffectedArea(FlexFieldSpecs_Ens,
                                         FlexFieldSpecs_AffectedArea):
    cls_var_specs = FlexVarSpecs_EnsMeanAffectedArea


class FlexFieldSpecs_EnsThrAgrmt_Concentration(FlexFieldSpecs_Ens,
                                              FlexFieldSpecs_Concentration):
    cls_var_specs = FlexVarSpecs_EnsThrAgrmt_Concentration


#----------------------------------------------------------------------

FlexFieldSpecs.Concentration = FlexFieldSpecs_Concentration
FlexFieldSpecs.Deposition = FlexFieldSpecs_Deposition
FlexFieldSpecs.AffectedArea = FlexFieldSpecs_AffectedArea
FlexFieldSpecs.Ens = FlexFieldSpecs_Ens
FlexFieldSpecs.EnsMean_Concentration = FlexFieldSpecs_EnsMean_Concentration
FlexFieldSpecs.EnsMean_Deposition = FlexFieldSpecs_EnsMean_Deposition
FlexFieldSpecs.EnsMeanAffectedArea = FlexFieldSpecs_EnsMeanAffectedArea
FlexFieldSpecs.EnsThrAgrmt_Concentration = (
    FlexFieldSpecs_EnsThrAgrmt_Concentration)

#======================================================================


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
        height_unit = {
            'en': 'm AGL',  #SR_HC
            'de': r'm $\mathrm{\"u}$. G.',  #SR_HC
        }[self.lang]

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

        # Variable names
        long_name = self.get_long_name(self.var_specs, lang=self.lang)
        short_name = self.get_short_name(self.var_specs, lang=self.lang)

        try:
            _i = self.var_specs.level
        except AttributeError:
            #SR_TMP<
            level_unit = ''
            level_bot = -1
            level_top = -1
            #SR_TMP>
        else:
            level_unit = {
                'en': 'm AGL',  #SR_HC
                'de': r'm $\mathrm{\"u}$. G.',  #SR_HC
            }[self.lang]
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

    @staticmethod
    def get_long_name(var_specs, *, type_=None, lang='en'):
        """Return long variable name.

        Args:
            var_specs (dict or FlexVarSpecs): Variable specifications.
                Must be either an instance of ``FlexVarSpecs`` or
                (most likely) a subclass thereof, or convertible to
                that. In the latter case, ``type_`` is mandatory.

            type_ (type, optional): Type to which ``var_specs`` is
                converted. Must be ``FlexVarSpecs`` or one of its
                subclasses. Mandatory if ``var_specs`` is not an
                instance of such a type. Defaults to None.

            lang (str, optional): Language, e.g., 'de' for German.
                Defaults to 'en' (English).

        """
        if type_ is None:
            type_ = type(var_specs)
        return type_.long_name(lang, var_specs)

    @staticmethod
    def get_short_name(var_specs, *, type_=None, lang='en'):
        """Return short variable name.

        Args: See ``get_long_name``.

        """
        if type_ is None:
            type_ = type(var_specs)
        return type_.short_name(lang, var_specs)

    def collect_species_attrs(self):
        """Collect species attributes."""

        substance = self._get_substance()

        # Get deposition and washout data
        if isinstance(self.var_specs, FlexVarSpecs.Concentration):
            name_core = self.field_var_name
        elif isinstance(self.var_specs, FlexVarSpecs.Deposition):
            name_core = self.field_var_name[3:]
        deposit_vel = self.ncattrs_vars[f'DD_{name_core}']['dryvel']
        washout_coeff = self.ncattrs_vars[f'WD_{name_core}']['weta']
        washout_exponent = self.ncattrs_vars[f'WD_{name_core}']['wetb']

        # Get half life information
        try:
            half_life, half_life_unit = {
                'Cs-137': (30.17, 'a'),  #SR_HC
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
        if isinstance(self.var_specs, FlexVarSpecs.Deposition):
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
