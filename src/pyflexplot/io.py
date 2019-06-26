# -*- coding: utf-8 -*-
"""
Input/output.
"""
import datetime
import logging as log
import netCDF4 as nc4
import numpy as np
import os
import re

from collections import namedtuple
from copy import copy

#from .data import FlexData
from .data import FlexAttrsCollection
from .data import FlexDataRotPole

from .utils_dev import ipython  #SR_DEV

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


class FlexFieldSpecs:
    """Specifications of FLEXPART field to be read from NetCDF file."""

    def __init__(
            self, time_ind, age_ind, release_point_ind, level_ind, species_id,
            source_ind, field_type):
        """Create an instance of ``FlexFieldSpecs``.

        Args:
            <TODO>

        """
        self.time_ind = int(time_ind)
        self.age_ind = int(age_ind)
        self.release_point_ind = int(release_point_ind)
        self.level_ind = int(level_ind)
        self.species_id = int(species_id)
        self.source_ind = int(source_ind)
        self.field_type = field_type.upper()

    def var_name(self):
        """Derive variable name from specifications."""
        var_name_base = f'spec{self.species_id:03d}'
        prefix = {'3D': '', 'DD': 'DD_', 'WD': 'WD_'}[self.field_type]
        return prefix + var_name_base

    def dim_inds_nc(self):
        """Derive indices along NetCDF dimensions."""

        inds = {}

        inds['nageclass'] = self.age_ind
        inds['numpoint'] = self.release_point_ind
        inds['time'] = self.time_ind

        if self.field_type == '3D':
            inds['level'] = self.level_ind

        inds['rlat'] = slice(None)
        inds['rlon'] = slice(None)

        return inds


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
        self._field_specs = None

    def read(self, fields_specs):
        """Read one or more fields from a file from disc.

        Args:
            fields_specs (list[FlexFieldSpecs]): Specifications for
                one or more input fields.

        Returns:
            list[FlexDataRotPole]: One data object for each field.

        """
        log.debug(f"read {self.path}")
        flex_data_lst = []
        with nc4.Dataset(self.path, 'r') as fi:
            self._fi = fi
            n_specs = len(fields_specs)
            log.debug(f"process {n_specs} field specs")
            for i_specs, field_specs in enumerate(fields_specs):
                log.debug(f"{i_specs + 1}/{n_specs}: {field_specs}")
                self._field_specs = field_specs
                flex_data = self._read()
                flex_data_lst.append(flex_data)
        self.reset()
        return flex_data_lst

    def _read(self):
        """Core routine reading one field from an open file."""

        # Grid coordinates
        log.debug("read grid")
        rlat = self._fi.variables['rlat'][:]
        rlon = self._fi.variables['rlon'][:]

        # Field
        log.debug("read field")
        fld = self._read_field()

        # Attributes
        log.debug("collect attributes")
        attrs = self._collect_attrs()

        #SR_TMP<
        log.debug("fix nc data")
        self._fix_nc_data(fld, attrs)
        #SR_TMP>

        # Collect data
        log.debug("create data object")
        flex_data = FlexDataRotPole(rlat, rlon, fld, attrs, self._field_specs)

        return flex_data

    def _read_field(self):

        # Select variable in file
        var_name = self._field_specs.var_name()
        var = self._fi.variables[var_name]

        # Indices of field along NetCDF dimensions
        dim_inds_nc = self._field_specs.dim_inds_nc()

        # Assemble indices for slicing
        inds = [None]*len(var.dimensions)
        for dim_name, dim_ind in dim_inds_nc.items():
            ind = var.dimensions.index(dim_name)
            inds[ind] = dim_ind
        if None in inds:
            raise Exception(
                f"variable '{var_name}': could not resolve all indices!"
                f"\ndim_inds   : {dim_inds_nc}"
                f"\ndimensions : {var.dimensions}"
                f"\ninds       : {inds}")

        # Read field
        fld = var[inds]

        return fld

    def _collect_attrs(self):
        """Collect attributes."""

        # Collect all variables attributes
        ncattrs_vars = {}
        for var in self._fi.variables.values():
            ncattrs_vars[var.name] = {
                attr: var.getncattr(attr) for attr in var.ncattrs()
            }

        # Select attributes of field variable
        var_name = self._field_specs.var_name()
        ncattrs_field = ncattrs_vars[var_name]

        # Collect release point information
        release_point = ReleasePoint.from_file(
            self._fi, self._field_specs.release_point_ind)

        raw = {}

        raw['grid'] = {}
        _lat = ncattrs_vars['rotated_pole']['grid_north_pole_latitude']
        _lon = ncattrs_vars['rotated_pole']['grid_north_pole_longitude']
        raw['grid']['north_pole_lat'] = _lat
        raw['grid']['north_pole_lon'] = _lon

        raw['release'] = {}
        raw['release']['site_lat'] = 47.37  #SR_HC
        raw['release']['site_lon'] = 7.97  #SR_HC
        raw['release']['site_name'] = 'Goesgen'  #SR_HC
        raw['release']['site_tz_name'] = 'Europe/Zurich'  #SR_HC
        raw['release']['height'] = (100, 'm AGL')  #SR_HC
        raw['release']['rate'] = (34722.2, 'Bq s-1')  #SR_HC
        raw['release']['mass'] = (1e9, 'Bq')  #SR_HC

        raw['variable'] = {}
        raw['variable']['name'] = 'Concentration'  #SR_HC
        raw['variable']['unit'] = ncattrs_field['units']
        raw['variable']['level_bot'] = (500, 'm AGL')  #SR_HC
        raw['variable']['level_top'] = (2000, 'm AGL')  #SR_HC

        raw['species'] = {}
        raw['species']['name'] = ncattrs_field['long_name']
        raw['species']['half_life'] = (30.0, 'years')  #SR_HC
        raw['species']['deposit_vel'] = (1.5e-3, 'm s-1')  #SR_HC
        raw['species']['sediment_vel'] = (0.0, 'm s-1')  #SR_HC
        raw['species']['washout_coeff'] = (7.0e-5, 's-1')  #SR_HC
        raw['species']['washout_exponent'] = 0.8  #SR_HC

        #SR_HC<
        ts_start = datetime.datetime(
            year=2019,
            month=5,
            day=28,
            hour=0,
            minute=0,
            tzinfo=datetime.timezone.utc,
        )
        ts_end = datetime.datetime(
            year=2019,
            month=5,
            day=28,
            hour=8,
            minute=0,
            tzinfo=datetime.timezone.utc,
        )
        ts_now = datetime.datetime(
            year=2019,
            month=5,
            day=28,
            hour=9,
            minute=0,
            tzinfo=datetime.timezone.utc,
        )
        #SR_HC>

        raw['simulation'] = {}
        raw['simulation']['model_name'] = 'COSMO-1'  #SR_HC
        raw['simulation']['start'] = ts_start
        raw['simulation']['end'] = ts_end
        raw['simulation']['now'] = ts_now

        ipython(globals(), locals(), 'FlexFile._collect_attrs')

        return FlexAttrsCollection(**raw)

    #SR_TMP<
    def _fix_nc_data(self, fld, attrs):
        if attrs.species.name in ['Cs-137', 'I-131a']:
            if attrs.variable.unit == 'ng kg-1':
                attrs.variable.unit = 'Bq m-3'
                fld *= 1e-12
            else:
                raise NotImplementedError(
                    f"species '{ncattrs_var['long_name']}': "
                    f"unknown unit '{ncattrs_var['units']}'")

    #SR_TMP>


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
                raise ValueError(f"unexpected argument {key}={val}")

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
