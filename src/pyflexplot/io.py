# -*- coding: utf-8 -*-
"""
Input/output.
"""
import logging as log
import netCDF4 as nc4
import numpy as np
import re

from collections import namedtuple
from copy import copy

from .data import FlexData

from .utils_dev import ipython  #SRU_DEV

ReleasePoint = namedtuple(
    'ReleasePoint',
    (
        'name age_id kind lllat lllon urlat urlon'
        ' zbot ztop start end n_parts ms_parts'),
)

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


class FlexFileReader:
    """Read FLEXPART output files.

    Kwargs:
        TODO

    """

    # Width of variable names in debug output
    wvar = 10

    # Reg. expr. to extract field type and species id from variable name
    rx_fld_var_name = re.compile(r'((?P<fld>[A-Z]{2})_)?spec(?P<id>[0-9]{3})')

    def __init__(
            self,
            *,
            age_inds=None,
            relpt_inds=None,
            species_ids=None,
            source_inds=None,
            time_inds=None,
            field_types=None,
            level_inds=None):

        def prep_attr(lst):
            return set(lst) if lst else None

        self.age_inds = prep_attr(age_inds)
        self.relpt_inds = prep_attr(relpt_inds)
        self.species_ids = prep_attr(species_ids)
        self.time_inds = prep_attr(time_inds)
        self.field_types = prep_attr(field_types)
        self.level_inds = prep_attr(level_inds)

    def read(self, file_path):
        """Read a FLEXPART NetCDF file and return it's contents.

        Args:
            file_path (str): Input file.

        Returns:
            FlexData object with the file content.

        """

        def prep_todo(vals):
            if vals is None:
                return None
            return {v: False for v in vals}

        self._todo = {
            'age_ind': prep_todo(self.age_inds),
            'relpt_ind': prep_todo(self.relpt_inds),
            'time_ind': prep_todo(self.time_inds),
            'level_ind': prep_todo(self.level_inds),
            'species_id': prep_todo(self.species_ids),
            'field_type': prep_todo(self.field_types),
        }

        def set2str(lst):
            if lst is None:
                return 'None'
            return ', '.join([str(i) for i in sorted(lst)])

        log.debug("particle field selection criteria:")
        log.debug(f" - age inds     : {set2str(self._todo['age_ind'])}")
        log.debug(f" - relpt inds   : {set2str(self._todo['relpt_ind'])}")
        log.debug(f" - time inds    : {set2str(self._todo['time_ind'])}")
        log.debug(f" - levels inds  : {set2str(self._todo['level_ind'])}")
        log.debug(f" - species ids  : {set2str(self._todo['species_id'])}")
        log.debug(f" - field types  : {set2str(self._todo['field_type'])}")

        log.debug(f"open netcdf file {file_path}")
        with nc4.Dataset(file_path, 'r') as fi:

            log.debug("read data setup (grid etc.)")
            setup = {}  #SRU_TMP
            flex_data = FlexData(setup)

            # Grid
            flex_data.set_grid(
                rlat=fi.variables['rlat'][:],
                rlon=fi.variables['rlon'][:],
            )

            relpts = self._collect_release_points(fi)

            time = fi.variables['time'][:]
            self._levels_all = fi.variables['level'][:]

            #ipython(globals(), locals(), "FlexFileReader.read()")

            # Field variables
            log.debug("read variables: particle fields")
            for var in fi.variables.values():
                if (var.dimensions[-2:]) != ('rlat', 'rlon'):
                    log.debug(
                        f" - {var.name:{self.wvar}} : skip (non-field var)")
                    continue
                self._process_fld_var(var, flex_data)

        # Check that all 'todo' values have been processed
        for name, vals in self._todo.items():
            if vals is not None:
                vals_false = {v for v, b in vals.items() if not b}
                if vals_false:
                    raise Exception(
                        f"invalid {name.replace('_', ' ')}: {vals_false}")

        return flex_data

    def _collect_release_points(self, fi):
        """Collect information about particle release points.

        Args:
            fi (netCDF4.Dataset): NetCDF file handle.

        Returns:
            list[ReleasePoint]: List of release points.

        """
        relpts = []
        n = fi.variables['RELCOM'].shape[0]
        for ipt in range(n):
            kwargs = {}

            # Name -- byte character array
            var = fi.variables['RELCOM'][ipt]
            kwargs['name'] = var[~var.mask].tostring().decode('utf-8').rstrip()

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
                kwargs[key_out] = fi.variables[key_in][ipt].tolist()

            relpts.append(ReleasePoint(**kwargs))

        return relpts

    def _process_fld_var(self, var, flex_data):
        """Check if field variable is to be read, and if so, read it."""

        #SRU_TMP<
        if var.name == 'fptot':
            #SRU Let's ignore fptot for now...
            log.debug(f" - {var.name:{self.wvar}} : !TMP! skip")
            return
        #SRU_TMP>

        # Parse var name for field type and species id
        match = self.rx_fld_var_name.match(var.name)
        field_type = match.group('fld')
        if field_type is None:
            field_type = '3D'
        species_id = int(match.group('id').lstrip('0'))
        log.debug(f" - {var.name:{self.wvar}} : particle field")

        # Determine levels to be read
        if field_type == '3D':
            if self.level_inds is not None:
                level_inds = list(self.level_inds)
                levels_sel = self._levels_all[level_inds]
            else:
                levels_sel = copy(self._levels_all)
                level_inds = np.arange(len(levels_sel))
        else:
            levels_sel = None
            level_inds = None

        # Get variable attributes
        #SRU_TMP<
        age_inds = np.arange(var.shape[0])
        relpt_inds = np.arange(var.shape[1])
        time_inds = np.arange(var.shape[2])
        #SRU_TMP>

        log.debug(f"age inds    : {age_inds}")
        log.debug(f"relpt inds  : {relpt_inds}")
        log.debug(f"time inds   : {time_inds}")
        log.debug(f"level inds  : {level_inds}")

        for age_ind in age_inds:
            for relpt_ind in relpt_inds:
                for time_ind in time_inds:
                    if level_inds is not None:
                        for level_ind, level in zip(level_inds, levels_sel):
                            self._proc_fld2d(
                                flex_data, var, species_id, field_type,
                                age_ind, relpt_ind, time_ind, level_ind)
                    else:
                        self._proc_fld2d(
                            flex_data,
                            var,
                            species_id,
                            field_type,
                            age_ind,
                            relpt_ind,
                            time_ind,
                            level_ind=None)

    def _proc_fld2d(
            self, flex_data, var, species_id, field_type, age_ind, relpt_ind,
            time_ind, level_ind):

        # Check whether to skip the field
        kwargs = {
            'species_id': species_id,
            'field_type': field_type,
            'age_ind': age_ind,
            'relpt_ind': relpt_ind,
            'time_ind': time_ind,
            'level_ind': level_ind,
        }
        read_field = self._check_read_field(**kwargs)
        if not read_field:
            # Skip it!
            return
        self._remove_todo(**kwargs)

        var_attrs = {var.getncattr(attr) for attr in var.ncattrs()}

        log.debug(f"({age_ind}, {relpt_ind}, {time_ind}, {level_ind})")

        # Compile indices to slice array
        if level_ind is None:
            inds = (age_ind, relpt_ind, time_ind)
        else:
            inds = (age_ind, relpt_ind, time_ind, level_ind)

        # Store array slice (horizontal 2d field)
        flex_data.add_field(
            var[inds],
            var_attrs,
            age_ind=age_ind,
            relpt_ind=relpt_ind,
            time_ind=time_ind,
            level_ind=level_ind,
            species_id=species_id,
            field_type=field_type,
        )

    def _check_read_field(self, **kwargs):
        """Check whether to read the current particle field."""
        log.debug("    - consider reading field")

        # For each variable, check whether it is still 'todo',
        # meaning whether it is still in the repsective todo set.
        read = {}
        for key, val in sorted(kwargs.items()):
            todo = self._todo[key]
            val = kwargs[key]
            if val is None:
                read[key] = False
                val = 'None'
            else:
                read[key] = todo is None or val in todo
            action = 'read' if read[key] else 'skip'
            log.debug(f"      - {key}: {val:2} -> {action} (todo: {todo})")

        if not all(read.values()):
            # Not all variables are still 'todo' -> skip field
            return False

        # All variables were still 'todo' -> read field
        return True

    def _remove_todo(self, **kwargs):
        """Remove value of each variable from respective todo set."""
        for name, key in kwargs.items():
            if self._todo[name] is not None:
                self._todo[name][key] = True
