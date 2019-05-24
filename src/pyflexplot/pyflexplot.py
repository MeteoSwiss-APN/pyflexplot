# -*- coding: utf-8 -*-
"""
Main module.
"""
import copy
import logging as log
import netCDF4 as nc4
import numpy as np
import re

#SRU_DEV<
try:
    from .utils_dev import ipython
except ImportError:
    pass
#SRU_DEV>


class FlexFileReader:
    """Read FLEXPART output files."""

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

    def __init__(
            self,
            species_ids=None,
            age_class_inds=None,
            source_inds=None,
            time_inds=None,
            field_types=None,
            level_inds=None):

        def prep_attr(lst):
            return set(lst) if lst else None

        self.species_ids = prep_attr(species_ids)
        self.age_class_inds = prep_attr(age_class_inds)
        self.time_inds = prep_attr(time_inds)
        self.field_types = prep_attr(field_types)
        self.level_inds = prep_attr(level_inds)

    def read(self, filename):
        """Read a FLEXPART NetCDF file and return it's contents.

        Args:
            filename (str): Input file name, incl. path.

            varname (str): Variable to read.

        Returns:
            FlexData object with the file content.

        """

        self._todo = {
            'spec_id': copy.copy(self.species_ids),
            'age_ind': copy.copy(self.age_class_inds),
            'tim_ind': copy.copy(self.time_inds),
            'fld_typ': copy.copy(self.field_types),
            'lvl_ind': copy.copy(self.level_inds),
        }

        #SRU_TMP<
        for key in ['age_ind', 'tim_ind', 'lvl_ind']:
            if self._todo[key] is not None:
                raise NotImplementedError(f"todo[{key}]: {self._todo[key]}")
        #SRU_TMP>

        def set2str(lst):
            if lst is None:
                return 'None'
            return ', '.join([str(i) for i in sorted(lst)])

        log.debug("particle field selection criteria:")
        log.debug(f" - species ids    : {set2str(self._todo['spec_id'])}")
        log.debug(f" - age class inds : {set2str(self._todo['age_ind'])}")
        log.debug(f" - time inds      : {set2str(self._todo['tim_ind'])}")
        log.debug(f" - field types    : {set2str(self._todo['fld_typ'])}")
        log.debug(f" - levels inds    : {set2str(self._todo['lvl_ind'])}")

        rx_var_name = re.compile(r'((?P<fld>[A-Z]{2})_)?spec(?P<id>[0-9]{3})')

        log.debug(f"open netcdf file {filename}")
        wvar = 10 # width of variable names in debug output
        with nc4.Dataset(filename, 'r') as fi:

            log.debug("read data setup (grid etc.)")
            setup = {}
            setup['rlon'] = fi.variables['rlon'][:]
            setup['rlat'] = fi.variables['rlat'][:]
            flex_data = FlexData(setup)

            levels = fi.variables["level"]

            log.debug("read variables: particle fields")
            variables = {}
            for var_name, var in fi.variables.items():
                if (var.dimensions[-2:]) != ('rlat', 'rlon'):
                    log.debug(f" - {var_name:{wvar}} : skip (non-field var)")
                    continue

                #SRU_TMP<
                if var_name == 'fptot':
                    #SRU Let's ignore fptot for now...
                    log.debug(f" - {var_name:{wvar}} : !TMP! skip")
                    continue
                #SRU_TMP>

                # Parse var name for field type and species id
                match = rx_var_name.match(var_name)
                fld_typ = match.group('fld')
                if fld_typ is None:
                    fld_typ = '3D'
                spec_id = int(match.group('id').lstrip('0'))
                log.debug(f" - {var_name:{wvar}} : particle field")

                if self._check_read_field(spec_id=spec_id, fld_typ=fld_typ):
                    self._remove_todo(spec_id=spec_id, fld_typ=fld_typ)

                    # Read array -- possibly only certain level(s)
                    if fld_typ == '3D' and self.level_inds is not None:
                        inds = list(self.level_inds)
                        levels_sel = levels[inds]
                        #SRU_TMP<
                        assert var.dimensions[::-1].index('level') == 2
                        #SRU_TMP>
                        arr = var[..., inds, :, :]
                    else:
                        arr = var[:]

                    var_attrs = {}  #SRU_TMP

                    flex_data.add_field(arr, spec_id, fld_typ, var_attrs)

                #ipython(globals(), locals(), "FlexReader.read()")

        if self._todo['spec_id']:
            raise Exception(f"invalid species ids: {self._todo['spec_id']}")
        if self._todo['fld_typ']:
            raise Exception(f"invalid field types: {self._todo['fld_typ']}")

        return flex_data

    def _check_read_field(self, **kwargs):
        """Check whether to read the current particle field."""
        log.debug("    - consider reading field")

        # For each variable, check whether it is still 'todo',
        # meaning whether it is still in the repsective todo set.
        read = {}
        for key, val in sorted(kwargs.items()):
            todo = self._todo[key]
            read[key] = todo is None or kwargs[key] in todo
            log.debug(
                f"      - {key}: {kwargs[key]:2}"
                f" -> {'read' if read[key] else 'skip'}")

        if not all(read.values()):
            # Not all variables are still 'todo' -> skip field
            return False

        # All variables were still 'todo' -> read field
        return True

    def _remove_todo(self, **kwargs):
        """Remove value of each variable from respective todo set."""
        for key, val in kwargs.items():
            if self._todo[key] is not None:
                self._todo[key].remove(val)


class FlexData:
    """Hold FLEXPART output data."""

    def __init__(self, setup):
        self._setup = setup

        self._fields = {}
        self._field_attrs = {}

    def _key(self, species_id, field_type):
        return species_id, field_type

    def add_field(self, arr, species_id, field_type, attrs=None):
        """Add a field array with optional attributes."""
        key = self._key(species_id, field_type)
        self._fields[key] = arr
        self._field_attrs[key] = attrs

    def field(self, species_id, field_type):
        """Return a field."""
        return self._fields[self._key(species_id, field_type)]

    def field_attrs(self, species_id, field_type):
        """Return field attributes."""
        return self._field_attrs[self._key(species_id, field_type)]
