# -*- coding: utf-8 -*-
"""
Main module.
"""
import logging as log
import netCDF4 as nc4
import copy
import re

#SRU_DEV<
try:
    from .utils_dev import ipython
except ImportError:
    pass
#SRU_DEV>


class FlexReader:
    """Read FLEXPART output files."""

    def __init__(
            self,
            species_ids=None,
            timestep_inds=None,
            field_types=None,
            level_inds=None):

        self.species_ids = set(species_ids) if species_ids else None
        self.timestep_inds = set(timestep_inds) if timestep_inds else None
        self.field_types = set(field_types) if field_types else None
        self.level_inds = set(level_inds) if level_inds else None

    def read(self, filename):
        """Read a FLEXPART NetCDF file and return it's contents.

        Args:
            filename (str): Input file name, incl. path.

            varname (str): Variable to read.

        Returns:
            FlexData object with the file content.

        """

        #
        # grid_conc_20190514000000.nc
        # ---------------------------
        #
        # Dimensions:
        #   <name>          <size>      <description>
        #   time            11          timestep
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

        #SR_TMP<
        if self.level_inds is not None:
            raise NotImplementedError(f"levels: {self.level_inds}")
        #SR_TMP>

        self._spec_ids_todo = copy.copy(self.species_ids)
        self._fld_typs_todo = copy.copy(self.field_types)

        log.debug(f"species ids to read: {self._spec_ids_todo}")
        log.debug(f"field types to read: {self._fld_typs_todo}")

        rx_var_name = re.compile(r'((?P<fld>[A-Z]{2})_)?spec(?P<id>[0-9]{3})')

        log.debug(f"open netcdf file {filename}")
        with nc4.Dataset(filename, 'r') as fi:

            log.debug("read data setup (grid etc.)")
            setup = {}
            setup['rlon'] = fi.variables['rlon'][:]
            setup['rlat'] = fi.variables['rlat'][:]
            flex_data = FlexData(setup)

            log.debug("read variables: particle fields")
            variables = {}
            for var_name, var in fi.variables.items():
                if (var.dimensions[-2:]) != ('rlat', 'rlon'):
                    log.debug(f" - {var_name}: skip (non-field var)")
                    continue

                #SRU_TMP<
                if var_name == 'fptot':
                    #SRU Let's ignore fptot for now...
                    log.debug(f" - {var_name}: skip !TMP!")
                    continue
                #SRU_TMP>

                # Parse var name for field type and species id
                match = rx_var_name.match(var_name)
                fld_typ = match.group('fld')
                if fld_typ is None:
                    fld_typ = '3D'
                spec_id = int(match.group('id').lstrip('0'))
                log.debug(f" - {var_name}: particle field")

                if self._check_read_field(spec_id, fld_typ):
                    log.debug("   - read the field")
                    var_attrs = {}  #SRU_TMP
                    flex_data.add_field(var[:], spec_id, fld_typ, var_attrs)

                #ipython(globals(), locals(), "FlexReader.read()")

        if self._spec_ids_todo:
            raise Exception(f"invalid species ids: {self._spec_ids_todo}")
        if self._fld_typs_todo:
            raise Exception(f"invalid field types: {self._fld_typs_todo}")

        return flex_data

    def _check_read_field(self, spec_id, fld_typ):
        """Check whether to read the current particle field."""

        log.debug("    - consider reading field")

        rd_sid = self._spec_ids_todo is None or spec_id in self._spec_ids_todo
        log.debug(
            f"      - species id: {spec_id:2}"
            f" -> {'read' if rd_sid else 'skip'}")

        rd_ftp = self._fld_typs_todo is None or fld_typ in self._fld_typs_todo
        log.debug(
            f"      - field type: {fld_typ:2}"
            f" -> {'read' if rd_ftp else 'skip'}")

        rd_tot = rd_sid and rd_ftp
        log.debug(f"      - verdict: {'read' if rd_tot else 'skip'}")

        if rd_tot:
            # Update todo sets
            if self._spec_ids_todo is not None:
                self._spec_ids_todo.remove(spec_id)
            if self._fld_typs_todo is not None:
                self._fld_typs_todo.remove(fld_typ)

        return rd_tot


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
