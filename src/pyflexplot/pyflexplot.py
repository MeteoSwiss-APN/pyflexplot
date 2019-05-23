# -*- coding: utf-8 -*-
"""
Main module.
"""
import logging
import netCDF4 as nc4

#SRU_DEV<
try:
    from .utils_dev import ipython
except ImportError:
    pass
#SRU_DEV>


class FlexReader:
    """Read FLEXPART output files."""

    def __init__(self, vars=None):

        self.vars = vars

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

        with nc4.Dataset(filename, 'r') as fi:

            for var_name, var in fi.variables.items():
                print(var_name, var.dimensions)

            print(fi.dimensions)

            ipython(globals(), locals(), "FlexReader.read()")


class FlexData:
    """Hold FLEXPART output data."""

    def __init__(self):
        pass
