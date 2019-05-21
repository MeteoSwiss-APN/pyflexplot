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

    def __init__(self):
        pass

    def read(self, filename, varname):
        """Read a FLEXPART NetCDF file and return it's contents.

        Args:
            filename (str): Input file name, incl. path.

            varname (str): Variable to read.

        Returns:
            FlexData object with the file content.

        """
        with nc4.Dataset(filename, 'r') as fi:

            var = fi.variables[varname]

            ipython(globals(), locals(), "FlexReader.read()")


class FlexData:
    """Hold FLEXPART output data."""

    def __init__(self):
        pass


