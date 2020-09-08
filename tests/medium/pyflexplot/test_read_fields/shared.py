# -*- coding: utf-8 -*-
"""
Utilities for testing.
"""
# Standard library
import distutils.dir_util
import os

# Third-party
import netCDF4 as nc4
import numpy as np
import pytest  # type: ignore

# First-party
from pyflexplot.setup import CoreSetup


@pytest.fixture
def datadir_rel(tmpdir, request):
    """Return path to temporary data directory named like test file.

    Pytest fixture to find a data folder with the same name as the test
    module and -- if found -- mirror it to a temporary directory, which
    allows the tests to use the data files freely, even in parallel.

    Adapted from `https://stackoverflow.com/a/29631801`.
    """
    file = request.module.__file__
    dir, _ = os.path.splitext(file)
    if os.path.isdir(dir):
        distutils.dir_util.copy_tree(dir, str(tmpdir))
    return tmpdir


@pytest.fixture
def datadir_original(tmpdir, request):
    """Return path to temporary data directory with original data files."""
    return _datadir_core("original", tmpdir, request)


@pytest.fixture
def datadir_artificial(tmpdir, request):
    """Return path to temporary data directory with artificial data files."""
    return _datadir_core("artificial", tmpdir, request)


@pytest.fixture
def datadir_reduced(tmpdir, request):
    """Return path to temporary data directory with reduced data files."""
    return _datadir_core("reduced", tmpdir, request)


def _datadir_core(subdir, tmpdir, request):
    file = request.module.__file__
    dir, _ = os.path.splitext(file)
    data_root = os.path.abspath(f"{os.path.abspath(dir)}/../../../../data")
    data_dir = f"{data_root}/pyflexplot/input/{subdir}"
    if os.path.isdir(data_dir):
        distutils.dir_util.copy_tree(data_dir, str(tmpdir))
    return tmpdir


def fix_nc_fld(fld, model):
    """Fix field read directly from NetCDF file."""
    if model.startswith("COSMO-"):
        fld[:] *= 1e-12
    elif model.startswith("IFS-"):
        pass
    else:
        raise NotImplementedError("model", model)


def read_nc_var(path, var_name, setup, model):
    # + assert isinstance(setup, CoreSetup)  # SR_TODO (if that's indeed the goal)
    with nc4.Dataset(path, "r") as fi:
        var = fi.variables[var_name]

        # Collect dimension indices
        idcs = []
        for dim_name in var.dimensions:
            if dim_name in ["rlat", "rlon", "latitude", "longitude"]:
                idx = slice(None)
            elif dim_name == "time":
                # Read all timesteps until the selected one
                if isinstance(setup, CoreSetup):
                    idx = slice(setup.core.dimensions.time + 1)
                else:
                    assert isinstance(setup.core.dimensions.time, int)
                    idx = slice(setup.core.dimensions.time + 1)
            elif dim_name in ["level", "height"]:
                if isinstance(setup, CoreSetup) or setup.core.dimensions.level is None:
                    idx = setup.core.dimensions.level
                else:
                    assert isinstance(setup.core.dimensions.level, int)
                    idx = setup.core.dimensions.level
            elif dim_name in ["nageclass", "numpoint", "noutrel", "pointspec"]:
                idx = 0
            else:
                raise NotImplementedError(f"dimension '{dim_name}'")
            idcs.append(idx)

        # Read field
        fld = var[idcs]
        assert len(fld.shape) == 3

        # Fix some issues with the input data
        fix_nc_fld(fld, model)  # SR_TMP

        # Reduce time dimension
        if setup.core.input_variable == "concentration":
            if setup.core.integrate:
                # Integrate concentration field over time
                fld = np.cumsum(fld, axis=0)
        elif setup.core.input_variable == "deposition":
            if not setup.core.integrate:
                # De-integrate deposition field over time
                fld[1:] -= fld[:-1].copy()
        else:
            raise NotImplementedError(f"variable '{setup.core.input_variable}'")
        fld = fld[-1]

        return fld
