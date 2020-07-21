# -*- coding: utf-8 -*-
"""
Fix issues with NetCDF input.
"""
# Standard library
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Mapping
from typing import Sequence
from typing import Union

# Third-party
import numpy as np

# Local
from .meta_data import MetaData
from .species import get_species
from .utils.logging import log


class FlexPartDataFixer:
    """Fix issues with FlexPart NetCDF output."""

    unit_conversion_factors: Dict[str, float] = {
        "ng kg-1": 1.0e-12,
        "1e-12 kg m-2": 1.0e-12,
    }
    unit_replacement_names: Dict[str, str] = {
        "ng kg-1": "Bq m-3",
        "1e-12 kg m-2": "Bq m-2",
    }
    valid_units: List[str] = list(unit_replacement_names.values())

    def __init__(self, file_reader):
        self.file_reader = file_reader

    def fix_nc_var_fld(
        self, fld: np.ndarray, model: str, var_ncattrs: Mapping[str, Any]
    ) -> None:
        if model in ["cosmo2", "cosmo1"]:
            self._fix_nc_var_cosmo(fld, var_ncattrs)
        elif model in ["ifs", "ifs-hres"]:
            pass
        else:
            raise NotImplementedError("model", model)

    def fix_meta_data(
        self, model: str, mdata: Union[MetaData, Sequence[MetaData]],
    ) -> None:
        if model in ["cosmo2", "cosmo1"]:
            self._fix_meta_data_cosmo(mdata)
        elif model in ["ifs", "ifs-hres"]:
            pass
        else:
            raise NotImplementedError("model", model)

    def fix_global_grid(self, lon, fld_time, idx_lon=-1):
        """Shift global grid longitudinally to fit into (-180..180) range."""

        # Check longitude dimension index
        if (
            (idx_lon < 0 and -idx_lon > len(fld_time.shape))
            or (idx_lon >= 0 and idx_lon >= len(fld_time.shape))
            or fld_time.shape[idx_lon] != lon.size
        ):
            raise ValueError("invalid idx_lon", idx_lon, fld_time.shape, lon.size)

        # Check longitudinal range
        if lon.max() - lon.min() > 360.0:
            raise ValueError("longitutinal range too large", lon.max() - lon.min())

        # Check that longitude is evenly spaced and seamless across date line
        dlons_raw = np.r_[lon, lon[0]] - np.r_[lon[-1], lon]
        dlons = np.abs(np.stack([dlons_raw, 360 - np.abs(dlons_raw)])).min(axis=0)
        if np.unique(dlons).size > 1:
            raise ValueError("longitude not evenly spaced/seamless", np.unique(dlons))
        dlon = next(iter(dlons))

        # Shift the grid
        if lon[-1] > 180.0:
            # Eastward shift
            n_shift = 0
            while lon[-1] > 180.0:
                n_shift += 1
                lon[:] = np.r_[lon[0] - dlon, lon[:-1]]
                if lon[0] < -180.0 or n_shift >= lon.size:
                    raise Exception(
                        "unexpected error while shifting lon eastward", lon, n_shift,
                    )
                idcs = np.arange(fld_time.shape[idx_lon] - 1)
                fld_time[:] = np.concatenate(
                    [
                        np.take(fld_time, [-1], idx_lon),
                        np.take(fld_time, idcs, idx_lon),
                    ],
                    axis=idx_lon,
                )
            log(wrn=f"fix global data: shift eastward by {n_shift} * {dlon} deg")
            return

        elif lon[0] < -180.0:
            # Westward shift
            n_shift = 0
            while lon[0] < -180.0:
                n_shift += 1
                lon[:] = np.r_[lon[1:], lon[-1] + dlon]
                if lon[-1] < -180.0 or n_shift >= lon.size:
                    raise Exception(
                        "unexpected error while shifting lon eastward", lon, n_shift,
                    )
                idcs = np.arange(1, fld_time.shape[idx_lon])
                fld_time[:] = np.concatenate(
                    [
                        np.take(fld_time, idcs, idx_lon),
                        np.take(fld_time, [0], idx_lon),
                    ],
                    axis=idx_lon,
                )
            log(wrn=f"fix global data: shift westward by {n_shift} * {dlon} deg")
            return

    def _fix_nc_var_cosmo(
        self, fld: np.ndarray, var_ncattrs: Mapping[str, Any]
    ) -> None:
        name = var_ncattrs["long_name"]
        if name.endswith("_dry_deposition") or name.endswith("_wet_deposition"):
            name = name.split("_")[0]
        try:
            get_species(name=name)
        except ValueError:
            log(wrn=f"unrecognized variable name '{name}'; skip input data fixes")
            return
        unit = var_ncattrs["units"]
        if unit in self.valid_units:
            pass
        else:
            try:
                fact = self.unit_conversion_factors[unit]
            except KeyError:
                raise NotImplementedError("conversion factor", name, unit)
        fld[:] *= fact

    def _fix_meta_data_cosmo(self, mdata: Union[MetaData, Sequence[MetaData]]) -> None:
        if isinstance(mdata, Sequence):
            for mdata_i in mdata:
                self._fix_meta_data_cosmo(mdata_i)
            return
        assert isinstance(mdata, MetaData)  # mypy
        name = mdata.species_name.value
        unit = mdata.variable_unit.value
        try:
            get_species(name=name)
        except ValueError:
            log(wrn=f"unrecognized variable name '{name}'; skip input meta data fixes")
            return
        if unit in self.valid_units:
            pass
        else:
            try:
                new_unit = self.unit_replacement_names[cast(str, unit)]
            except KeyError:
                raise NotImplementedError("unit for variable", unit, name)
            mdata.variable_unit.value = new_unit
