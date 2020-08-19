# -*- coding: utf-8 -*-
"""
Fix issues with NetCDF input.
"""
# Standard library
from typing import Any
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

    def __init__(self, file_reader):
        self.file_reader = file_reader

    def fix_nc_var_fld(
        self, fld: np.ndarray, model: str, var_ncattrs: Mapping[str, Any]
    ) -> None:
        if model in ["COSMO-2", "COSMO-1"]:
            self._fix_nc_var_cosmo(fld, var_ncattrs)
        elif model in ["IFS", "IFS-HRES"]:
            pass
        else:
            raise NotImplementedError("model", model)

    def fix_meta_data(
        self, model: str, integrate: bool, mdata: Union[MetaData, Sequence[MetaData]],
    ) -> None:
        if model in ["COSMO-2", "COSMO-1"]:
            self._fix_meta_data_cosmo(integrate, mdata)
        elif model in ["IFS", "IFS-HRES"]:
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
        if unit in ["ng kg-1", "1e-12 kg m-2"]:
            fact = 1.0e-12
        else:
            return
        fld[:] *= fact

    def _fix_meta_data_cosmo(
        self, integrate: bool, mdata: Union[MetaData, Sequence[MetaData]]
    ) -> None:
        if isinstance(mdata, Sequence):
            for mdata_i in mdata:
                self._fix_meta_data_cosmo(integrate, mdata_i)
            return
        assert isinstance(mdata, MetaData)  # mypy
        unit = str(mdata.variable.unit)
        if unit == "ng kg-1":
            new_unit = "Bq h m-3" if integrate else "Bq m-3"
        elif unit == "1e-12 kg m-2":
            new_unit = "Bq h m-2" if integrate else "Bq m-2"
        else:
            return
        mdata.variable.unit = new_unit
