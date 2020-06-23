# -*- coding: utf-8 -*-
"""
Fix issues with NetCDF input.
"""
# Standard library
from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import Sequence
from typing import Tuple
from typing import Union

# Third-party
import numpy as np

# Local
from .logging import log
from .meta_data import get_integr_type
from .meta_data import MetaData


class FlexPartDataFixer:
    """Fix issues with FlexPart NetCDF output."""

    possible_var_names: List[Union[str, Tuple[str, ...]]] = [
        "Cs-137",
        "I-131a",
        ("Cs-137", "I-131a"),
        ("I-131a", "Cs-137"),
    ]
    conversion_factor_by_unit: Dict[str, float] = {
        "ng kg-1": 1.0e-12,
        "1e-12 kg m-2": 1.0e-12,
    }

    def __init__(self, file_reader):
        self.file_reader = file_reader

    def fix_nc_var_fld(
        self, fld: np.ndarray, model: str, var_ncattrs: Mapping[str, Any]
    ) -> None:
        if model in ["cosmo2", "cosmo1"]:
            self._fix_nc_var_cosmo(fld, var_ncattrs)
        elif model == "ifs":
            pass
        else:
            raise NotImplementedError("model", model)

    def fix_meta_data(
        self, model: str, mdata: Union[MetaData, Sequence[MetaData]],
    ) -> None:
        if model in ["cosmo2", "cosmo1"]:
            self._fix_meta_data_cosmo(mdata)
        elif model == "ifs":
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
        unit = var_ncattrs["units"]
        if name not in self.possible_var_names:
            raise NotImplementedError("input_variable", name)
        try:
            fact = self.conversion_factor_by_unit[unit]
        except KeyError:
            raise NotImplementedError("conversion factor", name, unit)
        fld[:] *= fact

    def _fix_meta_data_cosmo(self, mdata):
        if isinstance(mdata, Sequence):
            for mdata_i in mdata:
                self._fix_meta_data_cosmo(mdata_i)
            return
        assert isinstance(mdata, MetaData)  # mypy

        # Variable unit
        var_name = mdata.species_name.value
        if var_name not in self.possible_var_names:
            raise NotImplementedError("input_variable", var_name)
        integr_type = get_integr_type(mdata.setup)
        old_unit = mdata.variable_unit.value
        new_unit = "Bq"
        if integr_type == "mean":
            pass
        elif integr_type in ["sum", "accum"]:
            new_unit += " h"
        else:
            raise NotImplementedError(
                "integration type for variable", integr_type, var_name
            )
        if old_unit == "ng kg-1":
            new_unit += " m-3"
        elif old_unit == "1e-12 kg m-2":
            new_unit += " m-2"
        else:
            raise NotImplementedError("unit for variable", old_unit, var_name)
        mdata.variable_unit.value = new_unit
