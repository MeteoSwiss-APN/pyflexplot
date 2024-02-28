"""Fix issues with NetCDF input."""
# Standard library
from typing import Any
from typing import Mapping
from typing import Sequence
from typing import Union

# Third-party
import numpy as np

# Local
from ..utils.logging import log
from .meta_data import MetaData


class FlexPartDataFixer:
    """Fix issues with FlexPart NetCDF output."""

    cosmo_models = [
        "COSMO-2",
        "COSMO-1",
        "COSMO-E",
        "COSMO-1E",
        "COSMO-2E",
        "ICON-CH1-CTRL",
        "ICON-CH2-CTRL",
        "ICON-CH1-EPS",
        "ICON-CH2-EPS",
    ]
    ifs_models = ["IFS-HRES", "IFS-HRES-EU"]

    def __init__(self, file_reader):
        """Create an instance of ``FlexPartDataFixer``."""
        self.file_reader = file_reader

    def fix_nc_var_fld(
        self, fld: np.ndarray, model: str, var_ncattrs: Mapping[str, Any]
    ) -> None:
        unit = var_ncattrs["units"]
        if model in self.cosmo_models:
            if unit in ["ng kg-1", "1e-12 kg m-2"]:
                fact = 1.0e-12
            else:
                msg = f"model {model}: unrecognized unit '{unit}'; skip variable fixes"
                log(wrn=msg)
                return
        elif model in self.ifs_models:
            if unit in ["ng m-3", "1e-12 kg m-2"]:
                fact = 1.0e-12
            else:
                msg = f"model {model}: unrecognized unit '{unit}'; skip variable fixes"
                log(wrn=msg)
                return
        else:
            raise NotImplementedError(f"model '{model}'")
        fld[:] *= fact
        return

    # pylint: disable=R0912  # too-many-branches
    def fix_meta_data(
        self,
        model: str,
        plot_variable: str,
        integrate: bool,
        mdata: Union[MetaData, Sequence[MetaData]],
    ) -> None:
        if isinstance(mdata, Sequence):
            for mdata_i in mdata:
                self.fix_meta_data(model, plot_variable, integrate, mdata_i)
            return
        unit = str(mdata.variable.unit)
        if model in self.cosmo_models:
            if unit == "ng kg-1":
                new_unit = "Bq h m-3" if integrate else "Bq m-3"
            elif unit == "1e-12 kg m-2":
                # new_unit = "Bq h m-2" if integrate else "Bq m-2"
                new_unit = "Bq m-2"
            # SR_TMP <
            elif plot_variable in [
                "affected_area",
                "cloud_arrival_time",
                "cloud_departure_time",
            ]:
                new_unit = unit
            elif plot_variable.endswith("deposition") and unit == "N/A":
                new_unit = "Bq m-2"
            # SR_TMP >
            else:
                msg = f"model {model}: unrecognized unit '{unit}'; skip meta data fixes"
                log(wrn=msg)
                return
        elif model in self.ifs_models:
            if unit == "ng m-3":
                new_unit = "Bq h m-3" if integrate else "Bq m-3"
            elif unit == "1e-12 kg m-2":
                # new_unit = "Bq h m-2" if integrate else "Bq m-2"
                new_unit = "Bq m-2"
            # SR_TMP <
            elif plot_variable in [
                "affected_area",
                "cloud_arrival_time",
                "cloud_departure_time",
            ]:
                new_unit = unit
            elif plot_variable.endswith("deposition") and unit == "N/A":
                new_unit = "Bq m-2"
            # SR_TMP >
            else:
                msg = f"model {model}: unrecognized unit '{unit}'; skip meta data fixes"
                log(wrn=msg)
                return
        else:
            raise NotImplementedError(f"model '{model}'")
        mdata.variable.unit = new_unit

    @staticmethod
    def fix_global_grid(
        lon: np.ndarray, fld_time: np.ndarray, lon_axis: int = -1
    ) -> None:
        """Shift global grid longitudinally to fit into (-180..180) range."""
        # Check longitude dimension index
        if (
            (lon_axis < 0 and -lon_axis > len(fld_time.shape))
            or (lon_axis >= 0 and lon_axis >= len(fld_time.shape))
            or fld_time.shape[lon_axis] != lon.size
        ):
            raise ValueError(
                f"invalid idx_lon {lon_axis}"
                f" (fld_time.shape={fld_time.shape}, lon.size={lon.size})"
            )

        # Check longitudinal range
        if lon.max() - lon.min() > 360.0:
            raise ValueError(
                f"longitutinal range too large: {lon.max() - lon.min()} > 360"
            )

        # Check that longitude is evenly spaced and seamless across date line
        dlons_raw = np.r_[lon, lon[0]] - np.r_[lon[-1], lon]
        dlons = np.abs(np.stack([dlons_raw, 360 - np.abs(dlons_raw)])).min(axis=0)
        if np.unique(dlons).size > 1:
            raise ValueError(
                f"longitude not evenly spaced/seamless: {np.unique(dlons).tolist()}"
            )
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
                        f"unexpected error while shifting lon eastward by {n_shift}"
                    )
                idcs = np.arange(fld_time.shape[lon_axis] - 1)
                fld_time[:] = np.concatenate(
                    [
                        np.take(fld_time, [-1], lon_axis),
                        np.take(fld_time, idcs, lon_axis),
                    ],
                    axis=lon_axis,
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
                        f"unexpected error while shifting lon eastward by {n_shift}"
                    )
                idcs = np.arange(1, fld_time.shape[lon_axis])
                fld_time[:] = np.concatenate(
                    [
                        np.take(fld_time, idcs, lon_axis),
                        np.take(fld_time, [0], lon_axis),
                    ],
                    axis=lon_axis,
                )
            log(wrn=f"fix global data: shift westward by {n_shift} * {dlon} deg")
            return
