"""Data structures."""
# Standard library
import dataclasses as dc
import warnings
from typing import Any
from typing import Dict
from typing import Iterator
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

# Third-party
import numpy as np
from cartopy.crs import PlateCarree
from cartopy.crs import Projection
from cartopy.crs import RotatedPole

# First-party
from srutils.str import join_multilines

# Local
from ..setups.model_setup import ModelSetup
from ..setups.plot_panel_setup import PlotPanelSetup
from ..setups.plot_setup import PlotSetup
from ..utils.exceptions import ArrayDimensionError
from ..utils.exceptions import FieldAllNaNError
from ..utils.exceptions import InconsistentArrayShapesError
from ..utils.formatting import format_ens_file_path
from ..utils.summarize import summarizable
from ..utils.summarize import summarize
from .meta_data import MetaData


@summarizable(
    summarize=lambda self: summarize(
        {
            "type": type(self).__name__,
            "fld": {
                "dtype": str(self.fld.dtype),
                "shape": self.fld.shape,
                "nanmin": np.nanmin(self.fld),
                "nanmean": np.nanmean(self.fld),
                "nanmedian": np.nanmedian(self.fld),
                "nanmax": np.nanmax(self.fld),
                "nanmin_nonzero": np.nanmin(np.where(self.fld == 0, np.nan, self.fld)),
                "nanmean_nonzero": np.nanmean(
                    np.where(self.fld == 0, np.nan, self.fld)
                ),
                "nanmedian_nonzero": np.nanmedian(
                    np.where(self.fld == 0, np.nan, self.fld)
                ),
                "nanmax_nonzero": np.nanmax(np.where(self.fld == 0, np.nan, self.fld)),
                "n_nan": np.count_nonzero(np.isnan(self.fld)),
                "n_zero": np.count_nonzero(self.fld == 0),
            },
            "lat": {
                "dtype": str(self.lat.dtype),
                "shape": self.lat.shape,
                "min": self.lat.min(),
                "max": self.lat.max(),
                "start": self.lat[:10].tolist(),
                "end": self.lat[-10:].tolist(),
            },
            "lon": {
                "dtype": str(self.lon.dtype),
                "shape": self.lon.shape,
                "min": self.lon.min(),
                "max": self.lon.max(),
                "start": self.lon[:10].tolist(),
                "end": self.lon[-10:].tolist(),
            },
            "mdata": self.mdata,
            "time_props": self.time_props,
            "panel_setup": self.panel_setup,
            "model_setup": self.model_setup,
            "proj": self.proj,
        }
    )
)
# pylint: disable=R0902  # too-many-instance-attributes
class Field:
    """FLEXPART field on rotated-pole grid."""

    def __init__(
        self,
        fld: np.ndarray,
        lat: np.ndarray,
        lon: np.ndarray,
        *,
        mdata: MetaData,
        panel_setup: PlotPanelSetup,
        model_setup: ModelSetup,
        time_props: "FieldTimeProperties",
    ) -> None:
        """Create an instance of ``Field``.

        Args:
            fld: Field array (2D) with dimensions (lat, lon).

            lat: Latitude array (1D).

            lon: Longitude array (1D).

            mdata: Meta data for plot for labels etc.

            panel_setup: Setup of plot panel in which the field is plotted.

            model_setup: Setup of the model.

            time_props: Properties of the field across all time steps.

        """
        self.fld: np.ndarray = fld
        self.lat: np.ndarray = lat
        self.lon: np.ndarray = lon
        self.mdata: MetaData = mdata
        self.panel_setup: PlotPanelSetup = panel_setup
        self.model_setup: ModelSetup = model_setup
        self.time_props: "FieldTimeProperties" = time_props
        try:
            self.check_consistency()
        except Exception as e:
            raise ValueError(f"{type(e).__name__}: {e}") from e

        # pylint: disable=E0110  # abstract-class-instatiated (PlateCarree/RotatedPole)
        self.proj: Projection
        if self.mdata.simulation.grid_is_rotated:
            self.proj = RotatedPole(
                pole_latitude=self.mdata.simulation.grid_north_pole_lat,
                pole_longitude=self.mdata.simulation.grid_north_pole_lon,
            )
        else:
            self.proj = PlateCarree(central_longitude=0.0)
        self._proj_geo = PlateCarree(central_longitude=0.0)

    def check_consistency(self):
        """Check consistency of field, dimensions, etc."""
        # Check dimensionalities
        for name, arr, ndim in [
            ("fld", self.fld, 2),
            ("lat", self.lat, 1),
            ("lon", self.lon, 1),
        ]:
            shape = arr.shape
            if len(shape) != ndim:
                raise ArrayDimensionError(f"{name}: len({shape}) != {ndim}")

        # Check consistency
        grid_shape = (self.lat.size, self.lon.size)
        if self.fld.shape[-2:] != grid_shape:
            raise InconsistentArrayShapesError(f"{self.fld.shape} != {grid_shape}")

    def locate_max(self) -> Tuple[float, float]:
        """Find location of field maximum in geographical coordinates."""
        if np.isnan(self.fld).all():
            raise FieldAllNaNError(self.fld.shape)
        assert len(self.fld.shape) == 2  # pylint
        # pylint: disable=W0632  # unbalanced-tuple-unpacking
        jmax, imax = np.unravel_index(np.nanargmax(self.fld), self.fld.shape)
        p_lon, p_lat = self._proj_geo.transform_point(
            self.lon[imax], self.lat[jmax], self.proj
        )
        return (p_lat, p_lon)

    def __repr__(self):
        lines = [
            f"fld=array[shape={self.fld.shape}, dtype={self.fld.dtype}],",
            f"lat=array[shape={self.lat.shape}, dtype={self.lat.dtype}],",
            f"lon=array[shape={self.lon.shape}, dtype={self.lon.dtype}],",
            f"mdata={self.mdata},",
            f"time_stats={self.time_props},",
            f"var_setups={self.panel_setup},",
            f"projs={self.proj},",
        ]
        body = join_multilines(lines, indent=2)
        return "\n".join([f"{type(self).__name__}(", body, ")"])


@summarizable
@dc.dataclass
class FieldStats:
    min: float = np.nan
    max: float = np.nan
    mean: float = np.nan
    median: float = np.nan

    @classmethod
    def create(cls, arr: np.ndarray) -> "FieldStats":
        arr = np.where(np.isfinite(arr), arr, np.nan)
        # Avoid zero-size errors below
        if arr.size == 0:
            return cls()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning, message="All-NaN slice encountered"
            )
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning, message="Mean of empty slice"
            )
            return cls(
                min=np.nanmin(arr),
                max=np.nanmax(arr),
                mean=np.nanmean(arr),
                median=np.nanmedian(arr),
            )


@summarizable(attrs=["stats", "stats_nz"])
class FieldTimeProperties:
    """Properties of a 2D field over time."""

    def __init__(self, arr: np.ndarray) -> None:
        """Create an instance of ``FieldTimeProperties``."""
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message="Invalid value encountered in greater",
            )
            arr_nz = np.where(arr > 0, arr, np.nan)
        self.stats = FieldStats.create(arr)
        self.stats_nz = FieldStats.create(arr_nz)
        self.mask: np.ndarray = (~np.isnan(arr)).sum(axis=0) > 0
        self.mask_nz: np.ndarray = (~np.isnan(arr_nz)).sum(axis=0) > 0


@dc.dataclass
class FieldGroupAttrs:
    """Attributes of a ``FieldGroup`` instance."""

    raw_path: str
    paths: Sequence[str]
    ens_member_ids: Optional[Sequence[int]]

    def format_path(self) -> str:
        return format_ens_file_path(self.raw_path, self.ens_member_ids)


@dc.dataclass
class FieldGroup:
    """A group of related ``Field`` objects."""

    fields: Sequence[Field]
    plot_setup: PlotSetup
    attrs: FieldGroupAttrs

    def __len__(self) -> int:
        return len(self.fields)

    def __iter__(self) -> Iterator[Field]:
        return iter(self.fields)

    @classmethod
    def create(
        cls,
        fields: Sequence[Field],
        plot_setup: Union[PlotSetup, Dict[str, Any]],
        attrs: Union[FieldGroupAttrs, Dict[str, Any]],
    ) -> "FieldGroup":
        if not isinstance(plot_setup, PlotSetup):
            plot_setup = PlotSetup.create(plot_setup)
        if not isinstance(attrs, FieldGroupAttrs):
            attrs = FieldGroupAttrs(**attrs)
        return cls(fields, plot_setup, attrs)
