"""Matplotlib coordinate transformations."""
# Standard library
import warnings
from dataclasses import dataclass
from typing import overload
from typing import Tuple

# Third-party
import numpy as np
from cartopy.crs import PlateCarree
from cartopy.crs import Projection

# First-party
from srutils.iter import isiterable


@dataclass
class CoordinateTransformer:
    """Transform points between different matplotlib coordinate types.

    Args:
        trans_axes: Coordinate system of the axes, ranging from (0, 0) in the
            bottom-left corner to (1, 1) in the top-right corner.

        trans_data: Coordinate system for the data, controlled by xlim and ylim.

        proj_map: Projection of the map plot.

        proj_data: Projection of the input data.

        proj_geo (optional): Geographical projection.

        invalid_ok (bool): Don't raise an exception when encountering invalid
            coordinates.

        invalid_warn (bool): Show a warning when encountering invalid
            coordinates.

    """

    trans_axes: Projection
    trans_data: Projection
    proj_map: Projection = PlateCarree(central_longitude=0.0)
    proj_data: Projection = PlateCarree(central_longitude=0.0)
    proj_geo: Projection = PlateCarree(central_longitude=0.0)
    invalid_ok: bool = True
    invalid_warn: bool = True

    @overload
    def axes_to_geo(self, x: float, y: float) -> Tuple[float, float]:
        ...

    @overload
    def axes_to_geo(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        ...

    def axes_to_geo(self, x, y):
        """Transform from axes to geographic coordinates."""
        if isiterable(x) or isiterable(y):
            check_same_sized_iterables(x, y)
            assert isinstance(x, np.ndarray)  # mypy
            assert isinstance(y, np.ndarray)  # mypy
            # pylint: disable=E0633  # unpacking-non-sequence
            x, y = np.array([self.axes_to_geo(xi, yi) for xi, yi in zip(x, y)]).T
            return x, y

        assert isinstance(x, float)  # mypy
        assert isinstance(y, float)  # mypy
        check_valid_coords((x, y), self.invalid_ok, self.invalid_warn)

        # Axes -> Display
        xy_dis = self.trans_axes.transform((x, y))
        check_valid_coords(xy_dis, self.invalid_ok, self.invalid_warn)

        # Display -> Plot
        x_plt, y_plt = self.trans_data.inverted().transform(xy_dis)
        check_valid_coords((x_plt, y_plt), self.invalid_ok, self.invalid_warn)

        # Plot -> Geo
        xy_geo = self.proj_geo.transform_point(x_plt, y_plt, self.proj_map, trap=True)
        check_valid_coords(xy_geo, self.invalid_ok, self.invalid_warn)

        return xy_geo

    @overload
    def geo_to_axes(self, x: float, y: float) -> Tuple[float, float]:
        ...

    @overload
    def geo_to_axes(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        ...

    def geo_to_axes(self, x, y):
        """Transform from geographic to axes coordinates."""
        if isiterable(x) or isiterable(y):
            check_same_sized_iterables(x, y)
            assert isinstance(x, np.ndarray)  # mypy
            assert isinstance(y, np.ndarray)  # mypy
            # pylint: disable=E0633  # unpacking-non-sequence
            x, y = np.array([self.geo_to_axes(xi, yi) for xi, yi in zip(x, y)]).T
            return x, y

        check_valid_coords((x, y), self.invalid_ok, self.invalid_warn)

        # Geo -> Plot
        xy_plt = self.proj_map.transform_point(x, y, self.proj_geo, trap=True)
        # SR_TMP < Suppress NaN warning -- TODO investigate origin of NaNs
        # check_valid_coords(xy_plt, invalid_ok, invalid_warn)
        check_valid_coords(xy_plt, self.invalid_ok, warn=False)
        # SR_TMP >

        # Plot -> Display
        xy_dis = self.trans_data.transform(xy_plt)
        # SR_TMP < Suppress NaN warning -- TODO investigate origin of NaNs
        # check_valid_coords(xy_dis, invalid_ok, invalid_warn)
        check_valid_coords(xy_dis, self.invalid_ok, warn=False)
        # SR_TMP >

        # Display -> Axes
        xy_axs = self.trans_axes.inverted().transform(xy_dis)
        # SR_TMP < Suppress NaN warning -- TODO investigate origin of NaNs
        # check_valid_coords(xy_axs, invalid_ok, invalid_warn)
        check_valid_coords(xy_axs, self.invalid_ok, warn=False)
        # SR_TMP >

        return xy_axs

    @overload
    def data_to_geo(self, x: float, y: float) -> Tuple[float, float]:
        ...

    @overload
    def data_to_geo(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        ...

    def data_to_geo(self, x, y):
        """Transform from data to geographic coordinates."""
        if isiterable(x) or isiterable(y):
            check_same_sized_iterables(x, y)
            x, y = self.proj_geo.transform_points(self.proj_data, x, y)[:, :2].T
            return x, y
        return self.proj_geo.transform_point(x, y, self.proj_data, trap=True)

    @overload
    def geo_to_data(self, x: float, y: float) -> Tuple[float, float]:
        ...

    @overload
    def geo_to_data(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        ...

    def geo_to_data(self, x, y):
        """Transform from geographic to data coordinates."""
        if isiterable(x) or isiterable(y):
            check_same_sized_iterables(x, y)
            x, y = self.proj_data.transform_points(self.proj_geo, x, y)[:, :2].T
            return x, y
        return self.proj_data.transform_point(x, y, self.proj_geo, trap=True)


@overload
def check_valid_coords(xy: Tuple[float, float], allow, warn):
    ...


@overload
def check_valid_coords(xy: Tuple[np.ndarray, np.ndarray], allow, warn):
    ...


def check_valid_coords(xy, allow: bool, warn: bool) -> None:
    """Check that xy coordinate is valid."""
    if np.isnan(xy).any() or np.isinf(xy).any():
        if not allow:
            raise ValueError("invalid coordinates", xy)
        elif warn:
            warnings.warn(f"invalid coordinates: {xy}")


def check_same_sized_iterables(x: np.ndarray, y: np.ndarray) -> None:
    """Check that x and y are iterables of the same size."""
    if isiterable(x) and not isiterable(y):
        raise ValueError("x is iterable but y is not", (x, y))
    if isiterable(y) and not isiterable(x):
        raise ValueError("y is iterable but x is not", (x, y))
    if len(x) != len(y):
        raise ValueError("x and y differ in length", (len(x), len(y)), (x, y))
