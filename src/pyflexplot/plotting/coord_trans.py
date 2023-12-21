"""Matplotlib coordinate transformations."""
from __future__ import annotations

# Standard library
import dataclasses as dc
import warnings
from collections.abc import Sequence
from typing import overload
from typing import Tuple
from typing import Union

# Third-party
import numpy as np
import numpy.typing as npt
from cartopy.crs import PlateCarree
from cartopy.crs import Projection

# First-party
from srutils.iter import isiterable

# Custom types
FloatArray1DLike_T = Union[Sequence[float], npt.NDArray[np.float_]]


@dc.dataclass
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
    proj_map: Projection = dc.field(
        # pylint: disable=E0110  # abstract-class-instatiated (PlateCarree)
        default_factory=lambda: PlateCarree(central_longitude=0.0)
    )
    proj_data: Projection = dc.field(
        # pylint: disable=E0110  # abstract-class-instatiated (PlateCarree)
        default_factory=lambda: PlateCarree(central_longitude=0.0)
    )
    proj_geo: Projection = dc.field(
        # pylint: disable=E0110  # abstract-class-instatiated (PlateCarree)
        default_factory=lambda: PlateCarree(central_longitude=0.0)
    )
    invalid_ok: bool = True
    invalid_warn: bool = True

    @overload
    def axes_to_data(self, x: float, y: float) -> Tuple[float, float]:
        ...

    @overload
    def axes_to_data(
        self, x: FloatArray1DLike_T, y: FloatArray1DLike_T
    ) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        ...

    def axes_to_data(self, x, y):
        """Transform from axes to data coordinates."""
        # pylint: disable=E0633  # unpacking-non-sequence
        x_geo, y_geo = self.axes_to_geo(x, y)
        x, y = self.geo_to_data(x_geo, y_geo)
        return (x, y)

    @overload
    def axes_to_geo(self, x: float, y: float) -> Tuple[float, float]:
        ...

    @overload
    def axes_to_geo(
        self, x: FloatArray1DLike_T, y: FloatArray1DLike_T
    ) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        ...

    def axes_to_geo(self, x, y):
        """Transform from axes to geographic coordinates."""
        if isiterable(x) or isiterable(y):
            check_same_sized_iterables(x, y)
            if len(x) > 0:
                # pylint: disable=E0633  # unpacking-non-sequence
                x, y = np.array([self.axes_to_geo(xi, yi) for xi, yi in zip(x, y)]).T
            return (x, y)

        x = float(x)
        y = float(y)
        check_valid_coords((x, y), self.invalid_ok, self.invalid_warn)

        # Axes -> Display
        xy_dis = self.trans_axes.transform((x, y))
        check_valid_coords(xy_dis, self.invalid_ok, self.invalid_warn)

        # Display -> Plot
        x_plt, y_plt = self.trans_data.inverted().transform(xy_dis)
        check_valid_coords((x_plt, y_plt), self.invalid_ok, self.invalid_warn)

        # Plot -> Geo
        # pylint: disable=E1101  # no-member [pylint 2.7.4]
        # (pylint 2.7.4 does not support dataclasses.field)
        xy_geo = self.proj_geo.transform_point(x_plt, y_plt, self.proj_map, trap=True)
        check_valid_coords(xy_geo, self.invalid_ok, self.invalid_warn)

        x, y = xy_geo
        return (x, y)

    @overload
    def axes_to_map(self, x: float, y: float) -> Tuple[float, float]:
        ...

    @overload
    def axes_to_map(
        self, x: FloatArray1DLike_T, y: FloatArray1DLike_T
    ) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        ...

    def axes_to_map(self, x, y):
        """Transform from axes to map coordinates."""
        # pylint: disable=E0633  # unpacking-non-sequence
        x_geo, y_geo = self.axes_to_geo(x, y)
        x_map, y_map = self.geo_to_map(x_geo, y_geo)

        if (
            isinstance(self.proj_map, PlateCarree)
            and not isinstance(x_map, float)
            and len(x) == 2
        ):
            # Fix edge case where lon range (-180, 180) can invert to (180, -180)
            # during the transformation
            if np.isclose(np.abs(x_map[0]), 180, atol=0.5) and np.isclose(
                x_map[0], -x[0], atol=0.5
            ):
                x_map[0] = np.sign(x[0]) * 180.0
            if np.isclose(np.abs(x_map[1]), 180, atol=0.5) and np.isclose(
                x_map[1], -x[1], atol=0.5
            ):
                x_map[1] = np.sign(x[1]) * 180.0
        return (x_map, y_map)

    @overload
    def data_to_axes(self, x: float, y: float) -> Tuple[float, float]:
        ...

    @overload
    def data_to_axes(
        self, x: FloatArray1DLike_T, y: FloatArray1DLike_T
    ) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        ...

    def data_to_axes(self, x, y):
        """Transform from data to axes coordinates."""
        # pylint: disable=E0633  # unpacking-non-sequence
        x_geo, y_geo = self.data_to_geo(x, y)
        x, y = self.geo_to_axes(x_geo, y_geo)
        return (x, y)

    @overload
    def data_to_geo(self, x: float, y: float) -> Tuple[float, float]:
        ...

    @overload
    def data_to_geo(
        self, x: FloatArray1DLike_T, y: FloatArray1DLike_T
    ) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        ...

    def data_to_geo(self, x, y):
        """Transform from data to geographic coordinates."""
        if isiterable(x) or isiterable(y):
            check_same_sized_iterables(x, y)
            if len(x) > 0:
                # pylint: disable=E1101  # no-member [pylint 2.7.4]
                # (pylint 2.7.4 does not support dataclasses.field)
                x, y = self.proj_geo.transform_points(self.proj_data, x, y)[:, :2].T
            return (x, y)
        # pylint: disable=E1101  # no-member [pylint 2.7.4]
        # (pylint 2.7.4 does not support dataclasses.field)
        x, y = self.proj_geo.transform_point(x, y, self.proj_data, trap=True)
        return (x, y)

    @overload
    def data_to_map(self, x: float, y: float) -> Tuple[float, float]:
        ...

    @overload
    def data_to_map(
        self, x: FloatArray1DLike_T, y: FloatArray1DLike_T
    ) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        ...

    def data_to_map(self, x, y):
        """Transform from data to map coordinates."""
        # pylint: disable=E0633  # unpacking-non-sequence
        x_geo, y_geo = self.data_to_geo(x, y)
        x, y = self.geo_to_map(x_geo, y_geo)
        return (x, y)

    @overload
    def geo_to_axes(self, x: float, y: float) -> Tuple[float, float]:
        ...

    @overload
    def geo_to_axes(
        self, x: FloatArray1DLike_T, y: FloatArray1DLike_T
    ) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        ...

    def geo_to_axes(self, x, y):
        """Transform from geographic to axes coordinates."""
        if isiterable(x) or isiterable(y):
            check_same_sized_iterables(x, y)
            if len(x) > 0:
                # pylint: disable=E0633  # unpacking-non-sequence
                x, y = np.array([self.geo_to_axes(xi, yi) for xi, yi in zip(x, y)]).T
            return (x, y)

        check_valid_coords((x, y), self.invalid_ok, self.invalid_warn)

        # Geo -> Plot
        # pylint: disable=E1101  # no-member [pylint 2.7.4]
        # (pylint 2.7.4 does not support dataclasses.field)
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

        x, y = xy_axs
        return (x, y)

    @overload
    def geo_to_data(self, x: float, y: float) -> Tuple[float, float]:
        ...

    @overload
    def geo_to_data(
        self, x: FloatArray1DLike_T, y: FloatArray1DLike_T
    ) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        ...

    def geo_to_data(self, x, y):
        """Transform from geographic to data coordinates."""
        if isiterable(x) or isiterable(y):
            check_same_sized_iterables(x, y)
            if len(x) > 0:
                # pylint: disable=E1101  # no-member [pylint 2.7.4]
                # (pylint 2.7.4 does not support dataclasses.field)
                x, y = self.proj_data.transform_points(self.proj_geo, x, y)[:, :2].T
            return (x, y)
        # pylint: disable=E1101  # no-member [pylint 2.7.4]
        # (pylint 2.7.4 does not support dataclasses.field)
        x, y = self.proj_data.transform_point(x, y, self.proj_geo, trap=True)
        return (x, y)

    @overload
    def geo_to_map(self, x: float, y: float) -> Tuple[float, float]:
        ...

    @overload
    def geo_to_map(
        self, x: FloatArray1DLike_T, y: FloatArray1DLike_T
    ) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        ...

    def geo_to_map(self, x, y):
        """Transform from geographical to map coordinates."""
        if isiterable(x) or isiterable(y):
            check_same_sized_iterables(x, y)
            if len(x) > 0:
                # pylint: disable=E1101  # no-member [pylint 2.7.4]
                # (pylint 2.7.4 does not support dataclasses.field)
                x, y = self.proj_map.transform_points(self.proj_geo, x, y)[:, :2].T
            return (x, y)
        # pylint: disable=E1101  # no-member [pylint 2.7.4]
        # (pylint 2.7.4 does not support dataclasses.field)
        x, y = self.proj_map.transform_point(x, y, self.proj_geo, trap=True)
        return (x, y)

    @overload
    def map_to_axes(self, x: float, y: float) -> Tuple[float, float]:
        ...

    @overload
    def map_to_axes(
        self, x: FloatArray1DLike_T, y: FloatArray1DLike_T
    ) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        ...

    def map_to_axes(self, x, y):
        """Transform from map to axes coordinates."""
        # pylint: disable=E0633  # unpacking-non-sequence
        x_geo, y_geo = self.map_to_geo(x, y)
        x, y = self.geo_to_axes(x_geo, y_geo)
        return (x, y)

    @overload
    def map_to_data(self, x: float, y: float) -> Tuple[float, float]:
        ...

    @overload
    def map_to_data(
        self, x: FloatArray1DLike_T, y: FloatArray1DLike_T
    ) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        ...

    def map_to_data(self, x, y):
        """Transform from map to data coordinates."""
        # pylint: disable=E0633  # unpacking-non-sequence
        x_geo, y_geo = self.map_to_geo(x, y)
        x, y = self.geo_to_data(x_geo, y_geo)
        return (x, y)

    @overload
    def map_to_geo(self, x: float, y: float) -> Tuple[float, float]:
        ...

    @overload
    def map_to_geo(
        self, x: FloatArray1DLike_T, y: FloatArray1DLike_T
    ) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        ...

    def map_to_geo(self, x, y):
        """Transform from map to geographical coordinates."""
        if isiterable(x) or isiterable(y):
            check_same_sized_iterables(x, y)
            if len(x) > 0:
                # pylint: disable=E1101  # no-member [pylint 2.7.4]
                # (pylint 2.7.4 does not support dataclasses.field)
                x, y = self.proj_geo.transform_points(self.proj_map, x, y)[:, :2].T
            return (x, y)
        # pylint: disable=E1101  # no-member [pylint 2.7.4]
        # (pylint 2.7.4 does not support dataclasses.field)
        x, y = self.proj_geo.transform_point(x, y, self.proj_map, trap=True)
        return (x, y)

    def __repr__(self) -> str:
        """Return a string representation."""
        return "\n".join(
            [
                f"{type(self).__name__}(",
                f"  trans_axes=<{type(self.trans_axes).__name__} object>,",
                f"  trans_data=<{type(self.trans_data).__name__} object>,",
                f"  proj_map=<{type(self.proj_map).__name__} object>,",
                f"  proj_data=<{type(self.proj_data).__name__} object>,",
                f"  proj_geo=<{type(self.proj_geo).__name__} object>,",
                f"  invalid_ok={self.invalid_ok},",
                f"  invalid_warn={self.invalid_ok},",
                ")",
            ]
        )


@overload
def check_valid_coords(xy: Tuple[float, float], allow, warn):
    ...


@overload
def check_valid_coords(xy: Tuple[FloatArray1DLike_T, FloatArray1DLike_T], allow, warn):
    ...


def check_valid_coords(xy, allow: bool, warn: bool) -> None:
    """Check that xy coordinate is valid."""
    if np.isnan(xy).any() or np.isinf(xy).any():
        if not allow:
            raise ValueError("invalid coordinates", xy)
        elif warn:
            warnings.warn(f"invalid coordinates: {xy}")


def check_same_sized_iterables(x: FloatArray1DLike_T, y: FloatArray1DLike_T) -> None:
    """Check that x and y are iterables of the same size."""
    if isiterable(x) and not isiterable(y):
        raise ValueError("x is iterable but y is not", (x, y))
    if isiterable(y) and not isiterable(x):
        raise ValueError("y is iterable but x is not", (x, y))
    if len(x) != len(y):
        raise ValueError("x and y differ in length", (len(x), len(y)), (x, y))
