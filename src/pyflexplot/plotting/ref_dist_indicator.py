"""Reference distance indicator for map plot."""
# Standard library
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import Optional

# Third-party
import geopy.distance
import matplotlib as mpl
from cartopy.crs import Projection  # type: ignore
from matplotlib.axes import Axes

# Local
from ..utils.exceptions import MaxIterationError
from ..utils.summarize import summarizable


@summarizable
@dataclass
class RefDistIndConfig:
    """Configuration of ``ReferenceDistanceIndicator``.

    Args:
        dist: Reference distance in ``unit``.

        font_size: Font size of distance label.

        min_w_box: Minimum width of the box.

        line_width: Line width of distance indicator.

        pos: Position of reference distance indicator box (corners of the plot).
            Options: "tl" (top-left), "tr" (top-right), "bl" (bottom-left), "br"
            (bottom-right).

        unit: Unit of reference distance ``val``.

    """

    dist: int = 100
    font_size: float = 11.0
    min_w_box: float = 0.075
    line_width: float = 2.0
    pos: str = "bl"
    unit: str = "km"

    def scale(self, factor: float) -> "RefDistIndConfig":
        kwargs: Dict[str, Any] = {
            **asdict(self),
            "font_size": self.font_size * factor,
            "line_width": self.line_width * factor,
        }
        return type(self)(**kwargs)


# pylint: disable=R0902  # too-many-instance-attributes
class ReferenceDistanceIndicator:
    """Reference distance indicator on a map plot."""

    def __init__(
        self, ax: Axes, axes_to_geo: Projection, config: RefDistIndConfig, zorder: int
    ) -> None:
        """Create an instance of ``ReferenceDistanceIndicator``.

        Args:
            ax: Axes.

            axes_to_geo: Projection to convert from axes to geographical
                coordinates.

            config: Configuration.

            zorder: Vertical order in plot.

        """
        self.config: RefDistIndConfig = config

        # Position in the plot (one of the corners)
        pos_choices = ["tl", "bl", "br", "tr"]
        if config.pos not in pos_choices:
            s_choices = ", ".join([f"'{p}'" for p in pos_choices])
            raise ValueError(f"invalid position '{config.pos}' (choices: {s_choices}")
        self.pos_y: str = config.pos[0]
        self.pos_x: str = config.pos[1]

        self.h_box: float = 0.06
        self.xpad_box: float = 0.2 * self.h_box
        self.ypad_box: float = 0.2 * self.h_box

        self._calc_box_y_params()
        self._calc_box_x_params(axes_to_geo)

        self._add_to(ax, zorder)

    @property
    def x_text(self) -> float:
        return self.x0_box + 0.5 * self.w_box

    @property
    def y_text(self) -> float:
        return self.y1_box - self.ypad_box

    def _add_to(self, ax: Axes, zorder: int) -> None:

        # Draw box
        ax.add_patch(
            mpl.patches.Rectangle(
                xy=(self.x0_box, self.y0_box),
                width=self.w_box,
                height=self.h_box,
                transform=ax.transAxes,
                zorder=zorder,
                fill=True,
                facecolor="white",
                edgecolor="black",
                linewidth=1,
            )
        )

        # Draw line
        ax.plot(
            [self.x0_line, self.x1_line],
            [self.y_line] * 2,
            transform=ax.transAxes,
            zorder=zorder,
            linestyle="-",
            linewidth=self.config.line_width,
            color="k",
        )

        # Add label
        ax.text(
            x=self.x_text,
            y=self.y_text,
            s=f"{self.config.dist:g} {self.config.unit}",
            transform=ax.transAxes,
            zorder=zorder,
            ha="center",
            va="top",
            fontsize=self.config.font_size,
        )

    def _calc_box_y_params(self) -> None:
        if self.pos_y == "t":
            self.y1_box = 1.0
            self.y0_box = self.y1_box - self.h_box
        elif self.pos_y == "b":
            self.y0_box = 0.0
            self.y1_box = self.y0_box + self.h_box
        else:
            raise ValueError(f"invalid y-position '{self.pos_y}'")
        self.y_line = self.y0_box + self.ypad_box

    def _calc_box_x_params(self, axes_to_geo) -> None:
        if self.pos_x == "l":
            self.x0_box = 0.0
            self.x0_line = self.x0_box + self.xpad_box
            self.x1_line = self._calc_horiz_dist(self.x0_line, "east", axes_to_geo)
        elif self.pos_x == "r":
            self.x1_box = 1.0
            self.x1_line = self.x1_box - self.xpad_box
            self.x0_line = self._calc_horiz_dist(self.x1_line, "west", axes_to_geo)
        else:
            raise ValueError(f"invalid x-position '{self.pos_x}'")
        w_line = self.x1_line - self.x0_line
        self.w_box = max(self.config.min_w_box, w_line + 2 * self.xpad_box)
        if self.w_box > 1.0:
            raise Exception(f"ref dist indicator box too wide: {self.w_box} > 1.0")
        if self.pos_x == "l":
            self.x1_box = self.x0_box + self.w_box
        elif self.pos_x == "r":
            self.x0_box = self.x1_box - self.w_box
        if self.w_box == self.config.min_w_box:
            x_box_center = self.x0_box + 0.5 * self.w_box
            self.x0_line = x_box_center - 0.5 * w_line
            self.x1_line = x_box_center + 0.5 * w_line

    def _calc_horiz_dist(
        self, x0: float, direction: str, axes_to_geo: Projection
    ) -> float:
        calculator = MapDistanceCalculator(axes_to_geo, self.config.unit)
        x1, _, _ = calculator.run(x0, self.y_line, self.config.dist, direction)
        return x1


# SR_TODO Refactor to reduce number of instance attributes!
# pylint: disable=R0902  # too-many-instance-attributes
class MapDistanceCalculator:
    """Calculate geographic distance along a line on a map plot."""

    def __init__(self, axes_to_geo, unit="km", p=0.001):
        """Initialize an instance of MapDistanceCalculator.

        Args:
            axes_to_geo (callable): Function to transform a point
                (x, y) from axes coordinates to geographic coordinates (x, y
                may be arrays).

            unit (str, optional): Unit of ``dist``. Defaults to 'km'.

            p (float, optional): Required precision as a fraction of ``dist``.
                Defaults to 0.001.

        """
        self.axes_to_geo = axes_to_geo
        self.unit = unit
        self.p = p

        # Declare attributes
        self.dist: Optional[float]
        self.direction: str
        self._step_ax_rel: float
        self._dx_unit: int
        self._dy_unit: int

    def reset(self, dist=None, dir_=None):
        self._set_dist(dist)
        self._set_dir(dir_)
        self._step_ax_rel = 0.1

    def _set_dist(self, dist):
        """Check and set the target distance."""
        self.dist = dist
        if dist is not None:
            if dist <= 0.0:
                raise ValueError(f"dist not above zero: {dist}")

    def _set_dir(self, direction):
        """Check and set the direction."""
        if direction is not None:
            if direction in "east":
                self._dx_unit = 1
                self._dy_unit = 0
            elif direction in "west":
                self._dx_unit = -1
                self._dy_unit = 0
            elif direction in "north":
                self._dx_unit = 0
                self._dy_unit = 1
            elif direction in "south":
                self._dx_unit = 0
                self._dy_unit = -1
            else:
                raise NotImplementedError(f"direction='{direction}'")
        self.direction = direction

    # pylint: disable=R0914  # too-many-locals
    def run(self, x0, y0, dist, dir_="east"):
        """Measure geo. distance along a straight line on the plot."""
        self.reset(dist=dist, dir_=dir_)

        step_ax_rel = 0.1
        refine_quot = 3

        dist0 = 0.0
        x, y = x0, y0

        iter_max = 9999
        for _ in range(iter_max):

            # Step away from point until target distance exceeded
            path, dists = self._overstep(x, y, dist0, step_ax_rel)

            # Select the largest distance
            i_sel = -1
            dist = dists[i_sel]
            # Note: We could also check `dists[-2]` and return that
            # if it were sufficiently close (based on the relative
            # error) and closer to the targest dist than `dist[-1]`.

            # Compute the relative error
            err = abs(dist - self.dist) / self.dist
            # Note: `abs` only necessary if `dist` could be `dist[-2]`

            if err < self.p:
                # Error sufficiently small: We're done!
                x1, y1 = path[i_sel]
                break

            dist0 = dists[-2]
            x, y = path[-2]
            step_ax_rel /= refine_quot

        else:
            raise MaxIterationError(iter_max)

        return x1, y1, dist

    def _overstep(self, x0_ax, y0_ax, dist0, step_ax_rel):
        """Move stepwise until the target distance is exceeded."""
        path_ax = [(x0_ax, y0_ax)]
        dists = [dist0]

        # Transform starting point to geographical coordinates
        x0_geo, y0_geo = self.axes_to_geo(x0_ax, y0_ax)

        # Step away until target distance exceeded
        dist = 0.0
        x1_ax, y1_ax = x0_ax, y0_ax
        while dist < self.dist - dist0:

            # Move one step farther away from starting point
            x1_ax += self._dx_unit * step_ax_rel
            y1_ax += self._dy_unit * step_ax_rel

            # Transform current point to gegographical coordinates
            x1_geo, y1_geo = self.axes_to_geo(x1_ax, y1_ax)

            # Compute geographical distance from starting point
            dist = self.comp_dist(x0_geo, y0_geo, x1_geo, y1_geo)

            path_ax.append((x1_ax, y1_ax))
            dists.append(dist + dist0)

        return path_ax, dists

    def comp_dist(self, lon0, lat0, lon1, lat1):
        """Compute the great circle distance between two points."""
        dist_obj = geopy.distance.great_circle((lat0, lon0), (lat1, lon1))
        if self.unit == "km":
            return dist_obj.kilometers
        else:
            raise NotImplementedError(f"great circle distance in {self.unit}")
