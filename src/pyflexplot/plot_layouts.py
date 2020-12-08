"""Plot layouts."""
# Standard library
from dataclasses import dataclass
from typing import Union

# Local
from .utils.typing import RectType

BoxedPlotLayoutType = Union[
    "BoxedPlotLayoutVintage",
    "BoxedPlotLayoutDeterministic",
    "BoxedPlotLayoutEnsemble",
]


# SR_TODO: Find a cleaner solution to define these layouts!
# This is again a case of misuse of classes...


@dataclass
# pylint: disable=R0902  # too-many-instance-attributes
class BoxedPlotLayoutVintage:
    aspect: float
    x0_tot: float = 0.0
    x1_tot: float = 1.0
    y0_tot: float = 0.0
    y1_tot: float = 1.0
    y_pad: float = 0.02
    h_top: float = 0.08
    w_right: float = 0.2
    h_rigbot: float = 0.42
    h_bottom: float = 0.05

    def __post_init__(self):
        self._x_pad: float = self.y_pad / self.aspect
        self._h_tot: float = self.y1_tot - self.y0_tot
        self._y1_top: float = self.y1_tot
        self._y0_top: float = self._y1_top - self.h_top
        self._y1_center: float = self._y0_top - self.y_pad
        self._y0_center: float = self.y0_tot + self.h_bottom
        self._h_center: float = self._y1_center - self._y0_center
        self._w_tot: float = self.x1_tot - self.x0_tot
        self._x1_right: float = self.x1_tot
        self._x0_right: float = self._x1_right - self.w_right
        self._x1_left: float = self._x0_right - self._x_pad
        self._x0_left: float = self.x0_tot
        self._w_left: float = self._x1_left - self._x0_left
        self._y0_rigbot: float = self._y0_center
        self._y1_rigbot: float = self._y0_rigbot + self.h_rigbot
        self._y1_rigmid: float = self._y0_top - self.y_pad
        self._y0_rigmid: float = self._y1_rigbot + self.y_pad
        self._h_rigmid: float = self._y1_rigmid - self._y0_rigmid

    def rect_top(self) -> RectType:
        return (self._x0_left, self._y0_top, self._w_tot, self.h_top)

    def rect_center(self) -> RectType:
        return (
            self._x0_left,
            self._y0_center,
            self._w_left,
            self._h_center,
        )

    def aspect_center(self) -> float:
        _, _, w, h = self.rect_center()
        return w / h

    def rect_right_middle(self) -> RectType:
        return (
            self._x0_right,
            self._y0_rigmid,
            self.w_right,
            self._h_rigmid,
        )

    def rect_right_bottom(self) -> RectType:
        return (
            self._x0_right,
            self._y0_rigbot,
            self.w_right,
            self.h_rigbot,
        )

    def rect_bottom_left(self) -> RectType:
        return (
            self._x0_left,
            self.y0_tot,
            self._w_left,
            self.h_bottom,
        )

    def rect_bottom_right(self) -> RectType:
        return (
            self._x0_right,
            self.y0_tot,
            self.w_right,
            self.h_bottom,
        )


@dataclass
# pylint: disable=R0902  # too-many-instance-attributes
class BoxedPlotLayoutModern:
    aspect: float
    x0_tot: float = 0.0
    x1_tot: float = 1.0
    y0_tot: float = 0.0
    y1_tot: float = 1.0
    y_pad: float = 0.02
    h_top: float = 0.08
    w_right: float = 0.2
    h_rigtop: float = 0.08
    h_rigbot: float = 0.42
    h_bottom: float = 0.05

    def __post_init__(self):
        self._x_pad: float = self.y_pad / self.aspect
        self._h_tot: float = self.y1_tot - self.y0_tot
        self._y1_top: float = self.y1_tot
        self._y0_top: float = self._y1_top - self.h_top
        self._y1_center: float = self._y0_top - self.y_pad
        self._y0_center: float = self.y0_tot + self.h_bottom
        self._h_center: float = self._y1_center - self._y0_center
        self._w_tot: float = self.x1_tot - self.x0_tot
        self._x1_right: float = self.x1_tot
        self._x0_right: float = self._x1_right - self.w_right
        self._x1_left: float = self._x0_right - self._x_pad
        self._x0_left: float = self.x0_tot
        self._w_left: float = self._x1_left - self._x0_left
        self._y0_rigbot: float = self._y0_center
        self._y1_rigtop: float = self.y1_tot
        self._y0_rigtop: float = self._y1_rigtop - self.h_rigtop
        self._y1_rigbot: float = self._y0_rigbot + self.h_rigbot
        self._y1_rigmid: float = self._y0_rigtop - self.y_pad
        self._y0_rigmid: float = self._y1_rigbot + self.y_pad
        self._h_rigmid: float = self._y1_rigmid - self._y0_rigmid

    def rect_top(self) -> RectType:
        return self._x0_left, self._y0_top, self._w_left, self.h_top

    def rect_center(self) -> RectType:
        return (
            self._x0_left,
            self._y0_center,
            self._w_left,
            self._h_center,
        )

    def aspect_center(self) -> float:
        _, _, w, h = self.rect_center()
        return w / h

    def rect_right_top(self) -> RectType:
        return (
            self._x0_right,
            self._y0_rigtop,
            self.w_right,
            self.h_rigtop,
        )

    def rect_right_middle(self) -> RectType:
        return (
            self._x0_right,
            self._y0_rigmid,
            self.w_right,
            self._h_rigmid,
        )

    def rect_right_bottom(self) -> RectType:
        return (
            self._x0_right,
            self._y0_rigbot,
            self.w_right,
            self.h_rigbot,
        )

    def rect_bottom_left(self) -> RectType:
        return (
            self._x0_left,
            self.y0_tot,
            self._w_left,
            self.h_bottom,
        )

    def rect_bottom_right(self) -> RectType:
        return (
            self._x0_right,
            self.y0_tot,
            self.w_right,
            self.h_bottom,
        )


@dataclass
# pylint: disable=R0902  # too-many-instance-attributes
class BoxedPlotLayoutDeterministic(BoxedPlotLayoutModern):
    pass


@dataclass
# pylint: disable=R0902  # too-many-instance-attributes
class BoxedPlotLayoutEnsemble(BoxedPlotLayoutModern):
    h_rigtop: float = 0.2
