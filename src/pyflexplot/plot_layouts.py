"""Plot layouts."""
# Standard library
from typing import Dict
from typing import Mapping
from typing import Tuple

# Local
from .utils.summarize import summarizable
from .utils.typing import RectType


@summarizable(attrs=["name", "aspects", "rects"])
class BoxedPlotLayout:
    def __init__(
        self, name: str, aspects: Mapping[str, float], rects: Mapping[str, RectType]
    ) -> None:
        """Create an instance of ``BoxerdPlotLayout``."""
        self.name: str = name
        self.aspects: Dict[str, float] = dict(aspects)
        self.rects: Dict[str, RectType] = dict(rects)

    def get_aspect(self, name: str = "tot") -> float:
        return self.aspects[name]

    def get_rect(self, name: str = "tot") -> RectType:
        return self.rects[name]

    @classmethod
    def create(cls, name: str, aspect: float = 1.0) -> "BoxedPlotLayout":
        """Create the predefined plot layout ``name`` with ``aspect``."""
        if name == "vintage":
            aspects, rects = create_layout_vintage(aspect)
        elif name == "post_vintage":
            aspects, rects = create_layout_post_vintage(aspect)
        elif name == "post_vintage_ens":
            aspects, rects = create_layout_post_vintage_ens(aspect)
        elif name == "standalone_details":
            aspects, rects = create_layout_standalone_details(aspect)
        else:
            raise ValueError(f"invalid name '{name}'")
        return cls(name, aspects=aspects, rects=rects)


# pylint: disable=R0914  # too-many-locals (>15)
def create_layout_vintage(
    aspect: float,
) -> Tuple[Dict[str, float], Dict[str, RectType]]:
    # Primary
    x0_tot: float = 0.0
    x1_tot: float = 1.0
    y0_tot: float = 0.0
    y1_tot: float = 1.0
    y_pad: float = 0.02
    h_top: float = 0.08
    h_rigbot: float = 0.42
    h_bottom: float = 0.05
    w_right: float = 0.2

    # Derived
    x_pad: float = y_pad / aspect
    h_tot: float = y1_tot - y0_tot
    y1_top: float = y1_tot
    y0_top: float = y1_top - h_top
    y1_center: float = y0_top - y_pad
    y0_center: float = y0_tot + h_bottom
    h_center: float = y1_center - y0_center
    w_tot: float = x1_tot - x0_tot
    x1_right: float = x1_tot
    x0_right: float = x1_right - w_right
    x1_left: float = x0_right - x_pad
    x0_left: float = x0_tot
    w_left: float = x1_left - x0_left
    y0_rigbot: float = y0_center
    y1_rigbot: float = y0_rigbot + h_rigbot
    y1_rigmid: float = y0_top - y_pad
    y0_rigmid: float = y1_rigbot + y_pad
    h_rigmid: float = y1_rigmid - y0_rigmid
    aspects: Dict[str, float] = {
        "tot": aspect,
        "center": w_left / h_center * aspect,
    }
    rects: Dict[str, RectType] = {
        "tot": (x0_tot, y0_tot, w_tot, h_tot),
        "top": (x0_left, y0_top, w_tot, h_top),
        "center": (x0_left, y0_center, w_left, h_center),
        "right_middle": (x0_right, y0_rigmid, w_right, h_rigmid),
        "right_bottom": (x0_right, y0_rigbot, w_right, h_rigbot),
        "bottom_left": (x0_left, y0_tot, w_left, h_bottom),
        "bottom_right": (x0_right, y0_tot, w_right, h_bottom),
    }
    return (aspects, rects)


# pylint: disable=R0914  # too-many-locals (>15)
def create_layout_post_vintage(
    aspect: float,
    *,
    h_rigtop: float = 0.08,
    h_rigbot: float = 0.42,
) -> Tuple[Dict[str, float], Dict[str, RectType]]:
    # Primary
    x0_tot: float = 0.0
    x1_tot: float = 1.0
    y0_tot: float = 0.0
    y1_tot: float = 1.0
    y_pad: float = 0.02
    h_top: float = 0.08
    # h_rigtop: float = ...
    # h_rigbot: float = ...
    h_bottom: float = 0.05
    w_right: float = 0.2

    # Derived
    x_pad: float = y_pad / aspect
    # h_tot: float = y1_tot - y0_tot
    y1_top: float = y1_tot
    y0_top: float = y1_top - h_top
    y1_center: float = y0_top - y_pad
    y0_center: float = y0_tot + h_bottom
    h_center: float = y1_center - y0_center
    # w_tot: float = x1_tot - x0_tot
    x1_right: float = x1_tot
    x0_right: float = x1_right - w_right
    x1_left: float = x0_right - x_pad
    x0_left: float = x0_tot
    w_left: float = x1_left - x0_left
    y0_rigbot: float = y0_center
    y1_rigtop: float = y1_tot
    y0_rigtop: float = y1_rigtop - h_rigtop
    y1_rigbot: float = y0_rigbot + h_rigbot
    y1_rigmid: float = y0_rigtop - y_pad
    y0_rigmid: float = y1_rigbot + y_pad
    h_rigmid: float = y1_rigmid - y0_rigmid

    aspects: Dict[str, float] = {
        "tot": aspect,
        "center": w_left / h_center * aspect,
    }
    rects: Dict[str, RectType] = {
        "top": (x0_left, y0_top, w_left, h_top),
        "center": (x0_left, y0_center, w_left, h_center),
        "right_top": (x0_right, y0_rigtop, w_right, h_rigtop),
        "right_middle": (x0_right, y0_rigmid, w_right, h_rigmid),
        "right_bottom": (x0_right, y0_rigbot, w_right, h_rigbot),
        "bottom_left": (x0_left, y0_tot, w_left, h_bottom),
        "bottom_right": (x0_right, y0_tot, w_right, h_bottom),
    }
    return (aspects, rects)


def create_layout_post_vintage_ens(
    aspect: float,
) -> Tuple[Dict[str, float], Dict[str, RectType]]:
    return create_layout_post_vintage(aspect, h_rigtop=0.2)


def create_layout_standalone_details(
    aspect: float,
) -> Tuple[Dict[str, float], Dict[str, RectType]]:
    return create_layout_post_vintage(aspect, h_rigtop=0.25, h_rigbot=0.25)
