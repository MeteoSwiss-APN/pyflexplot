# -*- coding: utf-8 -*-
"""
Plot layouts.
"""
# Standard library
from dataclasses import dataclass

# Local
from .typing import RectType


@dataclass
# pylint: disable=R0902  # too-many-instance-attributes
class BoxedPlotLayout:
    aspect: float
    x0_tot: float = 0.0
    x1_tot: float = 1.0
    y0_tot: float = 0.0
    y1_tot: float = 1.0
    y_pad: float = 0.02
    h_top: float = 0.08
    w_right: float = 0.2
    h_rigtop: float = 0.15
    h_rigbot: float = 0.42
    h_bottom: float = 0.05

    def __post_init__(self):
        self.x_pad: float = self.y_pad / self.aspect
        self.h_tot: float = self.y1_tot - self.y0_tot
        self.y1_top: float = self.y1_tot
        self.y0_top: float = self.y1_top - self.h_top
        self.y1_center: float = self.y0_top - self.y_pad
        self.y0_center: float = self.y0_tot + self.h_bottom
        self.h_center: float = self.y1_center - self.y0_center
        self.w_tot: float = self.x1_tot - self.x0_tot
        self.x1_right: float = self.x1_tot
        self.x0_right: float = self.x1_right - self.w_right
        self.x1_left: float = self.x0_right - self.x_pad
        self.x0_left: float = self.x0_tot
        self.w_left: float = self.x1_left - self.x0_left
        self.y0_rigbot: float = self.y0_center
        self.y1_rigtop: float = self.y1_tot
        self.y0_rigtop: float = self.y1_rigtop - self.h_rigtop
        self.y1_rigbot: float = self.y0_rigbot + self.h_rigbot
        self.y1_rigmid: float = self.y0_rigtop - self.y_pad
        self.y0_rigmid: float = self.y1_rigbot + self.y_pad
        self.h_rigmid: float = self.y1_rigmid - self.y0_rigmid
        self.rect_top: RectType = (self.x0_left, self.y0_top, self.w_left, self.h_top)
        self.rect_center: RectType = (
            self.x0_left,
            self.y0_center,
            self.w_left,
            self.h_center,
        )
        self.rect_right_top: RectType = (
            self.x0_right,
            self.y0_rigtop,
            self.w_right,
            self.h_rigtop,
        )
        self.rect_right_middle: RectType = (
            self.x0_right,
            self.y0_rigmid,
            self.w_right,
            self.h_rigmid,
        )
        self.rect_right_bottom: RectType = (
            self.x0_right,
            self.y0_rigbot,
            self.w_right,
            self.h_rigbot,
        )
        self.rect_bottom: RectType = (
            self.x0_tot,
            self.y0_tot,
            self.w_tot,
            self.h_bottom,
        )
