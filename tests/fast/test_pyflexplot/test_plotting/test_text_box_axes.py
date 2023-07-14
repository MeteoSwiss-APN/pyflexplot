"""Test module ``pyflexplot.map_axes``."""
# Third-party
from matplotlib import pyplot as plt

# First-party
from pyflexplot.plotting.text_box_axes import TextBoxAxes
from pyflexplot.utils.summarize import summarize
from srutils.testing import check_is_sub_element


class Test_TextBoxAxes_Summarize:
    """Summarize all relevant information about an ``TextBoxAxes``."""

    def create_text_box(self, name, **kwargs):
        self.fig, self.ax = plt.subplots(figsize=(1, 1))
        self.rect_lbwh = [1, 1, 3, 2]
        self.box = TextBoxAxes(self.fig, self.rect_lbwh, name=name, **kwargs)
        return self.box

    sol_base = {
        "type": "TextBoxAxes",
        "rect": [1.0, 1.0, 3.0, 2.0],
        "fig": {
            "type": "Figure",
            "bbox": {"type": "TransformedBbox", "bounds": (0.0, 0.0, 100.0, 100.0)},
            "axes": [{},
                #{"type": "AxesSubplot"},
                {
                    "type": "Axes",
                    "bbox": {
                        "type": "TransformedBbox",
                        "bounds": (100.0, 100.0, 300.0, 200.0),
                    },
                },
            ],
        },
    }

    def test_text_line(self):
        box = self.create_text_box("text_line")
        box.text("lower-left", loc="bl", dx=1.6, dy=0.8)
        res = {**summarize(box), "fig": summarize(box.fig)}
        sol = {
            "name": "text_line",
            **self.sol_base,
            "elements": [
                {
                    "type": "TextBoxElementText",
                    "s": "lower-left",
                    "loc": {
                        "type": "TextBoxLocation",
                        "va": "baseline",
                        "ha": "left",
                        "dx": 1.6,
                        "dy": 0.8,
                    },
                }
            ],
        }
        check_is_sub_element(
            name_super="results", obj_super=res, name_sub="solution", obj_sub=sol
        )

    def test_text_block(self):
        box = self.create_text_box("text_block")
        box.text_block(loc="mc", block=[("foo", "bar"), ("hello", "world")])
        res = {**summarize(box), "fig": summarize(box.fig)}
        sol = {
            **self.sol_base,
            "elements": [
                {"type": "TextBoxElementText", "s": ("foo", "bar")},
                {"type": "TextBoxElementText", "s": ("hello", "world")},
            ],
        }
        check_is_sub_element(
            name_super="results", obj_super=res, name_sub="solution", obj_sub=sol
        )

    def test_color_rect(self):
        box = self.create_text_box("color_rect")
        box.color_rect("tr", "red", "black")
        res = {**summarize(box), "fig": summarize(box.fig)}
        sol = {
            "name": "color_rect",
            **self.sol_base,
            "elements": [
                {"type": "TextBoxElementColorRect", "fc": "red", "ec": "black"}
            ],
        }
        check_is_sub_element(
            name_super="results", obj_super=res, name_sub="solution", obj_sub=sol
        )
