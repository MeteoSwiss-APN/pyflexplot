"""
Text boxes.
"""
# Standard library
from dataclasses import dataclass
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

# Third-party
import matplotlib as mpl
from matplotlib.figure import Figure

# Local
from ..utils.exceptions import MinFontSizeReachedError
from ..utils.exceptions import MinStrLenReachedError
from ..utils.summarize import post_summarize_plot
from ..utils.summarize import summarizable
from ..utils.typing import ColorType
from ..utils.typing import LocationType
from ..utils.typing import MarkerStyleType
from ..utils.typing import RawTextBlocksType
from ..utils.typing import RawTextBlockType
from ..utils.typing import RectType
from ..utils.typing import TextBlocksType


@summarizable(post_summarize=post_summarize_plot)
class TextBoxElement:
    """Base class for elements in text box."""

    def __init__(self, *args, **kwargs):
        raise Exception(f"{type(self).__name__} must be subclassed")


@summarizable(
    attrs=["loc", "s", "replace_edge_spaces", "edge_spaces_replacement_char", "kwargs"],
    overwrite=True,
)
class TextBoxElementText(TextBoxElement):
    """Text element in text box."""

    # pylint: disable=W0231  # super-init-not-called
    def __init__(
        self,
        box,
        loc,
        *,
        s,
        replace_edge_spaces=False,
        edge_spaces_replacement_char="\u2423",
        **kwargs,
    ):
        """Create an instance of ``TextBoxElementText``.

        Args:
            box (TextBoxAxes): Parent text box.

            loc (TextBoxLocation): Location in parent text box.

            s (str): Text.

            replace_edge_spaces (bool): Replace the first and/and last
                character in ``s`` by ``edge_spaces_replacement_char`` if it is
                a space. This can be useful for debugging, as trailing spaces
                may be dropped during rendering. Defaults to False.

            edge_spaces_replacement_char (str): Replacement character for space
                if ``edge_spaces_replacement_char == True``. Defaults to
                u'\u2423' (the 'open-box' character below the text baseline
                that commonly represents a space).

            **kwargs: Additional keyword arguments for ``ax.text``.

        """
        self.box = box
        self.loc = loc
        self.s = s
        self.replace_edge_spaces = replace_edge_spaces
        self.edge_spaces_replacement_char = edge_spaces_replacement_char
        self.kwargs = kwargs

        # SR_TMP < TODO consider removing this
        # Add alignment parameters, unless specified in input kwargs
        self.kwargs["ha"] = self.kwargs.get(
            "horizontalalignment", self.kwargs.get("ha", self.loc.ha)
        )
        self.kwargs["va"] = self.kwargs.get(
            "verticalalignment", self.kwargs.get("va", self.loc.va)
        )
        # SR_TMP >

        # SR_TMP <
        if kwargs["va"] == "top_baseline":
            # SR_NOTE: [2019-06-11]
            # Ideally, we would like to align text by a `top_baseline`,
            # analogous to baseline and center_baseline, which does not
            # depend on the height of the letters (e.g., '$^\circ$'
            # lifts the top of the text, like 'g' at the bottom). This
            # does not exist, however, and attempts to emulate it by
            # determining the line height (e.g., draw an 'M') and then
            # shifting y accordingly (with `baseline` alignment) were
            # not successful.
            raise NotImplementedError(f"verticalalignment='{kwargs['vs']}'")
        # SR_TMP >

    def draw(self):
        """Draw text element onto text bot axes."""
        s = self.s
        if self.replace_edge_spaces:
            # Preserve trailing whitespace by replacing the first and
            # last space by a visible character
            if self.edge_spaces_replacement_char == " ":
                raise Exception("edge_spaces_replacement_char == ' '")
            if self.s[0] == " ":
                s = self.edge_spaces_replacement_char + s[1:]
            if self.s[-1] == " ":
                s = s[:-1] + self.edge_spaces_replacement_char
        self.box.ax.text(x=self.loc.x, y=self.loc.y, s=s, **self.kwargs)


@summarizable(attrs=["loc", "w", "h", "fc", "ec", "x_anker", "kwargs"], overwrite=True)
# pylint: disable=R0902  # too-many-instance-attributes
class TextBoxElementColorRect(TextBoxElement):
    """A colored box element inside a text box axes."""

    # pylint: disable=W0231  # super-init-not-called
    def __init__(self, box, loc, *, w, h, fc, ec, x_anker=None, **kwargs):
        """Create an instance of ``TextBoxElementBolorBox``.

        Args:
            box (TextBoxAxes): Parent text box.

            loc (TextBoxLocation): Location in parent text box.

            w (float): Width (box coordinates).

            h (float): Height (box coordinates).

            fc (str or tuple[float]): Face color.

            ec (str or tuple[float]): Edge color.

            x_anker (str): Horizontal anker. Options: 'l' or 'left'; 'c' or
                'center'; 'r' or 'right'; and None, in which case it is derived
                from the horizontal location in ``loc``. Defaults to None.

            **kwargs: Additional keyword arguments for
                ``mpl.patches.Rectangle``.

        """
        self.box = box
        self.loc = loc
        self.w = w
        self.h = h
        self.fc = fc
        self.ec = ec
        self.x_anker = x_anker
        self.kwargs = kwargs

    def draw(self):
        x = self.loc.x
        y = self.loc.y
        w = self.w * self.loc.dx_unit
        h = self.h * self.loc.dy_unit

        # Adjust horizontal position
        if self.x_anker in ["l", "left"]:
            pass
        elif self.x_anker in ["c", "center"]:
            x -= 0.5 * w
        elif self.x_anker in ["r", "right"]:
            x -= w
        elif self.x_anker is None:
            x -= w * {"l": 0.0, "c": 0.5, "r": 1.0}[self.loc.loc_x]
        else:
            raise Exception(f"invalid x_anker '{self.x_anker}'")

        p = mpl.patches.Rectangle(
            xy=(x, y),
            width=w,
            height=h,
            fill=True,
            fc=self.fc,
            ec=self.ec,
            **self.kwargs,
        )
        self.box.ax.add_patch(p)


@summarizable(attrs=["loc", "m", "kwargs"], overwrite=True)
class TextBoxElementMarker(TextBoxElement):
    """A marker element in a text box axes."""

    # pylint: disable=W0231  # super-init-not-called
    def __init__(self, box, loc, *, m, **kwargs):
        """Create an instance of ``TextBoxElementMarker``.

        Args:
            box (TextBoxAxes): Parent text box.

            loc (TextBoxLocation): Position in parent text box.

            m (str or int): Marker type.

            **kwargs: Additional keyword arguments for ``ax.plot``.

        """
        self.box = box
        self.loc = loc
        self.m = m
        self.kwargs = kwargs

    def draw(self):
        self.box.ax.plot([self.loc.x], [self.loc.y], marker=self.m, **self.kwargs)


@summarizable(attrs=["loc", "c", "lw"], overwrite=True)
class TextBoxElementHLine(TextBoxElement):
    """Horizontal line in a text box axes."""

    # pylint: disable=W0231  # super-init-not-called
    def __init__(self, box, loc, *, c="k", lw=1.0):
        """Create an instance of ``TextBoxElementHLine``.

        Args:
            box (TextBoxAxes): Parent text box.

            loc (TextBoxLocation): Location in parent text box.

            c (<color>, optional): Line color. Defaults to 'k' (black).

            lw (float, optional): Line width. Defaults to 1.0.

        """
        self.box = box
        self.loc = loc
        self.c = c
        self.lw = lw

    def draw(self):
        self.box.ax.axhline(self.loc.y, color=self.c, linewidth=self.lw)


@summarizable(
    attrs=["name", "rect", "lw_frame", "dx_unit", "dy_unit"],
    post_summarize=lambda self, summary: {
        **post_summarize_plot(self, summary),
        "elements": [e.summarize() for e in self.elements],
    },
)
# SR_TODO Refactor to reduce instance attributes and arguments!
@dataclass
# pylint: disable=R0902  # too-many-instance-attributes
# pylint: disable=R0913  # too-many-arguments
class TextBoxAxes:
    """Text box axes for FLEXPART plot.

    Args:
        fig: Figure to which to add the text box axes.

        rect: Rectangle [left, bottom, width, height].

        name: Name of the text box.

        lw_frame (optional): Line width of frame around box. Frame is omitted if
            ``lw_frame`` is None.

        ec (optional): Edge color.

        fc (optional): Face color.

    """

    fig: Figure
    rect: RectType
    name: str
    lw_frame: Optional[float] = 1.0
    ec: ColorType = "black"
    fc: ColorType = "none"
    show_baselines: bool = False  # useful for debugging

    def __post_init__(self):
        self.ax = self.fig.add_axes(self.rect)
        self.ax.axis("off")
        self.ax.set_xlim(0.0, 1.0)
        self.ax.set_ylim(0.0, 1.0)

        # SR_TMP < TODO Clean up!
        w_rel_fig, h_rel_fig = ax_w_h_in_fig_coords(self.fig, self.ax)
        self.dx_unit = 0.0075 / w_rel_fig
        self.dy_unit = 0.009 / h_rel_fig
        self.pad_x = 1.0 * self.dx_unit
        self.pad_y = 1.0 * self.dy_unit
        # SR_TMP >

        self.elements = []

        # Uncomment to populate the box with nine sample labels
        # self.sample_labels()

    def draw(self):
        """Draw the defined text boxes onto the plot axes."""

        # Create box without frame
        p = mpl.patches.Rectangle(
            xy=(0, 0),
            width=1,
            height=1,
            transform=self.ax.transAxes,
            facecolor=self.fc,
            edgecolor="none",
            clip_on=False,
        )
        self.ax.add_patch(p)

        if self.lw_frame:
            # Enable frame around box
            p.set(edgecolor=self.ec, linewidth=self.lw_frame)

        for element in self.elements:
            element.draw()

    def text(
        self, s: str, loc: LocationType, dx: float = 0.0, dy: float = 0.0, **kwargs
    ) -> None:
        """Add text positioned relative to a reference location.

        Args:
            loc: Reference location parameter used to initialize an instance of
                ``TextBoxLocation``.

            s: Text string.

            dx (optional): Horizontal offset in unit distances. May be negative.

            dy (optional): Vertical offset in unit distances. May be negative.

            **kwargs: Formatting options passed to ``ax.text()``.

        """
        fancy_loc = TextBoxLocation(self, loc, dx, dy)
        self.elements.append(TextBoxElementText(self, loc=fancy_loc, s=s, **kwargs))
        if self.show_baselines:
            self.elements.append(TextBoxElementHLine(self, fancy_loc))

    def text_block(
        self,
        block: RawTextBlockType,
        loc: LocationType,
        colors: Optional[Sequence[ColorType]] = None,
        **kwargs,
    ) -> None:
        """Add a text block comprised of multiple lines.

        Args:
            loc: Reference location. For details see
                ``TextBoxAxes.text``.

            block: Text block.

            colors (optional): Line-specific colors. Defaults to None. If not
                None, must have same length as ``block``. Omit individual lines
                with None.

            **kwargs: Positioning and formatting options passed to
                ``TextBoxAxes.text_blocks``.

        """
        blocks_colors: Optional[Sequence[Sequence[ColorType]]]
        if colors is None:
            blocks_colors = None
        else:
            blocks_colors = [colors]
        self.text_blocks(blocks=[block], loc=loc, colors=blocks_colors, **kwargs)

    # pylint: disable=R0914  # too-many-locals
    def text_blocks(
        self,
        blocks: RawTextBlocksType,
        loc: LocationType,
        *,
        dy_unit: Optional[float] = None,
        dy_line: Optional[float] = None,
        dy_block: Optional[float] = None,
        colors: Optional[Sequence[Sequence[ColorType]]] = None,
        **kwargs,
    ) -> None:
        """Add multiple text blocks.

        Args:
            loc: Reference location. For details see ``TextBoxAxes.text``.

            blocks: List of text blocks, each of which constitutes a list of
                lines.

            dy_unit (optional): Initial vertical offset in unit distances. May
                be negative. Defaults to ``dy_line``.

            dy_line (optional): Incremental vertical offset between lines. May
                be negative. Defaults to 2.5.

            dy_block (optional): Incremental vertical offset between
                blocks of lines. Can be negative. Defaults to ``dy_line``.

            dx (optional): Horizontal offset in unit distances. May be negative.
                Defaults to 0.0.

            colors (optional): Line-specific colors in each block. If not None,
                must have same shape as ``blocks``. Omit individual blocks or
                lines in blocks with None.

            **kwargs: Formatting options passed to ``ax.text``.

        """
        if dy_line is None:
            dy_line = 2.5
        if dy_unit is None:
            dy_unit = dy_line
        if dy_block is None:
            dy_block = dy_line

        default_color = kwargs.pop("color", kwargs.pop("c", "black"))
        colors_blocks = self._prepare_line_colors(blocks, colors, default_color)

        dy = dy_unit
        for i, block in enumerate(blocks):
            for j, line in enumerate(block):
                self.text(line, loc=loc, dy=dy, color=colors_blocks[i][j], **kwargs)
                dy -= dy_line
            dy -= dy_block

    @staticmethod
    def _prepare_line_colors(blocks, colors, default_color):

        if colors is None:
            colors_blocks = [None] * len(blocks)
        elif len(colors) == len(blocks):
            colors_blocks = colors
        else:
            raise ValueError(
                f"different no. colors than blocks: {len(colors)} != {len(blocks)}"
            )

        for i, block in enumerate(blocks):
            if colors_blocks[i] is None:
                colors_blocks[i] = [None] * len(block)
            elif len(colors_blocks) != len(blocks):
                ith = f"{i}{({1: 'st', 2: 'nd', 3: 'rd'}.get(i, 'th'))}"
                raise ValueError(
                    f"colors of {ith} block must have same length as block: "
                    f"{len(colors_blocks[i])} != {len(block)}"
                )
            for j in range(len(block)):
                if colors_blocks[i][j] is None:
                    colors_blocks[i][j] = default_color

        return colors_blocks

    def text_block_hfill(
        self, block: RawTextBlockType, loc_y: LocationType = "t", **kwargs
    ) -> None:
        """Single block of horizontally filled lines.

        Args:
            block: Text block. See docstring of method ``text_blocks_hfill``
                for details.

            loc_y: Vertical reference location. For details see
                ``TextBoxAxes.text`` (vertical component only).

            **kwargs: Additional keyword arguments passed on to method
                ``text_blocks_hfill``.

        """
        blocks: Sequence[RawTextBlockType] = [block]
        self.text_blocks_hfill(blocks, loc_y, **kwargs)

    def text_blocks_hfill(
        self, blocks: RawTextBlocksType, loc_y: LocationType = "t", **kwargs
    ) -> None:
        r"""Add blocks of horizontally-filling lines.

        Lines are split at a tab character ('\t'), with the text before the tab
        left-aligned, and the text after right-aligned.

        Args:
            blocks: Text blocks, each of which consists of lines, each of which
                in turn consists of a left and right part. Possible formats:

                 - The blocks can be a multiline string, with empty lines
                   separating the individual blocks; or a list.

                 - In case of list blocks, each block can in turn constitute a
                   multiline string, or a list of lines.

                 - In case of a list block, each line can in turn constitute a
                   string, or a two-element string tuple.

                 - Lines represented by a string are split into a left and
                   right part at the first tab character ('\t').

            loc_y: Vertical reference location. For details see
                ``TextBoxAxes.text`` (vertical component only).

            **kwargs: Location and formatting options passed to
                ``TextBoxAxes.text_blocks``.

        """
        prepared_blocks = self._prepare_text_blocks(blocks)
        blocks_l, blocks_r = self._split_lines_horizontally(prepared_blocks)
        self.text_blocks(blocks_l, f"{loc_y}l", **kwargs)
        self.text_blocks(blocks_r, f"{loc_y}r", **kwargs)

    def _prepare_text_blocks(self, blocks: RawTextBlocksType) -> TextBlocksType:
        """Turn multiline strings (shorthand notation) into lists of strings."""
        blocks_lst: TextBlocksType = []
        block_or_blocks: Union[Sequence[str], Sequence[Sequence[str]]]
        if isinstance(blocks, str):
            blocks = blocks.strip().split("\n\n")
        assert isinstance(blocks, Sequence)  # mypy
        for block_or_blocks in blocks:
            blocks_i: RawTextBlocksType
            if isinstance(block_or_blocks, str):
                blocks_i = block_or_blocks.strip().split("\n\n")
            else:
                blocks_i = [block_or_blocks]
            block: RawTextBlockType
            for block in blocks_i:
                blocks_lst.append([])
                if isinstance(block, str):
                    block = block.strip().split("\n")
                assert isinstance(block, Sequence)  # mypy
                line: str
                for line in block:
                    blocks_lst[-1].append(line)
        return blocks_lst

    def _split_lines_horizontally(
        self, blocks: Sequence[Sequence[str]]
    ) -> Tuple[List[List[str]], List[List[str]]]:
        blocks_l: TextBlocksType = []
        blocks_r: TextBlocksType = []
        for block in blocks:
            blocks_l.append([])
            blocks_r.append([])
            for line in block:
                # Obtain left and right part of line
                if isinstance(line, str):
                    if "\t" not in line:
                        raise ValueError("no '\t' in line", line)
                    str_l, str_r = line.split("\t", 1)
                elif len(line) == 2:
                    str_l, str_r = line
                else:
                    raise ValueError(f"invalid line: {line}")
                blocks_l[-1].append(str_l)
                blocks_r[-1].append(str_r)
        return blocks_l, blocks_r

    def sample_labels(self):
        """Add sample text labels in corners etc."""
        kwargs = dict(fontsize=9)
        self.text("bl", "bot. left", **kwargs)
        self.text("bc", "bot. center", **kwargs)
        self.text("br", "bot. right", **kwargs)
        self.text("ml", "middle left", **kwargs)
        self.text("mc", "middle center", **kwargs)
        self.text("mr", "middle right", **kwargs)
        self.text("tl", "top left", **kwargs)
        self.text("tc", "top center", **kwargs)
        self.text("tr", "top right", **kwargs)

    def color_rect(
        self,
        loc: LocationType,
        fc: ColorType,
        ec: Optional[ColorType] = None,
        dx: float = 0.0,
        dy: float = 0.0,
        w: float = 3.0,
        h: float = 2.0,
        **kwargs,
    ) -> None:
        """Add a colored rectangle.

        Args:
            loc: Reference location parameter used to initialize an instance of
                ``TextBoxLocation``.

            fc: Face color.

            ec (optional): Edge color. Defaults to face color.

            dx (optional): Horizontal offset in unit distances. May be negative.

            dy (optional): Vertical offset in unit distances. May be negative.

            w (optional): Width in unit distances.

            h (optional): Height in unit distances.

            **kwargs: Keyword arguments passed to
                ``matplotlib.patches.Rectangle``.

        """
        if ec is None:
            ec = fc
        fancy_loc = TextBoxLocation(self, loc, dx, dy)
        self.elements.append(
            TextBoxElementColorRect(self, fancy_loc, w=w, h=h, fc=fc, ec=ec, **kwargs)
        )
        if self.show_baselines:
            self.elements.append(TextBoxElementHLine(self, fancy_loc))

    def marker(
        self,
        loc: LocationType,
        marker: MarkerStyleType,
        dx: float = 0.0,
        dy: float = 0.0,
        **kwargs,
    ) -> None:
        """Add a marker symbol.

        Args:
            loc: Reference location parameter used to initialize an instance of
                ``TextBoxLocation``.

            marker: Marker style passed to ``mpl.plot``. See
                ``matplotlib.markers`` for more information.

            dx (optional): Horizontal offset in unit distances. May be negative.

            dy (optional): Vertical offset in unit distances. May be negative.

            **kwargs: Keyword arguments passed to ``mpl.plot``.

        """
        fancy_loc = TextBoxLocation(self, loc, dx, dy)
        self.elements.append(TextBoxElementMarker(self, fancy_loc, m=marker, **kwargs))
        if self.show_baselines:
            self.elements.append(TextBoxElementHLine(self, fancy_loc))

    def fit_text(self, s: str, size: float, **kwargs) -> str:
        return TextFitter(self.ax, dx_unit=self.dx_unit, **kwargs).fit(s, size)


def ax_w_h_in_fig_coords(fig, ax):
    """Get the dimensions of an axes in figure coords."""
    trans = fig.transFigure.inverted()
    _, _, w, h = ax.bbox.transformed(trans).bounds
    return w, h


class TextFitter:
    """Fit a text string into the box by shrinking and/or truncation."""

    sizes = ["xx-small", "x-small", "small", "medium", "large", "x-large", "xx-large"]

    def __init__(self, ax, *, dx_unit, n_shrink_max=None, pad_rel=None, dots=".."):
        """Create an instance of ``TextFitter``.

        Args:
            ax (Axes): Axes.

            dx_unit (float): Horizontal unit distance.

            n_shrink_max (int, optional): Maximum number of times the font size
                can be reduced before the string is truncated. If it is None or
                negative, the font size is reduced all the way to "xx-small"
                if necessary. Defaults to None.

            pad_rel (float, optional): Total horizontal padding as a fraction
                of the box width. Defaults to twice the default horizontal
                offset ``2 * dx_unit``.

            dots (str, optional): String replacing the end of the retained part
                of ``s`` in case it must be truncated. Defaults to "..".

        """
        self.ax = ax
        self.dx_unit = dx_unit

        if n_shrink_max is not None:
            try:
                n_shrink_max = int(n_shrink_max)
            except ValueError:
                raise ValueError(
                    f"n_shrink_max of type {type(n_shrink_max).__name__} not "
                    f"int-compatible: {n_shrink_max}"
                )
            if n_shrink_max < 0:
                n_shrink_max = None
        self.n_shrink_max = n_shrink_max

        if pad_rel is None:
            pad_rel = 2 * self.dx_unit
        self.pad_rel = pad_rel

        self.dots = dots

    def fit(self, s, size):
        """
        Fit a string with a certain target size into the box.

        Args:
            s (str): Text string to fit into the box.

            size (str): Initial font size (e.g., "medium", "x-large").

        """
        if size not in self.sizes:
            raise ValueError(f"unknown font size '{size}'; must be one of {self.sizes}")

        w_rel_max = 1.0 - self.pad_rel
        while len(s) >= len(self.dots) and self.w_rel(s, size) > w_rel_max:
            try:
                size = self.shrink(size)
            except MinFontSizeReachedError:
                try:
                    s = self.truncate(s)
                except MinStrLenReachedError:
                    break

        return s, size

    def w_rel(self, s, size):
        """Returns the width of a string as a fraction of the box width."""

        # Determine width of text in display coordinates
        # src: https://stackoverflow.com/a/36959454
        renderer = self.ax.get_figure().canvas.get_renderer()
        txt = self.ax.text(0, 0, s, size=size)
        w_disp = txt.get_window_extent(renderer=renderer).width

        # Remove the text again from the axes
        self.ax.texts.pop()

        return w_disp / self.ax.bbox.width

    # pylint: disable=W0102  # dangerous-default-value
    def shrink(self, size, _n=[0]):
        """Shrink the relative font size by one increment."""
        i = self.sizes.index(size)
        if i == 0 or (self.n_shrink_max is not None and _n[0] >= self.n_shrink_max):
            raise MinFontSizeReachedError(size)
        size = self.sizes[i - 1]
        _n[0] += 1
        return size

    # pylint: disable=W0102  # dangerous-default-value
    def truncate(self, s, _n=[0]):
        """Truncate a string by one character and end with ``self.dots``."""
        if len(s) <= len(self.dots):
            raise MinStrLenReachedError(s)
        _n[0] += 1
        return s[: -(len(self.dots) + 1)] + self.dots


@summarizable(
    attrs=[
        "loc",
        "loc_y",
        "loc_x",
        "dx_unit",
        "dy_unit",
        "dx",
        "dy",
        "x0",
        "y0",
        "x",
        "y",
        "va",
        "ha",
    ],
    post_summarize=post_summarize_plot,
)
# SR_TODO Refactor to remove number of instance attributes!
# pylint: disable=R0902  # too-many-instance-attributes
class TextBoxLocation:
    """A reference location (like bottom-left) inside a box on a 3x3 grid."""

    def __init__(self, parent, loc, dx=None, dy=None):
        """Initialize an instance of TextBoxLocation.

        Args:
            parent (TextBoxAxes): Parent text box axes.

            loc (int or str): Location parameter. Takes one of three formats:
                integer, short string, or long string.

                Choices:

                    int     short   long
                    00      bl      bottom left
                    01      bc      bottom center
                    02      br      bottom right
                    10      ml      middle left
                    11      mc      middle
                    12      mr      middle right
                    20      tl      top left
                    21      tc      top center
                    22      tr      top right

            dx (float, optional): Horizontal offset in unit distances. Defaults
                to 0.0.

            dy (float, optional): Vertical offset in unit distances. Defaults
                to 0.0.

        """
        self.dx = dx or 0.0
        self.dy = dy or 0.0

        self._determine_loc_components(loc)

        self.dx_unit = parent.dx_unit
        self.dy_unit = parent.dy_unit
        self.pad_x = parent.pad_x
        self.pad_y = parent.pad_y

    def _determine_loc_components(self, loc):
        """Evaluate vertical and horizontal location parameter components."""
        loc = str(loc)
        if len(loc) == 2:
            loc_y, loc_x = loc
        elif loc == "center":
            loc_y, loc_x = loc, loc
        elif " " in loc:
            loc_y, loc_x = loc.split(" ", 1)
        else:
            raise ValueError("invalid location parameter", loc)
        self.loc_y = self._standardize_loc_y(loc_y)
        self.loc_x = self._standardize_loc_x(loc_x)
        self.loc = f"{self.loc_y}{self.loc_x}"

    def _standardize_loc_y(self, loc):
        """Standardize vertical location component."""
        if loc in (0, "0", "b", "bottom"):
            return "b"
        elif loc in (1, "1", "m", "middle"):
            return "m"
        elif loc in (2, "2", "t", "top"):
            return "t"
        raise ValueError(f"invalid vertical location component '{loc}'")

    def _standardize_loc_x(self, loc):
        """Standardize horizontal location component."""
        if loc in (0, "0", "l", "left"):
            return "l"
        elif loc in (1, "1", "c", "center"):
            return "c"
        elif loc in (2, "2", "r", "right"):
            return "r"
        raise ValueError(f"invalid horizontal location component '{loc}'")

    @property
    def va(self):
        """Vertical alignment variable."""
        return {"b": "baseline", "m": "center_baseline", "t": "top"}[self.loc_y]

    @property
    def ha(self):
        """Horizontal alignment variable."""
        return {"l": "left", "c": "center", "r": "right"}[self.loc_x]

    @property
    def y0(self):
        """Vertical baseline position."""
        if self.loc_y == "b":
            return 0.0 + self.pad_y
        elif self.loc_y == "m":
            return 0.5
        elif self.loc_y == "t":
            return 1.0 - self.pad_y
        else:
            raise Exception(
                f"invalid {type(self).__name__} instance attr loc_y: '{self.loc_y}'"
            )

    @property
    def x0(self):
        """Horizontal baseline position."""
        if self.loc_x == "l":
            return 0.0 + self.pad_x
        elif self.loc_x == "c":
            return 0.5
        elif self.loc_x == "r":
            return 1.0 - self.pad_x
        else:
            raise Exception(
                f"invalid {type(self).__name__} instance attr loc_x: '{self.loc_x}'"
            )

    @property
    def x(self):
        """Horizontal position."""
        return self.x0 + self.dx * self.dx_unit

    @property
    def y(self):
        """Vertical position."""
        return self.y0 + self.dy * self.dy_unit
