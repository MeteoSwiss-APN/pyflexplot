# -*- coding: utf-8 -*-
"""
Plots.
"""
import logging as log
import matplotlib.pyplot as plt
import numpy as np
import os

from copy import copy
from textwrap import dedent

from .plot_utils import AxesMap
from .plot_utils import ax_dims_fig_coords
from .plot_utils import TextBoxAxes
from .utils import SummarizableClass
from .plot_utils import SummarizablePlotClass
from .utils import Degrees
from .utils import ParentClass
from .utils_dev import ipython  #SR_DEV

#======================================================================
# Plot Labels
#======================================================================


class PlotLabels(SummarizableClass):

    summarizable_attrs = []  #SR_TODO

    def __init__(self):
        if self.__class__.__name__.endswith(f'_Base'):
            raise Exception(
                f"{type(self).__name__} must be subclassed, not instatiated")

    def __getattribute__(self, name):
        """Intersept attribute access to format labels.

        Attributes that do not start with an underscore are passed to
        the method ``self.format_attr``, which can be overridden by
        subclasses to format labels in a language-specific way.

        Args:
            name (str): Attribute name.

        """
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        try:
            val = object.__getattribute__(self, '__dict__')[name]
        except KeyError:
            try:
                val = object.__getattribute__(type(self), '__dict__')[name]
            except KeyError:
                raise AttributeError(name) from None
        return object.__getattribute__(self, 'format_attr')(name, val)

    def format_attr(self, name, val):
        """Override to format labels in a language-specific way."""
        return val


class PlotLabels_En(PlotLabels):
    pass


class PlotLabels_De(PlotLabels):

    def format_attr(self, name, val):
        """Format an attribute."""

        def umlaut(v):
            return f'$\\mathrm{{\\ddot{v}}}$'

        return val.format(
            ae=umlaut('a'),
            oe=umlaut('o'),
            ue=umlaut('u'),
        )


class DispersionPlotLabels_Simulation_En(PlotLabels_En):
    """FLEXPART dispersion plot labels in English (simulation)."""

    start = 'Start'
    end = 'End'
    flexpart_based_on = 'FLEXPART based on'
    copyright = u"\u00a9MeteoSwiss"


class DispersionPlotLabels_Simulation_De(PlotLabels_De):
    """part dispersion plot labels in German (simulation)."""

    start = 'Start'
    end = 'Ende'
    flexpart_based_on = 'FLEXPART basierend auf'
    copyright = u"\u00a9MeteoSchweiz"


class DispersionPlotLabels_Release_En(PlotLabels_En):
    """FLEXPART dispersion plot labels in English (release)."""

    lat = 'Latitude'
    lon = 'Longitude'
    height = 'Height'
    rate = 'Rate'
    mass = 'Total mass'
    site_name = 'Release site'
    max = 'Max.'


class DispersionPlotLabels_Release_De(PlotLabels_De):
    """part dispersion plot labels in German (release)."""

    lat = 'Breite'
    lon = 'L{ae}nge'
    height = 'H{oe}he'
    rate = 'Rate'
    mass = 'Totale Masse'
    site_name = 'Austrittsort'
    max = 'Max.'


class DispersionPlotLabels_Species_En(PlotLabels_En):
    """FLEXPART dispersion plot labels in English (species)."""

    name = 'Substance'
    half_life = 'Half-life'
    deposit_vel = 'Deposit. vel.'
    sediment_vel = 'Sediment. vel.'
    washout_coeff = 'Washout coeff.'
    washout_exponent = 'Washout exponent'


class DispersionPlotLabels_Species_De(PlotLabels_De):
    """part dispersion plot labels in German (species)."""

    name = 'Substanz'
    half_life = 'Halbwertszeit'
    deposit_vel = 'Deposit.-Geschw.'
    sediment_vel = 'Sediment.-Geschw.'
    washout_coeff = 'Auswaschkoeff.'
    washout_exponent = 'Auswaschexponent'


class DispersionPlotLabels(SummarizableClass):

    summarizable_attrs = []  #SR_TODO

    def __init__(self, lang):

        if lang == 'en':
            self.simulation = DispersionPlotLabels_Simulation_En()
            self.release = DispersionPlotLabels_Release_En()
            self.species = DispersionPlotLabels_Species_En()

        elif lang == 'de':
            self.simulation = DispersionPlotLabels_Simulation_De()
            self.release = DispersionPlotLabels_Release_De()
            self.species = DispersionPlotLabels_Species_De()

        else:
            raise ValueError(f"lang='{lang}'")


#======================================================================
# Plots
#======================================================================


class Plot(SummarizablePlotClass, ParentClass):
    """Base class for FLEXPART plots."""

    name = '__base__'
    summarizable_attrs = ['name', 'field', 'dpi', 'figsize', 'fig', 'ax_map']
    map_conf = {}

    def __init__(self, field, *, dpi=None, figsize=None):
        """Create an instance of ``Plot``.

        Args:
            field (Field): FLEXPART field.

            dpi (float, optional): Plot resolution (dots per inch).
                Defaults to 100.0.

            figsize (tuple[float, float], optional): Figure size in
                inches. Defaults to (12.0, 9.0).
        """
        self.field = field
        self.dpi = dpi or 100.0
        self.figsize = figsize or (12.0, 9.0)

    def prepare_plot(self):

        self.fig = plt.figure(figsize=self.figsize)

        self.ax_map = AxesMap(
            self.fig,
            self.field.rlat,
            self.field.rlon,
            self.field.attrs.grid.north_pole_lat.value,
            self.field.attrs.grid.north_pole_lon.value,
            **self.map_conf,
        )

    def save(self, file_path, format=None):
        """Save the plot to disk.

        Args:
            file_path (str): Output file name, incl. path.

            format (str): Plot format (e.g., 'png', 'pdf'). Defaults to
                None. If ``format`` is None, the plot format is derived
                from the extension of ``file_path``.

        """
        if format is None:
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in ['.pdf', '.png']:
                raise ValueError(
                    f"Cannot derive format from extension '{ext}'"
                    f"derived from '{os.path.basename(file_path)}'")
            format = ext[1:]

        self.fig.savefig(
            file_path,
            facecolor=self.fig.get_facecolor(),
            edgecolor=self.fig.get_edgecolor(),
            bbox_inches='tight',
            pad_inches=0.15,
        )
        plt.close(self.fig)

    def summarize(self, *args, **kwargs):
        data = super().summarize(*args, **kwargs)
        return data


class DispersionPlot(Plot):
    """Base class for FLEXPART dispersion plots."""

    name = '__base__dispersion__'
    cmap = 'flexplot'
    extend = 'max'
    n_levels = 9
    draw_colors = True
    draw_contours = False
    mark_field_max = True
    mark_release_site = True
    text_box_setup = {
        'h_rel_t': 0.1,
        'h_rel_b': 0.03,
        'w_rel_r': 0.25,
        'pad_hor_rel': 0.015,
        'h_rel_box_rt': 0.46,
    }
    level_range_style = 'simple'
    # options:
    # - 'simple'     : 10-15 / 15-20
    # - 'simple-int' : 10-14 / 15-19
    # - 'math'       : [10, 20)
    # - 'down'       : < 15 /  < 20
    # - 'up'         : >= 10 / >= 15
    # - 'and'        : >= 10 & < 15 / >= 15 & < 20
    # - 'var'        : 10 <= v < 15 / 15 <= v < 20
    #

    summarizable_attrs = Plot.summarizable_attrs + [
        'lang', 'labels', 'extend', 'level_range_style', 'draw_colors',
        'draw_contours', 'mark_field_max', 'mark_release_site',
        'text_box_setup', 'boxes'
    ]

    def __init__(self, *args, lang=None, **kwargs):
        """Create an instance of ``DispersionPlot``.

        Args:
            *args: Additional positional arguments passed to
                ``Plot.__init__``.

            lang (str, optional): Language, e.g., 'de' for German.
                Defaults to 'en' (English).

            **kwargs: Additional keyword arguments passed to
                ``Plot.__init__``.
        """
        super().__init__(*args, **kwargs)
        self.lang = lang or 'en'

        self.labels = DispersionPlotLabels(lang)

        # Formatting arguments
        self._max_marker_kwargs = {
            'marker': '+',
            'color': 'black',
            'markersize': 10,
            'markeredgewidth': 1.5,
        }
        self._site_marker_kwargs = {
            'marker': '^',
            'markeredgecolor': 'red',
            'markerfacecolor': 'white',
            'markersize': 7.5,
            'markeredgewidth': 1.5,
        }

        # Map plot configuration
        self.map_conf = {
            **super().map_conf,
            'bbox_pad_rel': -0.01,
            'geogr_res': '10m',
            #'geogr_res': '50m',
            'ref_dist_x0': 0.046,
            'ref_dist_y0': 0.96,
        }

        self.define_colors()
        self.define_levels()

        self.create()

    def create(self):
        """Create plot."""

        # Prepare plot
        self.prepare_plot()

        # Plot particle concentration field
        self.draw_map_plot()

        # Add text boxes around plot
        self.fig_add_text_boxes()
        self.fill_box_top()
        self.fill_box_right_top()
        self.fill_box_right_bottom()
        self.fill_box_bottom()
        self.draw_boxes()

    def summarize(self, *args, **kwargs):
        data = super().summarize(*args, **kwargs)
        # ...  SR_TODO
        return data

    def fld_nonzero(self):
        return np.where(self.field.fld > 0, self.field.fld, np.nan)

    def draw_map_plot(self):
        """Plot the particle concentrations onto the map."""

        # Add reference distance indicator
        self.ax_map.add_ref_dist_indicator()

        # Plot concentrations
        h_con = self._draw_colors_contours()

        if self.mark_release_site:
            # Add marker at release site
            self.ax_map.marker(
                self.field.attrs.release.lon.value,
                self.field.attrs.release.lat.value,
                **self._site_marker_kwargs,
            )

        if self.mark_field_max:
            # Add marker at location of maximum value
            self.ax_map.mark_max(self.field.fld, **self._max_marker_kwargs)

        return h_con

    def _draw_colors_contours(self):

        if not self.draw_colors:
            h_col = None
        else:
            h_col = self.ax_map.contourf(
                np.log10(self.fld_nonzero()),
                levels=self.levels_log10,
                colors=self.colors,
                extend=self.extend,
            )

        if not self.draw_contours:
            h_con = None
        else:
            h_con = self.ax_map.contour(
                np.log10(self.fld_nonzero()),
                levels=self.levels_log10,
                colors='black',
                linewidths=1,
            )
            return (h_col, h_con)

    #==================================================================
    # Text Boxes
    #==================================================================

    def fig_add_text_boxes(self):
        """Add empty text boxes to the figure around the map plot.

        Args:
            h_rel_t (float, optional): Height of top box as a fraction
                of the height of the map plot. Defaults to <TODO>.

            h_rel_b (float, optional): Height of bottom box as a
                fraction of the height of the map plot. Defaults to
                <TODO>.

            w_rel_r (float, optional): Width of the right boxes as a
                fraction of the width of the map plot. Default to <TODO>.

            pad_hor_rel (float, optional): Padding between map plot and
                the text boxes as a fraction of the map plot width. The
                same absolute padding is used in the horizontal and
                vertical direction. Defaults to <TODO>.

            h_rel_box_rt (float, optional): Height of the top box to
                the right of the map plot as a fraction of the combined
                height of both right boxees. Defaults to <TODO>.

        """

        h_rel_t = self.text_box_setup['h_rel_t']
        h_rel_b = self.text_box_setup['h_rel_b']
        w_rel_r = self.text_box_setup['w_rel_r']
        pad_hor_rel = self.text_box_setup['pad_hor_rel']
        h_rel_box_rt = self.text_box_setup['h_rel_box_rt']

        # Freeze the map plot in order to fix it's coordinates (bbox)
        self.fig.canvas.draw()

        # Obtain aspect ratio of figure
        fig_pxs = self.fig.get_window_extent()
        fig_aspect = fig_pxs.width/fig_pxs.height

        # Get map dimensions in figure coordinates
        w_map, h_map = ax_dims_fig_coords(self.fig, self.ax_map.ax)

        # Relocate the map close to the lower left corner
        x0_map, y0_map = 0.05, 0.05
        self.ax_map.ax.set_position([x0_map, y0_map, w_map, h_map])

        # Determine height of top box and width of right boxes
        w_box = w_rel_r*w_map
        h_box_t = h_rel_t*h_map
        h_box_b = h_rel_b*h_map

        # Determine padding between plot and boxes
        pad_hor = pad_hor_rel*w_map
        pad_ver = pad_hor*fig_aspect

        # Add axes for text boxes (one on top, two to the right)
        self.boxes = [
            # Top
            TextBoxAxes(
                self.fig, self.ax_map.ax, [
                    x0_map,
                    y0_map + pad_ver + h_map,
                    w_map + pad_hor + w_box,
                    h_box_t,
                ]),
            # Right/top
            TextBoxAxes(
                self.fig, self.ax_map.ax, [
                    x0_map + pad_hor + w_map,
                    y0_map + 0.5*pad_ver + (1.0 - h_rel_box_rt)*h_map,
                    w_box,
                    h_rel_box_rt*h_map - 0.5*pad_ver,
                ]),
            # Right/bottom
            TextBoxAxes(
                self.fig, self.ax_map.ax, [
                    x0_map + pad_hor + w_map,
                    y0_map,
                    w_box,
                    (1.0 - h_rel_box_rt)*h_map - 0.5*pad_ver,
                ]),
            # Bottom
            TextBoxAxes(
                self.fig,
                self.ax_map.ax, [
                    x0_map,
                    y0_map - h_box_b,
                    w_map + pad_hor + w_box,
                    h_box_b,
                ],
                show_border=False),
        ]

    def draw_boxes(self):
        for box in self.boxes:
            box.draw()

    #------------------------------------------------------------------
    # Top
    #------------------------------------------------------------------

    def fill_box_top(self, *, skip_pos=None):
        """Fill the box above the map plot.

        Args:
            skip_pos (set, optional): Positions (e.g., 'tl' for top-left;
                see ``pyflexplot.plot.BoxLocation`` for all options) to
                be skipped. Defaults to ``{}``.

        """
        box = self.boxes[0]

        if skip_pos is None:
            skip_pos = {}

        if not 'tl' in skip_pos:
            # Top left: variable
            s = f"{self.field.attrs.variable.long_name.value}"
            box.text('tl', s, size='x-large')

        if not 'tc' in skip_pos:
            # Top center: species
            s = f"{self.field.attrs.species.name.format(join=' + ')}"
            box.text('tc', s, size='x-large')

        if not 'tr' in skip_pos:
            # Top right: datetime
            timestep_fmtd = self.field.attrs.simulation.now.format()
            s = f"{timestep_fmtd}"
            box.text('tr', s, size='x-large')

        if not 'bl' in skip_pos:
            # Bottom left: integration time & level range
            _sim = self.field.attrs.simulation
            if self.lang == 'en':
                _sum, _since = 'Sum', 'since'
            elif self.lang == 'de':
                _sum, _since = r'Summe', 'seit'
            s = (
                f"{_sim.format_integr_period()} {_sum} "
                f"({_since} {_sim.integr_start.format(relative=True)})")
            lvl_range = self.field.attrs.variable.format_level_range()
            if lvl_range:
                s = f"{s}, {lvl_range}"
            box.text('bl', s, size='large')

        if not 'bc' in skip_pos:
            # Bottom center: release site
            s = f"{self.field.attrs.release.site_name.value}"
            box.text('bc', s, size='large')

        if not 'br' in skip_pos:
            # Bottom right: time into simulation
            s = self.field.attrs.simulation.now.format(relative=True)
            box.text('br', s, size='large')

        return box

    #------------------------------------------------------------------
    # Right/Top
    #------------------------------------------------------------------

    def fill_box_right_top(
            self, dx=1.2, dy_line=3.0, dy0_markers=0.5, w_box=4, h_box=2):
        """Fill the top box to the right of the map plot."""
        box = self.boxes[1]

        # Vertical position of legend (depending on number of levels)
        _f = (
            self.n_levels + int(self.extend in ['min', 'both']) +
            int(self.extend in ['max', 'both']))
        dy0_labels = 23 - 1.5*_f
        dy0_boxes = dy0_labels - 0.2*h_box

        # Add box title
        s = self._format_legend_box_title()
        box.text('tc', s=s, dy=1, size='large')

        # Format level ranges (contour plot legend)
        labels = self._format_level_ranges()

        # Add level labels
        box.text_block(
            'br',
            labels,
            dy0=dy0_labels,
            dy_line=dy_line,
            dx=-dx,
            reverse=True,
            size='small',
            family='monospace',
        )

        # Add color boxes
        dy = dy0_boxes
        for color in self.colors[::-1]:
            box.color_rect(
                loc='bl',
                fc=color,
                ec='black',
                dx=dx,
                dy=dy,
                w=w_box,
                h=h_box,
                lw=1.0,
            )
            dy += dy_line

        # Field maximum
        if self.mark_field_max:

            # Add maximum value marker
            dy_max = dy0_markers + dy_line
            box.marker(
                loc='bl',
                dx=dx + 1.0,
                dy=dy_max + 0.7,
                **self._max_marker_kwargs,
            )

            # Add text
            fld_max_fmtd = f'{self.labels.release.max}: '
            if np.isnan(self.field.fld).all():
                fld_max_fmtd += 'NaN'
            else:
                fld_max_fmtd += (
                    f'{np.nanmax(self.field.fld):.2E}'
                    f' {self.field.attrs.variable.unit.format()}')
            box.text(
                loc='bl',
                s=fld_max_fmtd,
                dx=5.5,
                dy=dy_max,
                size='small',
            )

        if self.mark_release_site:
            # Add release site marker
            dy_site = dy0_markers
            box.marker(
                loc='bl',
                dx=dx + 1.0,
                dy=dy_site + 0.7,
                **self._site_marker_kwargs,
            )
            s = (
                f"{self.labels.release.site_name}: "
                f"{self.field.attrs.release.site_name.value}")
            box.text(loc='bl', s=s, dx=5.5, dy=dy_site, size='small')

    def _format_legend_box_title(self):
        s = f"{self.field.attrs.variable.short_name.format()}"
        s += f" ({self.field.attrs.variable.unit.format()})"
        return s

    def _format_level_ranges(self, n=15):
        """Format the levels ranges for the contour plot legend."""

        labels = []

        # Under range
        if self.extend in ('min', 'both'):
            labels.append(self._format_level_range(None, self.levels[0], n))

        # In range
        for lvl0, lvl1 in zip(self.levels[:-1], self.levels[1:]):
            labels.append(self._format_level_range(lvl0, lvl1, n))

        # Over range
        if self.extend in ('max', 'both'):
            labels.append(self._format_level_range(self.levels[-1], None, n))

        #SR_TMP<
        _n = len(self.colors)
        assert len(labels) == _n, f'{len(labels)} != {_n}'
        #SR_TMP>

        return labels

    def _format_level_range(self, lvl0, lvl1, n):

        level_range_var = 'v'

        if lvl0 is not None and lvl1 is not None:
            # Closed range
            lvl0_fmtd = self._format_level(lvl0)
            lvl1_fmtd = self._format_level(lvl1)
            if self.level_range_style == 'simple':
                label = f"{lvl0_fmtd:>6} - {lvl1_fmtd:<6}"
            elif self.level_range_style == 'simple-int':
                label = self._format_level_range_simple_int(lvl0, lvl1)
            elif self.level_range_style == 'math':
                label = f"[{lvl0_fmtd:>}, {lvl1_fmtd:<})"
            elif self.level_range_style == 'down':
                label = f"< {lvl1_fmtd:<}"
            elif self.level_range_style == 'up':
                label = r'$\geq$' + f" {lvl0_fmtd:<}"
            elif self.level_range_style == 'and':
                label = r'$\geq$' + f" {lvl0_fmtd:>} & < {lvl1_fmtd:<}"
            elif self.level_range_style == 'var':
                label = (
                    f"{lvl0_fmtd:>} " + r'$\leq$' + f" {level_range_var} < "
                    f"{lvl1_fmtd:<}")
            else:
                raise Exception(
                    f"unknown level range style '{self.level_range_style}'")

        else:
            # Open-ended range
            if lvl0 is not None:
                op = r'$\geq$'
                lvl = lvl0
            elif lvl1 is not None:
                op = '<'
                lvl = lvl1
            lvl_fmtd = self._format_level(lvl)
            label = f"{op+' '+lvl_fmtd:^15}"
            if self.level_range_style == 'var':
                label = f'{level_range_var} {label}'

        if n is not None:
            label = f'{{label:^{n}}}'.format(label=label)

        return label

    def _format_level_range_simple_int(self, lvl0, lvl1):

        # Check input types
        if int(lvl0) != float(lvl0):
            raise ValueError(
                f"lvl0 is not an integer: {int(lvl0)} != {float(lvl0)}")
        if int(lvl1) != float(lvl1):
            raise ValueError(
                f"lvl1 is not an integer: {int(lvl1)} != {float(lvl1)}")

        # Determine level increment
        dlvls = sorted(set((self.levels[1:] - self.levels[:-1]).tolist()))
        if len(dlvls) != 1:
            raise Exception(
                f"varying level increments: {dlvls} (levels: {self.levels})")
        dlvl = next(iter(dlvls))
        if int(dlvl) != float(dlvl):
            raise ValueError(
                f"dlvl is not an integer: {int(dlvl)} != {float(dlvl)}")

        lvl0_fmtd = self._format_level(lvl0)
        lvl1 = lvl1 - 1
        if lvl1 == lvl0:
            return f"{lvl0_fmtd}"
        lvl1_fmtd = self._format_level(lvl1)
        return f"{lvl0_fmtd:>6} - {lvl1_fmtd:<6}"

    def _format_level(self, lvl):
        if lvl is None:
            return ''
        fmtd = f'{lvl:.0E}'
        n = len(fmtd)
        ll = np.log10(lvl)
        if ll >= -n + 2 and ll <= n - 1:
            fmtd = f'{lvl:f}' [:n]
            assert '1' in fmtd
        return fmtd

    #------------------------------------------------------------------
    # Right/Bottom
    #------------------------------------------------------------------

    def fill_box_right_bottom(self):
        """Fill the bottom box to the right of the map plot."""
        box = self.boxes[2]

        # Add box title
        s = {'en': 'Release', 'de': 'Austritt'}[self.lang]
        #box.text('tc', s, size='large')
        box.text('tc', s, dy=-1.5, size='large')

        # Release site coordinates
        _lat = Degrees(self.field.attrs.release.lat.value)
        lat_fmtd = (
            f"{_lat.degs()}" + r'$^\circ\,$' + f"{_lat.mins()}'" + r'$\,$N'
            f" ({_lat.frac():.2f}" + r'$^\circ\,$N)')
        _lon = Degrees(self.field.attrs.release.lon.value)
        _east = {'en': 'E', 'de': 'O'}[self.lang]
        lon_fmtd = (
            f"{_lon.degs()}" + r'$^\circ\,$' + f"{_lon.mins()}'" + r'$\,$'
            f"{_east} ({_lon.frac():.2f}" + r'$^\circ\,$' + f"{_east})")

        l = self.labels
        a = self.field.attrs
        info_blocks = dedent(
            f"""\
            {l.release.lat}:\t{lat_fmtd}
            {l.release.lon}:\t{lon_fmtd}
            {l.release.height}:\t{a.release.height.format()}

            {l.simulation.start}:\t{a.simulation.start.format()}
            {l.simulation.end}:\t{a.simulation.end.format()}
            {l.release.rate}:\t{a.release.rate.format()}
            {l.release.mass}:\t{a.release.mass.format()}

            {l.species.name}:\t{a.species.name.format()}
            {l.species.half_life}:\t{a.species.half_life.format()}
            {l.species.deposit_vel}:\t{a.species.deposit_vel.format()}
            {l.species.sediment_vel}:\t{a.species.sediment_vel.format()}
            {l.species.washout_coeff}:\t{a.species.washout_coeff.format()}
            {l.species.washout_exponent}:\t{a.species.washout_exponent.format()}
            """)

        # Add lines bottom-up (to take advantage of baseline alignment)
        dy0 = 2
        dy = 2.5
        box.text_blocks_hfill(
            'b',
            dy0=dy0,
            dy_line=dy,
            blocks=info_blocks,
            reverse=True,
            size='small')

    #------------------------------------------------------------------
    # Bottom
    #------------------------------------------------------------------

    def fill_box_bottom(self):
        """Fill the box to the bottom of the map plot."""
        box = self.boxes[3]

        # FLEXPART/model info
        s = self._flexpart_model_info()
        box.text('tl', dx=-0.7, dy=0.5, s=s, size='small')

        # MeteoSwiss Copyright
        cpright_fmtd = self.labels.simulation.copyright
        box.text('tr', dx=0.7, dy=0.5, s=cpright_fmtd, size='small')

    def _flexpart_model_info(self):
        model = self.field.attrs.simulation.model_name.value
        simstart_fmtd = self.field.attrs.simulation.start.format()
        return (
            f"{self.labels.simulation.flexpart_based_on} {model}, "
            f"{simstart_fmtd}")

    #==================================================================

    def define_colors(self):
        if self.cmap == 'flexplot':
            self.colors = colors_flexplot(self.n_levels, self.extend)
        else:
            cmap = mpl.cm.get_cmap(self.cmap)
            colors_core = [
                cmap(i/(self.n_levels - 1)) for i in range(self.n_levels)
            ]

    def define_levels(self):
        self.levels = self.auto_levels_log10()

    def auto_levels_log10(self, n_levels=None, val_max=None):
        if n_levels is None:
            n_levels = self.n_levels
        if val_max is None:
            val_max = self.field.time_stats['max']
        log10_max = int(np.floor(np.log10(val_max)))
        log10_d = 1
        levels_log10 = np.arange(
            log10_max - (n_levels - 1)*log10_d,
            log10_max + 0.5*log10_d,
            log10_d,
        )
        return 10**levels_log10

    @property
    def levels_log10(self):
        return np.log10(self.levels)


def colors_flexplot(n_levels, extend):

    #color_under = [i/255.0 for i in (255, 155, 255)]
    color_under = [i/255.0 for i in (200, 200, 200)]
    color_over = [i/255.0 for i in (200, 200, 200)]

    # yapf: disable
    colors_core_8 = (np.array([
        (224, 196, 172),  #
        (221, 127, 215),  #
        ( 99,   0, 255),  #
        (100, 153, 199),  #
        ( 34, 139,  34),  #
        ( 93, 255,   2),  #
        (199, 255,   0),  #
        (255, 239,  57),  #
    ], float)/255).tolist()
    # yapf: enable
    colors_core_7 = [colors_core_8[i] for i in (0, 1, 2, 3, 5, 6, 7)]
    colors_core_6 = [colors_core_8[i] for i in (1, 2, 3, 4, 5, 7)]
    colors_core_5 = [colors_core_8[i] for i in (1, 2, 4, 5, 7)]

    try:
        colors_core = {
            6: colors_core_5,
            7: colors_core_6,
            8: colors_core_7,
            9: colors_core_8,
        }[n_levels]
    except KeyError:
        raise ValueError(f"n_levels={n_levels}")

    if extend == 'none':
        return colors_core
    elif extend == 'min':
        return [color_under] + colors_core
    elif extend == 'max':
        return colors_core + [color_over]
    elif extend == 'both':
        return [color_under] + colors_core + [color_over]
    raise ValueError(f"extend='{extend}'")


#----------------------------------------------------------------------
# Deterministic Simulation
#----------------------------------------------------------------------


class Plot_Concentration(DispersionPlot):
    """FLEXPART plot of particle concentration at a certain level."""

    name = 'concentration'
    n_levels = 8


class Plot_Deposition(DispersionPlot):
    """FLEXPART plot of surface deposition."""

    name = 'deposition'
    n_levels = 9


class Plot_AffectedArea(DispersionPlot):
    """FLEXPART plot of area affected by surface deposition."""

    name = 'affected_area'


class Plot_AffectedAreaMono(Plot_AffectedArea):
    """FLEXPART plot of area affected by surface deposition (mono)."""

    name = 'affected_area_mono'
    extend = 'none'
    n_levels = 1

    def define_colors(self):
        self.colors = (np.array([(200, 200, 200)])/255).tolist()

    def define_levels(self):
        levels = self.auto_levels_log10(n_levels=9)
        self.levels = np.array([levels[0], np.inf])

    def fill_box_right_top(self):
        super().fill_box_right_top()


#----------------------------------------------------------------------
# Ensemble Simulation
#----------------------------------------------------------------------


class Plot_Ens(Plot):
    name = 'ens'

    text_box_setup = {
        'h_rel_t': 0.12,
        'h_rel_b': 0.03,
        'w_rel_r': 0.25,
        'pad_hor_rel': 0.015,
        'h_rel_box_rt': 0.46,
    }

    def fill_box_top(self, *, skip_pos=None):

        skip_pos_parent = {'tl'}
        if skip_pos is None:
            skip_pos = {}
        else:
            skip_pos_parent.update(set(skip_pos))

        box = super().fill_box_top(skip_pos=skip_pos_parent)

        #SR_TMP< TODO separate input and ensemble variables
        if not 'tl' in skip_pos:
            # Top left: variable
            s = f"{self.field.attrs.variable.long_name.value}"
            box.text('tl', s, size='x-large')
        #SR_TMP>

    def _flexpart_model_info(self):
        model = self.field.attrs.simulation.model_name.value
        simstart_fmtd = self.field.attrs.simulation.start.format()
        return (
            f"{self.labels.simulation.flexpart_based_on} {model} Ensemble, "
            #+f"{simstart_fmtd} (??? Members: ???)")
            f"{simstart_fmtd} (21 Members: 000-020)")


class Plot_EnsMean_Concentration(Plot_Ens, Plot_Concentration):
    name = 'ens_mean_concentration'


class Plot_EnsMean_Deposition(Plot_Ens, Plot_Deposition):
    name = 'ens_mean_deposition'


class Plot_EnsMeanAffectedArea(Plot_Ens, Plot_AffectedArea):
    name = 'ens_mean_affected_area'


class Plot_EnsMeanAffectedAreaMono(Plot_Ens, Plot_AffectedAreaMono):
    name = 'ens_mean_affected_area_mono'


class Plot_EnsThrAgrmt(Plot_Ens):
    name = 'ens_thr_agrmt'

    n_levels = 6
    d_level = 2
    extend = 'min'
    level_range_style = 'simple-int'  # 10-14 / 15-20
    mark_field_max = False

    def define_levels(self):
        n_max = 20  #SR_TMP SR_HC
        d = self.d_level
        self.levels = np.arange(
            n_max - self.d_level*
            (self.n_levels - 1), n_max + self.d_level, self.d_level) + 1

    def _draw_colors_contours(self):

        # If upper end of range is closed, color areas beyond black
        colors_plot = copy(self.colors)
        if self.extend in ['none', 'min']:
            colors_plot.append('black')
            extend_plot = {'none': 'max', 'min': 'both'}[self.extend]
        else:
            extend_plot = self.extend

        if not self.draw_colors:
            h_col = None
        else:
            h_col = self.ax_map.contourf(
                self.fld_nonzero(),
                levels=self.levels,
                colors=colors_plot,
                extend=extend_plot,
            )

        if not self.draw_contours:
            h_con = None
        else:
            h_con = self.ax_map.contour(
                self.fld_nonzero(),
                levels=self.levels,
                colors='black',
                linewidths=1,
            )

        return (h_col, h_con)

    def _format_legend_box_title(self):
        no = {'en': 'No.', 'de': 'Anz.'}[self.lang]
        name = self.field.attrs.variable.short_name.format()
        thresh = self.field.field_specs.ens_var_setup['thr']
        unit = self.field.attrs.variable.unit.format()
        geq = r'$\geq$'
        return f"{no} {name}\n({geq} {thresh} {unit})"

    def _format_level(self, lvl):
        return f'{lvl}'


class Plot_EnsThrAgrmt_Concentration(Plot_EnsThrAgrmt, Plot_Concentration):
    name = 'ens_thr_agrmt_concentration'


class Plot_EnsThrAgrmt_Deposition(Plot_EnsThrAgrmt, Plot_Deposition):
    name = 'ens_thr_agrmt_deposition'


#----------------------------------------------------------------------
