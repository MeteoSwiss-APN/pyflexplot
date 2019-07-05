# -*- coding: utf-8 -*-
"""
Plots.
"""
import logging as log
import matplotlib.pyplot as plt
import numpy as np
import os

from textwrap import dedent

from .plot import AxesMapRotPole
from .plot import ax_dims_fig_coords
from .plot import AxesTextBox
from .utils import Degrees
from .utils_dev import ipython  #SR_DEV


class FlexPlotBase:
    """Base class for FLEXPART plots."""

    def __init__(self, rlat, rlon, fld, attrs, time_stats):
        """Create an instance of ``FlexPlotBase``.

        Args:
            rlat (ndarray[float]): Rotated latitude (1d).

            rlon (ndarray[float]): Rotated longitude (1d).

            fld (ndarray[float, float]): Concentration field (2d).

            attrs (dict): Instance of ``FlexAttrsCollection``.

            time_stats (dict): Some statistics across all time steps.

        """
        self.rlat = rlat
        self.rlon = rlon
        self.fld = np.where(fld > 0, fld, np.nan)
        self.attrs = attrs
        self.time_stats = time_stats

    def prepare_plot(self):

        self.fig = plt.figure(figsize=self.figsize)

        self.ax_map = AxesMapRotPole(
            self.fig,
            self.rlat,
            self.rlon,
            self.attrs.grid.north_pole_lat,
            self.attrs.grid.north_pole_lon,
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


class FlexPlotBase_Dispersion(FlexPlotBase):
    """Base class for FLEXPART dispersion plots."""

    figsize = (12, 9)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
            'bbox_pad_rel': -0.01,
            'geogr_res': '10m',
            #'geogr_res': '50m',
            'ref_dist_x0': 0.046,
            'ref_dist_y0': 0.96,
        }

        self.find_contour_levels(n=8, extend='max')

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

    def draw_map_plot(self):
        """Plot the particle concentrations onto the map."""

        # Define colors
        # yapf: disable
        self.colors = (np.array([
            (255, 155, 255),  # -> under
            (224, 196, 172),  # \
            (221, 127, 215),  # |
            ( 99,   0, 255),  # |
            (100, 153, 199),  #  > range
            ( 93, 255,   2),  # |
            (199, 255,   0),  # |
            (255, 239,  57),  # /
            (200, 200, 200),  # -> over
        ], float)/255).tolist()
        # yapf: enable
        if self.extend in ['none', 'max']:
            self.colors.pop(0)  # Remove `under`
        if self.extend in ['none', 'min']:
            self.colors.pop(-1)  # Remove `over`

        # Add reference distance indicator
        self.ax_map.add_ref_dist_indicator()

        # Plot concentrations
        fld_log10 = np.log10(self.fld)
        handle = self.ax_map.contourf(
            fld_log10,
            levels=self.levels_log10,
            colors=self.colors,
            extend=self.extend,
        )

        # Add marker at release site
        self.ax_map.marker(
            self.attrs.release.site_lon,
            self.attrs.release.site_lat,
            **self._site_marker_kwargs,
        )

        # Add marker at location of maximum value
        self.ax_map.mark_max(self.fld, **self._max_marker_kwargs)

        return handle

    def fig_add_text_boxes(
            self,
            h_rel_t=0.1,
            h_rel_b=0.03,
            w_rel_r=0.25,
            pad_hor_rel=0.015,
            h_rel_box_rt=0.44):
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
        self.axs_box = np.array([
            # Top
            AxesTextBox(
                self.fig, self.ax_map.ax, [
                    x0_map,
                    y0_map + pad_ver + h_map,
                    w_map + pad_hor + w_box,
                    h_box_t,
                ]),
            # Right/top
            AxesTextBox(
                self.fig, self.ax_map.ax, [
                    x0_map + pad_hor + w_map,
                    y0_map + 0.5*pad_ver + (1.0 - h_rel_box_rt)*h_map,
                    w_box,
                    h_rel_box_rt*h_map - 0.5*pad_ver,
                ]),
            # Right/bottom
            AxesTextBox(
                self.fig, self.ax_map.ax, [
                    x0_map + pad_hor + w_map,
                    y0_map,
                    w_box,
                    (1.0 - h_rel_box_rt)*h_map - 0.5*pad_ver,
                ]),
            # Bottom
            AxesTextBox(
                self.fig,
                self.ax_map.ax, [
                    x0_map,
                    y0_map - h_box_b,
                    w_map + pad_hor + w_box,
                    h_box_b,
                ],
                show_border=False),
        ])

    def fill_box_top(self, *, skip_pos=None):
        """Fill the box above the map plot."""
        box = self.axs_box[0]

        if skip_pos is None:
            skip_pos = []

        if not 'tl' in skip_pos:
            # Top left: variable
            s = f"{self.attrs.variable.long_name}"
            box.text('tl', s, size='xx-large')

        if not 'tc' in skip_pos:
            # Top center: species
            s = f"{self.attrs.species.name}"
            box.text('tc', s, size='xx-large')

        if not 'tr' in skip_pos:
            # Top right: datetime
            timestep_fmtd = self.attrs.simulation.format_now()
            s = f"{timestep_fmtd}"
            box.text('tr', s, size='xx-large')

        if not 'bl' in skip_pos:
            # Bottom left: level range
            s = f"{self.attrs.variable.format_level_range()}"
            box.text('bl', s, size='large')

        if not 'br' in skip_pos:
            # Bottom center: release site
            #s = f"Release site: {self.attrs.release.site_name}"
            s = f"{self.attrs.release.site_name}"
            box.text('bc', s, size='large')

        if not 'br' in skip_pos:
            # Bottom right: time into simulation
            delta = self.attrs.simulation.now - self.attrs.simulation.start
            hours = int(delta.total_seconds()/3600)
            mins = int((delta.total_seconds()/3600)%1*60)
            s = f"T$_{0}$ + {hours:02d}:{mins:02d} h"
            box.text('br', s, size='large')

    def fill_box_right_top(self):
        """Fill the top box to the right of the map plot."""
        box = self.axs_box[1]

        # Add box title
        s = f"{self.attrs.variable.format_short_name()}"
        #box.text('tc', s=s, size='large')
        box.text('tc', s=s, dy=1, size='large')  #SR_DBG

        # Format level ranges (contour plot legend)
        labels = self._format_level_ranges()

        # Positioning parameters
        dx = 1.2
        dy_line = 3.0
        dy0_labels = 8.0
        w_box, h_box = 4, 2
        dy0_boxes = dy0_labels - 0.4

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

        # Spacing between colors and markers
        dy_spacing_markers = 0.5*dy_line

        # Field maximum
        fld_max_fmtd = 'Max.: '
        if np.isnan(self.fld).all():
            fld_max_fmtd += 'NaN'
        else:
            fld_max_fmtd += (
                f'{np.nanmax(self.fld):.2E}'
                f' {self.attrs.variable.format_unit()}')

        # Add maximum value marker
        dy_max = dy0_labels - dy_spacing_markers - dy_line
        box.marker(
            loc='bl',
            dx=dx + 1.0,
            dy=dy_max + 0.7,
            **self._max_marker_kwargs,
        )
        box.text(
            loc='bl',
            s=fld_max_fmtd,
            dx=5.5,
            dy=dy_max,
            size='small',
        )

        # Add release site marker
        dy_site = dy0_labels - dy_spacing_markers - 2*dy_line
        box.marker(
            loc='bl',
            dx=dx + 1.0,
            dy=dy_site + 0.7,
            **self._site_marker_kwargs,
        )
        s = f"Release Site: {self.attrs.release.site_name}"
        box.text(loc='bl', s=s, dx=5.5, dy=dy_site, size='small')

    def _format_level_ranges(self):
        """Format the levels ranges for the contour plot legend."""

        labels = []

        # Under range
        if self.extend in ('min', 'both'):
            labels.append(self._format_label(None, self.levels[0]))

        # In range
        for lvl0, lvl1 in zip(self.levels[:-1], self.levels[1:]):
            labels.append(self._format_label(lvl0, lvl1))

        # Over range
        if self.extend in ('max', 'both'):
            labels.append(self._format_label(self.levels[-1], None))

        #SR_TMP<
        _n = len(self.colors)
        assert len(labels) == _n, f'{len(labels)} != {_n}'
        #SR_TMP>

        return labels

    def _format_label(self, lvl0, lvl1):

        def format_level(lvl):
            return '' if lvl is None else f"{lvl:.0E}".strip()

        if lvl0 is not None and lvl1 is not None:
            op = '-'
            return f"{format_level(lvl0):>6} {op} {format_level(lvl1):<6}"
        else:
            if lvl0 is not None:
                op = '>'
                lvl = lvl0
            elif lvl1 is not None:
                op = '<'
                lvl = lvl1

            return f"{op+' '+format_level(lvl):^15}"

    def fill_box_right_bottom(self):
        """Fill the bottom box to the right of the map plot."""
        box = self.axs_box[2]

        # Add box title
        #box.text('tc', 'Release', size='large')
        box.text('tc', 'Release', dy=-1.5, size='large')

        # Release site coordinates
        lat = Degrees(self.attrs.release.site_lat)
        lat_fmtd = (
            f"{lat.degs()}$^\circ\,${lat.mins()}'$\,$N"
            f" ({lat.frac():.2f}$^\circ\,$N)")
        lon = Degrees(self.attrs.release.site_lon)
        lon_fmtd = (
            f"{lon.degs()}$^\circ\,${lon.mins()}'$\,$E"
            f" ({lon.frac():.2f}$^\circ\,$E)")

        info_blocks = dedent(
            f"""\
            Latitude:\t{lat_fmtd}
            Longitude:\t{lon_fmtd}
            Height:\t{self.attrs.release.format_height()}

            Start:\t{self.attrs.simulation.format_start()}
            End:\t{self.attrs.simulation.format_end()}
            Rate:\t{self.attrs.release.format_rate()}
            Total Mass:\t{self.attrs.release.format_mass()}

            Substance:\t{self.attrs.species.name}
            Half-Life:\t{self.attrs.species.format_half_life()}
            Deposit. Vel.:\t{self.attrs.species.format_deposit_vel()}
            Sediment. Vel.:\t{self.attrs.species.format_sediment_vel()}
            Washout Coeff.:\t{self.attrs.species.format_washout_coeff()}
            Washout Exponent:\t{self.attrs.species.washout_exponent:g}
            """)

        # Add lines bottom-up (to take advantage of baseline alignment)
        dy0 = 3
        dy = 2.5
        box.text_blocks_hfill(
            'b',
            dy0=dy0,
            dy_line=dy,
            blocks=info_blocks,
            reverse=True,
            size='small')

    def fill_box_bottom(self):
        """Fill the box to the bottom of the map plot."""
        box = self.axs_box[3]

        # FLEXPART/model info
        _model = self.attrs.simulation.model_name
        _simstart_fmtd = self.attrs.simulation.format_start()
        s = f"FLEXPART based on {_model} {_simstart_fmtd}"
        box.text('tl', dx=-0.7, dy=0.5, s=s, size='small')

        # MeteoSwiss Copyright
        cpright_fmtd = u"\u00a9MeteoSwiss"
        box.text('tr', dx=0.7, dy=0.5, s=cpright_fmtd, size='small')

    def find_contour_levels(self, n, extend):

        self.extend = extend

        log10_max = int(np.log10(self.time_stats['max']))

        log10_d = 1

        self.levels_log10 = np.arange(
            log10_max - (n - 1)*log10_d,
            log10_max + 0.5*log10_d,
            log10_d,
        )

        self.levels = 10**self.levels_log10

        print(self.levels_log10)
        print(self.levels)


class FlexPlotConcentration(FlexPlotBase_Dispersion):
    """FLEXPART plot of particle concentration at a certain level."""

    def __init__(self, *args, **kwargs):
        """Create an instance of ``FlexPlot``."""
        super().__init__(*args, **kwargs)
        self.create()


class FlexPlotDeposition(FlexPlotBase_Dispersion):
    """ """

    def __init__(self, *args, **kwargs):
        """Create an instance of ``FlexPlotDeposition``.

        """
        super().__init__(*args, **kwargs)
        self.create()

    def fill_box_top(self):
        super().fill_box_top(skip_pos=['bl'])
