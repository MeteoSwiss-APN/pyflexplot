# -*- coding: utf-8 -*-
"""
Plotters.
"""
#import numpy as np

from .io import FlexVarSpecs
from .flexplot import FlexPlot
from .utils_dev import ipython  #SR_DEV


class FlexPlotter:
    """Create one or more FLEXPLART plots of a certain type."""

    cls_plot = None

    # yapf: disable
    specs_fmt_keys = {
        'time'      : 'time_ind',
        'nageclass' : 'age_ind',
        'numpoint'  : 'rel_pt_ind',
        'level'     : 'level_ind',
        'species_id': 'species_id',
    }
    # yapf: enable

    def run(self, field, file_path_fmt, lang):
        """Create one or more plots.

        Args:
            field (FlexPlot*, list[FlexPlot*]): An instance or list of
                instances of the plot class.

            file_path_fmt (str): Format string of output file path.
                Must contain all necessary format keys to avoid that
                multiple files have the same name, but can be a plain
                string if no variable assumes more than one value.

        Yields:
            str: Output file paths.

        """
        self.file_path_fmt = file_path_fmt
        self.lang = lang

        data_lst = field if isinstance(field, (list, tuple)) else [field]

        _s = 's' if len(data_lst) > 1 else ''
        print(f"create {len(data_lst)} {self.cls_plot.name} plot{_s}")

        # Create plots one-by-one
        for i_data, field in enumerate(data_lst):
            file_path = self.format_file_path(field.field_specs)
            _w = len(str(len(data_lst)))
            print(f" {i_data+1:{_w}}/{len(data_lst)}  {file_path}")

            kwargs = {
                'rlat': field.rlat,
                'rlon': field.rlon,
                'fld': field.fld,
                'attrs': field.attrs,
                'time_stats': field.time_stats,
                'lang': lang,
            }

            self.cls_plot(**kwargs).save(file_path)

            yield file_path

    def format_file_path(self, field_specs):

        var_specs = field_specs.var_specs_merged()

        # Collect variable specifications
        kwargs = {}
        for specs_key, fmt_key in self.specs_fmt_keys.items():
            try:
                val = getattr(var_specs, specs_key)
            except AttributeError:
                pass
            else:
                try:
                    iter(val)
                except TypeError:
                    pass
                else:
                    val = '+'.join([str(i) for i in val])
            kwargs[fmt_key] = val

        # Special case: integrated variable
        plot_var = self.cls_plot.name
        if var_specs.integrate:
            plot_var += '-int'
        try:
            dep_type = var_specs.deposition
        except AttributeError:
            pass
        else:
            plot_var = f'{dep_type}-{plot_var}'
        kwargs['variable'] = plot_var

        # Language
        kwargs['lang'] = self.lang

        # Ensemble member ids
        if field_specs.member_ids is None:
            kwargs['member_ids'] = None
        else:
            member_ids_grouped = []
            for i, member_id in enumerate(field_specs.member_ids):
                if i == 0 or member_id - member_ids_grouped[-1][-1] > 1:
                    member_ids_grouped.append([member_id])
                else:
                    member_ids_grouped[-1].append(member_id)
            s = None
            for member_ids in member_ids_grouped:
                if len(member_ids) == 1:
                    s_i = f'{member_ids[0]:d}'
                elif len(member_ids) == 2:
                    s_i = f'{member_ids[0]:d}+{member_ids[1]:d}'
                else:
                    s_i = f'{member_ids[0]:d}-{member_ids[-1]:d}'
                if s is None:
                    s = s_i
                else:
                    s += f'+{s_i}'
            kwargs['member_ids'] = s

        #ipython(globals(), locals(), f"{type(self).__name__}.format_file_path")

        # Format file path
        try:
            file_path = self.file_path_fmt.format(**kwargs)
        except (TypeError, KeyError) as e:
            raise KeyError(
                f"cannot format '{self.file_path_fmt}' with {kwargs}: "
                f"{type(e).__name__}({e})")
        else:
            return file_path


#----------------------------------------------------------------------
# Deterministic Simulation
#----------------------------------------------------------------------


class FlexPlotter_Concentration(FlexPlotter):
    cls_plot = FlexPlot.Concentration


class FlexPlotter_Deposition(FlexPlotter):
    cls_plot = FlexPlot.Deposition


class FlexPlotter_AffectedArea(FlexPlotter):
    cls_plot = FlexPlot.AffectedArea


class FlexPlotter_AffectedAreaMono(FlexPlotter):
    cls_plot = FlexPlot.AffectedAreaMono


#----------------------------------------------------------------------

FlexPlotter.Concentration = FlexPlotter_Concentration
FlexPlotter.Deposition = FlexPlotter_Deposition
FlexPlotter.AffectedArea = FlexPlotter_AffectedArea
FlexPlotter.AffectedAreaMono = FlexPlotter_AffectedAreaMono

#----------------------------------------------------------------------
# Ensemble Simulation
#----------------------------------------------------------------------


class FlexPlotter_EnsMeanConcentration(FlexPlotter):
    cls_plot = FlexPlot.EnsMeanConcentration


class FlexPlotter_EnsMeanDeposition(FlexPlotter):
    cls_plot = FlexPlot.EnsMeanDeposition


class FlexPlotter_EnsMeanAffectedArea(FlexPlotter):
    cls_plot = FlexPlot.EnsMeanAffectedArea


class FlexPlotter_EnsMeanAffectedAreaMono(FlexPlotter):
    cls_plot = FlexPlot.EnsMeanAffectedAreaMono


#----------------------------------------------------------------------

FlexPlotter.EnsMeanConcentration = FlexPlotter_EnsMeanConcentration
FlexPlotter.EnsMeanDeposition = FlexPlotter_EnsMeanDeposition
FlexPlotter.EnsMeanAffectedArea = FlexPlotter_EnsMeanAffectedArea
FlexPlotter.EnsMeanAffectedAreaMono = FlexPlotter_EnsMeanAffectedAreaMono
