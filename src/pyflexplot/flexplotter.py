# -*- coding: utf-8 -*-
"""
Plotters.
"""
#import numpy as np

from .io import FlexVarSpecs
from .flexplot import FlexPlotConcentration
from .flexplot import FlexPlotDeposition
from .utils_dev import ipython  #SR_DEV


class FlexPlotter:
    """Create one or more FLEXPLART plots of a certain type.

    Attributes:
        <TODO>

    Methods:
        <TODO>

    """

    def __init__(self, type_):
        """Initialize instance of FlexPlotter.

        Args:
            type_ (str): Type of plot.

        """
        self.type_ = type_

        # Determine plot class
        cls_plot_by_type = {
            'concentration': FlexPlotConcentration,
            'deposition': FlexPlotDeposition,
        }
        try:
            self.cls_plot = cls_plot_by_type[type_]
        except KeyError:
            raise ValueError(f"no plot class defined for plot type '{type_}'")

        # Fetch specs keys
        self.specs_keys = FlexVarSpecs.specs()

    @classmethod
    def concentration(cls, *args, **kwargs):
        return cls('concentration').run(*args, **kwargs)

    @classmethod
    def deposition(cls, *args, **kwargs):
        return cls('deposition').run(*args, **kwargs)

    # yapf: disable
    specs_fmt_keys = {
        'time'      : 'time_ind',
        'nageclass' : 'age_ind',
        'numpoint'  : 'rel_pt_ind',
        'level'     : 'level_ind',
        'species_id': 'species_id',
    }
    # yapf: enable

    def run(self, field, file_path_fmt):
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

        data_lst = field if isinstance(field, (list, tuple)) else [field]

        _s = 's' if len(data_lst) > 1 else ''
        print(f"create {len(data_lst)} {self.type_} plot{_s}")

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
            }

            self.cls_plot(**kwargs).save(file_path)

            yield file_path

    def format_file_path(self, field_specs):

        var_specs = field_specs.var_specs_merged()

        kwargs = {
            f: getattr(var_specs, s)
            for s, f in self.specs_fmt_keys.items()
            if hasattr(var_specs, s)
        }

        plot_var = self.type_
        if var_specs.integrate:
            plot_var += '-int'
        kwargs['variable'] = plot_var

        try:
            file_path = self.file_path_fmt.format(**kwargs)
        except KeyError as e:
            raise KeyError(
                f"cannot format '{self.file_path_fmt}': "
                f"missing key '{e}' among {sorted(kwargs)}")
        else:
            return file_path
