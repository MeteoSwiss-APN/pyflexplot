# -*- coding: utf-8 -*-
"""
Plotters.
"""
#import numpy as np

from .io import FlexFieldSpecs
from .flexplot import FlexPlotConcentration
from .flexplot import FlexPlotDeposition


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
        try:
            f_cls_specs = getattr(FlexFieldSpecs, f'cls_{type_}')
        except AttributeError:
            raise ValueError(f"no specs class defined for plot type '{type_}'")
        else:
            self.specs_keys = f_cls_specs().keys()

    @classmethod
    def concentration(cls, *args, **kwargs):
        return cls('concentration').run(*args, **kwargs)

    @classmethod
    def deposition(cls, *args, **kwargs):
        return cls('deposition').run(*args, **kwargs)

    def run(self, *args, **kwargs):
        """Run specific plotter."""
        for path in self._run(*args, **kwargs):
            yield path

    specs_keys = [
        'time_ind',
        'age_ind',
        'rls_pt_ind',
        'level_ind',
        'species_id',
        'source_ind',
        'field_type',
    ]

    def run(self, data, file_path_fmt):
        """Create one or more plots.

        Args:
            data (FlexPlot*, list[FlexPlot*]): An instance or list of
                instances of the plot class.

            file_path_fmt (str): Format string of output file path.
                Must contain all necessary format keys to avoid that
                multiple files have the same name, but can be a plain
                string if no variable assumes more than one value.

        Yields:
            str: Output file paths.

        """
        self.file_path_fmt = file_path_fmt

        data_lst = data if isinstance(data, (list, tuple)) else [data]

        _s = 's' if len(data_lst) > 1 else ''
        print(f"create {len(data_lst)} {self.type_} plot{_s}")

        # Create plots one-by-one
        for i_data, data in enumerate(data_lst):
            file_path = self.format_file_path(data.field_specs)
            _w = len(str(len(data_lst)))
            print(f" {i_data+1:{_w}}/{len(data_lst)}  {file_path}")

            kwargs = {
                'rlat': data.rlat,
                'rlon': data.rlon,
                'fld': data.field,
                'attrs': data.attrs,
            }

            self.cls_plot(**kwargs).save(file_path)

            yield file_path

    def format_file_path(self, specs):
        kwargs = {k: getattr(specs, k) for k in self.specs_keys}

        plot_var = self.type_
        if specs.integrate:
            plot_var += '-int'
        kwargs['variable'] = plot_var

        return self.file_path_fmt.format(**kwargs)


