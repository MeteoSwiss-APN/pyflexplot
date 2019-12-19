# -*- coding: utf-8 -*-
"""
Plotters.
"""
# import numpy as np
import re

from srutils.various import isiterable

from .field_specs import FieldSpecs
from .plot import Plot
from .plot_utils import MapAxesConf_Cosmo1
from .plot_utils import MapAxesConf_Cosmo1_CH
from .utils import ParentClass
from .var_specs import VarSpecs


class Plotter:
    """Create one or more FLEXPLART plots of a certain type."""

    specs_fmt_keys = {
        "time": "time_ind",
        "nageclass": "age_ind",
        "numpoint": "rel_pt_ind",
        "level": "level_ind",
        "species_id": "species_id",
        "integrate": "integrate",
    }

    def run(
        self, name, field, file_path_fmt, *, domain="auto", lang="en", **kwargs_plot
    ):
        """Create one or more plots.

        Args:
            name (str): Name.

            field (Plot*, list[Plot*]): An instance or list of instances of the
                plot class.

            file_path_fmt (str): Format string of output file path. Must
                contain all necessary format keys to avoid that multiple files
                have the same name, but can be a plain string if no variable
                assumes more than one value.

            lang (str, optional): Language, e.g., 'de' for German. Defaults to
                'en' (English).

            **kwargs_plot: Keyword arguments used to instatiate the plot
                instance.

        Yields:
            str: Output file paths.

        """
        self.name = name
        self.file_path_fmt = file_path_fmt
        self.lang = lang

        data_lst = field if isinstance(field, (list, tuple)) else [field]

        _s = "s" if len(data_lst) > 1 else ""
        print(f"create {len(data_lst)} {self.name} plot{_s}")

        # SR_TMP < TODO Find less hard-coded solution
        if domain == "auto":
            domain = "cosmo1"
        if domain == "cosmo1":
            map_conf = MapAxesConf_Cosmo1(lang=lang)
        elif domain == "ch":
            map_conf = MapAxesConf_Cosmo1_CH(lang=lang)
        else:
            raise ValueError(f"unknown domain '{domain}'")
        # SR_TMP >

        # Create plots one-by-one
        for i_data, field in enumerate(data_lst):
            file_path = self.fmt_file_path(field.field_specs, domain)
            _w = len(str(len(data_lst)))
            print(f" {i_data+1:{_w}}/{len(data_lst)}  {file_path}")

            Plot.subcls(self.name)(
                field, map_conf=map_conf, lang=lang, **kwargs_plot
            ).save(file_path)

            yield file_path

    def fmt_file_path(self, field_specs, domain):

        var_specs_dct = field_specs.multi_var_specs.compressed_var_specs()

        # Collect variable specifications
        kwargs = {"domain": domain}
        for specs_key, fmt_key in self.specs_fmt_keys.items():
            try:
                val = var_specs_dct[specs_key]
            except KeyError:
                pass
            else:
                if specs_key == "integrate":
                    val = "int" if val else "no-int"
                kwargs[fmt_key] = val

        # Variable name
        plot_var = self.name
        try:
            dep_type = var_specs_dct["deposition"]
        except KeyError:
            pass
        else:
            if self.name == "deposition":
                plot_var = f"{dep_type}-{plot_var}"
            elif self.name.startswith("affected_area"):
                plot_var = f"{plot_var}_{dep_type}-deposition"
            else:
                raise NotImplementedError(
                    f"plot_var for deposition-based variable {plot_var}"
                )
        kwargs["variable"] = plot_var

        # Language
        kwargs["lang"] = self.lang

        # Ensemble member ids
        # if field_specs.issubcls("ens"):
        if "ens" in self.name:  # SR_TMP
            if not field_specs.member_ids:
                kwargs["member_ids"] = None
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
                        s_i = f"{member_ids[0]:d}"
                    elif len(member_ids) == 2:
                        s_i = f"{member_ids[0]:d}+{member_ids[1]:d}"
                    else:
                        s_i = f"{member_ids[0]:d}-{member_ids[-1]:d}"
                    if s is None:
                        s = s_i
                    else:
                        s += f"+{s_i}"
                kwargs["member_ids"] = s

        # Format the file path
        # Don't use str.format in order to handle multival elements
        file_path = self.file_path_fmt
        for key, val in kwargs.items():
            if not isiterable(val, str_ok=False):
                val = [val]

            n = len(val)

            # Iterate over relevant format keys
            rxs = r"{" + key + r"(:[^}]*)?}"
            for m in re.finditer(rxs, file_path):

                # Obtain format specifier (if there is one)
                try:
                    f = m.group(1)
                except IndexError:
                    f = None
                if not f:
                    f = ""

                # Format the string that replaces this format key in the path
                s = "+".join([f"{{{f}}}".format(v) for v in val])

                # Replace format key in the path by the just formatted string
                file_path = file_path[: m.span()[0]] + s + file_path[m.span()[1] :]

        return file_path
