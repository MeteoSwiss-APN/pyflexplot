# -*- coding: utf-8 -*-
"""
Plotters.
"""
# Standard library
# import numpy as np
import re

# First-party
from srutils.various import isiterable

# Local
from .plot import Plot
from .plot_utils import MapAxesConf_Cosmo1
from .plot_utils import MapAxesConf_Cosmo1_CH


class Plotter:
    """Create one or more FLEXPLART plots of a certain type."""

    specs_fmt_keys = {
        "time": "time_idx",
        "nageclass": "age_idx",
        "numpoint": "rel_pt_idx",
        "level": "level_idx",
        "species_id": "species_id",
        "integrate": "integrate",
    }

    def __init__(self):
        self.file_paths = []

    def run(self, name, field, config, **kwargs_plot):
        """Create one or more plots.

        Args:
            name (str): Name.

            field (Field*, list[Field*]): One or more Field instances.

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
        if config.outfile is None:
            raise ValueError("config.outfile is None")

        self.name = name
        self.config = config
        # SR_DBG <
        self.file_path_fmt = config.outfile
        self.domain = config.domain
        self.lang = config.lang
        # SR_DBG >

        fields = field if isinstance(field, (list, tuple)) else [field]
        assert all(type(obj).__name__.startswith("Field") for obj in fields)  # SR_DBG

        _s = "s" if len(fields) > 1 else ""
        print(f"create {len(fields)} {self.name} plot{_s}")

        # SR_TMP < TODO Find less hard-coded solution
        if self.domain == "auto":
            self.domain = "cosmo1"
        if self.domain == "cosmo1":
            map_conf = MapAxesConf_Cosmo1(lang=self.lang)
        elif self.domain == "ch":
            map_conf = MapAxesConf_Cosmo1_CH(lang=self.lang)
        else:
            raise ValueError(f"unknown domain '{self.domain}'")
        # SR_TMP >

        # Create plots one-by-one
        for i_data, field in enumerate(fields):
            file_path = self.format_file_path(field.field_specs)
            _w = len(str(len(fields)))
            print(f" {i_data+1:{_w}}/{len(fields)}  {file_path}")

            Plot.create(
                self.name, field, map_conf=map_conf, lang=self.lang, **kwargs_plot,
            ).save(file_path)

            yield file_path

    def format_file_path(self, field_specs):

        # SR_TMP <
        if re.search(r"{member_ids:[0-9]*d}", self.file_path_fmt):
            raise NotImplementedError(
                "number formatting of member ids of the form '{member_ids:[0-9]*d}'"
            )
        # SR_TMP >

        var_specs_dct = field_specs.multi_var_specs.compressed_dct()

        # Collect variable specifications
        kwargs = {"domain": self.domain}
        for specs_key, fmt_key in self.specs_fmt_keys.items():
            try:
                val = var_specs_dct[specs_key]
            except KeyError:
                pass
            else:
                if specs_key == "integrate":
                    val = "int" if val else "no-int"
                kwargs[fmt_key] = val

        kwargs["variable"] = self._fmt_variable(var_specs_dct)
        kwargs["plot_type"] = self.config.plot_type
        kwargs["lang"] = self.lang
        # if field_specs.issubcls("ens"):
        if "ens" in self.name:  # SR_TMP
            kwargs["member_ids"] = self._fmt_member_ids(field_specs)

        # Format the file path
        # Don't use str.format in order to handle multival elements
        file_path = self._fmt_file_path(kwargs)

        # Add number if file path not unique
        file_path = self.ensure_unique_path(file_path)

        return file_path

    def _fmt_variable(self, var_specs_dct):
        """
        Variable name.
        """
        if self.name == "deposition":
            deposition_type = var_specs_dct["deposition"]
            return f"{deposition_type}_{self.name}"
        return self.name

    def _fmt_member_ids(self, field_specs):
        """
        Ensemble member ids.
        """
        if not field_specs.member_ids:
            return None
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
        return s

    def _fmt_file_path(self, kwargs):
        file_path = self.file_path_fmt
        for key, val in kwargs.items():
            if not isiterable(val, str_ok=False):
                val = [val]
            # Iterate over relevant format keys
            rxs = r"{" + key + r"(:[^}]*)?}"
            re.finditer(rxs, file_path)
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

    def ensure_unique_path(self, path):
        """If file path has been used before, add/increment trailing number."""
        while path in self.file_paths:
            path = self.derive_unique_path(path)
        self.file_paths.append(path)
        return path

    @staticmethod
    def derive_unique_path(path):
        """Add/increment a trailing number to a file path."""

        # Extract suffix
        if path.endswith(".png"):
            suffix = ".png"
        else:
            raise NotImplementedError(f"unknown suffix: {path}")
        path_base = path[: -len(suffix)]

        # Reuse existing numbering if present
        match = re.search(r"-(?P<i>[0-9]+)$", path_base)
        if match:
            i = int(match.group("i")) + 1
            w = len(match.group("i"))
            path_base = path_base[: -w - 1]
        else:
            i = 1
            w = 1

        # Add numbering and suffix
        path = f"{path_base}-{{i:0{w}}}{suffix}".format(i=i)

        return path
