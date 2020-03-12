# -*- coding: utf-8 -*-
"""
Plotters.
"""
# Standard library
import re

# First-party
from srutils.iter import isiterable

# Local
from .plot import Plot
from .plot_lib import MapAxesConf_Cosmo1
from .plot_lib import MapAxesConf_Cosmo1_CH


def plot(fields, attrs_lst, setup):
    return Plotter().run(fields, attrs_lst, setup)


class Plotter:
    """Create one or more FLEXPLART plots of a certain type."""

    specs_fmt_keys = {
        "time": "time_idx",
        "nageclass": "age_idx",
        "numpoint": "rel_pt_idx",
        "level": "level",
        "species_id": "species_id",
        "integrate": "integrate",
    }

    def __init__(self):
        self.file_paths = []

    def run(self, fields, attrs_lst, setup):
        """Create one or more plots.

        Args:
            field (List[Field]): A list of fields.

            attrs_lst (List[Attr???]): A list of data attributes of equal
                length as ``fields``.

            setup (Setup): Plot setup.

        Yields:
            str: Output file paths.

        """
        if setup.outfile is None:
            raise ValueError("setup.outfile is None")

        self.setup = setup

        _s = "s" if len(fields) > 1 else ""
        print(f"create {len(fields)} {self.setup.plot_type} plot{_s}")

        # SR_TMP < TODO Find less hard-coded solution
        domain = self.setup.domain
        if domain == "auto":
            domain = "cosmo1"
        if domain == "cosmo1":
            map_conf = MapAxesConf_Cosmo1(lang=self.setup.lang)
        elif domain == "ch":
            map_conf = MapAxesConf_Cosmo1_CH(lang=self.setup.lang)
        else:
            raise ValueError("unknown domain", domain)
        # SR_TMP >

        # Create plots one-by-one
        for i_data, (field, attrs) in enumerate(zip(fields, attrs_lst)):
            out_file_path = self.format_out_file_path(attrs._setup)
            _w = len(str(len(fields)))
            print(f" {i_data+1:{_w}}/{len(fields)}  {out_file_path}")
            Plot(field, setup, attrs, map_conf=map_conf).save(out_file_path)
            yield out_file_path

    def format_out_file_path(self, setup):

        variable = setup.variable
        if setup.variable == "deposition":
            variable += f"_{setup.deposition_type}"

        kwargs = {
            "nageclass": setup.nageclass,
            "domain": setup.domain,
            "lang": setup.lang,
            "level": setup.level,
            "noutrel": setup.noutrel,
            "species_id": setup.species_id,
            "time": setup.time,
            "variable": variable,
        }

        # Format the file path
        # Don't use str.format in order to handle multival elements
        out_file_path = self._fmt_out_file_path(kwargs)

        # Add number if file path not unique
        out_file_path = self.ensure_unique_path(out_file_path)

        return out_file_path

    def _fmt_out_file_path(self, kwargs):
        out_file_path = self.setup.outfile
        for key, val in kwargs.items():
            if not isiterable(val, str_ok=False):
                val = [val]
            # Iterate over relevant format keys
            rxs = r"{" + key + r"(:[^}]*)?}"
            re.finditer(rxs, out_file_path)
            for m in re.finditer(rxs, out_file_path):

                # Obtain format specifier (if there is one)
                try:
                    f = m.group(1)
                except IndexError:
                    f = None
                if not f:
                    f = ""

                # Format the string that replaces this format key in the path
                formatted_key = "+".join([f"{{{f}}}".format(v) for v in val])

                # Replace format key in the path by the just formatted string
                start, end = out_file_path[: m.span()[0]], out_file_path[m.span()[1] :]
                out_file_path = f"{start}{formatted_key}{end}"

        # Check that all keys have been formatted
        if "{" in out_file_path or "}" in out_file_path:
            raise Exception(
                f"formatted output file path still contains format keys", out_file_path,
            )

        return out_file_path

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
