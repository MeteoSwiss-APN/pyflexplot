"""
Output.
"""
# Standard library
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from typing import cast
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Union

# Local
from .meta_data import MetaData
from .setup import Setup
from .utils.datetime import init_datetime
from .utils.logging import log


@dataclass
class FilePathFormatter:
    def __init__(self, previous: Optional[List[str]] = None) -> None:
        self.previous: List[str] = previous if previous is not None else []

    # pylint: disable=W0102  # dangerous-default-value ([])
    def format(
        self,
        template: str,
        setup: Setup,
        mdata: MetaData,
        nc_meta_data: Mapping[str, Any],
    ) -> str:
        log(dbg=f"formatting path '{template}'")
        path = self._format_template(template, setup, mdata, nc_meta_data)
        while path in self.previous:
            template = self.derive_unique_path(
                self._format_template(template, setup, mdata, nc_meta_data)
            )
            path = self._format_template(template, setup, mdata, nc_meta_data)
        self.previous.append(path)
        log(dbg=f"formatted path '{path}'")
        return path

    def _format_template(
        self,
        template: str,
        setup: Setup,
        mdata: MetaData,
        nc_meta_data: Mapping[str, Any],
    ) -> str:
        # Prepare base time
        base_time = self._format_time_step(cast(int, setup.base_time), setup)[0]

        # Prepare ens variable
        ens_variable = setup.core.ens_variable
        if ens_variable == "percentile":
            ens_variable += f"-{setup.core.ens_param_pctl:g}"

        # Prepare input variable
        input_variable = setup.core.input_variable
        if setup.core.input_variable == "deposition":
            input_variable += f"-{setup.deposition_type_str}"
            if not setup.core.integrate:
                input_variable += f"-instant"
        elif setup.core.input_variable == "concentration":
            if setup.core.integrate:
                input_variable += f"-integr"

        # Prepare release start
        release_start = self._format_time_step(
            mdata.simulation.start + mdata.release.start_rel, setup
        )

        # Prepare time steps
        time_steps = self._format_time_steps(
            nc_meta_data["derived"]["time_steps"], setup
        )

        # Format the file path
        # Don't use str.format in order to handle multival elements
        kwargs = {
            "base_time": base_time,
            "domain": setup.core.domain,
            "ens_variable": ens_variable,
            "input_variable": input_variable,
            "lang": setup.core.lang,
            "level": setup.core.dimensions.level,
            "model": setup.model,
            "nageclass": setup.core.dimensions.nageclass,
            "noutrel": setup.core.dimensions.noutrel,
            "plot_type": setup.core.plot_type,
            "plot_variable": setup.core.plot_variable,
            "release_site": nc_meta_data["derived"]["release_site"],
            "release_start": release_start,
            "species_id": setup.core.dimensions.species_id,
            "time_idx": setup.core.dimensions.time,
            "time_step": time_steps[setup.core.dimensions.time],
        }
        return self._replace_format_keys(template, kwargs)

    def _format_time_steps(
        self, tss_int: Sequence[Union[int, datetime]], setup: Setup
    ) -> List[str]:
        return [self._format_time_step(ts, setup) for ts in tss_int]

    def _format_time_step(self, ts: Union[int, datetime], setup: Setup) -> str:
        if not isinstance(ts, datetime):
            ts = init_datetime(str(ts))
        return ts.strftime(setup.outfile_time_format)

    def _replace_format_keys(self, path: str, kwargs: Mapping[str, Any]) -> str:
        for key, val in kwargs.items():
            if not (isinstance(val, Sequence) and not isinstance(val, str)):
                val = [val]
            # Iterate over relevant format keys
            rxs = r"{" + key + r"(:[^}]*)?}"
            re.finditer(rxs, path)
            for m in re.finditer(rxs, path):

                # Obtain format specifier (if there is one)
                try:
                    f = m.group(1) or ""
                except IndexError:
                    f = ""

                # Format the string that replaces this format key in the path
                formatted_key = "+".join([f"{{{f}}}".format(v) for v in val])

                # Replace format key in the path by the just formatted string
                start, end = path[: m.span()[0]], path[m.span()[1] :]
                path = f"{start}{formatted_key}{end}"

        # Check that all keys have been formatted
        if "{" in path or "}" in path:
            raise Exception(
                "formatted output file path still contains format keys", path
            )

        return path

    @staticmethod
    def derive_unique_path(path: str, sep: str = ".") -> str:
        """Add/increment a trailing number to a file path."""
        log(dbg=f"deriving unique path from '{path}'")

        # Extract suffix
        if path.endswith(".png") or path.endswith(".pdf"):
            suffix = f".{path.split('.')[-1]}"
        else:
            raise NotImplementedError(f"unknown suffix: {path}")
        path_base = path[: -len(suffix)]

        # Reuse existing numbering if present
        match = re.search(sep.replace(".", r"\.") + r"(?P<i>[0-9]+)$", path_base)
        if match:
            i = int(match.group("i")) + 1
            w = len(match.group("i"))
            path_base = path_base[: -w - 1]
        else:
            i = 1
            w = 1

        # Add numbering and suffix
        path = path_base + f"{sep}{{i:0{w}}}{suffix}".format(i=i)

        return path
