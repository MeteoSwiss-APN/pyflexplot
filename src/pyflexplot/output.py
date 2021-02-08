"""Output."""
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

# First-party
from srutils.datetime import init_datetime

# Local
from .setup import PlotSetup
from .utils.logging import log


@dataclass
class FilePathFormatter:
    """Format file paths."""

    def __init__(self, previous: Optional[List[str]] = None) -> None:
        """Create an instance of ``FilePathFormatter``."""
        self.previous: List[str] = previous if previous is not None else []

    # pylint: disable=W0102  # dangerous-default-value ([])
    def format(
        self,
        template: str,
        setup: PlotSetup,
        *,
        release_site: str,
        release_start: datetime,
        time_steps: Sequence[Union[int, datetime]],
    ) -> str:
        log(dbg=f"formatting path '{template}'")
        path = self._format_template(
            template,
            setup,
            release_site=release_site,
            release_start=release_start,
            time_steps=time_steps,
        )
        while path in self.previous:
            template = self.derive_unique_path(
                self._format_template(
                    template,
                    setup,
                    release_site=release_site,
                    release_start=release_start,
                    time_steps=time_steps,
                )
            )
            path = self._format_template(
                template,
                setup,
                release_site=release_site,
                release_start=release_start,
                time_steps=time_steps,
            )
        self.previous.append(path)
        log(dbg=f"formatted path '{path}'")
        return path

    def _format_template(
        self,
        template: str,
        setup: PlotSetup,
        *,
        release_site: str,
        release_start: datetime,
        time_steps: Sequence[Union[int, datetime]],
    ) -> str:
        # Prepare base time
        base_time = self._format_time_step(
            cast(int, setup.model.base_time), setup.outfile_time_format
        )[0]

        # Prepare ens variable
        ens_variable = setup.core.ens_variable
        if ens_variable == "percentile":
            ens_variable += f"-{setup.core.ens_params.pctl:g}"
        elif ens_variable == "probability":
            if setup.core.ens_params.thr_type == "lower":
                ens_variable += "-gt"
            elif setup.core.ens_params.thr_type == "upper":
                ens_variable += "-lt"
            ens_variable += f"-{setup.core.ens_params.thr:g}"

        # Prepare input variable
        input_variable = setup.core.input_variable
        if setup.core.input_variable == "deposition":
            input_variable += f"-{setup.deposition_type_str}"
            if not setup.core.integrate:
                input_variable += "-instant"
        elif setup.core.input_variable == "concentration":
            if setup.core.integrate:
                input_variable += "-integr"

        # Prepare release start
        release_start_fmtd: str = self._format_time_step(
            release_start, setup.outfile_time_format
        )

        # Prepare time steps
        time_steps_fmtd: List[str] = self._format_time_steps(
            time_steps, setup.outfile_time_format
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
            "model": setup.model.name,
            "nageclass": setup.core.dimensions.nageclass,
            "noutrel": setup.core.dimensions.noutrel,
            "plot_type": setup.plot_type,
            "release_site": release_site,
            "release_start": release_start_fmtd,
            "species_id": setup.core.dimensions.species_id,
            "time_idx": setup.core.dimensions.time,
            "time_step": time_steps_fmtd[setup.core.dimensions.time],
        }
        return self._replace_format_keys(template, kwargs)

    def _format_time_steps(
        self, tss_int: Sequence[Union[int, datetime]], outfile_time_format: str
    ) -> List[str]:
        return [self._format_time_step(ts, outfile_time_format) for ts in tss_int]

    @staticmethod
    def _format_time_step(ts: Union[int, datetime], outfile_time_format: str) -> str:
        if not isinstance(ts, datetime):
            ts = init_datetime(str(ts))
        return ts.strftime(outfile_time_format)

    @staticmethod
    def _replace_format_keys(template: str, kwargs: Mapping[str, Any]) -> str:
        path = template
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
                "formatted output file path still appears to contain format keys:"
                f" {path} (template: {template})"
            )

        return path

    @staticmethod
    def derive_unique_path(path: str, sep: str = ".") -> str:
        """Add/increment a trailing number to a file path."""
        # log(dbg=f"deriving unique path from '{path}'")

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
