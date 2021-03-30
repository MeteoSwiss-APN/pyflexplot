"""Output."""
# Standard library
import re
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
from .setups.plot_setup import PlotSetup
from .utils.logging import log


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
            cast(int, setup.model.base_time), setup.files.outfile_time_format
        )[0]

        # Prepare plot variable
        plot_variable = setup.panels.collect_equal("plot_variable")
        if plot_variable.endswith("deposition"):
            if not setup.panels.collect_equal("integrate"):
                plot_variable += "-instant"
        elif plot_variable == "concentration":
            if setup.panels.collect_equal("integrate"):
                plot_variable += "-integr"

        def prepare_ens_variable(ens_variable: str) -> str:
            if ens_variable == "percentile":
                ens_variable += f"-{setup.panels.collect_equal('ens_params').pctl:g}"
            elif ens_variable == "probability":
                if setup.panels.collect_equal("ens_params").thr_type == "lower":
                    ens_variable += "-gt"
                elif setup.panels.collect_equal("ens_params").thr_type == "upper":
                    ens_variable += "-lt"
                ens_variable += f"-{setup.panels.collect_equal('ens_params').thr:g}"
            return ens_variable

        # Prepare Prepare ens variable
        ens_variable: str
        if (
            setup.layout.plot_type == "multipanel"
            and setup.layout.multipanel_param == "ens_variable"
        ):
            ens_variable = "_".join(
                [
                    prepare_ens_variable(ens_variable_i)
                    for ens_variable_i in setup.panels.collect("ens_variable")
                ]
            )
        else:
            ens_variable = setup.panels.collect_equal("ens_variable")
            ens_variable = prepare_ens_variable(ens_variable)

        # Prepare release start
        release_start_fmtd: str = self._format_time_step(
            release_start, setup.files.outfile_time_format
        )

        # time steps
        time_steps_fmtd: List[str] = self._format_time_steps(
            time_steps, setup.files.outfile_time_format
        )

        # Format the file path
        # Don't use str.format in order to handle multival elements
        kwargs = {
            "base_time": base_time,
            "domain": setup.panels.collect_equal("domain"),
            "ens_variable": ens_variable,
            "plot_variable": plot_variable,
            "lang": setup.panels.collect_equal("lang"),
            "level": setup.panels.collect_equal("dimensions").level,
            "model": setup.model.name,
            "nageclass": setup.panels.collect_equal("dimensions").nageclass,
            "noutrel": setup.panels.collect_equal("dimensions").noutrel,
            "plot_type": setup.layout.plot_type,
            "release_site": release_site,
            "release_start": release_start_fmtd,
            "species_id": setup.panels.collect_equal("dimensions").species_id,
            "time_idx": setup.panels.collect_equal("dimensions").time,
            "time_step": time_steps_fmtd[setup.panels.collect_equal("dimensions").time],
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
