"""Output."""

# Standard library
import os
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
        species_name=str,
    ) -> str:
        log(dbg=f"formatting path '{template}'")
        path = self._format_template(
            template,
            setup,
            release_site=release_site,
            release_start=release_start,
            time_steps=time_steps,
            species_name=species_name,
        )

        # we need to check whether the file path has already been formatted
        # since the same path might be calculated, then we need to derive a unique path
        # until is not in any of the previously formatted paths
        path = formatted_file.path
        while path in self.previous:
            # template and path are recalculated on each loop
            # we need to first calculate the new template before re-formatting the file
            template = self.derive_unique_path(
                self._format_template(
                    template,
                    setup,
                    release_site=release_site,
                    release_start=release_start,
                    time_steps=time_steps,
                    species_name=species_name,
                )
            )
            path = self._format_template(
                template,
                setup,
                release_site=release_site,
                release_start=release_start,
                time_steps=time_steps,
                species_name=species_name,
            )
        self.previous.append(path)
        log(dbg=f"formatted path '{path}'")
        return path

    # pylint: disable=R0914  # tyy-many-locals (>15)
    def _format_template(
        self,
        template: str,
        setup: PlotSetup,
        *,
        release_site: str,
        release_start: datetime,
        time_steps: Sequence[Union[int, datetime]],
        species_name: str,
    ) -> str:
        # Prepare base time
        base_time = self._format_time_step(
            cast(int, setup.model.base_time), setup.files.output_time_format
        )

        # Prepare plot variable
        plot_variable = setup.panels.collect_equal("plot_variable")
        if plot_variable.endswith("deposition"):
            if not setup.panels.collect_equal("integrate"):
                plot_variable += "-instant"
        elif plot_variable == "concentration":
            if setup.panels.collect_equal("integrate"):
                plot_variable += "-integr"

        def prepare_ens_variable(ens_variable: str) -> str:
            plot_type = setup.layout.plot_type
            multipanel_param = setup.layout.multipanel_param
            if ens_variable == "percentile":
                if plot_type == "multipanel" and multipanel_param == "ens_params.pctl":
                    # pylint: disable=C0209  # consider-using-f-string (v2.11.1)
                    s_pctl = "+".join(
                        map("{:g}".format, setup.panels.collect("ens_params.pctl"))
                    )
                else:
                    s_pctl = f"{setup.panels.collect_equal('ens_params.pctl'):g}"
                ens_variable += f"-{s_pctl}"
            elif ens_variable == "probability":
                if setup.panels.collect_equal("ens_params.thr_type") == "lower":
                    ens_variable += "-gt"
                elif setup.panels.collect_equal("ens_params.thr_type") == "upper":
                    ens_variable += "-lt"
                ens_variable += f"-{setup.panels.collect_equal('ens_params.thr'):g}"
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
            release_start, setup.files.output_time_format
        )

        # time steps
        time_steps_fmtd: List[str] = self._format_time_steps(
            time_steps, setup.files.output_time_format
        )
        time_idx: Optional[int] = None
        time_step: Optional[str] = None
        time_idx_seq: Optional[Sequence[int]] = None
        time_step_seq: Optional[Sequence[str]] = None
        if (
            setup.layout.plot_type == "multipanel"
            and setup.layout.multipanel_param == "time"
        ):
            time_idx_seq = setup.panels.collect("dimensions.time")
            time_step_seq = [time_steps_fmtd[time_idx_i] for time_idx_i in time_idx_seq]
        else:
            time_idx = setup.panels.collect_equal("dimensions.time")
            assert time_idx is not None  # mypy
            time_step = time_steps_fmtd[time_idx]

        # Format the file path
        # Don't use str.format in order to handle multival elements
        kwargs = {
            "base_time": base_time,
            "domain": setup.panels.collect_equal("domain"),
            "ens_variable": ens_variable,
            "plot_variable": plot_variable,
            "lang": setup.panels.collect_equal("lang"),
            "level": setup.panels.collect_equal("dimensions.level"),
            "model": setup.model.name,
            "nageclass": setup.panels.collect_equal("dimensions.nageclass"),
            "plot_type": setup.layout.plot_type,
            "release": setup.panels.collect_equal("dimensions.release"),
            "release_site": release_site,
            "release_start": release_start_fmtd,
            "species_id": setup.panels.collect_equal("dimensions.species_id"),
            "species_name": species_name,
            "time_idx": time_idx if time_idx is not None else time_idx_seq,
            "time_step": time_step if time_step is not None else time_step_seq,
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
                start, end = path[:m.span()[0]], path[m.span()[1]:]
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
        _, suffix = os.path.splitext(path)
        if suffix not in [".pdf", ".png", ".shp"]:
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
