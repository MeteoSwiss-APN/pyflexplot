# -*- coding: utf-8 -*-
# pylint: disable=C0302  # too-many-lines (>1000)
"""
Attributes.
"""
# Standard library
import re
import warnings
from copy import copy
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import Any
from typing import cast
from typing import Collection
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

# Third-party
import netCDF4 as nc4
import numpy as np
from pydantic import BaseModel

# First-party
from srutils.dataclasses import dataclass_merge

# Local
from .setup import Setup
from .species import get_species
from .species import Species
from .utils.datetime import init_datetime

MetaDatumType = Union[int, float, str, datetime, timedelta]


def format_meta_datum(value: Any, join: Optional[str] = None) -> str:
    if isinstance(value, Collection) and not isinstance(value, str):
        # SR_TODO make sure this is covered by a test (it currently isn't)!
        return (join or " / ").join([format_meta_datum(v) for v in value])
    elif isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M %Z")
    elif isinstance(value, timedelta):
        hours = int(value.total_seconds() / 3600)
        minutes = int((value.total_seconds() / 60) % 60)
        return f"{hours:d}:{minutes:02d}$\\,$h"
    elif isinstance(value, (float, int)):
        if 1e-5 < value < 1e6:
            return f"{value:g}"
        else:
            return f"{value:.2g}"
    else:
        return str(value)


@dataclass
class MetaData:
    release: "ReleaseMetaData"
    simulation: "SimulationMetaData"
    variable: "VariableMetaData"
    species: "SpeciesMetaData"

    def merge_with(self, others: Collection["MetaData"],) -> "MetaData":
        """Merge this `MetaData` instance with one or more others.

        The nested functions are former methods that have been moved in here in
        order to eventually eliminate them.

        """
        if not others:
            return self

        def merge_release():
            return self.release.merge_with([other.release for other in others])

        def merge_simulation():
            if not all(other.simulation == self.simulation for other in others):
                raise ValueError("simulation meta data differ")
            return self.simulation

        def merge_variable():
            return self.variable.merge_with([other.variable for other in others])

        def merge_species():
            return self.species.merge_with([other.species for other in others])

        return type(self)(
            release=merge_release(),
            simulation=merge_simulation(),
            variable=merge_variable(),
            species=merge_species(),
        )

    def __deepcopy__(self, memo):
        return type(self)(
            release=copy(self.release),
            simulation=copy(self.simulation),
            variable=copy(self.variable),
            species=deepcopy(self.species, memo),
        )

    def format(
        self, param: str, *, add_unit: bool = False, join_combo: Optional[str] = None,
    ) -> str:
        """Format a parameter, optionally adding the unit (`~_unit`).

        This is left over after eliminating the type-specific MetaDatum[Combo]
        classes and will be cleaned up and likely turned into type-specific
        functions to be used explicitly to format individual attributes.

        """
        data_param, datum_param = param.split(".")
        try:
            datum = getattr(getattr(self, data_param), datum_param)
        except AttributeError:
            raise ValueError("invalid param", param)

        if param.endswith(".unit"):
            contains_unit = True
        elif not add_unit:
            contains_unit = False
        else:
            try:
                unit_datum = getattr(getattr(self, data_param), f"{datum_param}_unit")
            except AttributeError:
                raise Exception(f"missing unit {param}.unit of parameter {param}")
            if not isinstance(datum, tuple):
                values_fmtd = [format_meta_datum(datum), format_meta_datum(unit_datum)]
                datum = r"$\,$".join(values_fmtd)
            else:
                if not isinstance(unit_datum, tuple):
                    unit_datum = tuple([unit_datum] * len(datum))
                value = [
                    r"$\,$".join([format_meta_datum(d), format_meta_datum(u)])
                    for d, u in zip(datum, unit_datum)
                ]
                datum = tuple(value)
            contains_unit = True

        if isinstance(datum, tuple) and join_combo is not None:
            datum = join_combo.join(datum)

        datum_fmtd: str = format_meta_datum(datum)
        if contains_unit:
            return format_unit(datum_fmtd)
        return datum_fmtd


def format_unit(s: str) -> str:
    """Auto-format the unit by elevating superscripts etc."""
    s = str(s)
    old_new = [
        ("m-2", "m$^{-2}$"),
        ("m-3", "m$^{-3}$"),
        ("s-1", "s$^{-1}$"),
    ]
    for old, new in old_new:
        s = s.replace(old, new)
    return s


def collect_meta_data(
    fi: nc4.Dataset,
    setup: Setup,
    nc_meta_data: Mapping[str, Mapping[str, Any]],
    *,
    add_ts0: bool = False,
) -> MetaData:
    assert issubclass(MetaData, MetaData)  # SR_TMP
    return MetaData(
        release=ReleaseMetaData.from_file(fi, setup),
        simulation=SimulationMetaData.from_file(fi, setup, add_ts0),
        variable=VariableMetaData.from_file(fi, setup, nc_meta_data),
        species=SpeciesMetaData.from_file(fi, setup, nc_meta_data),
    )


def getncattr(nc_obj: Union[nc4.Dataset, nc4.Variable], attr: str) -> Any:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message="tostring() is deprecated. Use tobytes() instead.",
        )
        return nc_obj.getncattr(attr)


@dataclass
class VariableMetaData:
    unit: str
    bottom_level: float
    top_level: float
    level_unit: str

    @classmethod
    def from_file(
        cls,
        fi: nc4.Dataset,
        setup: Setup,
        nc_meta_data: Mapping[str, Mapping[str, Any]],
    ) -> "VariableMetaData":
        name = nc_var_name(setup, nc_meta_data["derived"]["model"])
        var = fi.variables[name]
        unit = getncattr(var, "units")
        idx: int
        if setup.core.dimensions.level is None:
            level_unit = ""
            level_bot = -1.0
            level_top = -1.0
        else:
            idx = setup.core.dimensions.level
            try:  # SR_TMP IFS
                var = fi.variables["level"]
            except KeyError:  # SR_TMP IFS
                var = fi.variables["height"]  # SR_TMP IFS
            level_bot = 0.0 if idx == 0 else float(var[idx - 1])
            level_top = float(var[idx])
            level_unit = getncattr(var, "units")
        return cls(
            unit=unit,
            bottom_level=level_bot,
            top_level=level_top,
            level_unit=level_unit,
        )

    def merge_with(self, others: Sequence["VariableMetaData"]) -> "VariableMetaData":
        unit = self.unit
        level_unit = self.level_unit
        bottom_level = self.bottom_level
        top_level = self.top_level
        for obj in others:
            assert unit == obj.unit
            assert level_unit == obj.level_unit
            bottom_level = min([bottom_level, obj.bottom_level])
            top_level = max([top_level, obj.top_level])
        return type(self)(
            unit=unit,
            level_unit=level_unit,
            bottom_level=bottom_level,
            top_level=top_level,
        )


@dataclass
class SimulationMetaData:
    start: datetime
    end: datetime
    now: datetime
    now_rel: timedelta
    reduction_start: datetime
    reduction_start_rel: timedelta

    @classmethod
    def from_file(
        cls, fi: nc4.Dataset, setup: Setup, add_ts0: bool
    ) -> "SimulationMetaData":

        # Start and end timesteps of simulation
        start = init_datetime(getncattr(fi, "ibdate") + getncattr(fi, "ibtime"))
        end = init_datetime(getncattr(fi, "iedate") + getncattr(fi, "ietime"))

        # Current time step and start time step of current integration period
        collector = TimeStepMetaDataCollector(fi, setup, add_ts0=add_ts0)
        now = collector.now()
        reduction_start = collector.reduction_start()
        now_rel: timedelta = collector.now_rel()
        reduction_start_rel = collector.reduction_start_rel()
        return cls(
            start=start,
            end=end,
            now=now,
            now_rel=now_rel,
            reduction_start=reduction_start,
            reduction_start_rel=reduction_start_rel,
        )


@dataclass
# pylint: disable=R0902  # too-many-instance-attrbutes
class ReleaseMetaData:
    duration: timedelta
    duration_unit: str
    end_rel: timedelta
    height: float
    height_unit: str
    lat: float
    lon: float
    mass: Union[float, Tuple[float, ...]]
    mass_unit: str
    site: str
    rate: Union[float, Tuple[float, ...]]
    rate_unit: str
    start_rel: timedelta

    @classmethod
    def from_file(cls, fi: nc4.Dataset, setup: Setup) -> "ReleaseMetaData":
        """Read information on a release from open file."""
        raw = RawReleaseMetaData.from_file(fi, setup)
        site = raw.site
        # SR_HC <
        if site == "Goesgen":
            site = r"G$\mathrm{\"o}$sgen"
        # SR_HC >
        start_rel = raw.rel_start
        end_rel = raw.rel_end
        duration = end_rel - start_rel
        duration_unit = "s"  # SR_HC
        assert raw.zbot_unit == raw.ztop_unit
        height_unit = raw.zbot_unit
        mass = raw.ms_parts
        mass_unit = "Bq"  # SR_HC
        return cls(
            duration=duration,
            duration_unit=duration_unit,
            end_rel=end_rel,
            height=np.mean([raw.zbot, raw.ztop]),
            height_unit=height_unit,
            lat=np.mean([raw.lllat, raw.urlat]),
            lon=np.mean([raw.lllon, raw.urlon]),
            mass=mass,
            mass_unit=mass_unit,
            site=site,
            rate=mass / duration.total_seconds() if duration else np.nan,
            rate_unit=f"{mass_unit} {duration_unit}-1",
            start_rel=start_rel,
        )

    def merge_with(self, others: Sequence["ReleaseMetaData"]) -> "ReleaseMetaData":
        return dataclass_merge(
            [self] + list(others), expect_equal_except=["mass", "rate"]
        )


class RawReleaseMetaData(BaseModel):
    age_id: int
    kind: str
    lllat: float
    lllon: float
    ms_parts: float
    n_parts: int
    rel_end: timedelta
    rel_start: timedelta
    site: str
    urlat: float
    urlon: float
    zbot: float
    zbot_unit: str
    ztop: float
    ztop_unit: str

    class Config:  # noqa
        arbitrary_types_allowed = True
        extra = "forbid"
        validate_all = True
        validate_assigment = True

    # pylint: disable=R0914  # too-many-locals
    @classmethod
    def from_file(cls, fi: nc4.Dataset, setup: Setup) -> "RawReleaseMetaData":
        """Read information on a release from open file."""

        assert setup.core.dimensions.numpoint is not None  # mypy
        idx_point = setup.core.dimensions.numpoint

        assert setup.core.dimensions.species_id is not None  # mypy
        # SR_TMP <
        idx_spec = setup.core.dimensions.species_id - 1
        assert 0 <= idx_spec < fi.dimensions["numspec"].size
        # SR_TMP >

        assert setup.core.dimensions.nageclass is not None  # mypy
        idx_age = setup.core.dimensions.nageclass

        var_name: str = "RELCOM"  # SR_HC TODO un-hardcode
        var = fi.variables[var_name]

        # Check index against no. release point and set it if necessary
        n = var.shape[0]
        if n == 0:
            raise ValueError(f"file '{fi.name}': no release points ('{var_name}')")
        elif n == 1:
            if idx_point is None:
                idx_point = 0
        elif n > 1:
            if idx_point is None:
                raise ValueError(
                    f"file '{fi.name}': idx is None despite {n} release points"
                )
        assert idx_point is not None  # mypy
        if idx_point < 0 or idx_point >= n:
            raise ValueError(
                f"file '{fi.name}': invalid index {idx_point} for {n} release points"
            )

        # Name: convert from byte character array
        site = var[idx_point][~var[idx_point].mask].tobytes().decode("utf-8").rstrip()

        # Other attributes
        key_pairs = [
            ("age_id", "LAGE"),
            ("kind", "RELKINDZ"),
            ("lllat", "RELLAT1"),
            ("lllon", "RELLNG1"),
            ("ms_parts", "RELXMASS"),
            ("n_parts", "RELPART"),
            ("rel_end", "RELEND"),
            ("rel_start", "RELSTART"),
            ("urlat", "RELLAT2"),
            ("urlon", "RELLNG2"),
            ("zbot", "RELZZ1"),
            ("ztop", "RELZZ2"),
        ]
        store_units = ["zbot", "ztop"]
        params = {"site": site}
        for key_out, key_in in key_pairs:
            var = fi.variables[key_in]
            # SR_TMP < TODO more flexible solution w/o hardcoding so much
            if var.dimensions == ("numpoint",):
                params[key_out] = var[idx_point].tolist()
            elif var.dimensions == ("numspec", "numpoint"):
                params[key_out] = var[idx_spec, idx_point].tolist()
            elif var.dimensions == ("nageclass",):
                params[key_out] = var[idx_age].tolist()
            else:
                raise NotImplementedError(
                    f"dimensions {var.dimensions} (variable {var.name})"
                )
            # SR_TMP >
            if key_out in store_units:
                unit = getncattr(fi.variables[key_in], "units")
                params[f"{key_out}_unit"] = unit
        return cls(**params)


class TimeStepMetaDataCollector:
    def __init__(self, fi: nc4.Dataset, setup: Setup, *, add_ts0: bool = False) -> None:
        self.fi = fi
        self.setup = setup
        self.add_ts0 = add_ts0

    def start(self) -> datetime:
        """Compute the time step when the simulation started."""
        var = self.fi.variables["time"]
        rx = re.compile(
            r"seconds since "
            r"(?P<year>[12][0-9][0-9][0-9])-"
            r"(?P<month>[01][0-9])-"
            r"(?P<day>[0-3][0-9]) "
            r"(?P<hour>[0-2][0-9]):"
            r"(?P<minute>[0-6][0-9])"
        )
        match = rx.match(var.units)
        if not match:
            raise Exception(f"cannot extract start from units '{var.units}'")
        return datetime(
            year=int(match["year"]),
            month=int(match["month"]),
            day=int(match["day"]),
            hour=int(match["hour"]),
            minute=int(match["minute"]),
            tzinfo=timezone.utc,
        )

    def now(self) -> datetime:
        """Current time step."""
        return self.start() + self.now_rel()

    def now_rel(self) -> timedelta:
        """Time since start."""
        var = self.fi.variables["time"]
        idx = self.time_step_idx()
        if idx < 0:
            return timedelta(0)
        return timedelta(seconds=int(var[idx]))

    def reduction_start(self) -> datetime:
        """Time step when reduction (mean/sum) started."""
        return self.now() - self.integration_duration()

    def reduction_start_rel(self) -> timedelta:
        """Time between simulation start and start of reduction (mean/sum)."""
        return self.reduction_start() - self.start()

    def integration_duration(self) -> timedelta:
        """Compute timestep delta of integration period."""
        if self.setup.core.integrate:
            return self.now_rel()
        n = self.time_step_idx() + 1
        if n == 0:
            return timedelta(0)
        return self.now_rel() / n

    def time_step_idx(self) -> int:
        """Index of current time step of current field."""
        # Default to timestep of current field
        assert self.setup.core.dimensions.time is not None  # mypy
        if self.add_ts0:
            return self.setup.core.dimensions.time - 1
        return self.setup.core.dimensions.time


@dataclass
# pylint: disable=R0902  # too-many-instance-attributes
class SpeciesMetaData:
    name: Union[str, Tuple[str, ...]]
    half_life: Union[float, Tuple[float, ...]]
    half_life_unit: Union[str, Tuple[str, ...]]
    deposition_velocity: Union[float, Tuple[float, ...]]
    deposition_velocity_unit: Union[str, Tuple[str, ...]]
    sedimentation_velocity: Union[float, Tuple[float, ...]]
    sedimentation_velocity_unit: Union[str, Tuple[str, ...]]
    washout_coefficient: Union[float, Tuple[float, ...]]
    washout_coefficient_unit: Union[str, Tuple[str, ...]]
    washout_exponent: Union[float, Tuple[float, ...]]

    @classmethod
    def from_file(
        cls,
        fi: nc4.Dataset,
        setup: Setup,
        nc_meta_data: Mapping[str, Mapping[str, Any]],
    ) -> "SpeciesMetaData":
        model: str = nc_meta_data["derived"]["model"]
        name: str = nc_var_name(setup, model)
        var: nc4.Variable = fi.variables[name]
        try:  # SR_TMP
            name = getncattr(var, "long_name")
        except AttributeError:
            # SR_TMP <
            # name = "N/A"
            if model.startswith("IFS"):
                # In the IFS NetCDF files, the deposition variables are missing
                # the basic meta data on species, like "long_name". Therefore,
                # try to # obtain the name from the activity variable of the
                # same species.
                if name.startswith("DD_") or name.startswith("WD_"):
                    alternative_name = f"{name[3:]}_mr"
                    try:
                        alternative_var = fi.variables[alternative_name]
                        name = getncattr(alternative_var, "long_name")
                    except (KeyError, AttributeError):
                        name = "N/A"
                    else:
                        name = name.split("_")[0]
            else:
                name = "N/A"
            # SR_TMP >
        else:
            name = name.split("_")[0]
        species: Species = get_species(name=name)
        return cls(
            name=species.name,
            half_life=species.half_life.value,
            half_life_unit=cast(str, species.half_life.unit),
            deposition_velocity=species.deposition_velocity.value,
            deposition_velocity_unit=cast(str, species.deposition_velocity.unit),
            sedimentation_velocity=species.sedimentation_velocity.value,
            sedimentation_velocity_unit=cast(str, species.sedimentation_velocity.unit),
            washout_coefficient=species.washout_coefficient.value,
            washout_coefficient_unit=cast(str, species.washout_coefficient.unit),
            washout_exponent=species.washout_exponent.value,
        )

    def merge_with(self, others: Sequence["SpeciesMetaData"]) -> "SpeciesMetaData":
        return dataclass_merge([self] + list(others))


def nc_var_name(setup: Setup, model: str) -> str:
    # SR_TMP <
    dimensions = setup.core.dimensions
    input_variable = setup.core.input_variable
    deposition_type = setup.deposition_type_str
    # SR_TMP >
    assert dimensions.species_id is not None  # mypy
    species_id = dimensions.species_id
    if input_variable == "concentration":
        if model in ["COSMO-2", "COSMO-1"]:
            return f"spec{species_id:03d}"
        elif model in ["IFS-HRES", "IFS-HRES-EU"]:
            return f"spec{species_id:03d}_mr"
        else:
            raise ValueError("unknown model", model)
    elif input_variable == "deposition":
        prefix = {"wet": "WD", "dry": "DD"}[deposition_type]
        return f"{prefix}_spec{species_id:03d}"
    raise ValueError("unknown variable", input_variable)
