# -*- coding: utf-8 -*-
"""
IO.
"""
# Standard library
import logging as log
from copy import deepcopy

# Third-party
import netCDF4 as nc4
import numpy as np

# First-party
from srutils.various import check_array_indices

# Local
from .attr import collect_attrs
from .data import Field
from .data import cloud_arrival_time
from .data import threshold_agreement
from .field_specs import FieldSpecs


class FileReader:
    """Reader of NetCDF files containing FLEXPART data.

    It represents a single input file for deterministic FLEXPART runs, or an
    ensemble of input files for ensemble FLEXPART runs (one file per ensemble
    member).

    """

    cls_field = Field

    choices_ens_var = ["mean", "median", "min", "max"]

    def __init__(self, in_file_path, *, cmd_open=nc4.Dataset):
        """Create an instance of ``FileReader``.

        Args:
            in_file_path (str): File path. In case of ensemble data, it must
                contain the format key '{ens_member[:0?d]}'.

            cmd_open (function, optional): Function to open the input file.
                Must support context manager interface, i.e., ``with
                cmd_open(in_file_path, mode) as f:``. Defaults to
                netCDF4.Dataset.

        """
        self.in_file_path_fmt = in_file_path
        self.cmd_open = cmd_open

        self.reset()

    def reset(self):
        self.lang = None
        self.n_members = None
        self.in_file_path_lst = None
        self.fi = None
        self.rlat = None
        self.rlon = None

        # SR_TMP < TODO don't cheat! _attrs_{all,none} are for checks only!
        if hasattr(self, "_attrs_all"):
            for attr in self.__dict__.keys():
                if attr in ["_attrs_all", "_attrs_none"]:
                    pass
                elif attr not in self._attrs_all:
                    del self.__dict__[attr]
                elif attr in self._attrs_none:
                    setattr(self, attr, None)
        # SR_TMP >

    # SR_DEV <<<
    def _store_attrs(self):
        """Store names of all attributes, and which are None."""
        if "_attrs_all" in self.__dict__:
            raise Exception("'_attrs_all' in self.__dict__")
        if "_attrs_none" in self.__dict__:
            raise Exception("'_attrs_none' in self.__dict__")
        self._attrs_all = self.__dict__.keys()
        self._attrs_none = [a for a in self._attrs_all if getattr(self, a) is None]

    # SR_DEV <<<
    def _check_attrs(self):
        """Check that attributes have been cleared up properly.

        Checks:
            There are no attributes that should not be there.

            All attributes that should be None, are.

        """
        attrs_all = self.__dict__.pop("_attrs_all")
        attrs_none = self.__dict__.pop("_attrs_none")

        # Check that there are no unexpected attributes
        attrs_unexp = [a for a in self.__dict__.keys() if a not in attrs_all]
        if attrs_unexp:
            raise Exception(f"{len(attrs_unexp)} unexpected attributes: {attrs_unexp}")

        # Check that all attributes that should be None, are
        attrs_nonone = [a for a in attrs_none if getattr(self, a) is not None]
        if attrs_nonone:
            raise Exception(
                f"{len(attrs_nonone)} attributes should be None but are not: "
                f"{attrs_nonone}"
            )

    def run(self, fld_specs, *, lang=None):
        """Read one or more fields from a file from disc.

        Args:
            fld_specs (FieldSpecs or list[FieldSpecs]): Specifications for one
                or more input fields.

            ens_var (str, optional): Name of ensemble variable, e.g., 'mean'.
                See ``Field.choices_ens_var`` for the full list. Mandatory in
                case of multiple ensemble members. Defaults to None.

            ens_var_opts (dict, optional): Options that can/must be passed
                alongside certain ensemble variables. Defaults to {}.

            lang (str, optional): Language, e.g., 'de' for German. Defaults to
                'en' (English).

        Returns:
            Field: Single data object; if ``fld_specs`` constitutes a single
            ``FieldSpecs`` instance.

            or

            list[Field]: One data object for each field; if ``fld_specs``
                constitutes a list of ``FieldSpecs`` instances.

        """
        self._store_attrs()  # SR_DEV

        # Set some attributes
        self.lang = lang or "en"

        if isinstance(fld_specs, FieldSpecs):
            multiple = False
            fld_specs_lst = [fld_specs]
        else:
            multiple = True
            fld_specs_lst = fld_specs
        del fld_specs

        # Group field specifications objects such that all in one group only
        # differ in time; collect the respective time indices; and merge the
        # group into one field specifications instance with # time dimension
        # 'slice(None)'. This allows for each such group to first read and
        # process all time steps (e.g., derive some statistics across all time
        # steps), and subsequently extract the requested time steps using the
        # separately stored indices.
        self._fld_specs_time_lst, self._time_inds_lst = self._group_fld_specs_by_time(
            fld_specs_lst
        )

        # Prepare array for fields
        self.n_fld_specs = len(self._fld_specs_time_lst)
        self.n_reqtime = self._determine_n_reqtime()
        flex_field_lst = []

        # Collect fields
        log.debug(f"process {self.n_fld_specs} field specs groups")
        for i_fld_specs in range(self.n_fld_specs):

            fld_specs_time = self._fld_specs_time_lst[i_fld_specs]
            time_idcs = self._time_inds_lst[i_fld_specs]

            ens_member_ids = fld_specs_time.multi_var_specs.setup.ens_member_ids
            self.n_members = 1 if not ens_member_ids else len(ens_member_ids)
            self.in_file_path_lst = self._prepare_in_file_path_lst(ens_member_ids)

            log.debug(f"{i_fld_specs + 1}/{self.n_fld_specs}: {fld_specs_time}")

            flex_field_lst.extend(
                self._create_fields_fld_specs(fld_specs_time, time_idcs)
            )

        # Return result field(s)
        result = flex_field_lst
        if not multiple:
            # Only one field type specified: remove fields dimension
            result = result[0]

        self.reset()
        self._check_attrs()  # SR_DEV

        return result

    def _prepare_in_file_path_lst(self, ens_member_ids):

        fmt_keys = ["{ens_member}", "{ens_member:"]
        fmt_key_in_path = any(k in self.in_file_path_fmt for k in fmt_keys)

        if not ens_member_ids:
            if fmt_key_in_path:
                raise ValueError(
                    f"input file path contains format key '{fmt_keys[0]}', but no "
                    f"ensemble member ids ('ens_member_ids') have been passed"
                )
            return [self.in_file_path_fmt]

        if not fmt_key_in_path:
            raise ValueError(
                f"input file path missing format key '{fmt_keys[0]}': "
                f"{self.in_file_path_fmt}"
            )
        return [self.in_file_path_fmt.format(ens_member=id) for id in ens_member_ids]

    def _determine_n_reqtime(self):
        """Determine the number of selected time steps."""
        n_reqtime_per_mem = [len(inds) for inds in self._time_inds_lst]
        if len(set(n_reqtime_per_mem)) > 1:
            raise Exception(
                f"numbers of timesteps differ across members: " f"{n_reqtime_per_mem}"
            )
        return next(iter(n_reqtime_per_mem))

    def _create_fields_fld_specs(self, fld_specs_time, time_idcs):

        # Read fields of all members at all time steps
        fld_time_mem = self._read_fld_time_mem(fld_specs_time)

        # Reduce fields array along member dimension
        # In other words: Compute single field from ensemble
        fld_time = self._reduce_ensemble(fld_time_mem, fld_specs_time)

        # Collect time stats
        time_stats = self._collect_time_stats(fld_time)

        # Create time-step-specific field specifications
        fld_specs_reqtime = self._create_specs_reqtime(fld_specs_time, time_idcs)

        # Collect attributes at requested time steps for all members
        attrs_reqtime_mem = self._collect_attrs_reqtime_mem(
            fld_specs_reqtime, time_idcs
        )

        # Merge attributes across members
        attrs_reqtime = self._merge_attrs_across_members(attrs_reqtime_mem)

        # Create fields at requested time steps for all members
        return self._create_fields_reqtime(
            fld_specs_reqtime, attrs_reqtime, fld_time, time_idcs, time_stats
        )

    def _read_fld_time_mem(self, fld_specs_time):
        """Read field over all time steps for each member."""

        fld_time_mem = None

        for i_mem, in_file_path in enumerate(self.in_file_path_lst):

            log.debug(f"read {in_file_path} (fields)")
            with self.cmd_open(in_file_path, "r") as self.fi:
                log.debug(f"extract {self.n_fld_specs} time steps")

                # Read grid variables
                rlat = self.fi.variables["rlat"][
                    slice(*fld_specs_time.multi_var_specs.collect_equal("rlat"))
                ]
                rlon = self.fi.variables["rlon"][
                    slice(*fld_specs_time.multi_var_specs.collect_equal("rlon"))
                ]
                if self.rlat is None:
                    self.rlat = rlat
                    self.rlon = rlon
                else:
                    if not (rlat == self.rlat).all():
                        raise Exception("inconsistent rlat")
                    if not (rlon == self.rlon).all():
                        raise Exception("inconsistent rlon")

                # Read field (all time steps)
                fld_time = self._import_field(fld_specs_time)

                # Store field for currentmember
                if fld_time_mem is None:
                    _shape = [self.n_members] + list(fld_time.shape)
                    fld_time_mem = np.full(_shape, np.nan, np.float32)
                fld_time_mem[i_mem] = fld_time

            self.fi = None

        return fld_time_mem

    def _reduce_ensemble(self, fld_time_mem, fld_specs_time):
        """Reduce the ensemble to a single field (time, rlat, rlon)."""
        if self.n_members == 1:
            return fld_time_mem[0]
        setup = fld_specs_time.multi_var_specs.setup
        plot_type = setup.plot_type
        if plot_type == "ens_mean":
            fld_time = np.nanmean(fld_time_mem, axis=0)
        elif plot_type == "ens_median":
            fld_time = np.nanmedian(fld_time_mem, axis=0)
        elif plot_type == "ens_min":
            fld_time = np.nanmin(fld_time_mem, axis=0)
        elif plot_type == "ens_max":
            fld_time = np.nanmax(fld_time_mem, axis=0)
        elif plot_type == "ens_thr_agrmt":
            thr = setup.ens_param_thr
            fld_time = threshold_agreement(fld_time_mem, thr, axis=0)
        elif plot_type == "ens_cloud_arrival_time":
            thr = setup.ens_param_thr
            mem_min = setup.ens_param_mem_min
            fld_time = cloud_arrival_time(fld_time_mem, thr, mem_min, mem_axis=0)
        else:
            raise NotImplementedError(f"plot var '{plot_type}'")
        return fld_time

    def _create_specs_reqtime(self, fld_specs_time, time_idcs):
        """Create time-step-specific field specifications."""
        fld_specs_reqtime = np.full([self.n_reqtime], None)
        for i_reqtime, time_idx in enumerate(time_idcs):
            fld_specs = deepcopy(fld_specs_time)
            # SR_TMP <
            for var_specs in fld_specs.multi_var_specs:
                var_specs._setup = var_specs._setup.derive({"time_idcs": [time_idx]})
            # SR_TMP >
            fld_specs_reqtime[i_reqtime] = fld_specs
        return fld_specs_reqtime

    def _collect_attrs_reqtime_mem(self, fld_specs_reqtime, time_idcs):
        """Collect attributes at requested time steps for all members."""
        attrs_reqtime_mem = np.full([self.n_reqtime, self.n_members], None)
        for i_mem, in_file_path in enumerate(self.in_file_path_lst):
            log.debug(f"read {in_file_path} (attributes)")
            with self.cmd_open(in_file_path, "r") as self.fi:
                for i_reqtime, time_idx in enumerate(time_idcs):
                    log.debug(f"{i_reqtime + 1}/{len(time_idcs)}: collect attributes")
                    fld_specs = fld_specs_reqtime[i_reqtime]
                    attrs_lst = []
                    for var_specs in fld_specs.multi_var_specs:
                        attrs = collect_attrs(self.fi, var_specs, lang=self.lang)
                        attrs_lst.append(attrs)
                    attrs = attrs_lst[0].merge_with(attrs_lst[1:])
                    attrs_reqtime_mem[i_reqtime, i_mem] = attrs
            self.fi = None
        return attrs_reqtime_mem

    def _group_fld_specs_by_time(self, fld_specs_lst):
        """Group specs that differ only in their time dimension."""

        fld_specs_time_inds_by_hash = {}
        for fld_specs in fld_specs_lst:
            fld_specs_time = deepcopy(fld_specs)

            # Extract time index and check its the same for all
            time_idx = fld_specs_time.multi_var_specs.collect_equal("time")

            # Store time-neutral fld specs alongside resp. time inds
            key = hash(fld_specs_time)
            if key not in fld_specs_time_inds_by_hash:
                fld_specs_time_inds_by_hash[key] = (fld_specs_time, [])
            if time_idx in fld_specs_time_inds_by_hash[key][1]:
                raise Exception(
                    f"duplicate time index {time_idx} in fld_specs:\n" f"{fld_specs}"
                )
            fld_specs_time_inds_by_hash[key][1].append(time_idx)

        # Regroup time-neutral fld specs and time inds into lists
        fld_specs_time_lst, time_inds_lst = [], []
        for _, (fld_specs_time, time_idcs) in fld_specs_time_inds_by_hash.items():
            fld_specs_time_lst.append(fld_specs_time)
            time_inds_lst.append(time_idcs)

        return fld_specs_time_lst, time_inds_lst

    def _merge_attrs_across_members(self, attrs_reqtime_mem):
        attrs_reqtime = attrs_reqtime_mem[:, 0]
        for i_mem in range(1, self.n_members):
            for i_reqtime, attrs in enumerate(attrs_reqtime_mem[:, i_mem]):
                attrs_ref = attrs_reqtime[i_reqtime]
                if attrs != attrs_ref:
                    raise Exception(
                        f"attributes differ between members 0 and {i_mem}: {attrs_ref} "
                        f"!= {attrs}"
                    )
        return attrs_reqtime

    def _create_fields_reqtime(
        self, fld_specs_reqtime, attrs_reqtime, fld_time, time_idcs, time_stats
    ):
        """Create fields at requested time steps for all members."""

        log.debug(f"select fields at requested time steps")

        flex_field_lst = []
        for i_reqtime, time_idx in enumerate(time_idcs):
            log.debug(f"{i_reqtime + 1}/{len(time_idcs)}")

            fld_specs = fld_specs_reqtime[i_reqtime]
            attrs = attrs_reqtime[i_reqtime]

            # Extract field
            fld = fld_time[time_idx]

            # Fix some known issues with the NetCDF input data
            log.debug("fix nc data: attrs")
            self._fix_nc_attrs(fld, attrs)

            # Collect data
            log.debug("create data object")
            flex_field = self.cls_field(
                fld, self.rlat, self.rlon, attrs, fld_specs, time_stats,
            )
            flex_field_lst.append(flex_field)
        return flex_field_lst

    def _collect_time_stats(self, fld_time):
        stats = {
            "mean": np.nanmean(fld_time),
            "median": np.nanmedian(fld_time),
            "mean_nz": np.nanmean(fld_time[fld_time > 0]),
            "median_nz": np.nanmedian(fld_time[fld_time > 0]),
            "max": np.nanmax(fld_time),
        }
        return stats

    def _import_field(self, fld_specs):

        # Read fields and attributes from var specifications
        fld_lst = []
        for var_specs in fld_specs.multi_var_specs:

            # Field
            log.debug("read field")
            fld = self._read_var(var_specs)

            fld_lst.append(fld)

        # Merge fields
        fld = fld_specs.merge_fields(fld_lst)

        return fld

    def _read_var(self, var_specs):

        # Select variable in file
        var_name = var_specs.var_name()
        var = self.fi.variables[var_name]

        # Indices of field along NetCDF dimensions
        dim_idxs_by_name = var_specs.dim_inds_by_name()

        # Assemble indices for slicing
        idxs = [None] * len(var.dimensions)
        for dim_name, dim_idx in dim_idxs_by_name.items():
            # Get the index of the dimension for this variable
            try:
                idx = var.dimensions.index(dim_name)
            except ValueError:
                # Potential issue: Dimension not among the variable dimensions!
                if dim_idx in (None, 0):
                    continue
                else:
                    # Index along the missing dimension cannot be non-trivial!
                    raise Exception(
                        f"dimension '{dim_name}' with non-zero index {dim_idx} missing",
                        {
                            "dim_idx": dim_idx,
                            "dim_name": dim_name,
                            "fi.filepath": self.fi.filepath(),
                            "var.dimensions": var.dimensions,
                            "var_name": var_name,
                        },
                    )

            # Check that the index along the dimension is valid
            if dim_idx is None:
                raise Exception(
                    f"dimension #{idx} '{dim_name}' is None",
                    {
                        "dim_idx": dim_idx,
                        "dim_name": dim_name,
                        "fi.filepath": self.fi.filepath(),
                        "idx": idx,
                    },
                )

            idxs[idx] = dim_idx

        # Check that all variable dimensions have been identified
        try:
            idx = idxs.index(None)
        except ValueError:
            # All good!
            pass
        else:
            raise Exception(
                f"unknown variable dimension #{idx} '{var.dimensions[idx]}'",
                {
                    "dim_idxs_by_name": dim_idxs_by_name,
                    "idxs": idxs,
                    "var.dimensions": var.dimensions,
                    "var_name": var_name,
                },
            )
        idxs = idxs
        log.debug(f"indices: {idxs}")
        check_array_indices(var.shape, idxs)

        # Read field
        log.debug(f"shape: {var.shape}")
        log.debug(f"indices: {idxs}")
        fld = var[idxs]

        log.debug(f"fix nc data: variable {var.name}")
        self._fix_nc_var(fld, var)

        # Time integration
        fld = self._time_integrations(fld, var_specs)

        return fld

    def _time_integrations(self, fld, var_specs):

        dt_hr = self._time_resolution()

        if var_specs._setup.variable == "concentration":  # SR_TMP
            if var_specs.integrate:
                # Integrate over time
                fld = np.cumsum(fld, axis=0) * dt_hr

        elif var_specs._setup.variable == "deposition":  # SR_TMP
            if not var_specs.integrate:
                # Revert integration over time
                fld[1:] = (fld[1:] - fld[:-1]) / dt_hr

        else:
            raise NotImplementedError(f"var_specs of type '{type(var_specs).__name__}'")

        return fld

    def _time_resolution(self):
        """Determine time resolution of input data."""
        time = self.fi.variables["time"]
        dts = set(time[1:] - time[:-1])
        if len(dts) > 1:
            raise Exception(f"Non-uniform time resolution: {sorted(dts)} ({time})")
        dt_min = next(iter(dts))
        dt_hr = dt_min / 3600.0
        return dt_hr

    # SR_TMP <<<
    def _fix_nc_var(self, fld, var):

        # SR_TMP < TODO more general solution to combined species
        names = [
            "Cs-137",
            "I-131a",
            ["Cs-137", "I-131a"],
            ["I-131a", "Cs-137"],
        ]
        # SR_TMP >

        name = var.getncattr("long_name").split("_")[0]
        unit = var.getncattr("units")
        if name in names:
            if unit == "ng kg-1":
                fld[:] *= 1e-12
            elif unit == "1e-12 kg m-2":
                fld[:] *= 1e-12
            else:
                raise NotImplementedError(f"species '{name}': unknown unit '{unit}'")
        else:
            raise NotImplementedError(f"species '{name}'")

    def _fix_nc_attrs(self, fld, attrs):

        # SR_TMP < TODO more general solution to combined species
        names = [
            "Cs-137",
            "I-131a",
            ["Cs-137", "I-131a"],
            ["I-131a", "Cs-137"],
        ]
        # SR_TMP >
        if attrs.species.name.value in names:

            new_unit = "Bq"  # SR_HC

            # Integration type
            if attrs.simulation.integr_type.value == "mean":
                pass
            elif attrs.simulation.integr_type.value in ["sum", "accum"]:
                new_unit += " h"
            else:
                raise NotImplementedError(
                    f"species '{attrs.species.name.value}': "
                    f"integration type '{attrs.simulation.integr_type.value}'"
                )

            # Original unit
            if attrs.variable.unit.value == "ng kg-1":
                new_unit += " m-3"  # SR_HC
            elif attrs.variable.unit.value == "1e-12 kg m-2":
                new_unit += " m-2"  # SR_HC
            else:
                raise NotImplementedError(
                    f"species '{attrs.species.name.value}': "
                    f"unit '{attrs.variable.unit.value}'"
                )

            attrs.variable.unit.value = new_unit
        else:
            raise NotImplementedError(f"species '{attrs.species.name.value}'")
