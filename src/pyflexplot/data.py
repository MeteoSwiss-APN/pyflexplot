# -*- coding: utf-8 -*-
"""
Data structures.
"""
import logging as log
import numpy as np

from collections import namedtuple

from .utils_dev import ipython  #SR_DEV

FieldKey = namedtuple(
    'FieldKey',
    'age_ind relpt_ind time_ind level_ind species_id field_type',
)


class FlexData:
    """Hold FLEXPART output data.

    Args:
        setup (dict): Setup. TODO: Better description!

    """

    def __init__(self, setup):
        self._setup = setup

        self._fields = {}
        self._field_attrs = {}

    def field_key(
            self, *, age_ind, relpt_ind, time_ind, level_ind, species_id,
            field_type):
        """Create key for field variable.

        Kwargs:
            age_ind (int): Index of age class.

            relpt_ind (int): Index of release point.

            time_ind (int): Index of time step.

            level_ind (int): Index of vertical level.

            species_id (int): Id of particle species.

            field_type (str): The type of the field ('3D', 'WD', 'DD').

        Returns:
            FieldKey instance (namedtuple).
        """
        return FieldKey(
            age_ind=age_ind,
            relpt_ind=relpt_ind,
            time_ind=time_ind,
            level_ind=level_ind,
            species_id=species_id,
            field_type=field_type,
        )

    def field_keys(
            self,
            *,
            age_inds=None,
            relpt_inds=None,
            time_inds=None,
            level_inds=None,
            species_ids=None,
            field_types=None):
        """Returns field keys, either all or a subset.

        Kwargs:
            age_inds (list[int], optional): Age class indices.
                Defaults to None. If None, all values are included.

            relpt_inds (list[int], optional): Release point indices.
                Defaults to None. If None, all values are included.

            time_inds (list[int], optional): Time step indices.
                Defaults to None. If None, all values are included.

            level_inds (list[int], optional): Vertical level indices.
                Defaults to None. If None, all values are included.

            species_ids (list[int], optional): Species ids.
                Defaults to None. If None, all values are included.

            field_types (list[int], optional): Field types.
                Defaults to None. If None, all values are included.

        Returns:
            list: List of keys.
        """

        def check_restriction(values, name):
            """Check validity of restriction."""
            if values is not None:
                choices = getattr(self, f'{name}s')()
                for value in values:
                    if value not in choices:
                        raise ValueError(
                            f"invalid {name.replace('_', ' ')} '{value}':"
                            f" not among {choices}")

        check_restriction(species_ids, 'age_ind')
        check_restriction(species_ids, 'relpt_ind')
        check_restriction(species_ids, 'time_ind')
        check_restriction(species_ids, 'level_ind')
        check_restriction(species_ids, 'species_id')
        check_restriction(field_types, 'field_type')

        # Collect keys
        keys = []
        for key in self._fields.keys():
            if (age_inds is not None and key.age_ind not in age_inds):
                continue
            if (relpt_inds is not None and key.relpt_ind not in relpt_inds):
                continue
            if (time_inds is not None and key.time_ind not in time_inds):
                continue
            if (level_inds is not None and key.level_ind not in level_inds):
                continue
            if (species_ids is not None and key.species_id not in species_ids):
                continue
            if (field_types is not None and key.field_type not in field_types):
                continue
            keys.append(key)

        return [key for key in self._fields.keys()]

    def age_inds(self, **restrictions):
        """Returns all age class ids, or a subset thereof."""
        return [key.age_ind for key in self.field_keys(**restrictions)]

    def relpt_inds(self, **restrictions):
        """Returns all relpt class ids, or a subset thereof."""
        return [key.relpt_ind for key in self.field_keys(**restrictions)]

    def time_inds(self, **restrictions):
        """Returns all time class ids, or a subset thereof."""
        return [key.time_ind for key in self.field_keys(**restrictions)]

    def level_inds(self, **restrictions):
        """Returns all level class ids, or a subset thereof."""
        return [key.level_ind for key in self.field_keys(**restrictions)]

    def species_ids(self, **restrictions):
        """Returns all species ids, or a subset thereof."""
        return [key.species_id for key in self.field_keys(**restrictions)]

    def field_types(self, **restrictions):
        """Returns all field types, or a subset thereof."""
        return [key.field_type for key in self.field_keys(**restrictions)]

    def set_grid(self, *, rlat, rlon):
        """Set grid variables.

        Kwargs:
            rlat (ndarray): Rotated latitude array (1D).

            rlon (ndarray): Rotated longitude array (1D).
        """
        self.rlat = rlat
        self.rlon = rlon

    def add_field(self, arr, attrs=None, **key_comps):
        """Add a field array with optional attributes.

        Args:
            arr (ndarray): Field array.

            attrs (dict, optional): Field attributes. Defaults to None.

            **key_comps: Components of field key. Passed on to method
                ``FlexData.field_key``.

        """
        key = self.field_key(**key_comps)
        self._fields[key] = arr
        self._field_attrs[key] = attrs

    def field(self, key_or_comps):
        """Access a field.

        Args:
            key_or_comps (FieldKey or dict): A FieldKey instance, or
                a dict containing the key components passed on to
                ``FlexData.field_key`` to create a FieldKey.

        Returns:
            ndarray: Field array.

        """
        if isinstance(key_or_comps, FieldKey):
            key = key_or_comps
        else:
            key = self.field_key(**key_or_comps)
        return self._fields[key]

    def field_attrs(self, **key_components):
        """Return field attributes.

        Args:
            **key_comps: Components of field key. Passed on to method
                ``FlexData.field_key``.

        Returns:
            Field attributes dict.

        """
        return self._field_attrs[self.field_key(**key_comps)]
