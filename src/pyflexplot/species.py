# -*- coding: utf-8 -*-
"""
Chemical species and their attributes.
"""
# Standard library
import dataclasses
from dataclasses import dataclass
from dataclasses import Field
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Union

# Third-party
import numpy as np


@dataclass
class SpeciesAttribute:
    """Attribute of a chemical species."""

    value: float = np.nan
    unit: Optional[str] = None

    @overload
    @classmethod
    def create(cls, arg1: float, arg2: Optional[str] = None):
        ...

    @overload
    @classmethod
    def create(cls, arg1: Tuple[float, str], arg2=None):
        ...

    @classmethod
    def create(cls, arg1, arg2=None) -> "SpeciesAttribute":
        if isinstance(arg1, Sequence):
            return cls.create(*arg1)
        value: float = float(arg1)
        unit: Optional[str] = None if arg2 is None else str(arg2)
        return cls(value, unit)

    @classmethod
    def field(cls, **kwargs: Any) -> "SpeciesAttribute":
        """Create a dataclass field that defaults to ``SpeciesAttribute()``.

        The return type is "SpeciesAttribute" rather than "dataclasses.Field"
        to prevent mypy error "incompatible types in assignment".

        Example:
        > @dataclass
        > class Species:
        >     age: SpeciesAttribute = SpeciesAttribute.field()

        """
        return dataclasses.field(default_factory=cls, **kwargs)  # type: ignore


@dataclass
class Species:
    """Chemical species."""

    id_: int
    name: str
    half_life: SpeciesAttribute = SpeciesAttribute.field()
    deposition_velocity: SpeciesAttribute = SpeciesAttribute.field()
    sedimentation_velocity: SpeciesAttribute = SpeciesAttribute.field()
    washout_coefficient: SpeciesAttribute = SpeciesAttribute.field()
    washout_exponent: SpeciesAttribute = SpeciesAttribute.field()

    @classmethod
    def create(cls, **params: Any) -> "Species":
        params_prep: Dict[str, Any] = {}
        fields = cls.__dataclass_fields__  # type: ignore  # pylint: disable=E1101
        for param, value in params.items():
            try:
                field: Field = fields[param]
            except KeyError:
                raise ValueError(param)
            if issubclass(field.type, SpeciesAttribute):
                value = SpeciesAttribute.create(value)
            params_prep[param] = value
        return cls(**params_prep)


defaults = {
    "half_life": (-999, "a"),
    "deposition_velocity": (-999, "m s-1"),
    "sedimentation_velocity": (-999, "m s-1"),
    "washout_coefficient": (-999, "s-1"),
    "washout_exponent": -999,
}


SPECIES: List[Species] = [
    Species.create(id_=1, name="TRACER", **defaults),
    Species.create(id_=2, name="O3", **defaults),
    Species.create(id_=3, name="NO", **defaults),
    Species.create(id_=4, name="NO2", **defaults),
    Species.create(id_=5, name="HNO3", **defaults),
    Species.create(id_=6, name="HNO2", **defaults),
    Species.create(id_=7, name="H2O2", **defaults),
    Species.create(id_=9, name="HCHO", **defaults),
    Species.create(id_=10, name="PAN", **defaults),
    Species.create(id_=11, name="NH3", **defaults),
    Species.create(id_=12, name="SO4", **defaults),
    Species.create(id_=13, name="NO3", **defaults),
    Species.create(id_=14, name="I2", **defaults),
    Species.create(
        id_=15,
        name="I-131a",
        half_life=(8.02, "d"),
        deposition_velocity=(1.5e-3, "m s-1"),
        sedimentation_velocity=(0.0, "m s-1"),
        washout_coefficient=(7.0e-5, "s-1"),
        washout_exponent=0.8,
    ),
    Species.create(
        id_=16,
        name="Cs-137",
        half_life=(30.17, "a"),
        deposition_velocity=(1.5e-3, "m s-1"),
        sedimentation_velocity=(0.0, "m s-1"),
        washout_coefficient=(7.0e-5, "s-1"),
        washout_exponent=0.8,
    ),
    Species.create(id_=17, name="Y", **defaults),
    Species.create(id_=18, name="Ru", **defaults),
    Species.create(id_=19, name="Kr", **defaults),
    Species.create(id_=20, name="Sr", **defaults),
    Species.create(id_=21, name="Xe", **defaults),
    Species.create(id_=22, name="CO", **defaults),
    Species.create(id_=23, name="SO2", **defaults),
    Species.create(id_=24, name="AIRTRACER", **defaults),
    Species.create(id_=25, name="AERO", **defaults),
    Species.create(id_=26, name="CH4", **defaults),
    Species.create(id_=27, name="C2H6", **defaults),
    Species.create(id_=28, name="C3H8", **defaults),
    Species.create(id_=29, name="Te", **defaults),
    Species.create(id_=30, name="I2", **defaults),
    Species.create(id_=31, name="PCB28", **defaults),
    Species.create(id_=32, name="Norm", **defaults),
    Species.create(id_=34, name="G", **defaults),
    Species.create(id_=40, name="BC", **defaults),
]


@overload
def get_species(*, id_: None = None, name: str) -> Species:
    ...


@overload
def get_species(*, id_: int, name: None = None) -> Species:
    ...


@overload
def get_species(
    *, id_: None = None, name: Union[List[str], Tuple[str, ...]]
) -> Tuple[Species]:
    ...


@overload
def get_species(*, id_: Sequence[int], name: None = None) -> Tuple[Species]:
    ...


def get_species(*, id_=None, name=None):
    """Identify one or more ``Species`` objects by an attribute."""
    if id_ is not None:
        attr, value = "id_", id_
    elif name is not None:
        attr, value = "name", name
    else:
        raise ValueError("must pass one argument")
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [get_species(**{attr: value_i}) for value_i in value]
    global SPECIES  # pylint: disable=W0603  # global-statement
    species: Species
    for species in SPECIES:
        try:
            value_i = getattr(species, attr)
        except AttributeError:
            raise ValueError(f"invalid attribute: {attr}")
        if value_i == value:
            return species
    s_value = f"'{value}'" if isinstance(value, str) else str(value)
    raise Exception(f"no species found with {attr} == {s_value}")
