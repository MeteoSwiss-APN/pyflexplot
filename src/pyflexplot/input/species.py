"""Chemical species and their attributes."""
# Standard library
import dataclasses as dc
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Union

# Third-party
import numpy as np


@dc.dataclass
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


@dc.dataclass
# pylint: disable=R0902  # too-many-instance-attributes
class Species:
    """Chemical species."""

    id_cosmo: int = -99
    id_ifs: int = -99
    name: str = "N/A"
    weight: SpeciesAttribute = dc.field(default=SpeciesAttribute(-99.9, "g mol-1"))
    half_life: SpeciesAttribute = dc.field(default=SpeciesAttribute(-99.9, "a"))
    deposition_velocity: SpeciesAttribute = dc.field(
        default=SpeciesAttribute(-99.9, "m s-1")
    )
    sedimentation_velocity: SpeciesAttribute = dc.field(
        default=SpeciesAttribute(-99.9, "m s-1")
    )
    washout_coefficient: SpeciesAttribute = dc.field(
        default=SpeciesAttribute(-99.9, "s-1")
    )
    washout_exponent: SpeciesAttribute = dc.field(default=SpeciesAttribute(-99.9))

    @classmethod
    def create(cls, **params: Any) -> "Species":
        params_prep: Dict[str, Any] = {}
        fields = cls.__dataclass_fields__  # type: ignore  # pylint: disable=E1101
        for param, value in params.items():
            try:
                field: dc.Field = fields[param]
            except KeyError as e:
                raise ValueError(param) from e
            if issubclass(field.type, SpeciesAttribute):
                value = SpeciesAttribute.create(value)
            params_prep[param] = value
        return cls(**params_prep)


# See `flexpart/options/SPECIES/SPECIES_???`
# Corresponding parameters:
#   Species()               : SPECIES_???
#   name                    : Tracer name
#   weight                  : molweight
#   half_life               : Species half life
#   deposition_velocity     : Alternative: dry deposition velocity
#   sedimentation_velocity  :
#   washout_coefficient     : Wet deposition - A
#   washout_exponent        : Wet deposition - B
SPECIES: List[Species] = [
    Species.create(),
    Species.create(
        id_cosmo=1,
        id_ifs=1,
        name="TRACER",
        weight=(350.0, "g mol-1"),
        washout_exponent=0.0,
    ),
    Species.create(id_cosmo=2, id_ifs=2, name="O3", weight=(48.0, "g mol-1")),
    Species.create(
        id_cosmo=3,
        id_ifs=3,
        name="NO",
        weight=(30.0, "g mol-1"),
        washout_coefficient=(8.0e-6, "s-1"),
        washout_exponent=0.62,
    ),
    Species.create(
        id_cosmo=4,
        id_ifs=4,
        name="NO2",
        weight=(46.0, "g mol-1"),
        washout_coefficient=(1.0e-5, "s-1"),
        washout_exponent=0.62,
    ),
    Species.create(
        id_cosmo=5,
        id_ifs=5,
        name="HNO3",
        weight=(63.0, "g mol-1"),
        washout_coefficient=(5.0e-5, "s-1"),
        washout_exponent=0.62,
    ),
    Species.create(id_cosmo=6, id_ifs=6, name="HNO2", weight=(47.0, "g mol-1")),
    Species.create(
        id_cosmo=7,
        id_ifs=7,
        name="H2O2",
        weight=(34.0, "g mol-1"),
        washout_coefficient=(1.0e-4, "s-1"),
        washout_exponent=0.62,
    ),
    Species.create(
        id_cosmo=8,
        id_ifs=23,
        name="SO2",
        weight=(64.0, "g mol-1"),
        washout_exponent=0.62,
    ),
    Species.create(id_cosmo=9, id_ifs=9, name="HCHO", weight=(30.0, "g mol-1")),
    Species.create(id_cosmo=10, id_ifs=10, name="PAN", weight=(121.0, "g mol-1")),
    Species.create(
        id_cosmo=11,
        id_ifs=11,
        name="NH3",
        weight=(17.0, "g mol-1"),
        washout_exponent=0.62,
    ),
    Species.create(
        id_cosmo=12,
        id_ifs=12,
        name="SO4-aero",
        washout_coefficient=(1.0e-4, "s-1"),
        washout_exponent=0.8,
    ),
    Species.create(
        id_cosmo=13,
        id_ifs=13,
        name="NO3-aero",
        washout_coefficient=(5.0e-6, "s-1"),
        washout_exponent=0.62,
    ),
    Species.create(
        id_cosmo=14,
        id_ifs=14,
        name="I2-131",
        half_life=(8.0, "d"),
        washout_coefficient=(8.0e-5, "s-1"),
        washout_exponent=0.62,
    ),
    Species.create(
        id_cosmo=15,
        id_ifs=15,
        name="I-131a",
        # half_life=(8.02, "d"),
        half_life=(8.04, "d"),
        deposition_velocity=(1.5e-3, "m s-1"),
        sedimentation_velocity=(0.0, "m s-1"),
        washout_coefficient=(7.0e-5, "s-1"),
        washout_exponent=0.8,
    ),
    Species.create(
        id_cosmo=16,
        id_ifs=16,
        name="Cs-137",
        # half_life=(30.17, "a"),
        half_life=(30.0, "a"),
        deposition_velocity=(1.5e-3, "m s-1"),
        sedimentation_velocity=(0.0, "m s-1"),
        washout_coefficient=(7.0e-5, "s-1"),
        washout_exponent=0.8,
    ),
    Species.create(
        id_cosmo=17,
        id_ifs=17,
        name="Y-91",
        half_life=(58.3, "d"),
        washout_coefficient=(1.0e-4, "s-1"),
        washout_exponent=0.8,
    ),
    Species.create(
        id_cosmo=18,
        id_ifs=18,
        name="Ru-106",
        half_life=(1.0, "a"),
        washout_coefficient=(1.0e-4, "s-1"),
        washout_exponent=0.8,
    ),
    Species.create(id_cosmo=19, id_ifs=19, name="Kr-85"),
    Species.create(
        id_cosmo=20,
        id_ifs=20,
        name="Sr-90",
        washout_coefficient=(1.0e-4, "s-1"),
        washout_exponent=0.8,
    ),
    Species.create(
        id_cosmo=21,
        id_ifs=21,
        name="Xe-133",
        half_life=(5.245, "d"),
        deposition_velocity=(0.0, "m s-1"),
        washout_coefficient=(0.0, "s-1"),
        washout_exponent=0.0,
    ),
    Species.create(id_cosmo=22, id_ifs=22, name="CO", weight=(28.0, "g mol-1")),
    Species.create(
        id_cosmo=24,
        id_ifs=24,
        name="AIRTRACER",
        weight=(29.0, "g mol-1"),
        washout_exponent=0.0,
    ),
    Species.create(
        id_cosmo=25,
        name="AERO-TRACER",
        weight=(29.0, "g mol-1"),
        washout_coefficient=(2.0e-5, "s-1"),
        washout_exponent=0.8,
    ),
    Species.create(
        id_cosmo=29,
        id_ifs=29,
        name="Te-132",
        half_life=(3.258, "d"),
        deposition_velocity=(1.5e-3, "m s-1"),
        washout_coefficient=(7.0e-5, "s-1"),
        washout_exponent=0.8,
    ),
    Species.create(
        id_cosmo=30,
        name="I-131e",
        half_life=(8.04, "d"),
        deposition_velocity=(1.5e-2, "m s-1"),
        washout_coefficient=(7.0e-5, "s-1"),
        washout_exponent=0.8,
    ),
    Species.create(
        id_cosmo=31,
        name="I-131o",
        half_life=(8.04, "d"),
        deposition_velocity=(1.5e-4, "m s-1"),
        washout_coefficient=(7.0e-5, "s-1"),
        washout_exponent=0.8,
    ),
    Species.create(
        id_cosmo=32,
        name="Co-60",
        half_life=(5.271, "a"),
        deposition_velocity=(1.5e-3, "m s-1"),
        washout_coefficient=(7.0e-5, "s-1"),
        washout_exponent=0.8,
    ),
    Species.create(
        id_cosmo=33,
        name="Ir-192",
        half_life=(74.02, "d"),
        deposition_velocity=(1.5e-3, "m s-1"),
        washout_coefficient=(7.0e-5, "s-1"),
        washout_exponent=0.8,
    ),
    Species.create(id_cosmo=34, name="H-3"),
    Species.create(
        id_cosmo=35,
        name="Ba-140",
        half_life=(12.74, "d"),
        deposition_velocity=(1.5e-3, "m s-1"),
        washout_coefficient=(7.0e-5, "s-1"),
        washout_exponent=0.8,
    ),
    Species.create(
        id_cosmo=36,
        name="Zr-95",
        half_life=(63.98, "d"),
        deposition_velocity=(1.5e-3, "m s-1"),
        washout_coefficient=(7.0e-5, "s-1"),
        washout_exponent=0.8,
    ),
    Species.create(
        id_cosmo=37,
        name="Ru-103",
        half_life=(39.28, "d"),
        deposition_velocity=(1.5e-3, "m s-1"),
        washout_coefficient=(7.0e-5, "s-1"),
        washout_exponent=0.8,
    ),
    Species.create(
        id_cosmo=38,
        name="Pu-241",
        half_life=(14.4, "a"),
        deposition_velocity=(1.5e-3, "m s-1"),
        washout_coefficient=(7.0e-5, "s-1"),
        washout_exponent=0.8,
    ),
    Species.create(
        id_cosmo=39,
        id_ifs=32,
        name="Norm",
        half_life=(30.0, "a"),
        deposition_velocity=(1.5e-3, "m s-1"),
        washout_coefficient=(7.0e-5, "s-1"),
        washout_exponent=0.8,
    ),
    Species.create(id_ifs=25, name="AERO-TRACE"),
    Species.create(id_ifs=26, name="CH4"),
    Species.create(id_ifs=27, name="C2H6"),
    Species.create(id_ifs=28, name="C3H8"),
    Species.create(id_ifs=30, name="I2"),
    Species.create(id_ifs=31, name="PCB28"),
    Species.create(id_ifs=34, name="G"),
    Species.create(id_ifs=40, name="BC"),
]


@overload
def get_species(*, name: str) -> Species:
    ...


@overload
def get_species(*, name: Union[Tuple[str, ...], List[str]]) -> Tuple[Species, ...]:
    ...


def get_species(*, name=None):
    """Identify one or more ``Species`` objects by an attribute."""
    if isinstance(name, (Tuple, List)):
        return tuple(get_species(name=name_i) for name_i in name)
    global SPECIES  # pylint: disable=W0603  # global-statement
    attr, value = "name", cast(str, name)
    species: Species
    for species in SPECIES:
        try:
            value_i = getattr(species, attr)
        except AttributeError as e:
            raise ValueError(f"invalid attribute: {attr}") from e
        if value_i == value:
            return species
    s_value = f"'{value}'" if isinstance(value, str) else str(value)
    raise Exception(f"no species found with {attr} == {s_value}")
