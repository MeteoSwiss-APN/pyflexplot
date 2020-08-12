# -*- coding: utf-8 -*-
"""
Tests for class ``pyflexplot.setup.Dimensions``.
"""

# First-party
from pyflexplot.dimensions import CoreDimensions
from pyflexplot.dimensions import Dimensions


class Test_CoreDimensions_Init:
    """Initialize ``CoreDimensions`` objects."""

    def test_no_args(self):
        cdims = CoreDimensions()
        res = cdims.dict()
        sol = {
            "deposition_type": None,
            "nageclass": None,
            "noutrel": None,
            "numpoint": None,
            "species_id": None,
            "time": None,
            "level": None,
        }
        assert res == sol

    def test_all_args(self):
        params = {
            "deposition_type": "dry",
            "nageclass": 0,
            "noutrel": 1,
            "numpoint": 3,
            "species_id": 2,
            "time": 0,
            "level": 2,
        }
        cdims = CoreDimensions(**params)
        res = cdims.dict()
        sol = params
        assert res == sol

    def test_some_args(self):
        params = {
            "noutrel": 1,
            "species_id": 2,
        }
        cdims = CoreDimensions(**params)
        res = cdims.dict()
        sol = {
            "deposition_type": None,
            "nageclass": None,
            "noutrel": 1,
            "numpoint": None,
            "species_id": 2,
            "time": None,
            "level": None,
        }
        assert res == sol


class Test_Init:
    """Initialize ``Dimensions`` objects with ``CoreDimensions`` objects."""

    def test_no_args(self):
        dims = Dimensions()
        res = dims.raw_dict()
        sol = {
            "deposition_type": (None,),
            "nageclass": (None,),
            "noutrel": (None,),
            "numpoint": (None,),
            "species_id": (None,),
            "time": (None,),
            "level": (None,),
        }
        assert res == sol

    def test_single_core(self):
        cdims = CoreDimensions(
            deposition_type="wet",
            nageclass=0,
            noutrel=1,
            numpoint=3,
            species_id=2,
            time=0,
            level=2,
        )
        dims = Dimensions([cdims])
        res = dims.raw_dict()
        sol = {
            "deposition_type": ("wet",),
            "nageclass": (0,),
            "noutrel": (1,),
            "numpoint": (3,),
            "species_id": (2,),
            "time": (0,),
            "level": (2,),
        }
        assert res == sol

    def test_double_default_core(self):
        core = [CoreDimensions(), CoreDimensions()]
        dims = Dimensions(core)
        res = dims.raw_dict()
        sol = {
            "deposition_type": (None, None),
            "nageclass": (None, None),
            "noutrel": (None, None),
            "numpoint": (None, None),
            "species_id": (None, None),
            "time": (None, None),
            "level": (None, None),
        }
        assert res == sol

    def test_multi_core(self):
        core = [
            CoreDimensions(nageclass=0, time=0,),
            CoreDimensions(deposition_type="wet"),
            CoreDimensions(
                deposition_type="dry", nageclass=3, species_id=0, time=2, level=1
            ),
            CoreDimensions(
                deposition_type="dry",
                nageclass=0,
                noutrel=1,
                species_id=2,
                time=1,
                level=1,
            ),
        ]
        dims = Dimensions(core)
        res = dims.raw_dict()
        sol = {
            "deposition_type": (None, "wet", "dry", "dry"),
            "nageclass": (0, None, 3, 0),
            "noutrel": (None, None, None, 1),
            "numpoint": (None, None, None, None),
            "species_id": (None, None, 0, 2),
            "time": (0, None, 2, 1),
            "level": (None, None, 1, 1),
        }
        assert res == sol


class Test_Dict:
    """Represent a ``dimensions`` object as a clean dict."""

    def test_no_args(self):
        dims = Dimensions()
        res = dims.dict()
        sol = {
            "deposition_type": None,
            "nageclass": None,
            "noutrel": None,
            "numpoint": None,
            "species_id": None,
            "time": None,
            "level": None,
        }
        assert res == sol

    def test_single_core(self):
        cdims = CoreDimensions(
            deposition_type="dry",
            nageclass=0,
            noutrel=1,
            numpoint=3,
            species_id=2,
            time=0,
            level=2,
        )
        dims = Dimensions([cdims])
        res = dims.dict()
        sol = {
            "deposition_type": "dry",
            "nageclass": 0,
            "noutrel": 1,
            "numpoint": 3,
            "species_id": 2,
            "time": 0,
            "level": 2,
        }
        assert res == sol

    def test_double_default_core(self):
        core = [CoreDimensions(), CoreDimensions()]
        dims = Dimensions(core)
        res = dims.dict()
        sol = {
            "deposition_type": None,
            "nageclass": None,
            "noutrel": None,
            "numpoint": None,
            "species_id": None,
            "time": None,
            "level": None,
        }
        assert res == sol

    def test_multi_core(self):
        core = [
            CoreDimensions(nageclass=0, time=0,),
            CoreDimensions(deposition_type="wet"),
            CoreDimensions(
                deposition_type="dry", nageclass=3, species_id=0, time=2, level=1
            ),
            CoreDimensions(
                deposition_type="dry",
                nageclass=0,
                noutrel=1,
                species_id=2,
                time=1,
                level=1,
            ),
        ]
        dims = Dimensions(core)
        res = dims.dict()
        sol = {
            "deposition_type": ("dry", "wet"),
            "nageclass": (0, 3),
            "noutrel": 1,
            "numpoint": None,
            "species_id": (0, 2),
            "time": (0, 1, 2),
            "level": 1,
        }
        assert res == sol


class Test_Create:
    """Create dimensions objects from a parameter dict."""

    def test_no_args(self):
        dims = Dimensions.create({})
        res = dims.dict()
        sol = {
            "deposition_type": None,
            "nageclass": None,
            "noutrel": None,
            "numpoint": None,
            "species_id": None,
            "time": None,
            "level": None,
        }
        assert res == sol

    def test_single_core(self):
        params = {
            "deposition_type": "dry",
            "nageclass": 0,
            "noutrel": 1,
            "numpoint": 3,
            "species_id": 2,
            "time": 0,
            "level": 2,
        }
        dims = Dimensions.create(params)
        res = dims.dict()
        sol = params
        assert res == sol

    def test_multi_core(self):
        params = {
            "deposition_type": ("wet", "dry"),
            "nageclass": (0, 3),
            "noutrel": 1,
            "numpoint": None,
            "species_id": (0, 2),
            "time": (2, 1, 0),
            "level": 1,
        }
        dims = Dimensions.create(params)
        res = dims.dict()
        sol = {
            key: (tuple(sorted(val)) if isinstance(val, tuple) else val)
            for key, val in params.items()
        }
        assert res == sol


class Test_Interact:
    """Acces and change paramters, derive new objects, etc."""

    params = {
        "deposition_type": ("dry", "wet", None, "dry"),
        "nageclass": (3, None, 0),
        "noutrel": (1, 1),
        "numpoint": None,
        "species_id": (0, 2),
        "time": (2, 0, 1),
        "level": 1,
    }

    def create_dims(self):
        return Dimensions.create(self.params)

    def assert_is_empty(self, dims: Dimensions) -> None:
        assert all(dims.get(param) is None for param in self.params)

    def test_get_raw(self):
        dims = self.create_dims()
        assert dims.get_raw("deposition_type") == ("dry", "wet", None, "dry")
        assert dims.get_raw("nageclass") == (3, None, 0, None)
        assert dims.get_raw("noutrel") == (1, 1, None, None)
        assert dims.get_raw("numpoint") == (None, None, None, None)
        assert dims.get_raw("species_id") == (0, 2, None, None)
        assert dims.get_raw("time") == (2, 0, 1, None)
        assert dims.get_raw("level") == (1, None, None, None)

    def test_get(self):
        dims = self.create_dims()
        assert dims.get("deposition_type") == ("dry", "wet")
        assert dims.get("nageclass") == (0, 3)
        assert dims.get("noutrel") == 1
        assert dims.get("numpoint") is None
        assert dims.get("species_id") == (0, 2)
        assert dims.get("time") == (0, 1, 2)
        assert dims.get("level") == 1

    def test_get_property(self):
        dims = self.create_dims()
        assert dims.deposition_type == dims.get("deposition_type")
        assert dims.nageclass == dims.get("nageclass")
        assert dims.noutrel == dims.get("noutrel")
        assert dims.numpoint == dims.get("numpoint")
        assert dims.species_id == dims.get("species_id")
        assert dims.time == dims.get("time")
        assert dims.level == dims.get("level")

    def test_get_getattr(self):
        dims = self.create_dims()
        for param in self.params:
            assert getattr(dims, param) == dims.get(param)

    def test_set_raw(self):
        dims = Dimensions()
        self.assert_is_empty(dims)
        dims.set("deposition_type", ("dry", "wet", None, "dry"))
        dims.set("nageclass", (3, None, 0, None))
        dims.set("noutrel", (1, 1, None, None))
        dims.set("numpoint", (None, None, None, None))
        dims.set("species_id", (0, 2, None, None))
        dims.set("time", (2, 0, 1, None))
        dims.set("level", (1, None, None, None))
        res = dims.dict()
        sol = self.create_dims().dict()
        assert res == sol

    def test_set_compact(self):
        dims = Dimensions()
        self.assert_is_empty(dims)
        dims.set("deposition_type", ("dry", "wet"))
        dims.set("nageclass", (0, 3))
        dims.set("noutrel", 1)
        dims.set("numpoint", None)
        dims.set("species_id", (0, 2))
        dims.set("time", (0, 1, 2))
        dims.set("level", 1)
        res = dims.dict()
        sol = self.create_dims().dict()
        assert res == sol

    def test_set_property_raw(self):
        dims = Dimensions()
        self.assert_is_empty(dims)
        dims.deposition_type = ("dry", "wet", None, "dry")
        dims.nageclass = (3, None, 0, None)
        dims.noutrel = (1, 1, None, None)
        dims.numpoint = (None, None, None, None)
        dims.species_id = (0, 2, None, None)
        dims.time = (2, 0, 1, None)
        dims.level = (1, None, None, None)
        res = dims.dict()
        sol = self.create_dims().dict()
        assert res == sol

    def test_set_property_compact(self):
        dims = Dimensions()
        self.assert_is_empty(dims)
        dims.deposition_type = ("dry", "wet")
        dims.nageclass = (0, 3)
        dims.noutrel = 1
        dims.numpoint = None
        dims.species_id = (0, 2)
        dims.time = (0, 1, 2)
        dims.level = 1
        res = dims.dict()
        sol = self.create_dims().dict()
        assert res == sol

    def test_update_empty_with_full(self):
        dims = Dimensions()
        self.assert_is_empty(dims)
        dims.update(self.params)
        res = dims.dict()
        sol = self.create_dims().dict()
        assert res == sol

    def test_update_empty_with_partial(self):
        dims = Dimensions()
        self.assert_is_empty(dims)
        dims.update({"deposition_type": "wet"})
        dims.update({"nageclass": (3, None, 0)})
        dims.update({"noutrel": (1, 1)})
        dims.update({"level": 1})
        res = dims.dict()
        sol = {
            "deposition_type": "wet",
            "nageclass": (0, 3),
            "noutrel": 1,
            "numpoint": None,
            "species_id": None,
            "time": None,
            "level": 1,
        }
        assert res == sol

    def test_derive_empty(self):
        dims = self.create_dims()
        derived = dims.derive({})
        res = dims.dict()
        sol = derived.dict()
        assert res == sol

    def test_derive_some(self):
        dims = self.create_dims()
        derived = dims.derive({"level": 2, "deposition_type": ("dry", "wet")})
        sol = {**dims.dict(), "deposition_type": ("dry", "wet"), "level": 2}
        res = derived.dict()
        assert res == sol
