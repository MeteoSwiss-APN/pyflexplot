"""Test class ``pyflexplot.setup.Dimensions``."""
# Third-party
import pytest

# First-party
from pyflexplot.setups.dimensions import CoreDimensions
from pyflexplot.setups.dimensions import Dimensions
from srutils.testing import assert_is_sub_element


class Test_Init:
    """Initialize ``Dimensions`` objects with ``CoreDimensions`` objects."""

    def test_no_args(self):
        dims = Dimensions()
        res = dims.raw_dict()
        sol = {
            "level": (None,),
            "nageclass": (None,),
            "release": (None,),
            "species_id": (None,),
            "time": (None,),
            "variable": (None,),
            "multiplier": (None,),
        }
        assert res == sol

    def test_single_core(self):
        cdims = CoreDimensions(
            level=2,
            nageclass=0,
            release=3,
            species_id=2,
            time=0,
            variable="wet_deposition",
            multiplier=1,
        )
        dims = Dimensions([cdims])
        res = dims.raw_dict()
        sol = {
            "level": (2,),
            "nageclass": (0,),
            "release": (3,),
            "species_id": (2,),
            "time": (0,),
            "variable": ("wet_deposition",),
            "multiplier": (1,),
        }
        assert res == sol

    def test_double_default_core(self):
        core = [CoreDimensions(), CoreDimensions()]
        dims = Dimensions(core)
        res = dims.raw_dict()
        sol = {
            "level": (None, None),
            "nageclass": (None, None),
            "release": (None, None),
            "species_id": (None, None),
            "time": (None, None),
            "variable": (None, None),
            "multiplier": (None, None),
        }
        assert res == sol

    def test_multi_core(self):
        core = [
            CoreDimensions(nageclass=0, time=0, variable="concentration"),
            CoreDimensions(variable="wet_deposition"),
            CoreDimensions(
                level=1,
                nageclass=3,
                species_id=0,
                time=2,
                variable="dry_deposition",
                multiplier=1,
            ),
            CoreDimensions(
                level=1,
                nageclass=0,
                release=1,
                species_id=2,
                time=1,
                variable="dry_deposition",
                multiplier=2,
            ),
        ]
        dims = Dimensions(core)
        res = dims.raw_dict()
        sol = {
            "level": (None, None, 1, 1),
            "nageclass": (0, None, 3, 0),
            "release": (None, None, None, 1),
            "species_id": (None, None, 0, 2),
            "time": (0, None, 2, 1),
            "variable": (
                "concentration",
                "wet_deposition",
                "dry_deposition",
                "dry_deposition",
            ),
            "multiplier": (None, None, 1, 2),
        }
        assert res == sol


class Test_Create:
    """Create dimensions objects from a parameter dict."""

    def test_no_args_fail(self):
        with pytest.raises(ValueError):
            Dimensions.create({})

    def test_almost_no_args(self):
        dims = Dimensions.create({}, plot_variable="concentration")
        res = dims.dict()
        sol = {
            "level": None,
            "nageclass": None,
            "release": None,
            "species_id": None,
            "time": None,
            "variable": "concentration",
            "multiplier": None,
        }
        assert res == sol

    def test_single_core(self):
        params = {
            "level": 2,
            "nageclass": 0,
            "release": 3,
            "species_id": 2,
            "time": 0,
            "variable": "dry_deposition",
            "multiplier": 1,
        }
        dims = Dimensions.create(params)
        res = dims.dict()
        sol = params
        assert res == sol

    def test_multi_core(self):
        params = {
            "level": 1,
            "nageclass": (0, 3),
            "release": None,
            "species_id": (0, 2),
            "time": (2, 1, 0),
            "variable": ("wet_deposition", "dry_deposition"),
            "multiplier": (1, 10),
        }
        dims = Dimensions.create(params)
        res = dims.dict()
        sol = {
            key: (tuple(sorted(val)) if isinstance(val, tuple) else val)
            for key, val in params.items()
        }
        assert res == sol

    def test_variable_ok(self):
        params = {"variable": ("concentration", "dry_deposition", "wet_deposition")}
        try:
            Dimensions.create(params)
        except ValueError as e:
            raise AssertionError(f"unexpected exception: {repr(e)}") from e

    def test_variable_fail(self):
        params = {"variable": ("concentration", ("dry_deposition", "wet_deposition"))}
        with pytest.raises(ValueError):
            Dimensions.create(params)


class Test_Dict:
    """Represent a ``dimensions`` object as a clean dict."""

    def test_no_args(self):
        dims = Dimensions()
        res = dims.dict()
        sol = {
            "level": None,
            "nageclass": None,
            "release": None,
            "species_id": None,
            "time": None,
            "variable": None,
            "multiplier": None,
        }
        assert res == sol

    def test_single_core(self):
        cdims = CoreDimensions(
            level=2,
            nageclass=0,
            release=3,
            species_id=2,
            time=0,
            variable="dry_deposition",
            multiplier=1,
        )
        dims = Dimensions([cdims])
        res = dims.dict()
        sol = {
            "level": 2,
            "nageclass": 0,
            "release": 3,
            "species_id": 2,
            "time": 0,
            "variable": "dry_deposition",
            "multiplier": 1,
        }
        assert res == sol

    def test_double_default_core(self):
        core = [CoreDimensions(), CoreDimensions()]
        dims = Dimensions(core)
        res = dims.dict()
        sol = {
            "level": None,
            "nageclass": None,
            "release": None,
            "species_id": None,
            "time": None,
            "variable": None,
            "multiplier": None,
        }
        assert res == sol

    def test_multi_core(self):
        core = [
            CoreDimensions(nageclass=0, time=0, variable="concentration"),
            CoreDimensions(
                variable="wet_deposition",
                multiplier=1,
            ),
            CoreDimensions(
                nageclass=3,
                species_id=0,
                time=2,
                level=1,
                variable="dry_deposition",
                multiplier=10,
            ),
            CoreDimensions(
                nageclass=0,
                species_id=2,
                time=1,
                level=1,
                variable="dry_deposition",
                multiplier=100,
            ),
        ]
        dims = Dimensions(core)
        res = dims.dict()
        sol = {
            "level": 1,
            "nageclass": (0, 3),
            "release": None,
            "species_id": (0, 2),
            "time": (0, 1, 2),
            "variable": ("concentration", "dry_deposition", "wet_deposition"),
            "multiplier": (1, 10, 100),
        }
        assert res == sol


class TestDecompress:
    params = {
        "nageclass": (0,),
        "release": (0,),
        "species_id": [1, 2],
        "time": [1, 2, 3],
        "variable": ["dry_deposition", "wet_deposition"],
    }

    def test_full(self):
        dims = Dimensions.create(self.params)
        dims_lst = dims.decompress()
        dcts = [dims.dict() for dims in dims_lst]
        sol = [
            {"species_id": 1, "time": 1, "variable": "dry_deposition"},
            {"species_id": 1, "time": 1, "variable": "wet_deposition"},
            {"species_id": 1, "time": 2, "variable": "dry_deposition"},
            {"species_id": 1, "time": 2, "variable": "wet_deposition"},
            {"species_id": 1, "time": 3, "variable": "dry_deposition"},
            {"species_id": 1, "time": 3, "variable": "wet_deposition"},
            {"species_id": 2, "time": 1, "variable": "dry_deposition"},
            {"species_id": 2, "time": 1, "variable": "wet_deposition"},
            {"species_id": 2, "time": 2, "variable": "dry_deposition"},
            {"species_id": 2, "time": 2, "variable": "wet_deposition"},
            {"species_id": 2, "time": 3, "variable": "dry_deposition"},
            {"species_id": 2, "time": 3, "variable": "wet_deposition"},
        ]
        assert_is_sub_element(sol, dcts, "solution", "result")

    def test_skip(self):
        dims = Dimensions.create(self.params)
        dims_lst = dims.decompress(skip=["variable", "time"])
        dcts = [dims.dict() for dims in dims_lst]
        sol = [
            {
                "species_id": 1,
                "time": [1, 2, 3],
                "variable": ["dry_deposition", "wet_deposition"],
            },
            {
                "species_id": 2,
                "time": [1, 2, 3],
                "variable": ["dry_deposition", "wet_deposition"],
            },
        ]
        assert_is_sub_element(sol, dcts, "solution", "result")

    def test_select(self):
        dims = Dimensions.create(self.params)
        dims_lst = dims.decompress(select=["variable", "time"])
        dcts = [dims.dict() for dims in dims_lst]
        sol = [
            {"species_id": [1, 2], "time": 1, "variable": "dry_deposition"},
            {"species_id": [1, 2], "time": 1, "variable": "wet_deposition"},
            {"species_id": [1, 2], "time": 2, "variable": "dry_deposition"},
            {"species_id": [1, 2], "time": 2, "variable": "wet_deposition"},
            {"species_id": [1, 2], "time": 3, "variable": "dry_deposition"},
            {"species_id": [1, 2], "time": 3, "variable": "wet_deposition"},
        ]
        assert_is_sub_element(sol, dcts, "solution", "result")

    def test_select_skip(self):
        dims = Dimensions.create(self.params)
        dims_lst = dims.decompress(
            select=["variable", "time"], skip=["time", "species_id"]
        )
        dcts = [dims.dict() for dims in dims_lst]
        sol = [
            {"species_id": [1, 2], "time": [1, 2, 3], "variable": "dry_deposition"},
            {"species_id": [1, 2], "time": [1, 2, 3], "variable": "wet_deposition"},
        ]
        assert_is_sub_element(sol, dcts, "solution", "result")


class Test_Interact:
    """Access and change parameters, derive new objects, etc."""

    params = {
        "level": 1,
        "nageclass": (3, None, 0),
        "release": None,
        "species_id": (0, 2),
        "time": (2, 0, 1),
        "variable": (
            "dry_deposition",
            "wet_deposition",
            "concentration",
            "dry_deposition",
        ),
        "multiplier": (None, 1, 10),
    }

    def create_dims(self):
        return Dimensions.create(self.params)

    def assert_is_empty(self, dims: Dimensions) -> None:
        assert all(dims.get(param) is None for param in self.params)

    def test_get_raw(self):
        dims = self.create_dims()
        assert dims.get_raw("level") == (1, None, None, None)
        assert dims.get_raw("nageclass") == (3, None, 0, None)
        assert dims.get_raw("release") == (None, None, None, None)
        assert dims.get_raw("species_id") == (0, 2, None, None)
        assert dims.get_raw("time") == (2, 0, 1, None)
        assert dims.get_raw("variable") == (
            "dry_deposition",
            "wet_deposition",
            "concentration",
            "dry_deposition",
        )

    def test_get(self):
        dims = self.create_dims()
        assert dims.get("level") == 1
        assert dims.get("nageclass") == (0, 3)
        assert dims.get("release") is None
        assert dims.get("species_id") == (0, 2)
        assert dims.get("time") == (0, 1, 2)
        assert dims.get("variable") == (
            "concentration",
            "dry_deposition",
            "wet_deposition",
        )

    def test_get_property(self):
        dims = self.create_dims()
        assert dims.level == dims.get("level")
        assert dims.nageclass == dims.get("nageclass")
        assert dims.release == dims.get("release")
        assert dims.species_id == dims.get("species_id")
        assert dims.time == dims.get("time")
        assert dims.variable == dims.get("variable")

    def test_get_getattr(self):
        dims = self.create_dims()
        for param in self.params:
            assert getattr(dims, param) == dims.get(param)

    def test_set_raw(self):
        dims = Dimensions()
        self.assert_is_empty(dims)
        dims.set("level", (1, None, None, None))
        dims.set("nageclass", (3, None, 0, None))
        dims.set("release", (None, None, None, None))
        dims.set("species_id", (0, 2, None, None))
        dims.set("time", (2, 0, 1, None))
        dims.set(
            "variable",
            ("dry_deposition", "wet_deposition", "concentration", "dry_deposition"),
        )
        dims.set(
            "multiplier",
            (1, 10, None, None),
        )
        res = dims.dict()
        sol = self.create_dims().dict()
        assert res == sol

    def test_set_compact(self):
        dims = Dimensions()
        self.assert_is_empty(dims)
        dims.set("level", 1)
        dims.set("nageclass", (0, 3))
        dims.set("release", None)
        dims.set("species_id", (0, 2))
        dims.set("time", (0, 1, 2))
        dims.set("variable", ("concentration", "dry_deposition", "wet_deposition"))
        dims.set(
            "multiplier",
            (1, 10),
        )
        res = dims.dict()
        sol = self.create_dims().dict()
        assert res == sol

    def test_set_property_raw(self):
        dims = Dimensions()
        self.assert_is_empty(dims)
        dims.level = (1, None, None, None)
        dims.nageclass = (3, None, 0, None)
        dims.release = (None, None, None, None)
        dims.species_id = (0, 2, None, None)
        dims.time = (2, 0, 1, None)
        dims.variable = (
            "dry_deposition",
            "wet_deposition",
            "concentration",
            "dry_deposition",
        )
        dims.multiplier = (1, 10, None, None)
        res = dims.dict()
        sol = self.create_dims().dict()
        assert res == sol

    def test_set_property_compact(self):
        dims = Dimensions()
        self.assert_is_empty(dims)
        dims.level = 1
        dims.nageclass = (0, 3)
        dims.release = None
        dims.species_id = (0, 2)
        dims.time = (0, 1, 2)
        dims.variable = ("concentration", "dry_deposition", "wet_deposition")
        dims.multiplier = (1, 10, None, None)
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
        dims.update({"level": 1})
        dims.update({"nageclass": (3, None, 0)})
        dims.update({"variable": "wet_deposition"})
        res = dims.dict()
        sol = {
            "level": 1,
            "nageclass": (0, 3),
            "release": None,
            "species_id": None,
            "time": None,
            "variable": "wet_deposition",
            "multiplier": None,
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
        derived = dims.derive(
            {
                "level": 2,
                "variable": ("dry_deposition", "wet_deposition"),
            }
        )
        sol = {
            **dims.dict(),
            "level": 2,
            "variable": ("dry_deposition", "wet_deposition"),
        }
        res = derived.dict()
        assert res == sol
