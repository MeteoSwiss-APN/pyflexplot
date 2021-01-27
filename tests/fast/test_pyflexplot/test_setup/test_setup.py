"""Test module ``pyflexplot.setup``."""
# First-party
from pyflexplot.setup import Setup
from pyflexplot.setup import SetupGroup
from srutils.testing import check_is_sub_element

# Local
from .shared import DEFAULT_SETUP


class Test_Setup_Create:
    def test_some_dimensions(self):
        params = {
            "infile": "dummy.nc",
            "outfile": "dummy.png",
            "model": {
                "name": "COSMO-1",
                "ens_member_id": (0, 1, 5, 10, 15, 20),
            },
            "core": {
                "integrate": False,
                "ens_variable": "mean",
                "dimensions": {"time": 10, "level": 1},
                "input_variable": "concentration",
            },
        }
        setup = Setup.create(params)
        res = setup.dict()
        check_is_sub_element(obj_super=res, obj_sub=params)


class Test_Setup_Decompress:
    @property
    def setup(self):
        return DEFAULT_SETUP.derive(
            {
                "core": {
                    "input_variable": "deposition",
                    "dimensions": {
                        "deposition_type": ["dry", "wet"],
                        "nageclass": (0,),
                        "noutrel": (0,),
                        "numpoint": (0,),
                        "species_id": [1, 2],
                        "time": [1, 2, 3],
                    },
                },
            },
        )

    def test_full(self):
        """Decompress all params."""
        setups = self.setup.decompress()
        assert len(setups) == 12
        res = {
            (
                s.core.dimensions.deposition_type,
                s.core.dimensions.species_id,
                s.core.dimensions.time,
            )
            for s in setups
        }
        sol = {
            ("dry", 1, 1),
            ("dry", 1, 2),
            ("dry", 1, 3),
            ("dry", 2, 1),
            ("dry", 2, 2),
            ("dry", 2, 3),
            ("wet", 1, 1),
            ("wet", 1, 2),
            ("wet", 1, 3),
            ("wet", 2, 1),
            ("wet", 2, 2),
            ("wet", 2, 3),
        }
        assert res == sol

    def test_full_with_partially(self):
        """Decompress all params."""
        setups = self.setup.decompress_partially(None)
        assert len(setups) == 12
        assert isinstance(setups, SetupGroup)
        assert all(isinstance(setup, Setup) for setup in setups)
        res = {
            (
                s.core.dimensions.deposition_type,
                s.core.dimensions.species_id,
                s.core.dimensions.time,
            )
            for s in setups
        }
        sol = {
            ("dry", 1, 1),
            ("dry", 1, 2),
            ("dry", 1, 3),
            ("dry", 2, 1),
            ("dry", 2, 2),
            ("dry", 2, 3),
            ("wet", 1, 1),
            ("wet", 1, 2),
            ("wet", 1, 3),
            ("wet", 2, 1),
            ("wet", 2, 2),
            ("wet", 2, 3),
        }
        assert res == sol

    def test_partially_one(self):
        """Decompress only one select parameter."""
        setups = self.setup.decompress_partially(["dimensions.species_id"])
        assert isinstance(setups, SetupGroup)
        assert all(isinstance(setup, Setup) for setup in setups)
        assert len(setups) == 2
        res = {
            (
                s.core.dimensions.deposition_type,
                s.core.dimensions.species_id,
                s.core.dimensions.time,
            )
            for s in setups
        }
        sol = {(("dry", "wet"), 1, (1, 2, 3)), (("dry", "wet"), 2, (1, 2, 3))}
        assert res == sol

    def test_select_two(self):
        """Decompress two select parameters."""
        setups = self.setup.decompress_partially(
            ["dimensions.time", "dimensions.deposition_type"]
        )
        assert len(setups) == 6
        assert isinstance(setups, SetupGroup)
        assert all(isinstance(setup, Setup) for setup in setups)
        res = {
            (
                s.core.dimensions.deposition_type,
                s.core.dimensions.species_id,
                s.core.dimensions.time,
            )
            for s in setups
        }
        sol = {
            ("dry", (1, 2), 1),
            ("dry", (1, 2), 2),
            ("dry", (1, 2), 3),
            ("wet", (1, 2), 1),
            ("wet", (1, 2), 2),
            ("wet", (1, 2), 3),
        }
        assert res == sol
