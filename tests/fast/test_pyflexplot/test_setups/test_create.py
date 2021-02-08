"""Test class ``pyflexplot.setup.Setup.create``."""
# First-party
from pyflexplot.setups.setup import PlotSetup
from srutils.dict import merge_dicts

# Local
from .shared import DEFAULT_PARAMS
from .shared import DEFAULT_SETUP
from .shared import MANDATORY_RAW_DEFAULT_PARAMS
from .shared import OPTIONAL_RAW_DEFAULT_PARAMS
from .shared import RAW_DEFAULT_PARAMS


class Test_Empty:
    def test_init_dict_vs_init(self):
        assert PlotSetup(**DEFAULT_PARAMS).dict() == PlotSetup(**DEFAULT_PARAMS)

    def test_init_dict_vs_create_dict(self):
        res = PlotSetup(**DEFAULT_PARAMS).dict()
        sol = PlotSetup.create(MANDATORY_RAW_DEFAULT_PARAMS).dict()
        assert res == sol

    def test_init_dict_vs_create(self):
        res = PlotSetup(**DEFAULT_PARAMS).dict()
        sol = PlotSetup.create(MANDATORY_RAW_DEFAULT_PARAMS)
        assert res == sol

    def test_init_dict_vs_default_setup_dict(self):
        assert PlotSetup(**DEFAULT_PARAMS).dict() == DEFAULT_SETUP.dict()

    def test_init_dict_vs_default_setup(self):
        assert PlotSetup(**DEFAULT_PARAMS).dict() == DEFAULT_SETUP

    def test_init_dict_vs_default_params_dict(self):
        assert PlotSetup(**DEFAULT_PARAMS).dict() == RAW_DEFAULT_PARAMS

    def test_init_vs_init_dict(self):
        assert PlotSetup(**DEFAULT_PARAMS) == PlotSetup(**DEFAULT_PARAMS).dict()

    def test_init_vs_create(self):
        assert PlotSetup(**DEFAULT_PARAMS) == PlotSetup.create(
            MANDATORY_RAW_DEFAULT_PARAMS
        )

    def test_init_vs_default_setup(self):
        assert PlotSetup(**DEFAULT_PARAMS) == DEFAULT_SETUP

    def test_init_vs_default_params_dict(self):
        assert PlotSetup(**DEFAULT_PARAMS) == merge_dicts(
            MANDATORY_RAW_DEFAULT_PARAMS, OPTIONAL_RAW_DEFAULT_PARAMS
        )

    def test_create_vs_create_dict(self):
        res = PlotSetup.create(MANDATORY_RAW_DEFAULT_PARAMS)
        sol = PlotSetup.create(MANDATORY_RAW_DEFAULT_PARAMS).dict()
        assert res == sol

    def test_default_setup_vs_default_setup_dict(self):
        assert DEFAULT_SETUP == DEFAULT_SETUP.dict()


class Test_WildcardToNone:
    """Wildcard values passed to ``Setup.create`` turn into None.

    The wildcards can be used in a set file to explicitly specify that all
    available values of a respective dimension (etc.) shall be read from a
    NetCDF file, as es the case if the setup value is None (the default).

    """

    def test_species_id(self):
        params = merge_dicts(
            MANDATORY_RAW_DEFAULT_PARAMS,
            {"panels": {"dimensions": {"species_id": "*"}}},
        )
        setup = PlotSetup.create(params)
        assert setup.panels.collect_equal("dimensions").species_id is None

    def test_time(self):
        params = merge_dicts(
            MANDATORY_RAW_DEFAULT_PARAMS, {"panels": {"dimensions": {"time": "*"}}}
        )
        setup = PlotSetup.create(params)
        assert setup.panels.collect_equal("dimensions").time is None

    def test_level(self):
        params = merge_dicts(
            MANDATORY_RAW_DEFAULT_PARAMS, {"panels": {"dimensions": {"level": "*"}}}
        )
        setup = PlotSetup.create(params)
        assert setup.panels.collect_equal("dimensions").level is None

    def test_others(self):
        params = merge_dicts(
            MANDATORY_RAW_DEFAULT_PARAMS,
            {
                "panels": {
                    "dimensions": {
                        "nageclass": "*",
                        "noutrel": "*",
                        "numpoint": "*",
                    },
                },
            },
        )
        setup = PlotSetup.create(params)
        assert setup.panels.collect_equal("dimensions").nageclass is None
        assert setup.panels.collect_equal("dimensions").noutrel is None
        assert setup.panels.collect_equal("dimensions").numpoint is None
