# -*- coding: utf-8 -*-
"""
Tests for class ``pyflexplot.setup.Setup.create``.
"""
# First-party
from pyflexplot.setup import Setup

# Local
from .shared import DEFAULT_PARAMS
from .shared import DEFAULT_SETUP
from .shared import DUMMY_PARAMS


class Test_Empty:
    def test_init_dict_vs_init(self):
        assert Setup(**DUMMY_PARAMS).dict() == Setup(**DUMMY_PARAMS)

    def test_init_dict_vs_create_dict(self):
        assert Setup(**DUMMY_PARAMS).dict() == Setup.create({**DUMMY_PARAMS}).dict()

    def test_init_dict_vs_create(self):
        assert Setup(**DUMMY_PARAMS).dict() == Setup.create({**DUMMY_PARAMS})

    def test_init_dict_vs_default_setup_dict(self):
        assert Setup(**DUMMY_PARAMS).dict() == DEFAULT_SETUP.dict()

    def test_init_dict_vs_default_setup(self):
        assert Setup(**DUMMY_PARAMS).dict() == DEFAULT_SETUP

    def test_init_dict_vs_default_params_dict(self):
        assert Setup(**DUMMY_PARAMS).dict() == {**DUMMY_PARAMS, **DEFAULT_PARAMS}

    def test_init_vs_init_dict(self):
        assert Setup(**DUMMY_PARAMS) == Setup(**DUMMY_PARAMS).dict()

    def test_init_vs_create(self):
        assert Setup(**DUMMY_PARAMS) == Setup.create({**DUMMY_PARAMS})

    def test_init_vs_default_setup(self):
        assert Setup(**DUMMY_PARAMS) == DEFAULT_SETUP

    def test_init_vs_default_params_dict(self):
        assert Setup(**DUMMY_PARAMS) == {**DUMMY_PARAMS, **DEFAULT_PARAMS}

    def test_create_vs_create_dict(self):
        assert Setup.create({**DUMMY_PARAMS}) == Setup.create({**DUMMY_PARAMS}).dict()

    def test_default_setup_vs_default_setup_dict(self):
        assert DEFAULT_SETUP == DEFAULT_SETUP.dict()


class Test_WildcardToNone:
    """Wildcard values passed to ``Setup.create`` turn into None.

    The wildcards can be used in a set file to explicitly specify that all
    available values of a respective dimension (etc.) shall be read from a
    NetCDF file, as es the case if the setup value is None (the default).

    """

    def test_species_id(self):
        params = {**DUMMY_PARAMS, "dimensions": {"species_id": "*"}}
        setup = Setup.create(params)
        assert setup.core.dimensions.species_id is None

    def test_time(self):
        params = {**DUMMY_PARAMS, "dimensions": {"time": "*"}}
        setup = Setup.create(params)
        assert setup.core.dimensions.time is None

    def test_level(self):
        params = {**DUMMY_PARAMS, "dimensions": {"level": "*"}}
        setup = Setup.create(params)
        assert setup.core.dimensions.level is None

    def test_others(self):
        params = {
            **DUMMY_PARAMS,
            "dimensions": {"nageclass": "*", "noutrel": "*", "numpoint": "*"},
        }
        setup = Setup.create(params)
        assert setup.core.dimensions.nageclass is None
        assert setup.core.dimensions.noutrel is None
        assert setup.core.dimensions.numpoint is None
