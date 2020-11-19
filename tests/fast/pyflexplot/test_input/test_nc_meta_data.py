"""Tests for module ``pyflexplot.input.nc_meta_data``."""
# Third-party
import pytest

# First-party
from pyflexplot.input.nc_meta_data import derive_species_ids


@pytest.mark.parametrize(
    "vars, ids",
    [
        (["spec001"], (1,)),
        (["spec001_mr"], (1,)),
        (["DD_spec001"], (1,)),
        (["WD_spec001"], (1,)),
        (["spec000", "spec002_mr", "DD_spec004", "WD_spec006"], (0, 2, 4, 6)),
    ],
)
def test_derive_species_ids(vars, ids):
    res = derive_species_ids(vars)
    assert res == ids
