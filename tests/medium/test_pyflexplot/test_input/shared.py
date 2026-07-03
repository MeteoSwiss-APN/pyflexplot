"""Utilities for tests for module ``pyflexplot.input``."""
# Standard library
import shutil
from pathlib import Path

# Third-party
import pytest  # type: ignore

DATA_ROOT = Path(__file__).parents[3] / "data/pyflexplot"


@pytest.fixture
def datadir_flexpart_artificial(tmpdir, request):
    """Return path to temporary directory with artificial flexpart data."""
    return prepare_datadir("flexpart/artificial", tmpdir, request)


@pytest.fixture
def datadir_flexpart_reduced(tmpdir, request):
    """Return path to temporary directory with reduced flexpart data."""
    return prepare_datadir("flexpart/reduced", tmpdir, request)


def prepare_datadir(subdir, tmpdir, request):
    data_dir = DATA_ROOT / subdir
    if data_dir.is_dir():
        shutil.copytree(data_dir, str(tmpdir), dirs_exist_ok=True)
    return tmpdir
