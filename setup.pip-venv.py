"""Set up the project."""
# Standard library
from typing import List
from typing import Sequence

# Third-party
from setuptools import find_packages
from setuptools import setup


def read_present_files(paths: Sequence[str]) -> str:
    """Read the content of those files that are present."""
    contents: List[str] = []
    for path in paths:
        try:
            with open(path, "r") as f:
                contents += ["\n".join(map(str.strip, f.readlines()))]
        except FileNotFoundError:
            continue
    return "\n\n".join(contents)


description_files = [
    "README",
    "README.md",
    "README.rst",
    "HISTORY",
    "HISTORY.md",
    "HISTORY.rst",
]

metadata = {
    "name": "pyflexplot",
    "version": "1.0.4",
    "description": "PyFlexPlot visualizes FLEXPART particle dispersion simulations.",
    "long_description": read_present_files(description_files),
    "author": "Stefan Ruedisuehli",
    "author_email": "stefan.ruedisuehli@env.ethz.ch",
    "url": "https://github.com/ruestefa/pyflexplot",
    "keywords": "NWP dispersion ensemble modeling FLEXPART visualization",
    "classifiers": [
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
}

python = ">= 3.7"

# Runtime dependencies (unpinned: only critical version restrictions)
requirements = [
    # Install cartopy from github to build it against installed C libraries
    # Forked to add a pyproject.toml to specify cython as build dependency
    # Minimum requirement: Cartopy >= 0.18.0
    (
        "cartopy"
        "@git+ssh://git@github.com/MeteoSwiss-APN/cartopy.git"
        "#v0.18.0-MeteoSwiss-APN"
    ),
    # Specify shapely as dependency (although it is not used directly) because
    # cartopy depends on it, and install it from github instead of PyPI in order
    # to build (compile) it against installed C libraries
    "shapely@git+ssh://git@github.com/Toblerity/shapely.git@1.7.1",
    "click>=6.0",
    "geopy",
    "matplotlib",
    "netcdf4",
    "numpy",
    "pillow>=8.3.2",
    "pypdf2",
    "scipy",
    "toml",
    "typing-extensions",
]

scripts = [
    "pyflexplot=pyflexplot.cli.cli:cli",
    "pytrajplot=pytrajplot.cli:cli",
    "crop-netcdf=srtools.crop_netcdf:main",
]

setup(
    python_requires=python,
    install_requires=requirements,
    entry_points={"console_scripts": scripts},
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    zip_save=False,
    **metadata,
)
