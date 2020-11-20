#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The setup script.
"""
# Third-party
from setuptools import find_packages
from setuptools import setup


def read_file(path):
    with open(path, "r") as f:
        return "\n".join([l.strip() for l in f.readlines()])


description_files = ["README.rst", "HISTORY.rst"]

metadata = {
    "name": "pyflexplot",
    "version": "0.13.11",
    "description": "PyFlexPlot visualizes FLEXPART particle dispersion simulations.",
    "long_description": "\n\n".join([read_file(f) for f in description_files]),
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

# Runtime dependencies
try:
    # Try to read pinned dependencies from requirements.txt
    with open("requirements.txt") as fi:
        dependencies = [line.strip() for line in fi.readlines()]
except Exception:
    # Otherwise, use specified unpinned dependencies
    dependencies = [
        # Install cartopy from github to build it against installed C libraries
        # Forked to add a pyproject.toml to specify cython as build dependency
        # Minimum requirement: Cartopy >= 0.18.0
        (
            "cartopy"
            "@git+ssh://git@github.com/MeteoSwiss-APN/cartopy.git"
            "#v0.18.0-MeteoSwiss-APN"
        ),
        # Specify shapely as dependency because cartopy depends on it
        # Install shapely from github to build it against installed C libraries
        "shapely@git+ssh://git@github.com/Toblerity/shapely.git@1.7.1",
        "click>=6.0",
        "geopy",
        "matplotlib",
        "netcdf4",
        "numpy",
        "pillow>=7.1.0",
        "pydantic",
        "pypdf2",
        "scipy",
        "toml",
        "typing-extensions",
    ]

scripts = [
    "pyflexplot=pyflexplot.cli.cli:cli",
    "crop-netcdf=srtools.crop_netcdf:main",
]

setup(
    python_requires=python,
    install_requires=dependencies,
    entry_points={"console_scripts": scripts},
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    zip_save=False,
    **metadata,
)
