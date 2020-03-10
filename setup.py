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
    "version": "0.6.8",
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

dependencies = [
    "cartopy @ git+ssh://git@github.com/MeteoSwiss-APN/cartopy.git",
    "Click >= 6.0",
    "geopy",
    "matplotlib",
    "netCDF4",
    "numpy",
    "pillow >= 6.2.0",
    "pydantic",
    "scipy",
    "toml",
]

# Build shapely from source (dependency of Cartopy)
dependencies.append("shapely @ git+ssh://git@github.com/Toblerity/shapely.git")

scripts = [
    "pyflexplot=pyflexplot.cli:cli",
    "crop-netcdf=tools.crop_netcdf:main",
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
