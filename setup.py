"""Set up the project."""
# Standard library
from typing import List
from typing import Sequence

# Third-party
from pkg_resources import parse_requirements
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
    "version": "1.0.6.dev1",
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

# Runtime dependencies: top-level and unpinned (only critical version restrictions)
with open("requirements.in") as f:
    requirements = list(map(str, parse_requirements(f.readlines())))

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
