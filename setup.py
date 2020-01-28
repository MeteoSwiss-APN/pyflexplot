#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The setup script.
"""

import subprocess
import sys
import warnings

from setuptools import setup

def read_requirements(path):
    """
    Read a requirements file line-by-line, stripping away '#'-comments.
    """
    try:
        with open(path) as fi:
            requirements = []
            for line in fi.readlines():
                line = line.strip().split("#", 1)[0]
                if line:
                    requirements.append(line)
    except IOError as e:
        warnings.warn(
            f"cannot read requirements from {setup_requirements_file} "
            f"({type(e).__name__}: {str(e)})"
        )
    else:
        return requirements

def pip_install(packages, args=None, *, ordered=False):
    """
    Install a list of packages by calling `python -m pip install ...`.

    Args:
        packages (list[str]): List of packages.

        args (list[str], optional): Additional arguments for `pip install`.
            Defaults to [].

        ordered (bool, optional): Whether to preserve the package order by
            installing them in sequence or all at once. Defaults to False.

    """
    if args is None:
        args = []
    args = [sys.executable, "-m", "pip", "install"] + args
    if not ordered:
        packages = [s for ss in packages for s in ss.split()]
        subprocess.check_call(args + packages)
    else:
        for package in packages:
            subprocess.check_call(args + package.split())

# Install setup requirements specified in file with pip
pip_install(read_requirements("requirements/setup-ordered.txt"), ordered=True)

setup()
