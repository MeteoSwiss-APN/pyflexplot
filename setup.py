#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages
from glob import glob
from os.path import basename

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=6.0', ]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="Stefan Ruedisuehli",
    author_email='stefan.ruedisuehli@env.ethz.ch',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Python FLEXPART Plotting contains Python scripts to plot FLEXPART output",
    entry_points={
        'console_scripts': [
            'pyflexplot=pyflexplot.cli:main',
        ],
    },
    install_requires=requirements,
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='pyflexplot',
    name='pyflexplot',
    packages=find_packages('src'),
    package_dir={'': "src"},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/ruestefa/pyflexplot',
    version='0.1.0',
    zip_safe=False,
)
