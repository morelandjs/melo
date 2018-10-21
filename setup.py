#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

import melo

with open('README.rst') as f:
    long_description = f.read()

setup(
    name='melo',
    version=melo.__version__,
    description='Margin dependent Elo ratings and predictions.',
    long_description=long_description,
    author='J. Scott Moreland',
    author_email='morelandjs@gmail.com',
    url='https://github.com/melo.git',
    license='MIT',
    packages=['melo'],
    package_data={'melo': ['nfl_scores.dat']},
    install_requires=['numpy', 'scipy >= 0.18.0'],
)
