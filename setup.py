#!/usr/bin/env python
# -*- coding: utf-8 -*-

import setuptools


def version():
    with open('melo/__init__.py', 'r') as f:
        for l in f:
            if l.startswith('__version__ = '):
                return l.split("'")[1]

    raise RuntimeError('unable to determine version')


def long_description():
    with open('README.rst') as f:
        return f.read()


setuptools.setup(
    name='melo',
    version=version(),
    description='Margin dependent Elo ratings and predictions.',
    long_description=long_description(),
    author='J. Scott Moreland',
    author_email='morelandjs@gmail.com',
    url='https://github.com/melo.git',
    license='MIT',
    packages=['melo'],
    package_data={'melo': ['nfl.dat']},
    install_requires=['numpy', 'scipy >= 0.18.0'],
)
