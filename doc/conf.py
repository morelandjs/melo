# -*- coding: utf-8 -*-

import os
import sys

import melo


sys.path.insert(0, os.path.abspath(os.pardir))

project = 'melo'
version = release = '1.0.0'
author = 'J. Scott Moreland'
copyright = '2018 J. Scott Moreland'

source_suffix = '.rst'
master_doc = 'index'

templates_path = ['_templates']
html_static_path = ['_static']
exclude_patterns = ['_build']

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
]

default_role = 'math'

html_theme = 'sphinx_rtd_theme'
html_context = dict(show_source=False)
