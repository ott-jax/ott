# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
from datetime import datetime
import os
import sys

sys.path.insert(0, os.path.abspath('../'))

import ott  # noqa: 402

# -- Project information -----------------------------------------------------
needs_sphinx = "4.0"

project = "ott"
author = "OTT authors"
copyright = f"2021-{datetime.now():%Y}, {author}"

# The full version, including alpha/beta/rc tags
release = ott.__version__
version = ott.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinxcontrib.bibtex',
    'nbsphinx',
    'IPython.sphinxext.ipython_console_highlighting',
    'sphinx_autodoc_typehints',
    'recommonmark',
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "flax": ("https://flax.readthedocs.io/en/latest/", None),
    "scikit-sparse": ("https://scikit-sparse.readthedocs.io/en/latest/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
}

master_doc = 'index'
source_suffix = ['.rst']

autosummary_generate = True
autosummary_filename_map = {
    "ott.solvers.linear.sinkhorn.sinkhorn": "sinkhorn-function"
}

autodoc_typehints = 'description'

# bibliography
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"
bibtex_default_style = "alpha"

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', '**.ipynb_checkpoints']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = 'sphinx_book_theme'
html_logo = '_static/images/logoOTT.png'
html_favicon = '_static/images/logoOTT.ico'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

nbsphinx_codecell_lexer = "ipython3"
nbsphinx_execute = 'never'
nbsphinx_prolog = r"""
{% set docname = 'docs/' + env.doc2path(env.docname, base=None) %}
.. raw:: html

    <div class="docutils container">
        <a class="reference external"
           href="https://colab.research.google.com/github/ott-jax/ott/blob/main/{{ docname|e }}">
        <img alt="Open in Colab" src="../_static/images/colab-badge.svg" width="125px">
        </a>
    </div>
"""
