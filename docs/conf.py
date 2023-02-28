# Copyright OTT-JAX
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
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
import logging
from datetime import datetime

import ott
from sphinx.util import logging as sphinx_logging

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
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "myst_nb",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx_autodoc_typehints",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "flax": ("https://flax.readthedocs.io/en/latest/", None),
    "scikit-sparse": ("https://scikit-sparse.readthedocs.io/en/latest/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "pot": ("https://pythonot.github.io/", None),
    "jaxopt": ("https://jaxopt.github.io/stable", None),
    "optax": ("https://optax.readthedocs.io/en/latest/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

master_doc = "index"
source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
}
todo_include_todos = False

autosummary_generate = True
autodoc_typehints = "description"

# myst-nb
myst_heading_anchors = 2
nb_execution_mode = "off"
nb_mime_priority_overrides = [("spelling", "text/plain", 0)]
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "dollarmath",
]

# bibliography
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"
bibtex_default_style = "alpha"

# spelling
spelling_lang = "en_US"
spelling_warning = True
spelling_word_list_filename = ["spelling/technical.txt", "spelling/misc.txt"]
spelling_add_pypi_package_names = True
spelling_exclude_patterns = ["references.rst"]
spelling_filters = [
    "enchant.tokenize.URLFilter",
    "enchant.tokenize.EmailFilter",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = "sphinx_book_theme"
html_logo = "_static/images/logoOTT.png"
html_favicon = "_static/images/logoOTT.ico"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_theme_options = {
    "repository_url": "https://github.com/ott-jax/ott",
    "repository_branch": "main",
    "path_to_docs": "docs/",
    "use_repository_button": True,
    "use_fullscreen_button": False,
    "logo_only": True,
    "launch_buttons": {
        "colab_url": "https://colab.research.google.com",
        "binderhub_url": "https://mybinder.org",
        "notebook_interface": "jupyterlab",
    },
}


class AutodocExternalFilter(logging.Filter):

  def filter(self, record: logging.LogRecord) -> bool:
    msg = record.getMessage()
    return not (
        "name 'ArrayTree' is not defined" in msg or
        "PositiveDense.kernel_init" in msg
    )


class SpellingFilter(logging.Filter):

  def filter(self, record: logging.LogRecord) -> bool:
    msg = record.getMessage()
    return "_autosummary" not in msg


sphinx_logging.getLogger("sphinx_autodoc_typehints").logger.addFilter(
    AutodocExternalFilter()
)
sphinx_logging.getLogger("sphinxcontrib.spelling.builder").logger.addFilter(
    SpellingFilter()
)
