# Copyright OTT-JAX
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   https://www.apache.org/licenses/LICENSE-2.0
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

from sphinx.util import logging as sphinx_logging

import ott

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
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "myst_nb",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "jaxopt": ("https://jaxopt.github.io/stable", None),
    "lineax": ("https://docs.kidger.site/lineax/", None),
    "flax": ("https://flax.readthedocs.io/en/latest/", None),
    "optax": ("https://optax.readthedocs.io/en/latest/", None),
    "diffrax": ("https://docs.kidger.site/diffrax/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pot": ("https://pythonot.github.io/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

master_doc = "index"
source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
}
todo_include_todos = False
templates_path = ["_templates"]

autosummary_generate = True
autodoc_typehints = "description"
always_document_param_types = True

# myst-nb
myst_heading_anchors = 2
nb_execution_mode = "off"
nb_mime_priority_overrides = [("spelling", "text/plain", 0)]
myst_enable_extensions = [
    "colon_fence",
    "amsmath",
    "dollarmath",
]

# mathjax
mathjax3_config = {
    "tex": {
        "macros": {
            "prox": r"\mathrm{prox}",
            "min": r"\mathrm{min}",
            "argmin": r"\mathrm{argmin}",
        }
    }
}

# bibliography
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"
bibtex_default_style = "alpha"

# spelling
spelling_lang = "en_US"
spelling_warning = True
spelling_word_list_filename = ["spelling/technical.txt", "spelling/misc.txt"]
spelling_add_pypi_package_names = True
# flax misspelled words; `flax.linen.Module.{bind,module_paths}` is ignored in
# the `class.rst` because of indentation error that cannot be suppressed
spelling_exclude_patterns = [
    "bibliography.rst",
    "**setup.rst",
    "**lazy_init.rst",
    "**is_initializing.rst",
]
spelling_filters = [
    "enchant.tokenize.URLFilter",
    "enchant.tokenize.EmailFilter",
]

# linkcheck
linkcheck_ignore = [
    # 403 Client Error
    "https://doi.org/10.1561/2200000073",
    "https://doi.org/10.1089/cmb.2021.0446",
    "https://www.jstor.org/stable/3647580",
    "https://doi.org/10.1137/19M1301047",
    "https://doi.org/10.1137/17M1140431",
    "https://doi.org/10.1137/141000439",
    "https://doi.org/10.1002/mana.19901470121",
    "https://doi.org/10.1145/2516971.2516977",
    "https://doi.org/10.1145/2766963",
    "https://keras.io/examples/nlp/pretrained_word_embeddings/",
    "https://www.gnu.org/software/gsl/doc/html/statistics.html#weighted-samples",
    "https://doi.org/10.3390/a13090212",
    "https://doi.org/10.3390/a14050143",
    "https://www.mdpi.com/1999-4893/13/9/212",
]
linkcheck_report_timeouts_as_broken = False

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build"]

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
    "pygments_light_style": "tango",
    "pygments_dark_style": "monokai",
    "launch_buttons": {
        "colab_url": "https://colab.research.google.com",
        "binderhub_url": "https://mybinder.org",
        "notebook_interface": "jupyterlab",
    },
}


class ChexFilter(logging.Filter):
  """Filter warning related to :class:`chex.ArrayTree` missing link."""

  def filter(self, record: logging.LogRecord) -> bool:
    msg = record.getMessage()
    return "name 'ArrayTree' is not defined" not in msg


class SpellingAutosummaryFilter(logging.Filter):
  """Filter warning related to `sphinx.ext.autosummary`.

  ``spelling_warning`` must be set to ``True``.
  """

  def filter(self, record: logging.LogRecord) -> bool:
    """Filter warnings.

    Ignore misspelled words because it warns about total number of
    misspelled words, including ones coming from auto-generated files.

    Ignore everything auto-generated; note that using only "_autosummary"
    causes the warnings to appear twice, one corresponding to the docstring
    and the other to the generated `rST` file, e.g.:

    - geometry.rst:50:<autosummary>:1:
    - geometry.py:docstring of ott.geometry.geometry.Geometry:1:
    """
    msg = record.getMessage()
    return "autosummary" not in msg and "misspelled words" not in msg


sphinx_logging.getLogger("sphinx_autodoc_typehints").logger.addFilter(
    ChexFilter()
)

sphinx_logging.getLogger("sphinxcontrib.spelling.builder").logger.addFilter(
    SpellingAutosummaryFilter()
)
