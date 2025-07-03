# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import sys
from pathlib import Path

from mach import __version__

# find project
sys.path.insert(0, str(Path(__file__).parents[1]))

from sphinx_gallery.sorting import FileNameSortKey

# -- Project information -----------------------------------------------------

project = "mach"
copyright = "2025, Forest Neurotech Developers"  # noqa: A001
author = "Forest Neurotech Developers, Charles Guan"

# The full version, including alpha/beta/rc tags
release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

autosummary_generate = True
autosummary_generate_overwrite = False

# autodoc configuration
maximum_signature_line_length = 68  # wrap long signatures with each parameter on separate lines
autodoc_typehints = "signature"

# Source file suffixes
source_suffix = [".rst", ".md"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]

# Add paths for images referenced from the root README
html_static_path = ["../assets"]

# sphinx-gallery configuration
sphinx_gallery_conf = {
    # path to your example scripts
    "examples_dirs": ["../examples"],
    # path to where to save gallery generated output
    "gallery_dirs": ["examples"],
    # execute ALL examples/*.py files
    "filename_pattern": "/*",
    # specify that examples should be ordered according to filename
    "within_subsection_order": FileNameSortKey,
    # directory where function granular galleries are stored
    "backreferences_dir": "gen_modules/backreferences",
    # Modules for which function level galleries are created.  In
    # this case sphinx_gallery and numpy in a tuple of strings.
    "doc_module": ("mach"),
}

# configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": (f"https://docs.python.org/{sys.version_info.major}", None),
    "matplotlib": ("https://matplotlib.org/", None),
}
