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
import os
import sys
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = "Pybind11-OpenKE"
copyright = "2023, LuYF-Lemon-love"
author = "LuYF-Lemon-love"


# -- General configuration ---------------------------------------------------
# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_autopackagesummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    'sphinx_gallery.gen_gallery',
    'sphinx_copybutton',
]

sphinx_gallery_conf = {
     'examples_dirs': ['../examples', '../experiments'],   # path to your example scripts
     'gallery_dirs': ['auto_examples', 'auto_experiments'],  # path to where to save gallery generated output
     #'download_all_examples': False,
     #'line_numbers': True,
}

autosummary_generate = True

autodoc_mock_imports = ["base", "torch", "numpy", "tqdm", "sklearn"]

intersphinx_mapping = {
    "rtd": ("https://docs.readthedocs.io/en/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
    # "torch": ("https://pytorch.org/docs/stable/", None),
    "torch": ("https://pytorch.org/docs/1.7.0/", None),
    'numpy': ('https://numpy.org/doc/stable', None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for EPUB output
epub_show_urls = "footnote"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_logo = "_static/logo.png"

html_theme_options = {
    'style_nav_header_background': '#ED77B6',
    'logo_only': True,
}