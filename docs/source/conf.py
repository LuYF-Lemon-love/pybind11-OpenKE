# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'Pybind11-OpenKE'
copyright = '2023, LuYF-Lemon-love'
author = 'LuYF-Lemon-love'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx_autopackagesummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx_gallery.gen_gallery',
    'sphinx_copybutton',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
    # 'torch': ('https://pytorch.org/docs/stable/', None),
    'torch': ('https://pytorch.org/docs/1.7.0/', None),
    'numpy': ('https://numpy.org/doc/stable', None),
}

sphinx_gallery_conf = {
     'examples_dirs': ['../../examples', '../../experiments'],   # path to your example scripts
     'gallery_dirs': ['auto_examples', 'auto_experiments'],  # path to where to save gallery generated output
     #'download_all_examples': False,
     #'line_numbers': True,
}

intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
