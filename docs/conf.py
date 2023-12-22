import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'Pybind11-OpenKE'
copyright = '2023, LuYF-Lemon-love'
author = 'LuYF-Lemon-love'

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

sphinx_gallery_conf = {
     'examples_dirs': ['../examples', '../experiments'],
     'gallery_dirs': ['examples', 'experiments'],
}

autosummary_generate = True

autodoc_mock_imports = ['base', 'torch', 'numpy', 'tqdm', 'sklearn']

intersphinx_mapping = {
    'rtd': ('https://docs.readthedocs.io/en/stable/', None),
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
    # 'torch': ('https://pytorch.org/docs/stable/', None),
    'torch': ('https://pytorch.org/docs/1.7.0/', None),
    'numpy': ('https://numpy.org/doc/stable', None),
}

intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

epub_show_urls = 'footnote'

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'

html_static_path = ['_static']

html_logo = '_static/logo.png'

html_theme_options = {
    'style_nav_header_background': '#ED77B6',
    'logo_only': True,
}