import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))

import pybind11_ke as package

pkg_name = package.__name__
pkg_file = package.__file__
pkg_version = str(package.__version__)
pkg_location = os.path.dirname(os.path.dirname(pkg_file))

project = 'Pybind11-OpenKE'
author = 'LuYF-Lemon-love'
copyright = f'2023, {author}'

github_user = author
github_repo = 'pybind11-OpenKE'
github_version = 'pybind11-OpenKE-PyTorch'

github_url = f'https://github.com/{github_user}/{github_repo}/'
gh_page_url = f'https://pybind11-openke.readthedocs.io/zh-cn/latest/'
# gh_page_url = f'https://{github_user}.github.io/{github_repo}/'

html_baseurl = gh_page_url
html_context = {
    'display_github': True,
    'github_user': github_user,
    'github_repo': github_repo,
    'github_version': github_version,
    "conf_py_path": "/docs/",
}

html_theme_options = {
    'style_external_links': False,
    
    'github_url': github_url,

    'doc_items': {
        'Pybind11-OpenKE': 'https://pybind11-openke.readthedocs.io/zh-cn/latest/',
        'AD-KGE': 'https://github.com/LuYF-Lemon-love/AD-KGE',
    },

    'logo': '_static/logo.png',
    'logo_dark': '',
    'logo_icon': '',
}

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

html_theme = 'trojanzoo_sphinx_theme'

html_static_path = ['_static']