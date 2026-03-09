# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from dtw_loss_functions import __version__, __author__
version = __version__
author = __author__

project = 'dtw_loss_functions'
author  = __author__
version = __version__
copyright = '2026, Alberto Zancanaro'

print("%%%%%%%%%%%%%%%%%%%%")
print(project, author, version)
print("%%%%%%%%%%%%%%%%%%%%")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'numpydoc', # Used to support NumPy-style docstrings in the documentation
    'sphinx_copybutton', # Used to add a "copy" button to code blocks in the documentation
    'sphinx_design',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
    'sphinxcontrib.bibtex', # Used to include bibliography and citations in the documentation
]

autodoc_default_options = {
    'members': True,            # Document all members
    'undoc-members': True,      # Include members without docstrings
    'private-members': False,   # Hide private methods (starting with _)
    'inherited-members': False, # Set to False to hide inherited methods
    'show-inheritance': False,   # Optional: Still shows the "Bases: ParentClass" text
}

# This option is used to remove the common prefix from the module names in the documentation, so that the documentation is cleaner and more readable. 
# In this case, we want to remove the "dtw_loss_functions." prefix from the module names in the documentation, so that the modules are displayed as "otw", "soft_dtw", etc. instead of "dtw_loss_functions.otw", "dtw_loss_functions.soft_dtw", etc.
modindex_common_prefix = ["dtw_loss_functions."]

# Options for the numpydoc extension
numpydoc_show_class_members = True
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = False

# Name of the bibliography file(s) to use with the sphinxcontrib-bibtex extension
bibtex_bibfiles = ['biblio.bib']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

