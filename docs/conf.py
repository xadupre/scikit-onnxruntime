# Licensed under the MIT License.

# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.

import os
import sys
import shutil
import sphinx_gallery.gen_gallery
import skonnxrt
import skl2onnx
import onnxruntime
import sphinx_readable_theme


# -- Project information -----------------------------------------------------

project = 'scikit-onnxruntime'
copyright = '2019, Microsoft'
author = 'Microsoft'
version = skonnxrt.__version__
release = version

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.intersphinx',
    'sphinx.ext.imgmath',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    "sphinx.ext.autodoc",
    'sphinx.ext.githubpages',
    "sphinx.ext.napoleon",
    "sphinx_gallery.gen_gallery",
    'sphinx.ext.autodoc',
    "sphinxcontrib.blockdiag",
    'pyquickhelper.sphinxext.sphinx_epkg_extension',
]

templates_path = ['_templates']
source_suffix = ['.rst']

master_doc = 'index'
language = "en"
exclude_patterns = []
pygments_style = 'default'

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_mo"
html_static_path = ['_static']
html_theme = "readable"
html_theme_path = [sphinx_readable_theme.get_html_theme_path()]
html_logo = "logo_main.png"

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {'https://docs.python.org/': None}

# -- Options for Sphinx Gallery ----------------------------------------------

sphinx_gallery_conf = {
    'examples_dirs': 'examples',
    'gallery_dirs': 'auto_examples',
}

# -- shortcuts --

epkg_dictionary = {
    'DataFrame': 'https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html',
    'onnx': 'https://github.com/onnx/onnx',
    'ONNX': 'https://onnx.ai/',
    'onnxmltools': 'https://github.com/onnx/onnxmltools',
    'onnxruntime': 'https://xadupre.github.io/onnxruntime/index.html',
    'scikit-learn': 'https://scikit-learn.org/stable/',
    'skl2onnx': 'https://github.com/onnx/sklearn-onnx',
    'sklearn-onnx': 'https://github.com/onnx/sklearn-onnx',
}

# -- Setup actions -----------------------------------------------------------


def setup(app):
    # Placeholder to initialize the folder before
    # generating the documentation.
    return app
