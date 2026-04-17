import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

# -- Project information -----------------------------------------------------
project = "Classical_Laminate_Theory"
copyright = "2026, Senior Software Engineer"
author = "Senior Software Engineer"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Support for Google/NumPy style docstrings
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = "renku"

# If the renku theme is not found in the env, we fallback to sphinx_rtd_theme
try:
    import renku_sphinx_theme

    html_theme = "renku"
except ImportError:
    html_theme = "sphinx_rtd_theme"

html_static_path = ["_static"]
