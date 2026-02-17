# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
#
# Note: MkDocs is the canonical docs build/deploy path for this project
# (see docs/mkdocs.yml and .github/workflows/docs.yml). This Sphinx
# configuration is retained for local/legacy workflows.

import sys
from pathlib import Path

# Resolve the project root relative to this file's location for robustness
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root))

# Import version from package
from nonconform import __version__

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Nonconform"
copyright = "2025, Oliver Hennhöfer"
author = "Oliver Hennhöfer"
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "myst_parser",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",  # For linking to other projects' documentation
    "sphinx.ext.todo",  # For TODO items
]

# Make section labels unique across files
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 2

# Intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# Napoleon settings for better docstring parsing
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None
napoleon_attr_annotations = True

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Furo theme options
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#2B5B84",  # Dark blue
        "color-brand-content": "#2B5B84",
        "color-admonition-background": "#F0F4F8",
        "color-sidebar-background": "#F8F9FA",
        "color-sidebar-link-text": "#2B5B84",
        "color-sidebar-item-background--current": "#E8EEF4",
        "color-sidebar-item-text--current": "#2B5B84",
    },
    "dark_css_variables": {
        "color-brand-primary": "#4A9CD5",  # Lighter blue for dark mode
        "color-brand-content": "#4A9CD5",
        "color-admonition-background": "#1E293B",
        "color-sidebar-background": "#1A1F2E",
        "color-sidebar-link-text": "#4A9CD5",
        "color-sidebar-item-background--current": "#2B3A4F",
        "color-sidebar-item-text--current": "#4A9CD5",
        "color-background-primary": "#0F172A",  # Dark background
        "color-background-secondary": "#1E293B",
        "color-foreground-primary": "#E2E8F0",  # Light text
        "color-foreground-secondary": "#94A3B8",
    },
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
}

# -- Options for sphinx-autoapi ------------------------------------------------
autoapi_type = "python"
autoapi_dirs = ["../../nonconform"]  # Point specifically to the nonconform package
autoapi_root = "api"

# AutoAPI options
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "imported-members",
    "noindex",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

# MyST parser configuration
myst_enable_extensions = [
    "dollarmath",  # Enable $...$ and $$...$$ for math
    "amsmath",  # Enable AMS math environments
    "deflist",  # Definition lists
    "fieldlist",  # Field lists
    "tasklist",  # Task lists
]

# MathJax configuration
mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
    }
}

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "noindex": True,  # Add this to prevent duplicate object descriptions
}

autoapi_generate_api_docs = True
autoapi_add_toctree_entry = False
autoapi_root = "api"
autoapi_keep_files = False

# Suppress warnings for cross-reference ambiguity
suppress_warnings = ["ref.python"]


# Configure AutoAPI member skipping.
def autoapi_skip_member(app, what, name, obj, skip, options):
    """Apply custom member skipping rules (none at the moment)."""
    return skip


# Enable todo items
todo_include_todos = True


# Register the autoapi skip function
def setup(app):
    """Set up the Sphinx application."""
    app.connect("autoapi-skip-member", autoapi_skip_member)


master_doc = "index"
