"""Sphinx configuration."""

from datetime import UTC, datetime

project = "PEST"
copyright = f"{datetime.now(tz=UTC).year}, HITS gGmbH"
author = """Bernd Doser <bernd.doser@h-its.org>,
            Sebastian T. Gomez <sebastian.trujillogomez@h-its.org>"""

extensions = [
    "myst_parser",
    "sphinxcontrib.bibtex",
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

bibtex_bibfiles = ["references.bib"]
html_logo = "assets/logo.png"
html_favicon = "assets/logo.png"
html_theme = "sphinx_rtd_theme"
html_theme_options = {"logo_only": True}

autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
