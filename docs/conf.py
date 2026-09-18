"""Sphinx configuration."""

from datetime import datetime

project = "PEST"
copyright = f"{datetime.now().year}, HITS gGmbH"
author = """Bernd Doser <bernd.doser@h-its.org>,
            Sebastian T. Gomez <sebastian.trujillogomez@h-its.org>"""

extensions = [
    "myst_parser",
    "sphinxcontrib.bibtex",
    "sphinx_rtd_theme",
    # "recommonmark",
    # "sphinx.ext.autodoc",
    # "sphinx.ext.napoleon",
    # "sphinx_autodoc_typehints",
]

bibtex_bibfiles = ["references.bib"]
html_logo = "../docs/assests/logo.png"
html_theme = "sphinx_rtd_theme"
html_theme_options = {"logo_only": True, "display_version": False}
