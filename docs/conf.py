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

html_theme = "sphinx_rtd_theme"
bibtex_bibfiles = ["references.bib"]
