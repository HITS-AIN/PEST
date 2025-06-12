import importlib.metadata

from .fits_converter import FitsConverter
from .gaia_converter import GaiaConverter

__version__ = importlib.metadata.version("astro-pest")
__all__ = [
    "GaiaConverter",
    "FitsConverter",
]
