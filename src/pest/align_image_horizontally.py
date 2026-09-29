import numpy as np
from scipy.ndimage import rotate

from .orientation import estimate_geometry_weighted


class AlignImageHorizontally:
    """Rotate a (C, H, W) image so that the galaxy major axis is horizontal."""

    def __call__(self, image: np.ndarray) -> np.ndarray:
        stats = estimate_geometry_weighted(image)
        return rotate(image, np.degrees(stats["pa_rad"]), axes=(2, 1), reshape=True)
