import numpy as np
from scipy.ndimage import rotate

from .orientation import estimate_geometry_weighted


class AlignImageHorizontally:
    """Rotate a (C, H, W) image so that the galaxy major axis is horizontal."""

    def __call__(self, image: np.ndarray) -> np.ndarray:
        # Pipeline format is (C, H, W); orientation functions expect (H, W, C)
        image = image.transpose(1, 2, 0)
        stats = estimate_geometry_weighted(image)
        image = rotate(image, np.degrees(stats["pa_rad"]), reshape=True)
        return image.transpose(2, 0, 1)
