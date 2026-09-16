import numpy as np

from .orientation import estimate_geometry_weighted


class Crop:
    """Crop a square region around the galaxy centroid detected via weighted moments.

        Operates on (C, H, W) arrays. The half-size of the crop is given by ``relative_size``
        times the estimated major axis length, so by default it is half the major axis.


    Args:
        relative_size (float): Half-size of the crop relative to the estimated major axis (default 0.5).
    """

    def __init__(self, relative_size: float = 0.5):
        self.relative_size = relative_size

    def crop_(self, img, center, half_size):
        """Crop a square region around the center."""
        x_c, y_c = center
        H, W = img.shape[:2]

        col_min = int(max(0, x_c - half_size))
        col_max = int(min(W, x_c + half_size))
        row_min = int(max(0, y_c - half_size))
        row_max = int(min(H, y_c + half_size))

        return img[row_min:row_max, col_min:col_max]

    def __call__(self, image: np.ndarray) -> np.ndarray:
        # Pipeline format is (C, H, W); orientation functions expect (H, W, C)
        img_hwc = np.moveaxis(image, 0, -1)
        stats = estimate_geometry_weighted(img_hwc.copy())
        half_size = stats["major_axis"] / self.relative_size
        cropped_hwc = self.crop_(img_hwc, stats["centroid"], half_size)
        return np.moveaxis(cropped_hwc, -1, 0).astype(image.dtype)
