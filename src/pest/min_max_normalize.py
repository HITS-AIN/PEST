import numpy as np


class MinMaxNormalize:
    """Scale a (C, H, W) image array to a given feature range using min/max normalization.

    Args:
        feature_range (list[float]): Target [min, max] range. Defaults to [0, 1].
    """

    def __init__(self, feature_range: list[float] | None = None):
        if feature_range is None:
            feature_range = [0, 1]
        self.min_val = feature_range[0]
        self.max_val = feature_range[1]

    def __call__(self, image: np.ndarray) -> np.ndarray:
        img_min = image.min()
        img_max = image.max()
        if img_max == img_min:
            return np.full_like(image, self.min_val)
        scale = (self.max_val - self.min_val) / (img_max - img_min)
        return (image - img_min) * scale + self.min_val
