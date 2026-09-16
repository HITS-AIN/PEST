import numpy as np

from .orientation import estimate_geometry_weighted


class FilterTruncatedGalaxies:
    """Filter out images whose galaxy is truncated.

    Uses weighted image moments to estimate the major axis length. Returns True
    (keep) when the major axis is above the threshold, False (discard)
    otherwise.

    Args:
        threshold (float): Minimum allowed major axis in units of the image size (default 0.5).
    """

    is_filter = True

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def __call__(self, sample) -> bool:
        image = np.array(sample["image"] if isinstance(sample, dict) else sample)
        img_hwc = np.moveaxis(image, 0, -1)
        stats = estimate_geometry_weighted(img_hwc.copy(), bg_subtract=0.0)
        major_axis = stats["major_axis"]
        image_size = max(img_hwc.shape[:2])
        return major_axis / image_size < self.threshold
