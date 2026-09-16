import numpy as np

from .orientation import estimate_geometry_weighted


class FilterInclinationAngle:
    """Filter out images whose galaxy inclination exceeds a maximum angle.

    Uses weighted image moments to estimate the inclination. Returns True
    (keep) when the inclination is within the allowed range, False (discard)
    otherwise.

    Args:
        max_inclination (float): Maximum allowed inclination in degrees (default 90).
    """

    is_filter = True

    def __init__(self, max_inclination: float = 90.0):
        self.max_inclination = max_inclination

    def __call__(self, sample: dict) -> bool:
        image = np.array(sample["image"])
        # Pipeline format is (C, H, W); estimate_geometry_weighted expects (H, W, C)
        img_hwc = np.moveaxis(image, 0, -1)
        stats = estimate_geometry_weighted(img_hwc.copy())
        return stats["inclination_deg"] <= self.max_inclination
