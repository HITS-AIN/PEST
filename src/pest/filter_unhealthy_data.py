import numpy as np

from .orientation import estimate_geometry_weighted


class FilterUnhealthyData:
    """Filter out unhealthy images (NaN, Inf, or all pixels the same).

    Designed to be used with HuggingFace datasets.filter(batched=False).
    Returns True if the sample is healthy (keep), False otherwise (discard).
    """

    is_filter = True

    def __call__(self, sample: dict) -> bool:
        image = np.array(sample["image"])
        if np.isnan(image).any() or np.isinf(image).any() or np.all(image == image.flat[0]):
            return False
        image = image.transpose(1, 2, 0)
        stats = estimate_geometry_weighted(image)
        return not np.isnan(stats["pa_rad"])
