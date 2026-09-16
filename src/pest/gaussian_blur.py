import numpy as np
from scipy.ndimage import gaussian_filter


class GaussianBlur:
    """Apply a Gaussian blur to each channel of an image.

    Operates on (C, H, W) arrays.

    Args:
        kernel_size (int): Approximate kernel size. The standard deviation is
            derived as ``(kernel_size - 1) / 6``, matching the common
            OpenCV convention where ~99.7 % of the kernel weight falls within
            ``kernel_size`` pixels (default 3).
        sigma (float | None): Standard deviation for the Gaussian kernel.
            When provided, overrides ``kernel_size`` (default None).
    """

    def __init__(self, kernel_size: int = 3, sigma: float | None = None):
        if sigma is not None:
            self.sigma = float(sigma)
        else:
            self.sigma = max((kernel_size - 1) / 6.0, 1e-6)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        # image shape: (C, H, W)
        blurred = np.stack(
            [gaussian_filter(channel, sigma=self.sigma) for channel in image],
            axis=0,
        )
        return blurred.astype(image.dtype)
