import numpy as np


class AddCircularMask:
    """Zero out pixels outside a circle inscribed in the image.

    Operates on (C, H, W) arrays. The circle is centred at the image centre
    with a radius equal to ``radius_fraction`` times half the smaller image
    dimension, so by default it exactly inscribes the image.

    Args:
        radius_fraction (float): Fraction of the inscribed radius to use
            (default 1.0).
    """

    def __init__(self, radius_fraction: float = 1.0):
        self.radius_fraction = radius_fraction

    def __call__(self, image: np.ndarray) -> np.ndarray:
        # image shape: (C, H, W)
        _, H, W = image.shape
        cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
        radius = self.radius_fraction * min(H, W) / 2.0

        ys = np.arange(H)
        xs = np.arange(W)
        dist2 = (ys[:, None] - cy) ** 2 + (xs[None, :] - cx) ** 2
        mask = dist2 <= radius**2  # (H, W) bool

        return np.where(mask[None, :, :], image, 0).astype(image.dtype)
