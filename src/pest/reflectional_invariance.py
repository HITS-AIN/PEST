import numpy as np


class ReflectionalInvariance:
    """Ensure the left half of the image has more flux than the right half.

    Operates on (C, H, W) arrays. Flips horizontally if the right half has
    more total flux than the left half.
    """

    def __call__(self, image: np.ndarray) -> np.ndarray:
        # image shape: (C, H, W)
        mid = image.shape[2] // 2
        left_mass = image[:, :, :mid].sum()
        right_mass = image[:, :, mid:].sum()
        if right_mass > left_mass:
            image = np.flip(image, axis=2)
        return image
