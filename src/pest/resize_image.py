import numpy as np
from skimage.transform import resize


class ResizeImage:
    """Resize a (C, H, W) image array to the given spatial size.

    Args:
        size (list[int]): Target [height, width].
    """

    def __init__(self, size: list[int]):
        self.size = size

    def __call__(self, image: np.ndarray) -> np.ndarray:
        # image shape: (C, H, W)
        target_shape = (image.shape[0], self.size[0], self.size[1])
        resized = resize(image, target_shape, anti_aliasing=True, preserve_range=True)
        return resized.astype(image.dtype)
