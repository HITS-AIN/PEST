"""Reader for the Illustris SKIRT dataset in FITS format."""

from pathlib import Path

import numpy as np
from astropy.io import fits

from pest.create_normalized_rgb_colors import CreateNormalizedRGBColors


class IllustrisSkirtReader:
    """Reader for the Illustris SKIRT dataset in FITS format.

    Args:
        path (str): Path to the directory containing FITS files.
        count_only (bool): If True, only counts the number of valid FITS files without yielding image data. Default is False.
    """

    def __init__(
        self,
        path: str,
        count_only: bool = False,
    ):
        self.path = Path(path)
        self.count_only = count_only

        self.normalize_rgb = CreateNormalizedRGBColors(
            stretch=0.9,
            range=5,
            lower_limit=0.001,
            channel_combinations=[[2, 3], [1, 0], [0]],
            scalers=[0.7, 0.5, 1.3],
        )

    def extract(self):
        """Iterate over all FITS files in the path and yield normalized image data.

        If count_only is True, yields the total count of valid files instead of image data.

        Yields:
            int: Total count of valid FITS files (if count_only=True).
            np.ndarray: Normalized RGB image array of shape (3, H, W) with values in [0, 1].
        """

        if self.count_only:
            yield sum(1 for _ in self.path.rglob("*.fits"))
            return

        for fits_file in sorted(self.path.rglob("*.fits")):
            data = fits.getdata(fits_file, 0)
            data = np.array(data, dtype=np.float32)
            data = self.normalize_rgb(data)

            # Skip images that contain NaN, Inf, or are constant (all values are the same).
            if np.isnan(data).any() or np.isinf(data).any() or np.all(data == data.flat[0]):
                continue

            yield data
