from pathlib import Path

import numpy as np
from astropy.io import fits


class FitsDataset:
    """PyTorch Dataset for the Illustris SKIRT dataset in FITS format.

    Args:
        path (str): Path to the directory containing FITS files.
        columns (list[str] | None): List of columns to extract from the FITS files.
    """

    def __init__(
        self,
        path: str,
        columns: list[str] | None = None,
    ) -> None:
        self.path = Path(path)
        if self.path.is_file():
            self.files = [self.path]
        else:
            self.files = sorted(self.path.rglob("*.fits"))
        self.columns = columns

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict:
        fits_file = self.files[index]
        image = fits.getdata(fits_file, 0)
        image = np.array(image, dtype=np.float32)

        data: dict = {"image": image}
        if self.columns:
            splits = fits_file.parts
            for col in self.columns:
                if col == "simulation":
                    data["simulation"] = splits[-5]
                elif col == "snapshot":
                    data["snapshot"] = np.int32(splits[-3].split("_")[1])
                elif col == "subhalo_id":
                    data["subhalo_id"] = np.int32(splits[-1][: -len(".fits")].split("_")[1])

        return data
