from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits

MORPHS_FILENAME_TEMPLATE = "morphs_{band}.hdf5"


class IllustrisSkirtDataset:
    """PyTorch Dataset for the Illustris SKIRT dataset in FITS format.

    Args:
        path (str): Path to the directory containing FITS files.
        columns (list[str | tuple[str, str] | list[str]] | None): List of columns
            to extract from the FITS files. Besides the plain string columns
            "image", "simulation", "snapshot" and "subhalo_id", a `(field, band)`
            tuple (or two-element list, e.g. as parsed from YAML) extracts
            `field` from the morphology catalog of the given filter `band`.

    A `(field, band)` column is read from the morphology catalog
    `morphs_{band}.hdf5` located in the snapshot directory of the FITS file,
    e.g. `TNG50/sdss/snapshot_095/morphs_r.hdf5` for
    `TNG50/sdss/snapshot_095/data/broadband_90.fits`.
    """

    def __init__(
        self,
        path: str,
        columns: list[str | tuple[str, str]] | None = None,
    ) -> None:
        self.path = Path(path)
        if self.path.is_file():
            self.files = [self.path]
        else:
            self.files = sorted(self.path.rglob("*.fits"))
        self.columns = columns

        self._hdf5_field_cache: dict[tuple[Path, str], dict[int, np.float32]] = {}

    def __len__(self) -> int:
        return len(self.files)

    def _hdf5_field(self, fits_file: Path, subhalo_id: np.int32, field: str, band: str) -> np.float32:
        morphs_path = fits_file.parents[1] / MORPHS_FILENAME_TEMPLATE.format(band=band)
        cache_key = (morphs_path, field)
        if cache_key not in self._hdf5_field_cache:
            with h5py.File(morphs_path, "r") as f:
                self._hdf5_field_cache[cache_key] = dict(zip(f["subfind_id"][:], f[field][:]))
        return np.float32(self._hdf5_field_cache[cache_key][subhalo_id])

    def __getitem__(self, index: int) -> dict:
        fits_file = self.files[index]

        data: dict = {}
        if self.columns is None or "image" in self.columns:
            image = fits.getdata(fits_file, 0)
            data["image"] = np.array(image, dtype=np.float32)

        if self.columns:
            splits = fits_file.parts
            subhalo_id = None
            if "subhalo_id" in self.columns or any(isinstance(col, list | tuple) for col in self.columns):
                subhalo_id = np.int32(splits[-1][: -len(".fits")].split("_")[1])
            for col in self.columns:
                if col == "simulation":
                    data["simulation"] = splits[-5]
                elif col == "snapshot":
                    data["snapshot"] = np.int32(splits[-3].split("_")[1])
                elif col == "subhalo_id":
                    data["subhalo_id"] = subhalo_id
                elif isinstance(col, list | tuple):
                    field, band = col
                    data[tuple(col)] = self._hdf5_field(fits_file, subhalo_id, field, band)

        return data
