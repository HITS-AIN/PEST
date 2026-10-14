import numpy as np
import pynbody
from pynbody.array import SimArray

# pynbody centering schemes that work with the star particles alone ("pot" needs the potential, which is not
# loaded), and "catalogue" to keep the center of the dataset.
CENTER_MODES = [None, "catalogue", "ssc", "com", "hyb"]


class RenderStars:
    """Transform step that adds a 3-color stellar image of a subhalo to the record.

    The pipeline binds the worker's dataset to this step, which must provide
    `stars(subhalo_id)` returning a snapshot of the star particles of the
    subhalo, centered on the subhalo (e.g. `PynbodyDataset`). The center is
    refined on the stars with `pynbody.analysis.center`, and the image is
    rendered with `pynbody.plot.stars.render`.

    Args:
        width (float | str): Image width, as a float in the snapshot units
            (kpc for `PynbodyDataset`) or a unit string, e.g. "50 kpc".
        resolution (int | None): Image size in pixels. If `None`, the pynbody
            default is used.
        center_mode (str | None): pynbody centering scheme to refine the center
            on the star particles: "ssc", "com" or "hyb". If `None`, the pynbody
            default ("ssc") is used. "catalogue" keeps the center of the
            dataset, e.g. the most bound particle of SUBFIND catalogues.
        column (str): Name of the added column.
        render_args (dict | None): Additional arguments for
            `pynbody.plot.stars.render`, e.g. `{"dynamic_range": 3.0}`. Dust
            (`with_dust`) is not supported, as pynbody computes it from the gas
            particles, which are not loaded.
    """

    def __init__(
        self,
        width: float | str = 50,
        resolution: int | None = None,
        center_mode: str | None = None,
        column: str = "image",
        render_args: dict | None = None,
    ):
        if center_mode not in CENTER_MODES:
            raise ValueError(
                f"center_mode must be one of {CENTER_MODES}, got {center_mode!r}. "
                "For the potential minimum, use 'catalogue' with SUBFIND catalogues."
            )
        self.width = width
        self.resolution = resolution
        self.center_mode = center_mode
        self.column = column
        self.render_args = render_args or {}
        if self.render_args.get("with_dust"):
            raise ValueError("with_dust is not supported: pynbody computes the dust from the gas, which is not loaded.")
        self.dataset = None

    def bind(self, dataset) -> None:
        if not hasattr(dataset, "stars"):
            raise TypeError(f"{type(dataset).__name__} does not provide star particles.")
        self.dataset = dataset

    def __call__(self, record: dict) -> dict:
        if self.dataset is None:
            raise RuntimeError("RenderStars must be bound to a dataset before use.")
        if "subhalo_id" not in record:
            raise KeyError("RenderStars requires the 'subhalo_id' column in the extracted records.")
        stars = self.dataset.stars(record["subhalo_id"])
        # pynbody fails with an obscure KeyError for a missing star family, e.g. in dark matter only runs.
        if len(stars) == 0:
            raise ValueError(f"Subhalo {record['subhalo_id']} has no star particles to render.")
        if self.center_mode != "catalogue":
            # The image is not rotated, so no velocity center is needed; cen_num_particles=0 keeps "ssc" from
            # computing one anyway.
            pynbody.analysis.center(
                stars, mode=self.center_mode, with_velocity=False, cen_num_particles=0, move_all=False
            )
        # The renderer multiplies the luminosity density, 10**(-0.4 mag) * rho / mass, by mass / rho, so the SPH
        # density cancels out. A placeholder saves computing it, about a third of the time of the smoothing.
        stars["rho"] = SimArray(
            np.ones(len(stars), dtype=stars["pos"].dtype), stars["mass"].units / stars["pos"].units ** 3
        )
        image = pynbody.plot.stars.render(
            stars,
            width=self.width,
            resolution=self.resolution,
            noplot=True,
            return_image=True,
            **self.render_args,
        )
        # pynbody puts the lowest y in the first row; flip to the usual top-down image layout.
        image = np.asarray(image[::-1], dtype=np.float32)
        # Channels first: (3, height, width).
        record[self.column] = np.moveaxis(image, -1, 0)
        return record
