import numpy as np
import pynbody
from pynbody.array import SimArray

from .render_particles import RenderParticles


class RenderStars(RenderParticles):
    """Filter step that adds a 3-color stellar image of a subhalo to the record.

    The pipeline binds the worker's dataset to this step, which must provide
    `stars(subhalo_id)` returning a snapshot of the star particles of the
    subhalo, centered on the subhalo (e.g. `PynbodyDataset`). The center is
    refined on the stars with `pynbody.analysis.center`, and the image is
    rendered with `pynbody.plot.stars.render`.

    The step returns the record with the image, or `None` for a subhalo that
    cannot be rendered, which the pipeline drops like a record rejected by a
    filter: one with fewer than `min_particles` stars, one that cannot be
    oriented (see `orientation`), and one whose rendering fails, which is
    reported with a warning.

    Args:
        width (float | str): Image width, as a float in the snapshot units
            (kpc for `PynbodyDataset`) or a unit string, e.g. "50 kpc".
        resolution (int | None): Image size in pixels. If `None`, the pynbody
            default is used.
        center_mode (str | None): pynbody centering scheme to refine the center
            on the star particles: "ssc", "com" or "hyb". If `None`, the pynbody
            default ("ssc") is used. "catalogue" keeps the center of the
            dataset, e.g. the most bound particle of SUBFIND catalogues.
        orientation (str | None): Rotate the galaxy before rendering with
            `pynbody.analysis.faceon` or `pynbody.analysis.sideon`, which orient
            the angular momentum of the stars within `disk_size` along the line
            of sight or perpendicular to it. If `None`, the image is not rotated.
            The velocities of the stars are loaded for this. A galaxy whose
            stars within `disk_size` have no angular momentum, e.g. with no
            star that close to the center, cannot be oriented and is dropped.
        disk_size (float | str): Radius around the center that determines the
            angular momentum for `orientation`, as a float in the snapshot units
            or a unit string.
        column (str): Name of the added column.
        render_args (dict | None): Additional arguments for
            `pynbody.plot.stars.render`, e.g. `{"dynamic_range": 3.0}`. Dust
            (`with_dust`) is not supported, as pynbody computes it from the gas
            particles, which are not loaded.
        min_particles (int): Drop subhalos with fewer star particles. The
            default is the number of neighbours of pynbody's SPH smoothing
            (32), which fails with fewer particles.
    """

    component = "stars"

    def __init__(
        self,
        width: float | str = 50,
        resolution: int | None = None,
        center_mode: str | None = None,
        orientation: str | None = None,
        disk_size: float | str = "5 kpc",
        column: str = "image",
        render_args: dict | None = None,
        min_particles: int = 32,
    ):
        super().__init__(width, resolution, center_mode, orientation, disk_size, column, render_args, min_particles)
        if self.render_args.get("with_dust"):
            raise ValueError("with_dust is not supported: pynbody computes the dust from the gas, which is not loaded.")

    def _particles(self, subhalo_id):
        # Without an orientation, the dataset only has to provide `stars(subhalo_id)`.
        return self.dataset.stars(subhalo_id, **({"velocities": True} if self.orientation else {}))

    def _image(self, stars) -> np.ndarray:
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
        return np.moveaxis(image, -1, 0)
