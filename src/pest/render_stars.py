import numpy as np
import pynbody
from pynbody.array import SimArray

from .render_particles import RenderParticles


class RenderStars(RenderParticles):
    """Filter step that adds a 3-color stellar image of a subhalo to the record.

    The pipeline binds the worker's dataset to this step, which must provide
    `stars(subhalo_id)` returning a snapshot of the star particles of the
    subhalo, centered on the subhalo (e.g. `PynbodyDataset`), and
    `particles(subhalo_id, component, fields)` if the width or the orientation
    refer to the gas. The image is rendered with `pynbody.plot.stars.render`.

    With the defaults, the image has the same view as the one of `RenderGas`:
    both are centered on the center of the dataset, and both are oriented by
    the stars. A width relative to a half-mass radius, e.g.
    "10 stars_half_mass_radius", is the same for both as well.

    The step returns the record with the image, or `None` for a subhalo that
    cannot be rendered, which the pipeline drops like a record rejected by a
    filter: one with fewer than `min_particles` stars, one without particles
    for the half-mass radius of `width`, one that cannot be oriented (see
    `orientation`), and one whose rendering fails, which is reported with a
    warning.

    Args:
        width (float | str): Image width, as a float in the snapshot units
            (kpc for `PynbodyDataset`), a unit string, e.g. "50 kpc", or a
            multiple of the half-mass radius of the stars or the gas of the
            subhalo, e.g. "10 stars_half_mass_radius" or
            "5 gas_half_mass_radius". The half-mass radius is the radius around
            the center of the dataset of the sphere that holds half of the mass
            of these particles.
        resolution (int | None): Image size in pixels. If `None`, the pynbody
            default is used.
        center_mode (str | None): "catalogue" (the default) keeps the center of
            the dataset, e.g. the most bound particle of SUBFIND catalogues.
            "ssc", "com" or "hyb" refine it on the star particles with
            `pynbody.analysis.center`, and `None` with the pynbody default
            ("ssc"). A refined center differs from the one of `RenderGas`,
            which refines it on the gas.
        orientation (str | None): Rotate the galaxy before rendering like
            `pynbody.analysis.faceon` or `pynbody.analysis.sideon`, which orient
            the angular momentum of the `orientation_by` particles within
            `disk_size` along the line of sight or perpendicular to it. If
            `None`, the image is not rotated. A galaxy whose `orientation_by`
            particles within `disk_size` have no angular momentum, e.g. with no
            particle that close to the center, cannot be oriented and is dropped.
        orientation_by (str): The particles whose angular momentum orients the
            galaxy, "stars" (the default) or "gas".
        disk_size (float | str): Radius around the center of the dataset that
            determines the angular momentum for `orientation`, as a float in
            the snapshot units or a unit string.
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
        center_mode: str | None = "catalogue",
        orientation: str | None = None,
        orientation_by: str = "stars",
        disk_size: float | str = "5 kpc",
        column: str = "image",
        render_args: dict | None = None,
        min_particles: int = 32,
    ):
        super().__init__(
            width, resolution, center_mode, orientation, orientation_by, disk_size, column, render_args, min_particles
        )
        if self.render_args.get("with_dust"):
            raise ValueError("with_dust is not supported: pynbody computes the dust from the gas, which is not loaded.")

    def _particles(self, subhalo_id):
        # Without an orientation by the stars, the dataset only has to provide `stars(subhalo_id)`.
        velocities = self.orientation is not None and self.orientation_by == self.component
        return self.dataset.stars(subhalo_id, **({"velocities": True} if velocities else {}))

    def _image(self, stars, width: float | str) -> np.ndarray:
        # The renderer multiplies the luminosity density, 10**(-0.4 mag) * rho / mass, by mass / rho, so the SPH
        # density cancels out. A placeholder saves computing it, about a third of the time of the smoothing.
        stars["rho"] = SimArray(
            np.ones(len(stars), dtype=stars["pos"].dtype), stars["mass"].units / stars["pos"].units ** 3
        )
        image = pynbody.plot.stars.render(
            stars,
            width=width,
            resolution=self.resolution,
            noplot=True,
            return_image=True,
            **self.render_args,
        )
        # pynbody puts the lowest y in the first row; flip to the usual top-down image layout.
        image = np.asarray(image[::-1], dtype=np.float32)
        # Channels first: (3, height, width).
        return np.moveaxis(image, -1, 0)
