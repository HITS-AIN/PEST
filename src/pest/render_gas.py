import numpy as np
import pynbody

from .render_particles import RenderParticles


class RenderGas(RenderParticles):
    """Filter step that adds a gas image of a subhalo to the record.

    The pipeline binds the worker's dataset to this step, which must provide
    `gas(subhalo_id)` returning a snapshot of the gas particles of the subhalo,
    centered on the subhalo (e.g. `PynbodyDataset`), and
    `particles(subhalo_id, component, fields)` if the width or the orientation
    refer to the stars, as they do by default with an `orientation`. The image
    is rendered with `pynbody.plot.image`, by default the projected gas
    surface density.

    With the defaults, the image has the same view as the one of `RenderStars`:
    both are centered on the center of the dataset, and both are oriented by
    the stars. A width relative to a half-mass radius, e.g.
    "10 stars_half_mass_radius", is the same for both as well.

    The step returns the record with the image, or `None` for a subhalo that
    cannot be rendered, which the pipeline drops like a record rejected by a
    filter: one with fewer than `min_particles` gas particles, one without
    particles for the half-mass radius of `width`, one that cannot be oriented
    (see `orientation`), one without a positive pixel for `log`, and one whose
    rendering fails, which is reported with a warning.

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
            "ssc", "com" or "hyb" refine it on the gas particles with
            `pynbody.analysis.center`, and `None` with the pynbody default
            ("ssc"). A refined center differs from the one of `RenderStars`,
            which refines it on the stars.
        orientation (str | None): Rotate the galaxy before rendering like
            `pynbody.analysis.faceon` or `pynbody.analysis.sideon`, which orient
            the angular momentum of the `orientation_by` particles within
            `disk_size` along the line of sight or perpendicular to it. If
            `None`, the image is not rotated. A galaxy whose `orientation_by`
            particles within `disk_size` have no angular momentum, e.g. with no
            particle that close to the center, cannot be oriented and is dropped.
        orientation_by (str): The particles whose angular momentum orients the
            galaxy, "stars" (the default, as for `RenderStars`) or "gas".
        disk_size (float | str): Radius around the center of the dataset that
            determines the angular momentum for `orientation`, as a float in
            the snapshot units or a unit string.
        qty (str): Gas array to render, e.g. "rho" or "temp".
        units (str | None): Units of the image. Units of `qty` integrated along
            the line of sight give a projected image, e.g. the default
            "Msol kpc**-2" for the surface density of "rho". If `None`, the
            units of `qty` are used, which gives a slice at z = 0 unless
            `render_args` sets a `weight`, e.g. `{"weight": "rho"}` for the
            density-weighted average of `qty` along the line of sight.
        log (bool): Store the log10 of the image. Pixels without a positive
            value are set to the smallest positive one, and a subhalo without
            any positive pixel is dropped. Only for quantities that are
            positive.
        column (str): Name of the added column.
        render_args (dict | None): Additional arguments for
            `pynbody.plot.image`, e.g. `{"weight": "rho"}` or
            `{"kernel": "WendlandC2"}`.
        min_particles (int): Drop subhalos with fewer gas particles. The
            default is the number of neighbours of pynbody's SPH smoothing
            (32), which fails with fewer particles.
    """

    component = "gas"

    def __init__(
        self,
        width: float | str = 50,
        resolution: int | None = None,
        center_mode: str | None = "catalogue",
        orientation: str | None = None,
        orientation_by: str = "stars",
        disk_size: float | str = "5 kpc",
        qty: str = "rho",
        units: str | None = "Msol kpc**-2",
        log: bool = True,
        column: str = "gas_image",
        render_args: dict | None = None,
        min_particles: int = 32,
    ):
        super().__init__(
            width, resolution, center_mode, orientation, orientation_by, disk_size, column, render_args, min_particles
        )
        self.qty = qty
        self.units = units
        self.log = log
        # Arrays besides the default ones of `gas()` that the renderer reads.
        self.arrays = [name for name in (qty, self.render_args.get("weight")) if isinstance(name, str)]

    def _particles(self, subhalo_id):
        velocities = self.orientation is not None and self.orientation_by == self.component
        return self.dataset.gas(subhalo_id, velocities=velocities, arrays=self.arrays)

    def _image(self, gas, width: float | str) -> np.ndarray | None:
        """The image of `gas`, or None if it has no positive pixel to take the log of."""
        image = pynbody.plot.image(
            gas,
            qty=self.qty,
            width=width,
            resolution=self.resolution,
            units=self.units,
            noplot=True,
            **self.render_args,
        )
        # pynbody puts the lowest y in the first row; flip to the usual top-down image layout.
        image = np.asarray(image[::-1], dtype=np.float64)
        if self.log:
            positive = image > 0
            if not positive.any():
                return None
            image = np.log10(np.where(positive, image, image[positive].min()))
        # Channels first: (1, height, width).
        return image[np.newaxis].astype(np.float32)
