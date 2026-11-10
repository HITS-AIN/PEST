import warnings

import numpy as np
import pynbody
from pynbody.array import SimArray

# pynbody centering schemes that work with the particles of one family alone ("pot" needs the potential, which is
# not loaded), and "catalogue" to keep the center of the dataset.
CENTER_MODES = [None, "catalogue", "ssc", "com", "hyb"]
# pynbody.analysis functions that rotate the galaxy by the angular momentum of its disk.
ORIENTATIONS = [None, "faceon", "sideon"]


def check_center_and_orientation(center_mode: str | None, orientation: str | None) -> None:
    if center_mode not in CENTER_MODES:
        raise ValueError(
            f"center_mode must be one of {CENTER_MODES}, got {center_mode!r}. "
            "For the potential minimum, use 'catalogue' with SUBFIND catalogues."
        )
    if orientation not in ORIENTATIONS:
        raise ValueError(f"orientation must be one of {ORIENTATIONS}, got {orientation!r}.")


def center_and_orient(particles, center_mode: str | None, orientation: str | None, disk_size: float | str) -> bool:
    """Refine the center of `particles` with `pynbody.analysis.center` and rotate them by their angular momentum.

    `particles` is a private snapshot of one family, so moving and rotating only it is enough. Returns whether
    they could be oriented, which needs an angular momentum within `disk_size`.
    """
    if center_mode != "catalogue":
        # The velocities are already relative to the subhalo velocity of the dataset, so no velocity center is
        # needed; cen_num_particles=0 keeps "ssc" from computing one anyway.
        pynbody.analysis.center(particles, mode=center_mode, with_velocity=False, cen_num_particles=0, move_all=False)
    if orientation is None:
        return True
    # pynbody orients by the angular momentum of the particles within disk_size. Without one, e.g. with no
    # particle that close to the center, it fails with "Transformation matrix is not orthogonal".
    inner = particles[pynbody.filt.Sphere(disk_size)]
    norm = np.linalg.norm(np.asarray(pynbody.analysis.angmom.ang_mom_vec(inner)))
    if not (np.isfinite(norm) and norm > 0):
        return False
    getattr(pynbody.analysis, orientation)(particles, disk_size=disk_size, already_centered=True, move_all=False)
    return True


def render_or_drop(render, particles, min_particles: int, subhalo_id, name: str):
    """The image `render(particles)` of the `name` particles of a subhalo, or None to drop the subhalo.

    A subhalo is dropped with fewer than `min_particles`, if `render` returns None, or if `render` fails, which
    is reported with a warning.
    """
    # No particles cannot be rendered either: pynbody fails with an obscure KeyError for a missing family, e.g.
    # the stars in dark matter only runs.
    if len(particles) < max(min_particles, 1):
        return None
    try:
        return render(particles)
    # Whatever pynbody fails with for the particles of this subhalo must not abort the run for all the others.
    except Exception as error:  # noqa: BLE001
        warnings.warn(
            f"Subhalo {subhalo_id} is dropped, its {name} could not be rendered: {type(error).__name__}: {error}",
            stacklevel=2,
        )
        return None


class RenderStars:
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

    is_filter = True

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
        check_center_and_orientation(center_mode, orientation)
        self.width = width
        self.resolution = resolution
        self.center_mode = center_mode
        self.orientation = orientation
        self.disk_size = disk_size
        self.column = column
        self.render_args = render_args or {}
        if self.render_args.get("with_dust"):
            raise ValueError("with_dust is not supported: pynbody computes the dust from the gas, which is not loaded.")
        self.min_particles = min_particles
        self.dataset = None

    def bind(self, dataset) -> None:
        if not hasattr(dataset, "stars"):
            raise TypeError(f"{type(dataset).__name__} does not provide star particles.")
        self.dataset = dataset

    def __call__(self, record: dict) -> dict | None:
        if self.dataset is None:
            raise RuntimeError("RenderStars must be bound to a dataset before use.")
        if "subhalo_id" not in record:
            raise KeyError("RenderStars requires the 'subhalo_id' column in the extracted records.")
        subhalo_id = record["subhalo_id"]
        stars = self.dataset.stars(subhalo_id, **({"velocities": True} if self.orientation else {}))
        image = render_or_drop(self._render, stars, self.min_particles, subhalo_id, "stars")
        if image is None:
            return None
        record[self.column] = image
        return record

    def _render(self, stars) -> np.ndarray | None:
        """The image of `stars`, or None if they cannot be oriented."""
        if not center_and_orient(stars, self.center_mode, self.orientation, self.disk_size):
            return None
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
