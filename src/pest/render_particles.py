import warnings
from abc import ABC, abstractmethod

import numpy as np
import pynbody

# pynbody centering schemes that work with the particles of one family alone ("pot" needs the potential, which is
# not loaded), and "catalogue" to keep the center of the dataset.
CENTER_MODES = [None, "catalogue", "ssc", "com", "hyb"]
# pynbody.analysis functions that rotate the galaxy by the angular momentum of its disk.
ORIENTATIONS = [None, "faceon", "sideon"]


class RenderParticles(ABC):
    """Base class of the filter steps that add an image of the particles of a subhalo to the record.

    A subclass renders one kind of particles: it sets `component` and
    implements `_particles()` and `_image()`. The pipeline binds the worker's
    dataset to the step, which must provide these particles with a method named
    like `component`, as a snapshot centered on the subhalo.

    The step refines the center on the particles, rotates them for
    `orientation`, and returns the record with their image. It returns `None`
    for a subhalo that cannot be rendered, which the pipeline drops like a
    record rejected by a filter: one with fewer than `min_particles` particles,
    one that cannot be oriented, one that `_image()` has no image for, and one
    whose rendering fails, which is reported with a warning.
    """

    is_filter = True
    # The particles to render, named like the dataset method that provides them, e.g. "stars".
    component: str

    def __init__(
        self,
        width: float | str,
        resolution: int | None,
        center_mode: str | None,
        orientation: str | None,
        disk_size: float | str,
        column: str,
        render_args: dict | None,
        min_particles: int,
    ):
        if center_mode not in CENTER_MODES:
            raise ValueError(
                f"center_mode must be one of {CENTER_MODES}, got {center_mode!r}. "
                "For the potential minimum, use 'catalogue' with SUBFIND catalogues."
            )
        if orientation not in ORIENTATIONS:
            raise ValueError(f"orientation must be one of {ORIENTATIONS}, got {orientation!r}.")
        self.width = width
        self.resolution = resolution
        self.center_mode = center_mode
        self.orientation = orientation
        self.disk_size = disk_size
        self.column = column
        self.render_args = render_args or {}
        self.min_particles = min_particles
        self.dataset = None

    def bind(self, dataset) -> None:
        if not hasattr(dataset, self.component):
            raise TypeError(f"{type(dataset).__name__} does not provide the {self.component} of a subhalo.")
        self.dataset = dataset

    def __call__(self, record: dict) -> dict | None:
        step = type(self).__name__
        if self.dataset is None:
            raise RuntimeError(f"{step} must be bound to a dataset before use.")
        if "subhalo_id" not in record:
            raise KeyError(f"{step} requires the 'subhalo_id' column in the extracted records.")
        subhalo_id = record["subhalo_id"]
        particles = self._particles(subhalo_id)
        # No particles cannot be rendered either: pynbody fails with an obscure KeyError for a missing family, e.g.
        # the stars in dark matter only runs.
        if len(particles) < max(self.min_particles, 1):
            return None
        try:
            image = self._image(particles) if self._center_and_orient(particles) else None
        # Whatever pynbody fails with for the particles of this subhalo must not abort the run for all the others.
        except Exception as error:  # noqa: BLE001
            warnings.warn(
                f"Subhalo {subhalo_id} is dropped, its {self.component} could not be rendered: "
                f"{type(error).__name__}: {error}"
            )
            return None
        if image is None:
            return None
        record[self.column] = image
        return record

    @abstractmethod
    def _particles(self, subhalo_id):
        """The snapshot of the particles of a subhalo, from the bound dataset."""

    @abstractmethod
    def _image(self, particles) -> np.ndarray | None:
        """The image of the centered and oriented `particles`, channels first, or None to drop the subhalo."""

    def _center_and_orient(self, particles) -> bool:
        """Refine the center of `particles` with `pynbody.analysis.center` and rotate them by their angular momentum.

        `particles` is a private snapshot of one family, so moving and rotating only it is enough. Returns whether
        they could be oriented, which needs an angular momentum within `disk_size`.
        """
        if self.center_mode != "catalogue":
            # The velocities are already relative to the subhalo velocity of the dataset, so no velocity center is
            # needed; cen_num_particles=0 keeps "ssc" from computing one anyway.
            pynbody.analysis.center(
                particles, mode=self.center_mode, with_velocity=False, cen_num_particles=0, move_all=False
            )
        if self.orientation is None:
            return True
        # pynbody orients by the angular momentum of the particles within disk_size. Without one, e.g. with no
        # particle that close to the center, it fails with "Transformation matrix is not orthogonal".
        inner = particles[pynbody.filt.Sphere(self.disk_size)]
        norm = np.linalg.norm(np.asarray(pynbody.analysis.angmom.ang_mom_vec(inner)))
        if not (np.isfinite(norm) and norm > 0):
            return False
        orient = getattr(pynbody.analysis, self.orientation)
        orient(particles, disk_size=self.disk_size, already_centered=True, move_all=False)
        return True
