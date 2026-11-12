import warnings
from abc import ABC, abstractmethod

import numpy as np
import pynbody

# pynbody centering schemes that work with the particles of one family alone ("pot" needs the potential, which is
# not loaded), and "catalogue" to keep the center of the dataset.
CENTER_MODES = [None, "catalogue", "ssc", "com", "hyb"]
# pynbody.analysis.angmom functions that give the rotation matrix of an orientation from the angular momentum.
ORIENTATION_MATRICES = {
    "faceon": pynbody.analysis.angmom.calc_faceon_matrix,
    "sideon": pynbody.analysis.angmom.calc_sideon_matrix,
}
ORIENTATIONS = [None, *ORIENTATION_MATRICES]
# Particles that the width and the orientation of the images can refer to, named like the dataset methods.
REFERENCE_COMPONENTS = ["stars", "gas"]
# Width units that are the half-mass radius of a component.
HALF_MASS_RADII = {f"{component}_half_mass_radius": component for component in REFERENCE_COMPONENTS}


def parse_width(width: float | str) -> tuple[float | str, str | None]:
    """Split a width into a value and the component whose half-mass radius is its unit.

    A width relative to a half-mass radius, e.g. "10 stars_half_mass_radius", gives `(10.0, "stars")`, and a bare
    "stars_half_mass_radius" a factor of 1. Any other width, e.g. 50 or "50 kpc", is returned unchanged with `None`.
    """
    if isinstance(width, str):
        value, _, unit = width.strip().rpartition(" ")
        if unit in HALF_MASS_RADII:
            return (float(value) if value.strip() else 1.0), HALF_MASS_RADII[unit]
    return width, None


def half_mass_radius(pos: np.ndarray, mass: np.ndarray) -> float:
    """The radius around the origin of the sphere that holds half of the mass of the particles."""
    radius = np.linalg.norm(np.asarray(pos, dtype=np.float64), axis=1)
    order = np.argsort(radius)
    cumulative = np.cumsum(np.asarray(mass, dtype=np.float64)[order])
    return float(radius[order][np.searchsorted(cumulative, 0.5 * cumulative[-1])])


class RenderParticles(ABC):
    """Base class of the filter steps that add an image of the particles of a subhalo to the record.

    A subclass renders one kind of particles: it sets `component` and
    implements `_particles()` and `_image()`. The pipeline binds the worker's
    dataset to the step, which must provide these particles with a method named
    like `component`, as a snapshot centered on the subhalo.

    The step determines the width and the rotation of the image, refines the
    center on the particles for `center_mode`, rotates them for `orientation`,
    and returns the record with their image. The width relative to a half-mass
    radius and the rotation are computed from the particles around the center
    of the dataset, before any refinement, so the steps of all components that
    refer to the same particles (e.g. star and gas images both oriented by the
    stars) get the same ones.

    It returns `None` for a subhalo that cannot be rendered, which the pipeline
    drops like a record rejected by a filter: one with fewer than
    `min_particles` particles, one without particles for the half-mass radius
    of the width, one that cannot be oriented, one that `_image()` has no image
    for, and one whose rendering fails, which is reported with a warning.
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
        orientation_by: str,
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
        if orientation_by not in REFERENCE_COMPONENTS:
            raise ValueError(f"orientation_by must be one of {REFERENCE_COMPONENTS}, got {orientation_by!r}.")
        self.width = width
        self.width_factor, self.width_component = parse_width(width)
        self.resolution = resolution
        self.center_mode = center_mode
        self.orientation = orientation
        self.orientation_by = orientation_by
        self.disk_size = disk_size
        self.column = column
        self.render_args = render_args or {}
        self.min_particles = min_particles
        self.dataset = None

    def bind(self, dataset) -> None:
        if not hasattr(dataset, self.component):
            raise TypeError(f"{type(dataset).__name__} does not provide the {self.component} of a subhalo.")
        references = {self.width_component, self.orientation_by if self.orientation else None}
        others = sorted(references - {None, self.component})
        if others and not hasattr(dataset, "particles"):
            raise TypeError(
                f"{type(dataset).__name__} does not provide the particles of a subhalo, which {type(self).__name__} "
                f"needs for the {', '.join(others)} that the width or the orientation refer to."
            )
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
            image = self._render(subhalo_id, particles)
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
    def _image(self, particles, width: float | str) -> np.ndarray | None:
        """The image of the centered and oriented `particles` with `width`, channels first, or None to drop the
        subhalo."""

    def _render(self, subhalo_id, particles) -> np.ndarray | None:
        """The image of `particles`, or None if the subhalo has no width or cannot be oriented."""
        # Both from the particles around the center of the dataset, before the center is refined.
        width = self._width(subhalo_id, particles)
        if width is None:
            return None
        rotation = None
        if self.orientation is not None:
            rotation = self._rotation(subhalo_id, particles)
            if rotation is None:
                return None
        if self.center_mode != "catalogue":
            # The velocities are already relative to the subhalo velocity of the dataset, so no velocity center is
            # needed; cen_num_particles=0 keeps "ssc" from computing one anyway. `particles` is a private snapshot
            # of one family, so moving only it is enough.
            pynbody.analysis.center(
                particles, mode=self.center_mode, with_velocity=False, cen_num_particles=0, move_all=False
            )
        if rotation is not None:
            particles.rotate(rotation, description=self.orientation)
        return self._image(particles, width)

    def _reference(self, subhalo_id, particles, component: str, fields: list[str]) -> dict[str, np.ndarray]:
        """The `fields` of the `component` particles of the subhalo around the center of the dataset.

        For the rendered component, they are taken from `particles`, otherwise read with the `particles()` method of
        the dataset. Both as float32, the dtype of `particles()`, so that the steps of all components get the same.
        """
        if component != self.component:
            return self.dataset.particles(subhalo_id, component, fields)
        return {field: np.asarray(particles[field], dtype=np.float32) for field in fields}

    def _width(self, subhalo_id, particles) -> float | str | None:
        """The image width, or None if there are no particles for the half-mass radius it refers to."""
        if self.width_component is None:
            return self.width
        reference = self._reference(subhalo_id, particles, self.width_component, ["pos", "mass"])
        if not (len(reference["mass"]) and np.sum(reference["mass"], dtype=np.float64) > 0):
            return None
        return self.width_factor * half_mass_radius(reference["pos"], reference["mass"])

    def _rotation(self, subhalo_id, particles) -> np.ndarray | None:
        """The matrix that rotates the subhalo for `orientation`, or None if it cannot be oriented.

        That is the matrix of pynbody's `faceon` or `sideon` for the angular momentum of the `orientation_by`
        particles within `disk_size`. Without an angular momentum, e.g. with no particle that close to the center,
        pynbody would fail with "Transformation matrix is not orthogonal".
        """
        reference = self._reference(subhalo_id, particles, self.orientation_by, ["pos", "vel", "mass"])
        disk_size = self.disk_size
        if isinstance(disk_size, str):
            disk_size = pynbody.units.Unit(disk_size).in_units("kpc")
        pos, vel, mass = (np.asarray(reference[field], dtype=np.float64) for field in ("pos", "vel", "mass"))
        inner = np.linalg.norm(pos, axis=1) < float(disk_size)
        angular_momentum = np.sum(mass[inner, None] * np.cross(pos[inner], vel[inner]), axis=0)
        norm = np.linalg.norm(angular_momentum)
        if not (np.isfinite(norm) and norm > 0):
            return None
        return ORIENTATION_MATRICES[self.orientation](angular_momentum)
