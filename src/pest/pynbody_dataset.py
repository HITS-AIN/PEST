import re
from pathlib import Path

import numpy as np
import pynbody
from pynbody.array import SimArray
from pynbody.halo.subfindhdf import TNGSubfindHDFCatalogue
from pynbody.snapshot.gadgethdf import ArepoHDFSnap

from .units import has_units, in_units, physical_units

COMPONENTS = {"stars": "st", "gas": "g", "dm": "dm"}
# pynbody tries the generic Arepo catalogue first, which reads the wrong multi-file header for TNG, and does not
# find the Illustris catalogue files (see `IllustrisSubfindHDFCatalogue`).
SUBFIND_CATALOGUES = ["TNGSubfindHDFCatalogue", "IllustrisSubfindHDFCatalogue"]
DEFAULT_COLUMNS = ["subhalo_id"]
# Columns that the dataset provides besides the halo catalogue properties.
RECORD_COLUMNS = ("subhalo_id", "snapshot", "simulation")
# Directories of a snapshot in several files, e.g. `snapdir_099` (IllustrisTNG) or `snapshot_099`.
SNAPSHOT_DIRECTORY = re.compile(r"snap(?:dir|shot)_\d+")
PARTICLE_FIELDS = {"pos", "vel", "mass"}
# Catalogue properties holding the (position, velocity) of the subhalo center, per catalogue format.
CENTER_PROPERTIES = [("SubhaloPos", "SubhaloVel"), ("sub_pos", "sub_vel")]
# Star arrays that `stars()` provides for rendering.
STAR_ARRAYS = ["pos", "mass", "aform", "metals"]
# Gas arrays that `gas()` provides for rendering.
GAS_ARRAYS = ["pos", "mass", "rho"]


def simulation_name(snapshot_path: str | Path) -> str:
    """The name of the simulation of a snapshot, taken from its directory, e.g. "TNG100-1".

    That is the directory containing the snapshot, or, for a snapshot in its own
    directory (e.g. `TNG100-1/snapdir_099/snap_099`), the one above it.
    """
    directory = Path(snapshot_path).absolute().parent
    if SNAPSHOT_DIRECTORY.fullmatch(directory.name):
        directory = directory.parent
    return directory.name


class IllustrisSubfindHDFCatalogue(TNGSubfindHDFCatalogue):
    """SUBFIND catalogue of the original Illustris simulations (e.g. Illustris-1).

    It has the HDF5 format of IllustrisTNG, but its files are named
    `groups_XXX/groups_XXX.N.hdf5` instead of `groups_XXX/fof_subhalo_tab_XXX.N.hdf5`.
    pynbody does not find them, and falls back to its reader of binary SUBFIND
    catalogues, which fails to open `groups_XXX/subhalo_tab_XXX.0`. A catalogue
    path passed by the user is the file name without the `.N.hdf5` suffix,
    e.g. `.../groups_135/groups_135`.
    """

    @classmethod
    def _catalogue_filename(cls, sim, user_provided_filename=None, namestem="groups_"):
        # `namestem` is "groups_" for the catalogue, or "subhalo_prog_"/"subhalo_desc_" for merger tree files, which
        # Illustris does not have next to the catalogue.
        if user_provided_filename is None:
            return super()._catalogue_filename(sim, None, namestem)
        path = Path(user_provided_filename)
        if not path.name.startswith("groups_"):
            raise ValueError("Filename must start with 'groups_'")
        return str(path.with_name(namestem + path.name.removeprefix("groups_")))


class PynbodyDataset:
    """Dataset of (sub)halos of any snapshot and halo catalogue that pynbody can read.

    One record per halo of the catalogue, e.g. IllustrisTNG, Arepo or Gadget-4
    SUBFIND subhalos, EAGLE FoF groups, AHF or AdaptaHOP halos. Catalogues that
    can list all subhalos (SUBFIND) are loaded as subhalo catalogues, all others
    with their default halos. The record index is the position in the
    catalogue, and the "subhalo_id" column is pynbody's halo number, which can
    differ from it (e.g. EAGLE group numbers start at 1).

    Only halo catalogue data is read here, so records are cheap. The selection
    of subhalos (e.g. a mass range) is done with filters in the transform stage,
    which can use `table()` to drop subhalos before any record is extracted.
    Particle data is added afterwards by a `pest.LoadParticles` transform step,
    using `particles()`, stellar images by a `pest.RenderStars` step, using
    `stars()`, and gas images by a `pest.RenderGas` step, using `gas()`.

    `snapshot_path` can also be a list of snapshots (see `MultiSnapshotDataset`),
    whose catalogues are concatenated into one dataset.

    Args:
        snapshot_path (str | list[str]): Path to the snapshot, e.g.
            `.../snapshot_099/snap_099`, or a list of such paths.
        halos_path (str | list[str] | None): Path to the halo catalogue, a list
            with one path per snapshot. If `None`, pynbody locates it next to
            the snapshot.
        columns (list[str] | None): Columns to extract: "subhalo_id", "snapshot",
            "simulation" (the name of the simulation directory, see
            `simulation_name()`), or any halo catalogue property (e.g. "SubhaloSFR"). Which properties
            exist depends on the catalogue format; some (e.g. EAGLE, AdaptaHOP)
            provide none. Defaults to `["subhalo_id"]`.
        halos_args (dict | None): Keyword arguments for pynbody's `halos()`,
            replacing the automatic catalogue selection, e.g.
            `{"priority": ["AHFCatalogue"]}`.
    """

    def __new__(cls, snapshot_path, *args, **kwargs):
        if cls is PynbodyDataset and not isinstance(snapshot_path, str | Path):
            return super().__new__(MultiSnapshotDataset)
        return super().__new__(cls)

    def __init__(
        self,
        snapshot_path: str | list[str],
        halos_path: str | list[str] | None = None,
        columns: list[str] | None = None,
        halos_args: dict | None = None,
    ) -> None:
        self.columns = DEFAULT_COLUMNS if columns is None else columns
        particle_columns = PARTICLE_FIELDS.intersection(self.columns)
        if particle_columns:
            raise ValueError(
                f"Particle fields {sorted(particle_columns)} are not dataset columns, "
                "load them with a pest.LoadParticles transform step."
            )

        self.snapshot = pynbody.load(snapshot_path)
        self.snapshot.physical_units()
        self.halos = self._load_halos(halos_path, halos_args)
        self.halos.physical_units()
        self.halo_numbers = np.asarray(self.halos.keys())

        # Only the catalogue arrays are read here, no particle data.
        self.properties = self.halos.get_properties_all_halos()
        missing = [col for col in self.columns if col not in (*RECORD_COLUMNS, *self.properties)]
        if missing:
            raise KeyError(f"Columns {missing} are not in the halo catalogue, available: {sorted(self.properties)}")
        self.center_properties = next(
            (keys for keys in CENTER_PROPERTIES if all(key in self.properties for key in keys)), None
        )
        # Formats without partial loading (e.g. Gadget binary, Ramses) fall back to copying out of the snapshot.
        self.partial_loading = True
        # Particles of IllustrisTNG-like snapshots are read directly from the snapshot files, see `_read_particles()`.
        self.direct_loading = True
        self._direct_groups: dict = {}

        name = Path(snapshot_path).name
        match = re.search(r"snap(?:shot)?_(\d+)", name) or re.search(r"(\d+)$", name.removesuffix(".hdf5"))
        self.snapshot_number = np.int32(match.group(1)) if match else None
        self.simulation = simulation_name(snapshot_path)

    def _load_halos(self, halos_path: str | None, halos_args: dict | None):
        kwargs = {} if halos_path is None else {"filename": halos_path}
        if halos_args is not None:
            return self.snapshot.halos(**kwargs, **halos_args)
        try:
            return self.snapshot.halos(subhalos=True, priority=SUBFIND_CATALOGUES, **kwargs)
        except (TypeError, RuntimeError):
            # Catalogues without a flat list of subhalos (e.g. EAGLE, AHF, AdaptaHOP) reject `subhalos`.
            return self.snapshot.halos(**kwargs)

    def __len__(self) -> int:
        return len(self.halo_numbers)

    def table(self) -> dict:
        """Halo catalogue properties of all subhalos, indexed like the records."""
        return self.properties

    def __getitem__(self, index: int) -> dict:
        data: dict = {}
        for col in self.columns:
            if col == "subhalo_id":
                data["subhalo_id"] = np.int32(self.halo_numbers[index])
            elif col == "snapshot":
                data["snapshot"] = self.snapshot_number
            elif col == "simulation":
                data["simulation"] = self.simulation
            else:
                data[col] = np.asarray(self.properties[col][index])
        return data

    def halo(self, subhalo_id: int | np.integer):
        """A standalone pynbody snapshot holding only the particles of one subhalo.

        The particles are read from disk with partial loading where the format
        supports it, otherwise copied out of the snapshot. Indexing the
        catalogue instead gives a view on the full snapshot, where loading any
        array reads it for all particles of the simulation, and where moving
        the particles (e.g. centering) moves the whole snapshot.
        """
        subhalo_id = int(subhalo_id)
        if self.partial_loading:
            try:
                halo = self.halos.load_copy(subhalo_id)
            except (TypeError, NotImplementedError):
                self.partial_loading = False
        if not self.partial_loading:
            halo = self.halos[subhalo_id].get_copy_on_access_simsnap()
        halo.physical_units()
        return halo

    def particles(self, subhalo_id: int, component: str, fields: list[str]) -> dict[str, np.ndarray]:
        """Read particle `fields` of one subhalo.

        "pos" and "vel" are returned relative to the subhalo center in kpc and
        km/s, any other particle field (e.g. "mass") as a float32 array. The
        center is taken from the catalogue, or computed with
        `pynbody.analysis.center` if the catalogue has none. Fields stored in
        IllustrisTNG-like snapshots are read directly from the snapshot files
        (see `_read_particles()`), others from the subhalo loaded with `halo()`.
        """
        if component not in COMPONENTS:
            raise ValueError(f"component must be one of {list(COMPONENTS)}, got {component!r}")
        family = pynbody.family.get_family(COMPONENTS[component])
        particles = self._read_particles(int(subhalo_id), family, fields)
        if particles is None:
            halo = self.halo(subhalo_id)
            if self.center_properties is None:
                pynbody.analysis.center(halo)
            particles = getattr(halo, COMPONENTS[component])
        return {field: self._particle_field(particles, field, subhalo_id) for field in fields}

    def stars(self, subhalo_id: int | np.integer, velocities: bool = False):
        """A pynbody snapshot of the star particles of one subhalo, e.g. for rendering.

        Positions are in kpc relative to the subhalo center, which is taken from
        the catalogue (wrapped at the periodic box boundary), or computed with
        `pynbody.analysis.center` on the whole subhalo if the catalogue has
        none, as for `particles()`, so all components share it. For IllustrisTNG-like
        snapshots, only the arrays needed for rendering ("pos", "mass", "aform",
        "metals") are read (see `_read_particles()`). Wind particles, which TNG
        stores with the stars, are kept as pynbody keeps them: they have no age
        and emit no light. Other formats use the stars of `halo()`. With
        `velocities`, the star velocities relative to the subhalo center are
        provided as well, e.g. to orient the galaxy by its angular momentum.
        """
        return self._component_snapshot(subhalo_id, "stars", STAR_ARRAYS, velocities)

    def gas(self, subhalo_id: int | np.integer, velocities: bool = False, arrays: list[str] | tuple = ()):
        """A pynbody snapshot of the gas particles of one subhalo, e.g. for rendering.

        Positions are in kpc relative to the subhalo center, as for `stars()`.
        For IllustrisTNG-like snapshots, only the arrays needed for rendering
        ("pos", "mass", "rho") and the additional `arrays` (e.g. the rendered
        quantity) are read (see `_read_particles()`). If any of them is not
        stored in the snapshot files, e.g. derived arrays like "temp", the gas
        of `halo()` is used instead, where pynbody derives them. With
        `velocities`, the gas velocities relative to the subhalo center are
        provided as well, e.g. to orient the galaxy by its angular momentum.
        """
        return self._component_snapshot(subhalo_id, "gas", [*GAS_ARRAYS, *arrays], velocities)

    def _component_snapshot(self, subhalo_id: int | np.integer, component: str, arrays: list[str], velocities: bool):
        """The `component` particles of one subhalo with `arrays`, positioned relative to the subhalo center."""
        subhalo_id = int(subhalo_id)
        family_name = COMPONENTS[component]
        # Each array once, as the additional ones may repeat the default ones. The velocities are needed to orient
        # the galaxy by its angular momentum.
        arrays = list(dict.fromkeys([*arrays, "vel"] if velocities else arrays))
        particles = self._read_particles(subhalo_id, pynbody.family.get_family(family_name), arrays)
        if particles is None:
            halo = self.halo(subhalo_id)
            if self.center_properties is None:
                # The center of the whole subhalo, as in `particles()`, so that all components share it.
                pynbody.analysis.center(halo)
            particles = getattr(halo, family_name)
        if self.center_properties is not None:
            particles["pos"] = self._relative_to_center(particles["pos"], "pos", subhalo_id)
            if velocities:
                particles["vel"] = self._relative_to_center(particles["vel"], "vel", subhalo_id)
        return particles

    def _read_particles(self, subhalo_id: int, family, names: list[str]):
        """Arrays `names` of the `family` particles of one subhalo, read directly from the snapshot files.

        IllustrisTNG-like snapshots (Arepo HDF5 with a SUBFIND catalogue) store
        the particles of a subhalo contiguously, so only these arrays of the
        subhalo are read, much faster than loading it with `halo()`. Returns a
        pynbody snapshot in physical units, or None if the arrays cannot be read
        directly, e.g. ones that pynbody derives.
        """
        if not self.direct_loading:
            return None
        if family not in self._direct_groups:
            self._direct_groups[family] = self._find_groups(family)
        if self._direct_groups[family] is None:
            return None
        ptype, groups, starts = self._direct_groups[family]
        index = self.halos.number_mapper.number_to_index(subhalo_id)
        start = int(self.halos._subfind_halo_offsets[ptype][index])
        stop = start + int(self.halos._subfind_halo_lengths[ptype][index])
        # The particles of a subhalo are contiguous, but may be spread over several files.
        parts = [
            (group, slice(max(start, first) - first, min(stop, end) - first))
            for group, first, end in zip(groups, starts[:-1], starts[1:])
            if first < stop and start < end
        ] or [(groups[0], slice(0, 0))]
        arrays = {name: self._read_array(parts, name) for name in names}
        if any(values is None for values in arrays.values()):
            return None

        particles = pynbody.new(**{family.name: stop - start})
        # The box size is left out, as positions get wrapped around the subhalo center: with it, pynbody would
        # treat them as periodic, e.g. the renderer would draw periodic copies.
        particles.properties.update({key: value for key, value in self.snapshot.properties.items() if key != "boxsize"})
        for name, values in arrays.items():
            # new() creates "pos", "vel" and "mass" as float64, while pynbody's loader keeps the dtypes of the files.
            if name in particles:
                del particles[name]
            particles[name] = values
        particles.physical_units()
        return particles

    def _find_groups(self, family) -> tuple[str, list, np.ndarray] | None:
        """The HDF5 particle type of `family`, its groups in the snapshot files and the index of their first
        particle, if `_read_particles()` can read them."""
        snapshot, halos = self.snapshot, self.halos
        if not (
            isinstance(snapshot, ArepoHDFSnap)
            and self.center_properties is not None
            # FoF group catalogues have subhalo offsets as well, but are indexed by group.
            and getattr(halos, "_sub_mode", False)
        ):
            return None
        ptypes = snapshot._family_to_group_map.get(family, [])
        if len(ptypes) != 1 or ptypes[0] not in getattr(halos, "_subfind_halo_offsets", {}):
            return None
        # pynbody keeps the snapshot files open and counts their particles like this.
        groups = list(snapshot._hdf_files.iter_particle_groups_with_name(ptypes[0]))
        if not groups:
            return None
        starts = np.cumsum([0] + [group["ParticleIDs"].shape[0] for group in groups])
        return ptypes[0], groups, starts

    def _read_array(self, parts: list, name: str) -> SimArray | None:
        """The pynbody array `name` of slices of particle groups, if it can be read from the snapshot files."""
        first = parts[0][0]
        dataset = next((dataset for dataset in self.snapshot._translate_array_name(name) if dataset in first), None)
        if dataset is not None and "length_scaling" in first[dataset].attrs:
            data = np.concatenate([group[dataset][part] for group, part in parts])
            units = self.snapshot._get_units_from_hdf_attr(first[dataset].attrs)
        elif dataset is None and name == "mass":
            # Masses that are the same for all particles of a type are only stored in the header.
            masses = [group.file["Header"].attrs["MassTable"][int(group.name[-1])] for group, _ in parts]
            if min(masses) <= 0:
                return None
            data = np.concatenate([np.full(part.stop - part.start, mass) for (_, part), mass in zip(parts, masses)])
            units = self.snapshot.infer_original_units("Msol")
        else:
            return None
        if name == "mass":
            # pynbody keeps masses in the dtype of the coordinates.
            data = data.astype(first["Coordinates"].dtype)
        return SimArray(data, units)

    def _particle_field(self, particles, field: str, subhalo_id: int) -> np.ndarray:
        values = particles[field]
        if field in ("pos", "vel") and self.center_properties is not None:
            values = self._relative_to_center(values, field, subhalo_id)
        return np.asarray(values, dtype=np.float32)

    def _relative_to_center(self, values, field: str, subhalo_id: int) -> np.ndarray:
        """Positions or velocities ("pos" or "vel") relative to the catalogue center of the subhalo."""
        center_key = self.center_properties[0 if field == "pos" else 1]
        index = self.halos.number_mapper.number_to_index(int(subhalo_id))
        center = in_units(self.properties[center_key][index], getattr(values, "units", None))
        offset = np.asarray(values) - center
        boxsize = self._boxsize(values) if field == "pos" else None
        if boxsize is not None:
            # Halos crossing the boundary of a periodic box.
            offset = (offset + boxsize / 2) % boxsize - boxsize / 2
        return offset

    def _boxsize(self, values) -> float | None:
        boxsize = self.snapshot.properties.get("boxsize")
        units = getattr(values, "units", None)
        if boxsize is None or units is None:
            return None
        return float(boxsize.in_units(units, **self.snapshot.conversion_context()))


class MultiSnapshotDataset(PynbodyDataset):
    """Concatenation of the `PynbodyDataset`s of several snapshots, created by `PynbodyDataset(list_of_paths)`.

    The records of the snapshots follow each other in the order of the list.
    Subhalo ids are only unique within a snapshot, so the particle methods
    (`particles()`, `stars()`, `gas()`, `halo()`) refer to the snapshot of the record
    extracted last with `__getitem__`, as the transform steps run on one record
    at a time. Add the "snapshot" column to tell the records apart in the output.
    """

    def __init__(
        self,
        snapshot_path: list[str],
        halos_path: list[str] | None = None,
        columns: list[str] | None = None,
        halos_args: dict | None = None,
    ) -> None:
        paths = [str(path) for path in snapshot_path]
        if not paths:
            raise ValueError("snapshot_path must contain at least one snapshot.")
        if halos_path is None:
            halos_paths = [None] * len(paths)
        elif isinstance(halos_path, str | Path) or len(halos_path) != len(paths):
            raise ValueError("halos_path must be a list with one path per snapshot.")
        else:
            halos_paths = list(halos_path)
        self.datasets = [
            PynbodyDataset(path, halos_path=halos, columns=columns, halos_args=halos_args)
            for path, halos in zip(paths, halos_paths)
        ]
        self.columns = self.datasets[0].columns
        self.offsets = np.cumsum([0] + [len(dataset) for dataset in self.datasets])
        self._current = self.datasets[0]
        self._table: dict | None = None

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def table(self) -> dict:
        """Halo catalogue properties of all subhalos of all snapshots, indexed like the records.

        Only properties that all catalogues provide are included.
        """
        if self._table is None:
            tables = [dataset.table() for dataset in self.datasets]
            keys = [key for key in tables[0] if all(key in table for table in tables)]
            self._table = {key: self._concatenate([table[key] for table in tables]) for key in keys}
        return self._table

    @staticmethod
    def _concatenate(arrays: list):
        # The concatenated array belongs to no simulation, so units depending on the scale factor `a` or on `h`
        # could not be converted any more, and `a` differs between the snapshots. Each array is converted to
        # physical units with the cosmology of its own snapshot first.
        arrays = [physical_units(array) for array in arrays]
        if not has_units(arrays[0]):
            return np.concatenate([np.asarray(array) for array in arrays])
        units = arrays[0].units
        return SimArray(np.concatenate([in_units(array, units) for array in arrays]), units)

    def __getitem__(self, index: int) -> dict:
        index = int(index)
        if not -len(self) <= index < len(self):
            raise IndexError(f"index {index} is out of range for a dataset of length {len(self)}")
        index %= len(self)
        number = int(np.searchsorted(self.offsets, index, side="right")) - 1
        self._current = self.datasets[number]
        return self._current[index - int(self.offsets[number])]

    def halo(self, subhalo_id: int | np.integer):
        return self._current.halo(subhalo_id)

    def particles(self, subhalo_id: int, component: str, fields: list[str]) -> dict[str, np.ndarray]:
        return self._current.particles(subhalo_id, component, fields)

    def stars(self, subhalo_id: int | np.integer, velocities: bool = False):
        return self._current.stars(subhalo_id, velocities=velocities)

    def gas(self, subhalo_id: int | np.integer, velocities: bool = False, arrays: list[str] | tuple = ()):
        return self._current.gas(subhalo_id, velocities=velocities, arrays=arrays)
