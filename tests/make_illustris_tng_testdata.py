"""Generate a tiny synthetic IllustrisTNG snapshot and SUBFIND catalogue for the tests.

The files follow the layout and format of the TNG data release, so that pynbody
reads them with ArepoHDFSnap and TNGSubfindHDFCatalogue:

    tests/data/illustris_tng/TNG50-1/snapdir_099/snap_099.{0,1}.hdf5
    tests/data/illustris_tng/TNG50-1/groups_099/fof_subhalo_tab_099.0.hdf5

Contents (code units: ckpc/h, 1e10 Msun/h, km/s * sqrt(a)):

- 2 FoF groups with 5 subhalos: group 0 holds subhalos 0, 1, 2, group 1 holds 3, 4.
- Gas (PartType0), dark matter (PartType1, mass from the MassTable) and stars (PartType4).
- Subhalo 0 has 64 star particles, enough for pynbody's SPH smoothing (32 neighbours) in RenderStars.
  3 of them are wind particles, with a negative GFM_StellarFormationTime as in TNG.
- Subhalo 2 is dark (no gas, no stars).
- Subhalo 3 crosses the periodic box boundary in x.
- Subhalo 4 has SubhaloFlag = 0.
- Particles are ordered by group and subhalo, followed by the group fuzz and the
  particles outside of any group, as in the TNG snapshots.
- The snapshot is split into 2 files with half of the particles of each type, so
  the stars of subhalo 0 are spread over both files.

Run `python tests/make_illustris_tng_testdata.py` to regenerate the files.
"""

from pathlib import Path

import h5py
import numpy as np

OUTPUT_PATH = Path(__file__).parent / "data" / "illustris_tng" / "TNG50-1"
SNAPSHOT = 99
SEED = 42
NUM_FILES = 2
WIND_STARS = [5, 30, 50]  # indices of the wind particles among the star particles, all in subhalo 0

BOXSIZE = 35000.0  # [ckpc/h]
HUBBLE = 0.6774
REDSHIFT = 0.0
DM_MASS = 3.0e-5  # [1e10 Msun/h]

UNIT_LENGTH = 3.085678e21  # [cm]
UNIT_MASS = 1.989e43  # [g]
UNIT_VELOCITY = 1.0e5  # [cm/s]

PART_TYPES = [0, 1, 4]  # gas, dm, stars

# Subhalo: parent group, center [ckpc/h], bulk velocity [km/s], number of (gas, dm, star) particles,
# mean (gas, star) particle mass [1e10 Msun/h], SFR [Msun/yr], flag
SUBHALOS = [
    {
        "group": 0,
        "pos": [10000.0, 12000.0, 8000.0],
        "vel": [100.0, -50.0, 20.0],
        "npart": [20, 40, 64],
        "mass": [1.0e-4, 1.0e-1],
        "sfr": 2.5,
        "flag": 1,
    },
    {
        "group": 0,
        "pos": [10060.0, 11950.0, 8030.0],
        "vel": [250.0, 10.0, -40.0],
        "npart": [5, 10, 8],
        "mass": [1.0e-4, 1.0e-2],
        "sfr": 0.3,
        "flag": 1,
    },
    {
        "group": 0,
        "pos": [9950.0, 12040.0, 7980.0],
        "vel": [-80.0, 130.0, 60.0],
        "npart": [0, 6, 0],
        "mass": [1.0e-4, 1.0e-2],
        "sfr": 0.0,
        "flag": 1,
    },
    {
        "group": 1,
        "pos": [1.0, 20000.0, 30000.0],
        "vel": [-30.0, 70.0, 10.0],
        "npart": [10, 20, 15],
        "mass": [1.0e-4, 2.0e-1],
        "sfr": 1.2,
        "flag": 1,
    },
    {
        "group": 1,
        "pos": [34990.0, 20040.0, 30020.0],
        "vel": [60.0, -20.0, 90.0],
        "npart": [3, 8, 5],
        "mass": [1.0e-4, 1.0e-3],
        "sfr": 0.05,
        "flag": 0,
    },
]
# Particles of a group that are not bound to any of its subhalos, per (gas, dm, star)
GROUP_FUZZ = [[2, 4, 1], [1, 2, 0]]
# Particles outside of any group, per (gas, dm, star)
OUTER_FUZZ = [5, 10, 2]


def unit_attrs(length=0.0, mass=0.0, velocity=0.0, a=0.0, h=0.0, to_cgs=0.0) -> dict:
    """Unit metadata attached to every TNG dataset."""
    return {
        "length_scaling": length,
        "mass_scaling": mass,
        "velocity_scaling": velocity,
        "a_scaling": a,
        "h_scaling": h,
        "to_cgs": to_cgs,
    }


POS_UNITS = unit_attrs(length=1.0, a=1.0, h=-1.0, to_cgs=UNIT_LENGTH)
VEL_UNITS = unit_attrs(velocity=1.0, a=0.5, to_cgs=UNIT_VELOCITY)
MASS_UNITS = unit_attrs(mass=1.0, h=-1.0, to_cgs=UNIT_MASS)
SFR_UNITS = unit_attrs(mass=1.0, length=-1.0, velocity=1.0, to_cgs=1.989e33 / 3.15576e7)  # Msun/yr
NO_UNITS = unit_attrs()


def write(group: h5py.Group, name: str, data, attrs: dict) -> None:
    dataset = group.create_dataset(name, data=data)
    dataset.attrs.update(attrs)


def make_particles(rng, center, vel, n, mass, scale):
    pos = (np.asarray(center) + rng.normal(0.0, scale, (n, 3))) % BOXSIZE
    vel = np.asarray(vel) + rng.normal(0.0, 50.0, (n, 3))
    masses = mass * rng.uniform(0.5, 1.5, n)
    return pos, vel, masses


def build(rng):
    """Particles per type, sorted by group, subhalo and fuzz, and the catalogue lengths."""
    particles = {ptype: {"pos": [], "vel": [], "mass": []} for ptype in PART_TYPES}
    sub_len = np.zeros((len(SUBHALOS), 6), dtype=np.int32)
    group_len = np.zeros((len(GROUP_FUZZ), 6), dtype=np.int32)

    def add(ptype, pos, vel, mass):
        particles[ptype]["pos"].append(pos)
        particles[ptype]["vel"].append(vel)
        particles[ptype]["mass"].append(mass)

    for ptype_index, ptype in enumerate(PART_TYPES):
        for group, fuzz in enumerate(GROUP_FUZZ):
            members = [i for i, sub in enumerate(SUBHALOS) if sub["group"] == group]
            for i in members:
                sub = SUBHALOS[i]
                n = sub["npart"][ptype_index]
                mass = DM_MASS if ptype == 1 else sub["mass"][0 if ptype == 0 else 1]
                scale = 2.0 if ptype == 4 else 5.0
                add(ptype, *make_particles(rng, sub["pos"], sub["vel"], n, mass, scale))
                sub_len[i, ptype] = n
            # Fuzz is spread around the central subhalo of the group
            central = SUBHALOS[members[0]]
            n = fuzz[ptype_index]
            add(ptype, *make_particles(rng, central["pos"], central["vel"], n, DM_MASS, 50.0))
            group_len[group, ptype] = sub_len[members, ptype].sum() + n
        n = OUTER_FUZZ[ptype_index]
        pos = rng.uniform(0.0, BOXSIZE, (n, 3))
        add(ptype, pos, rng.normal(0.0, 100.0, (n, 3)), np.full(n, DM_MASS))

    particles = {
        ptype: {key: np.concatenate(values) for key, values in fields.items()} for ptype, fields in particles.items()
    }
    for fields in particles.values():
        fields["pos"] = fields["pos"].astype(np.float64)
        fields["vel"] = fields["vel"].astype(np.float32)
        fields["mass"] = fields["mass"].astype(np.float32)
    particles[1]["mass"][:] = DM_MASS
    return particles, sub_len, group_len


def snapshot_datasets(particles, rng) -> dict:
    """The snapshot datasets of each particle type with their unit attributes, for all files together."""
    datasets = {}
    first_id = 1
    for ptype, fields in particles.items():
        n = len(fields["pos"])
        data = {
            "Coordinates": (fields["pos"], POS_UNITS),
            "Velocities": (fields["vel"], VEL_UNITS),
            "ParticleIDs": (np.arange(first_id, first_id + n, dtype=np.uint64), NO_UNITS),
        }
        first_id += n
        if ptype != 1:  # dark matter masses are in the MassTable
            data["Masses"] = (fields["mass"], MASS_UNITS)
            data["GFM_Metallicity"] = (rng.uniform(0.005, 0.03, n).astype(np.float32), NO_UNITS)
        if ptype == 4:
            formation_time = rng.uniform(0.2, 0.99, n).astype(np.float32)
            formation_time[WIND_STARS] *= -1
            data["GFM_StellarFormationTime"] = (formation_time, NO_UNITS)
        datasets[ptype] = data
    return datasets


def write_snapshot(directory: Path, datasets) -> None:
    """Write the snapshot as NUM_FILES files, each with an equal share of the particles of each type."""
    npart_total = np.zeros(6, dtype=np.uint32)
    for ptype, data in datasets.items():
        npart_total[ptype] = len(data["ParticleIDs"][0])
    bounds = [npart_total * i // NUM_FILES for i in range(NUM_FILES + 1)]
    mass_table = np.zeros(6)
    mass_table[1] = DM_MASS

    directory.mkdir(parents=True, exist_ok=True)
    for i in range(NUM_FILES):
        with h5py.File(directory / f"snap_{SNAPSHOT:03d}.{i}.hdf5", "w") as f:
            f.create_group("Config").attrs["VORONOI"] = 1
            header = f.create_group("Header")
            header.attrs.update(
                {
                    "BoxSize": BOXSIZE,
                    "HubbleParam": HUBBLE,
                    "MassTable": mass_table,
                    "NumFilesPerSnapshot": NUM_FILES,
                    "NumPart_ThisFile": (bounds[i + 1] - bounds[i]).astype(np.int32),
                    "NumPart_Total": npart_total,
                    "NumPart_Total_HighWord": np.zeros(6, dtype=np.uint32),
                    "Omega0": 0.3089,
                    "OmegaBaryon": 0.0486,
                    "OmegaLambda": 0.6911,
                    "Redshift": REDSHIFT,
                    "Time": 1.0 / (1.0 + REDSHIFT),
                    "UnitLength_in_cm": UNIT_LENGTH,
                    "UnitMass_in_g": UNIT_MASS,
                    "UnitVelocity_in_cm_per_s": UNIT_VELOCITY,
                }
            )
            f.create_group("Parameters").attrs.update(
                {
                    "BoxSize": BOXSIZE,
                    "HubbleParam": HUBBLE,
                    "Omega0": 0.3089,
                    "OmegaBaryon": 0.0486,
                    "OmegaLambda": 0.6911,
                    "UnitLength_in_cm": UNIT_LENGTH,
                    "UnitMass_in_g": UNIT_MASS,
                    "UnitVelocity_in_cm_per_s": UNIT_VELOCITY,
                }
            )

            for ptype, data in datasets.items():
                group = f.create_group(f"PartType{ptype}")
                for name, (values, attrs) in data.items():
                    write(group, name, values[bounds[i][ptype] : bounds[i + 1][ptype]], attrs)


def write_catalogue(path: Path, particles, sub_len, group_len) -> None:
    nsub = len(SUBHALOS)
    ngroups = len(GROUP_FUZZ)

    sub_mass_type = np.zeros((nsub, 6), dtype=np.float32)
    group_mass_type = np.zeros((ngroups, 6), dtype=np.float32)
    for ptype, fields in particles.items():
        offset = 0
        for group in range(ngroups):
            members = [i for i, sub in enumerate(SUBHALOS) if sub["group"] == group]
            group_start = offset
            for i in members:
                sub_mass_type[i, ptype] = fields["mass"][offset : offset + sub_len[i, ptype]].sum()
                offset += sub_len[i, ptype]
            offset = group_start + group_len[group, ptype]
            group_mass_type[group, ptype] = fields["mass"][group_start:offset].sum()

    sub_group = np.array([sub["group"] for sub in SUBHALOS], dtype=np.int32)
    first_sub = np.array([np.flatnonzero(sub_group == g)[0] for g in range(ngroups)], dtype=np.int32)
    sub_pos = np.array([sub["pos"] for sub in SUBHALOS], dtype=np.float32)
    group_pos = sub_pos[first_sub]

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_group("Config")
        f.create_group("Parameters")
        f.create_group("IDs")
        f.create_group("Header").attrs.update(
            {
                "BoxSize": BOXSIZE,
                "HubbleParam": HUBBLE,
                "Ngroups_ThisFile": np.int32(ngroups),
                "Ngroups_Total": np.int32(ngroups),
                "Nids_ThisFile": np.int32(0),
                "Nids_Total": np.int32(0),
                "Nsubgroups_ThisFile": np.int32(nsub),
                "Nsubgroups_Total": np.int32(nsub),
                "NumFiles": np.int32(1),
                "Omega0": 0.3089,
                "OmegaLambda": 0.6911,
                "Redshift": REDSHIFT,
                "Time": 1.0 / (1.0 + REDSHIFT),
            }
        )

        group = f.create_group("Group")
        write(group, "GroupFirstSub", first_sub, NO_UNITS)
        write(group, "GroupLen", group_len.sum(axis=1).astype(np.int32), NO_UNITS)
        write(group, "GroupLenType", group_len, NO_UNITS)
        write(group, "GroupMass", group_mass_type.sum(axis=1), MASS_UNITS)
        write(group, "GroupMassType", group_mass_type, MASS_UNITS)
        write(group, "GroupNsubs", np.bincount(sub_group, minlength=ngroups).astype(np.int32), NO_UNITS)
        write(group, "GroupPos", group_pos, POS_UNITS)

        subhalo = f.create_group("Subhalo")
        write(subhalo, "SubhaloFlag", np.array([sub["flag"] for sub in SUBHALOS], dtype=np.uint8), NO_UNITS)
        write(subhalo, "SubhaloGrNr", sub_group, NO_UNITS)
        write(subhalo, "SubhaloLen", sub_len.sum(axis=1).astype(np.int32), NO_UNITS)
        write(subhalo, "SubhaloLenType", sub_len, NO_UNITS)
        write(subhalo, "SubhaloMass", sub_mass_type.sum(axis=1), MASS_UNITS)
        write(subhalo, "SubhaloMassType", sub_mass_type, MASS_UNITS)
        write(subhalo, "SubhaloPos", sub_pos, POS_UNITS)
        write(subhalo, "SubhaloSFR", np.array([sub["sfr"] for sub in SUBHALOS], dtype=np.float32), SFR_UNITS)
        write(subhalo, "SubhaloVel", np.array([sub["vel"] for sub in SUBHALOS], dtype=np.float32), VEL_UNITS)


def main() -> None:
    rng = np.random.default_rng(SEED)
    particles, sub_len, group_len = build(rng)
    snapshot_dir = OUTPUT_PATH / f"snapdir_{SNAPSHOT:03d}"
    catalogue_path = OUTPUT_PATH / f"groups_{SNAPSHOT:03d}" / f"fof_subhalo_tab_{SNAPSHOT:03d}.0.hdf5"
    write_snapshot(snapshot_dir, snapshot_datasets(particles, rng))
    write_catalogue(catalogue_path, particles, sub_len, group_len)
    print(f"Written {NUM_FILES} snapshot files to {snapshot_dir} and {catalogue_path}")


if __name__ == "__main__":
    main()
