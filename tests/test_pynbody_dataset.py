import numpy as np
import pandas as pd
import pynbody
import pytest

import pest.pynbody_dataset
from pest import FilterRange, LoadParticles, Pipeline, PynbodyDataset, RenderStars

# 5 subhalos: stellar masses [Msun] and flags
STELLAR_MASS = np.array([1e8, 5e9, 2e10, 3e10, 1e12])
SUBHALO_FLAG = np.array([1, 1, 0, 1, 1])


def make_stars(pos, vel, mass):
    """A pynbody snapshot of star particles."""
    stars = pynbody.new(star=len(mass))
    stars["pos"] = pynbody.array.SimArray(pos, "kpc")
    stars["vel"] = pynbody.array.SimArray(vel, "km s**-1")
    stars["mass"] = pynbody.array.SimArray(mass, "Msol")
    return stars


class FakeHalo:
    def __init__(self, subhalo_id):
        n = subhalo_id + 1
        pos = np.full((n, 3), 10.0 * subhalo_id + 1.0)
        self.st = make_stars(pos, pos * 2, np.ones(n))

    def physical_units(self):
        pass


class FakeNumberMapper:
    def number_to_index(self, number):
        return number


class FakeHalos:
    def __init__(self):
        self.accessed = []
        self.number_mapper = FakeNumberMapper()

    def physical_units(self):
        pass

    def keys(self):
        return list(range(len(STELLAR_MASS)))

    def get_properties_all_halos(self):
        n = len(STELLAR_MASS)
        mass_type = np.zeros((n, 6))
        mass_type[:, 4] = STELLAR_MASS
        pos = 10.0 * np.arange(n)[:, None] * np.ones((1, 3))
        return {
            "SubhaloMassType": mass_type,
            "SubhaloMass": 10 * STELLAR_MASS,
            "SubhaloFlag": SUBHALO_FLAG,
            "SubhaloPos": pos,
            "SubhaloVel": 2 * pos,
            "SubhaloSFR": np.arange(n, dtype=float),
        }

    def __getitem__(self, subhalo_id):
        raise AssertionError("indexing the catalogue gives a view that loads arrays of the full snapshot")

    def load_copy(self, subhalo_id):
        self.accessed.append(subhalo_id)
        return FakeHalo(subhalo_id)


class FakeSnapshot:
    def __init__(self):
        self.fake_halos = FakeHalos()
        self.properties = {}

    def physical_units(self):
        pass

    def conversion_context(self):
        return {}

    def halos(self, **kwargs):
        return self.fake_halos


@pytest.fixture
def fake_pynbody(monkeypatch):
    snapshot = FakeSnapshot()
    monkeypatch.setattr(pest.pynbody_dataset.pynbody, "load", lambda path: snapshot)
    return snapshot


MASS_FILTER = {
    "class_path": "pest.FilterRange",
    "init_args": {"column": "SubhaloMassType", "index": 4, "min": 1e9, "max": 1e11, "units": "Msol"},
}
FLAG_FILTER = {"class_path": "pest.FilterRange", "init_args": {"column": "SubhaloFlag", "min": 1}}
LOAD_PARTICLES = {"class_path": "pest.LoadParticles", "init_args": {"fields": ["pos", "vel", "mass"]}}


def _run_pipeline(tmp_path, transform, columns=("subhalo_id", "snapshot", "SubhaloSFR")):
    output_path = tmp_path / "out.parquet"
    config = {
        "num_workers": 1,
        "shuffle": False,
        "extract": {
            "class_path": "pest.PynbodyDataset",
            "init_args": {"snapshot_path": "TNG50-1/snapshot_099/snap_099", "columns": list(columns)},
        },
        "transform": transform,
        "load": [{"class_path": "pest.ParquetWriter", "init_args": {"output_path": str(output_path)}}],
    }
    Pipeline(config).run()
    return pd.read_parquet(output_path)


def test_dataset_reads_catalogue_only(fake_pynbody):
    dataset = PynbodyDataset("TNG50-1/snapshot_099/snap_099", columns=["subhalo_id", "snapshot", "SubhaloSFR"])

    assert len(dataset) == 5
    item = dataset[3]
    assert item == {"subhalo_id": 3, "snapshot": 99, "SubhaloSFR": 3.0}
    assert fake_pynbody.fake_halos.accessed == []


def test_particle_columns_rejected(fake_pynbody):
    with pytest.raises(ValueError, match="LoadParticles"):
        PynbodyDataset("snap_099", columns=["subhalo_id", "pos"])


def test_particles(fake_pynbody):
    dataset = PynbodyDataset("snap_099")

    particles = dataset.particles(3, "stars", ["pos", "vel", "mass"])
    assert particles["pos"].shape == (4, 3)
    assert particles["pos"].dtype == np.float32
    np.testing.assert_allclose(particles["pos"], 1.0)
    np.testing.assert_allclose(particles["vel"], 2.0)
    np.testing.assert_allclose(particles["mass"], 1.0)
    assert fake_pynbody.fake_halos.accessed == [3]


def test_particles_wrap_periodic_box(fake_pynbody, monkeypatch):
    fake_pynbody.properties["boxsize"] = pynbody.array.SimArray(100.0, "kpc")
    halo = FakeHalo(3)
    pos = pynbody.array.SimArray([[99.0, 1.0, 50.0], [2.0, 98.0, 50.0]], "kpc")
    halo.st = make_stars(pos, np.zeros((2, 3)), np.ones(2))
    monkeypatch.setattr(fake_pynbody.fake_halos, "load_copy", lambda subhalo_id: halo)
    properties = fake_pynbody.fake_halos.get_properties_all_halos()
    properties["SubhaloPos"][3] = [1.0, 99.0, 50.0]
    monkeypatch.setattr(fake_pynbody.fake_halos, "get_properties_all_halos", lambda: properties)
    dataset = PynbodyDataset("snap_099")

    particles = dataset.particles(3, "stars", ["pos"])
    np.testing.assert_allclose(particles["pos"], [[-2.0, 2.0, 0.0], [1.0, -1.0, 0.0]])


def test_halo_numbers_differ_from_index(fake_pynbody, monkeypatch):
    # e.g. EAGLE group numbers start at 1
    monkeypatch.setattr(fake_pynbody.fake_halos, "keys", lambda: list(range(1, len(STELLAR_MASS) + 1)))
    monkeypatch.setattr(fake_pynbody.fake_halos.number_mapper, "number_to_index", lambda number: number - 1)
    dataset = PynbodyDataset("snap_099", columns=["subhalo_id", "SubhaloSFR"])

    assert len(dataset) == 5
    assert dataset[0] == {"subhalo_id": 1, "SubhaloSFR": 0.0}
    assert dataset[4] == {"subhalo_id": 5, "SubhaloSFR": 4.0}

    # Particles of halo number 3 are centered on the catalogue entry at index 2.
    particles = dataset.particles(3, "stars", ["pos", "vel"])
    np.testing.assert_allclose(particles["pos"], 11.0)
    np.testing.assert_allclose(particles["vel"], 22.0)
    assert fake_pynbody.fake_halos.accessed == [3]


def test_halo(fake_pynbody):
    dataset = PynbodyDataset("snap_099")

    halo = dataset.halo(np.int32(3))
    assert isinstance(halo, FakeHalo)
    assert len(halo.st["mass"]) == 4
    assert fake_pynbody.fake_halos.accessed == [3]


def test_stars_from_halo(fake_pynbody):
    dataset = PynbodyDataset("snap_099")

    stars = dataset.stars(np.int32(3))
    # Not an IllustrisTNG-like snapshot, so the stars come from the halo loaded by pynbody.
    assert fake_pynbody.fake_halos.accessed == [3]
    assert len(stars) == 4
    np.testing.assert_allclose(stars["pos"], 1.0)


def test_invalid_component(fake_pynbody):
    dataset = PynbodyDataset("snap_099")
    with pytest.raises(ValueError):
        dataset.particles(0, "bh", ["pos"])


def test_mass_filter_on_table(fake_pynbody):
    dataset = PynbodyDataset("snap_099")

    stellar = FilterRange("SubhaloMassType", index=4, min=1e9, max=1e11).mask(dataset.table())
    total = FilterRange("SubhaloMass", min=1e11).mask(dataset.table())
    assert list(np.flatnonzero(stellar)) == [1, 2, 3]
    assert list(np.flatnonzero(total)) == [2, 3, 4]


def test_mass_filter_converts_units(fake_pynbody, monkeypatch):
    properties = fake_pynbody.fake_halos.get_properties_all_halos()
    properties["SubhaloMassType"] = pynbody.array.SimArray(properties["SubhaloMassType"] / 1e10, "1e10 Msol")
    monkeypatch.setattr(fake_pynbody.fake_halos, "get_properties_all_halos", lambda: properties)
    dataset = PynbodyDataset("snap_099")

    mask = FilterRange("SubhaloMassType", index=4, min=1e9, max=1e11, units="Msol").mask(dataset.table())
    assert list(np.flatnonzero(mask)) == [1, 2, 3]


def test_pipeline_loads_particles_only_for_selected(fake_pynbody, tmp_path):
    df = _run_pipeline(tmp_path, [MASS_FILTER, FLAG_FILTER, LOAD_PARTICLES])

    assert list(df["subhalo_id"]) == [1, 3]
    assert list(df["SubhaloSFR"]) == [1.0, 3.0]
    assert set(df.columns) == {"subhalo_id", "snapshot", "SubhaloSFR", "pos", "vel", "mass"}
    np.testing.assert_allclose(np.stack(df.iloc[1]["pos"]), np.ones((4, 3)))
    assert fake_pynbody.fake_halos.accessed == [1, 3]


def test_pipeline_filter_after_loading(fake_pynbody, tmp_path):
    late_filter = {"class_path": "pest.FilterRange", "init_args": {"column": "SubhaloSFR", "max": 2.0}}
    df = _run_pipeline(tmp_path, [MASS_FILTER, LOAD_PARTICLES, late_filter])

    # SubhaloFlag is not applied here, so subhalo 2 survives the mass filter.
    assert list(df["subhalo_id"]) == [1, 2]
    assert fake_pynbody.fake_halos.accessed == [1, 2, 3]


def test_load_particles_requires_binding():
    with pytest.raises(RuntimeError):
        LoadParticles(fields=["pos"])({"subhalo_id": 0})


@pytest.fixture
def fake_render(monkeypatch):
    calls = []

    def render(halo, **kwargs):
        calls.append(("render", halo, kwargs))
        image = np.zeros((4, 4, 3))
        image[0] = 1.0  # first row = lowest y in pynbody
        return image

    monkeypatch.setattr(pynbody.analysis, "center", lambda halo, **kw: calls.append(("center", halo, kw)))
    monkeypatch.setattr(pynbody.plot.stars, "render", render)
    return calls


def test_render_stars(fake_pynbody, fake_render):
    step = RenderStars(width="30 kpc", resolution=4, center_mode="ssc", render_args={"dynamic_range": 3.0})
    step.bind(PynbodyDataset("snap_099"))

    record = step({"subhalo_id": np.int32(3)})

    assert record["image"].shape == (3, 4, 4)
    assert record["image"].dtype == np.float32
    np.testing.assert_allclose(record["image"][:, -1], 1.0)
    assert [call[0] for call in fake_render] == ["center", "render"]
    _, centered_stars, center_kwargs = fake_render[0]
    _, rendered_stars, render_kwargs = fake_render[1]
    assert centered_stars is rendered_stars
    assert center_kwargs == {"mode": "ssc", "with_velocity": False, "cen_num_particles": 0, "move_all": False}
    assert render_kwargs == {
        "width": "30 kpc",
        "resolution": 4,
        "noplot": True,
        "return_image": True,
        "dynamic_range": 3.0,
    }
    # Relative to the catalogue center, with a placeholder for the SPH density.
    np.testing.assert_allclose(rendered_stars["pos"], 1.0)
    np.testing.assert_allclose(rendered_stars["rho"], 1.0)


def test_render_stars_catalogue_center(fake_pynbody, fake_render):
    step = RenderStars(resolution=4, center_mode="catalogue")
    step.bind(PynbodyDataset("snap_099"))

    step({"subhalo_id": 3})
    assert [call[0] for call in fake_render] == ["render"]


@pytest.mark.parametrize("kwargs", [{"center_mode": "pot"}, {"render_args": {"with_dust": True}}])
def test_render_stars_rejects_unsupported(kwargs):
    with pytest.raises(ValueError):
        RenderStars(**kwargs)


def test_pipeline_renders_stars(fake_pynbody, fake_render, tmp_path):
    render = {"class_path": "pest.RenderStars", "init_args": {"resolution": 4}}
    df = _run_pipeline(tmp_path, [MASS_FILTER, FLAG_FILTER, render])

    assert list(df["subhalo_id"]) == [1, 3]
    assert np.stack([np.stack(row) for row in df.iloc[0]["image"]]).shape == (3, 4, 4)


def test_render_stars_without_stars(fake_pynbody, fake_render, monkeypatch):
    halo = FakeHalo(0)
    halo.st = make_stars(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0))
    monkeypatch.setattr(fake_pynbody.fake_halos, "load_copy", lambda subhalo_id: halo)
    step = RenderStars()
    step.bind(PynbodyDataset("snap_099"))

    with pytest.raises(ValueError, match="no star particles"):
        step({"subhalo_id": 0})
    assert fake_render == []


def test_render_stars_requires_binding():
    with pytest.raises(RuntimeError):
        RenderStars()({"subhalo_id": 0})


def test_render_stars_requires_stars():
    with pytest.raises(TypeError):
        RenderStars().bind(object())


# Synthetic TNG snapshot (in 2 files) and SUBFIND catalogue, generated by tests/make_illustris_tng_testdata.py
TNG_SNAPSHOT = "tests/data/illustris_tng/TNG50-1/snapdir_099/snap_099"


@pytest.mark.filterwarnings("ignore")
def test_tng_catalogue():
    dataset = PynbodyDataset(TNG_SNAPSHOT, columns=["subhalo_id", "snapshot", "SubhaloSFR", "SubhaloFlag"])

    assert type(dataset.halos).__name__ == "TNGSubfindHDFCatalogue"
    assert len(dataset) == 5
    assert dataset[3] == {"subhalo_id": 3, "snapshot": 99, "SubhaloSFR": np.float32(1.2), "SubhaloFlag": 1}
    mask = FilterRange("SubhaloMassType", index=4, min=2e9, units="Msol").mask(dataset.table())
    assert list(np.flatnonzero(mask)) == [0, 3]


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("subhalo_id, n_stars", [(0, 64), (2, 0), (3, 15), (4, 5)])
def test_tng_particles(subhalo_id, n_stars):
    # Subhalos 3 and 4 cross the periodic box boundary.
    dataset = PynbodyDataset(TNG_SNAPSHOT)

    particles = dataset.particles(subhalo_id, "stars", ["pos", "vel", "mass"])
    assert particles["pos"].shape == (n_stars, 3)
    assert np.all(np.abs(particles["pos"]) < 20.0)  # [kpc]
    assert np.all(np.abs(particles["vel"]) < 500.0)  # [km/s]
    assert np.all(particles["mass"] > 0)


@pytest.mark.filterwarnings("ignore")
def test_tng_render_stars():
    dataset = PynbodyDataset(TNG_SNAPSHOT)
    step = RenderStars(resolution=16)
    step.bind(dataset)

    assert step(dataset[0])["image"].shape == (3, 16, 16)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("subhalo_id", range(5))
def test_tng_stars_direct_matches_halo(subhalo_id):
    dataset = PynbodyDataset(TNG_SNAPSHOT)
    assert len(dataset.snapshot._hdf_files) == 2

    direct = dataset.stars(subhalo_id)
    dataset.direct_loading = False
    loaded = dataset.stars(subhalo_id)

    assert len(direct) == len(loaded)
    for name in ["pos", "mass", "aform", "metals"]:
        np.testing.assert_array_equal(direct[name], loaded[name])
        assert direct[name].units == loaded[name].units
    np.testing.assert_allclose(direct["age"].in_units("Gyr"), loaded["age"].in_units("Gyr"), equal_nan=True)
    # Subhalo 3 crosses the periodic box boundary.
    assert np.all(np.abs(direct["pos"]) < 20.0)  # [kpc]


@pytest.mark.filterwarnings("ignore")
def test_tng_stars_read_directly(monkeypatch):
    dataset = PynbodyDataset(TNG_SNAPSHOT)
    monkeypatch.setattr(dataset, "halo", lambda subhalo_id: pytest.fail("stars loaded with pynbody"))

    assert len(dataset.stars(0)) == 64


@pytest.mark.filterwarnings("ignore")
def test_tng_stars_wind_particles():
    stars = PynbodyDataset(TNG_SNAPSHOT).stars(0)

    # TNG stores wind particles with the stars, with a negative formation time; they have no age.
    wind = np.asarray(stars["aform"]) < 0
    assert list(np.flatnonzero(wind)) == [5, 30, 50]
    np.testing.assert_array_equal(np.isnan(stars["age"]), wind)


@pytest.mark.filterwarnings("ignore")
def test_tng_stars_box_boundary():
    # Subhalo 3 crosses the periodic box boundary, so its stars must be wrapped around the center to find it.
    stars = PynbodyDataset(TNG_SNAPSHOT).stars(3)

    center = pynbody.analysis.center(stars, with_velocity=False, cen_num_particles=0, return_cen=True)
    assert np.all(np.abs(center) < 5.0)  # [kpc]


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("center_mode", [None, "catalogue"])
def test_tng_render_stars_direct_matches_halo(center_mode):
    dataset = PynbodyDataset(TNG_SNAPSHOT)
    step = RenderStars(width=20, resolution=16, center_mode=center_mode)
    step.bind(dataset)

    direct = step({"subhalo_id": 0})["image"]
    dataset.direct_loading = False
    loaded = step({"subhalo_id": 0})["image"]
    np.testing.assert_allclose(direct, loaded, atol=1e-6)


@pytest.mark.filterwarnings("ignore")
def test_tng_render_stars_rho_placeholder():
    dataset = PynbodyDataset(TNG_SNAPSHOT)
    step = RenderStars(width=20, resolution=16, center_mode="catalogue")
    step.bind(dataset)

    image = step({"subhalo_id": 0})["image"]
    # The same stars, rendered with the SPH density computed by pynbody.
    expected = pynbody.plot.stars.render(dataset.stars(0), width=20, resolution=16, noplot=True, return_image=True)
    np.testing.assert_allclose(image, np.moveaxis(expected[::-1], -1, 0), atol=1e-6)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(
    "component, fields",
    [
        ("stars", ["pos", "vel", "mass", "metals", "aform"]),
        ("gas", ["pos", "vel", "mass", "metals"]),
        ("dm", ["pos", "vel", "mass", "iord"]),
    ],
)
@pytest.mark.parametrize("subhalo_id", range(5))
def test_tng_particles_direct_matches_halo(component, fields, subhalo_id):
    dataset = PynbodyDataset(TNG_SNAPSHOT)

    direct = dataset.particles(subhalo_id, component, fields)
    dataset.direct_loading = False
    loaded = dataset.particles(subhalo_id, component, fields)

    for field in fields:
        np.testing.assert_array_equal(direct[field], loaded[field])


@pytest.mark.filterwarnings("ignore")
def test_tng_particles_read_directly(monkeypatch):
    dataset = PynbodyDataset(TNG_SNAPSHOT)
    monkeypatch.setattr(dataset, "halo", lambda subhalo_id: pytest.fail("particles loaded with pynbody"))

    # Dark matter masses are only stored in the header of the snapshot files.
    particles = dataset.particles(0, "dm", ["pos", "vel", "mass"])
    assert particles["pos"].shape == (40, 3)
    assert np.all(particles["mass"] > 0)


@pytest.mark.filterwarnings("ignore")
def test_tng_particles_derived_field(monkeypatch):
    dataset = PynbodyDataset(TNG_SNAPSHOT)
    loaded = []
    halo = dataset.halo
    monkeypatch.setattr(dataset, "halo", lambda subhalo_id: loaded.append(subhalo_id) or halo(subhalo_id))

    # Ages are derived by pynbody, not stored in the snapshot files.
    ages = dataset.particles(0, "stars", ["mass", "age"])["age"]
    assert loaded == [0]
    assert np.isnan(ages).sum() == 3  # the wind particles
