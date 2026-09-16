from pest import PointCloudGenerator
from pest.galaxy import Galaxy, Gas, Star


def test_point_cloud_generation(tmp_path):
    """Test the point cloud generation from galaxy data."""
    star = Star(id=1, mass=1.0, position=[0.0, 0.0, 0.0], velocity=[0.0, 0.0, 0.0], luminosity=10.0)
    gas = Gas(id=2, mass=0.5, position=[1.0, 0.0, 0.0], velocity=[0.0, 0.1, 0.0], temperature=500.0)

    galaxy_1 = Galaxy(
        id=1,
        central=True,
        mass=1.0,
        position=[0.0, 0.0, 0.0],
        velocity=[0.0, 0.0, 0.0],
        stars=[star],
    )
    galaxy_2 = Galaxy(
        id=2,
        central=True,
        mass=2.0,
        position=[0.0, 0.0, 1.0],
        velocity=[0.0, 0.0, 0.0],
        gas=[gas],
    )

    generator = PointCloudGenerator(
        output_directory=str(tmp_path),
        columns=["id", "stars_position", "stars_mass", "stars_luminosity"],
    )
    generator.process(galaxy_1)
    generator.process(galaxy_2)

    assert (tmp_path / "part-0.parquet").exists(), "Parquet file was not created."
