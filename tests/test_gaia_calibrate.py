from pyarrow import parquet

from pest import gaia_calibrate


def test_gaia_calibrate(tmp_path):

    gaia_calibrate(
        input_path="tests/data/gaia",
        output_path=tmp_path,
        number_of_workers=1,
    )

    assert tmp_path.joinpath("XpContinuousMeanSpectrum_000000-003111.parquet").exists()

    table = parquet.read_table(
        tmp_path.joinpath("XpContinuousMeanSpectrum_000000-003111.parquet")
    )
    assert table.schema.metadata[b"flux_shape"] == b"(1, 343)"
