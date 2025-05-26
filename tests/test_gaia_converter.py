from pyarrow import parquet

from pest import GaiaConverter


def test_gaia_converter_file(tmp_path):
    gaia_converter = GaiaConverter()

    input_file = "tests/data/gaia/XpContinuousMeanSpectrum_000000-003111.csv.gz"
    output_file = tmp_path.joinpath("XpContinuousMeanSpectrum_000000-003111.parquet")

    gaia_converter.convert(input_file, output_file)
    assert output_file.exists()

    table = parquet.read_table(output_file)
    assert table.schema.metadata[b"flux_shape"] == b"(1, 343)"


def test_gaia_converter_directory(tmp_path):
    gaia_converter = GaiaConverter()
    gaia_converter.convert_all("tests/data/gaia", tmp_path)

    output_file = tmp_path.joinpath("XpContinuousMeanSpectrum_000000-003111.parquet")
    assert output_file.exists()

    table = parquet.read_table(output_file)
    assert table.schema.metadata[b"flux_shape"] == b"(1, 343)"
