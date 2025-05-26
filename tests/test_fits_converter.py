from pyarrow import parquet

from pest import FitsConverter

# def test_fits_converter_file(tmp_path):

#     fits_converter = FitsConverter()

#     input_file = "tests/data/fits/TNG100/sdss/snapnum_099/data/broadband_6.fits"
#     output_file = tmp_path.joinpath("0.parquet")

#     fits_converter.convert(input_file, output_file)
#     assert output_file.exists()

#     table = parquet.read_table(output_file)
#     assert table.schema.metadata[b"flux_shape"] == b"(3, 128, 128)"


def test_fits_converter_directory(tmp_path):
    fits_converter = FitsConverter(image_size=128)
    fits_converter.convert_all("tests/data/fits/TNG100/sdss/snapnum_099/data/", tmp_path)

    output_file = tmp_path.joinpath("0.parquet")
    assert output_file.exists()

    table = parquet.read_table(output_file)
    assert table.schema.metadata[b"data_shape"] == b"(3, 128, 128)"
    assert len(table) == 2
