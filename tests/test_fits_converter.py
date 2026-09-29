from pyarrow import parquet

from pest import FitsConverter


def test_fits_converter_directory(tmp_path):
    fits_converter = FitsConverter(image_size=128, datatype="float32", flatten=True)
    fits_converter.convert_all("tests/data/illustris_tng_skirt/TNG50/sdss/snapshot_095/data/", tmp_path)

    output_file = tmp_path.joinpath("0.parquet")
    assert output_file.exists()

    table = parquet.read_table(output_file)
    assert table.schema.metadata[b"data_shape"] == b"(3, 128, 128)"
    assert len(table) == 1
