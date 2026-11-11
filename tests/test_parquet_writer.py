from typing import ClassVar

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pest import Pipeline
from pest.parquet_writer import ParquetWriter, _column_array


def _reference(values: list) -> pa.Array:
    """The array built through Python lists, as `ParquetWriter` did before."""
    arrow_type = pa.from_numpy_dtype(values[0].dtype)
    for _ in range(values[0].ndim):
        arrow_type = pa.list_(arrow_type)
    return pa.array([v.tolist() for v in values], type=arrow_type)


def test_column_array_images():
    rng = np.random.default_rng(0)
    values = [rng.random((3, 4, 5), dtype=np.float32) for _ in range(3)]
    array = _column_array(values)
    assert array.type == pa.list_(pa.list_(pa.list_(pa.float32())))
    assert array.equals(_reference(values))


def test_column_array_varying_shapes():
    values = [np.arange(6, dtype=np.float64).reshape(2, 3), np.zeros((0, 3)), np.arange(4.0).reshape(4, 1)]
    assert _column_array(values).equals(_reference(values))


def test_column_array_scalars_and_none():
    assert _column_array([np.float32(1.5), np.float32(2.5)]).to_pylist() == [1.5, 2.5]
    assert _column_array([np.arange(3), None]).to_pylist() == [[0, 1, 2], None]


def test_parquet_writer_row_groups(tmp_path):
    output_path = tmp_path / "out.parquet"
    records = ({"id": np.int32(i), "image": np.full((2, 2), i, dtype=np.float32)} for i in range(5))

    ParquetWriter(str(output_path), chunk_size=2)(records)

    file = pq.ParquetFile(output_path)
    assert file.metadata.num_row_groups == 3
    table = file.read()
    assert table["id"].to_pylist() == list(range(5))
    assert table["image"].to_pylist()[4] == [[4.0, 4.0], [4.0, 4.0]]


class _Records:
    """Dataset of `n` records holding their index."""

    def __init__(self, n: int):
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> dict:
        return {"id": np.int32(index)}


class _Recorder:
    """Load step remembering which records it got, and whether they came as a stream."""

    received: ClassVar[list] = []

    def __init__(self, consumes_records: bool = True):
        self.consumes_records = consumes_records

    def __call__(self, records) -> None:
        _Recorder.received.append((isinstance(records, list), [int(r["id"]) for r in records]))


@pytest.mark.parametrize("num_workers", [1, 2])
def test_pipeline_streams_records(tmp_path, num_workers):
    output_path = tmp_path / "out.parquet"
    config = {
        "num_workers": num_workers,
        "batch_size": 2,
        "shuffle": False,
        "extract": {"class_path": "test_parquet_writer._Records", "init_args": {"n": 7}},
        "load": [
            {"class_path": "pest.ParquetWriter", "init_args": {"output_path": str(output_path), "chunk_size": 3}},
            {"class_path": "test_parquet_writer._Recorder", "init_args": {"consumes_records": False}},
        ],
    }
    _Recorder.received = []
    Pipeline(config).run()
    assert pq.read_table(output_path)["id"].to_pylist() == list(range(7))
    # The load step not consuming the records gets the exhausted stream.
    assert _Recorder.received == [(False, [])]

    # With several load steps consuming the records, each of them gets all of them.
    config["load"][1]["init_args"]["consumes_records"] = True
    _Recorder.received = []
    Pipeline(config).run()
    assert _Recorder.received == [(True, list(range(7)))]
    assert pq.read_table(output_path)["id"].to_pylist() == list(range(7))
