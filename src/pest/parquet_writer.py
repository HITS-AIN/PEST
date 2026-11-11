import warnings
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _column_array(values: list) -> pa.Array:
    """Build a pyarrow array for one column.

    Numpy array/scalar values keep their original dtype and nesting (so images
    stay e.g. float32 instead of being widened to float64), everything else is
    left to pyarrow's normal type inference.
    """
    sample = values[0]
    if isinstance(sample, (np.ndarray, np.generic)):
        dtype = np.asarray(sample).dtype
        ndim = np.ndim(sample)
        if ndim > 0 and dtype != object and all(isinstance(v, np.ndarray) and v.ndim == ndim for v in values):
            return _nested_list_array([np.asarray(v, dtype=dtype) for v in values], pa.from_numpy_dtype(dtype))
        arrow_type = pa.from_numpy_dtype(dtype)
        for _ in range(ndim):
            arrow_type = pa.list_(arrow_type)
        return pa.array([np.asarray(v).tolist() if v is not None else None for v in values], type=arrow_type)
    return pa.array(values)


def _nested_list_array(arrays: list[np.ndarray], value_type: pa.DataType) -> pa.Array:
    """Nested lists (one nesting level per dimension) of n-dimensional numpy arrays of equal ndim.

    Same result as `pa.array([a.tolist() for a in arrays])`, but built from the
    flat numpy buffers instead of Python lists, which take about ten times the
    memory of the arrays (e.g. 8 MB instead of 0.8 MB for a 3x256x256 image).
    The shapes may differ between the arrays, e.g. point clouds of different sizes.
    """
    result = pa.array(np.concatenate([a.ravel() for a in arrays]), type=value_type)
    shapes = np.array([a.shape for a in arrays], dtype=np.int64)
    # From the innermost dimension outwards: array i has prod(shape[:d]) lists of length shape[d] at dimension d.
    for d in reversed(range(shapes.shape[1])):
        counts = np.prod(shapes[:, :d], axis=1)
        lengths = np.repeat(shapes[:, d], counts)
        offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        if offsets[-1] > np.iinfo(np.int32).max:
            raise ValueError(
                "Too many values in one row group of the Parquet file, reduce the chunk_size of the ParquetWriter."
            )
        result = pa.ListArray.from_arrays(pa.array(offsets.astype(np.int32)), result)
    return result


def _column_name(column) -> str:
    """Flatten a (possibly nested) record key into a Parquet column name."""
    return column if isinstance(column, str) else "_".join(column)


def _records_to_table(records: list[dict]) -> pa.Table:
    columns = records[0].keys()
    arrays = [_column_array([record[column] for record in records]) for column in columns]
    return pa.Table.from_arrays(arrays, names=[_column_name(column) for column in columns])


class ParquetWriter:
    """Write a stream of transformed records to a Parquet file.

    Records are written in row-group batches as they arrive, so the whole
    dataset never needs to be held in memory at once. If the pipeline fails
    with an exception, the file is closed with the records written so far.

    Args:
        output_path (str): Destination path for the Parquet file.
        chunk_size (int | None): Number of records per row group (default: 256).
            `None` writes every record in a single row group, holding all of
            them in memory.
    """

    def __init__(self, output_path: str, chunk_size: int | None = 256):
        self.output_path = output_path
        self.chunk_size = chunk_size

    def __call__(self, records: Iterable[dict]) -> None:
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        writer = None
        batch: list[dict] = []
        chunk_size = self.chunk_size or float("inf")
        try:
            for record in records:
                batch.append(record)
                if len(batch) >= chunk_size:
                    writer = self._write_batch(writer, batch)
                    batch = []
            if batch:
                writer = self._write_batch(writer, batch)
            if writer is None:
                # No records at all: still produce an (empty) file, but it has no
                # columns, so readers such as DuckDB reject it.
                warnings.warn(
                    f"No records to write, {self.output_path} will be an empty file without columns. "
                    "Check whether the filters in the pipeline drop every record.",
                    stacklevel=2,
                )
                pq.write_table(pa.table({}), self.output_path)
        finally:
            if writer is not None:
                writer.close()

    def _write_batch(self, writer: pq.ParquetWriter | None, batch: list[dict]) -> pq.ParquetWriter:
        table = _records_to_table(batch)
        if writer is None:
            writer = pq.ParquetWriter(self.output_path, table.schema)
        writer.write_table(table)
        return writer
