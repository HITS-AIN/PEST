from collections.abc import Iterable

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
        arrow_type = pa.from_numpy_dtype(np.asarray(sample).dtype)
        for _ in range(np.ndim(sample)):
            arrow_type = pa.list_(arrow_type)
        return pa.array([np.asarray(v).tolist() for v in values], type=arrow_type)
    return pa.array(values)


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
    dataset never needs to be held in memory at once.

    Args:
        output_path (str): Destination path for the Parquet file.
        chunk_size (int | None): Number of records per row group. Defaults to
            writing every record in a single row group.
    """

    def __init__(self, output_path: str, chunk_size: int | None = None):
        self.output_path = output_path
        self.chunk_size = chunk_size

    def __call__(self, records: Iterable[dict]) -> None:
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
                # No records at all: still produce an (empty) file.
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
