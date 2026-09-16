import os

import pyarrow as pa
import pyarrow.parquet as pq

from pest.galaxy import Galaxy
from pest.generator import Generator


class PointCloudGenerator(Generator):
    def __init__(
        self,
        output_directory: str,
        columns: list[str],
        chunk_size: int = 1000,
        compression: str = "snappy",
    ):
        """Initialize the PointCloudGenerator."""

        super().__init__()
        self.columns = columns
        self.chunk_size = chunk_size
        self.compression = compression
        self.output_directory = output_directory
        self._writer = None
        self._idx = 0
        self._file_idx = 0

        self.schema = pa.schema([])

        fixed_field_types = {
            "id": pa.int64(),
            "stars_position": pa.list_(pa.list_(pa.float32(), list_size=3)),
            "stars_mass": pa.list_(pa.float32()),
            "stars_luminosity": pa.list_(pa.float32()),
            "gas_position": pa.list_(pa.list_(pa.float32(), list_size=3)),
            "gas_mass": pa.list_(pa.float32()),
            "gas_temperature": pa.list_(pa.float32()),
        }
        for column in self.columns:
            self.schema = self.schema.append(pa.field(column, fixed_field_types[column]))

        os.makedirs(output_directory, exist_ok=True)

    def process(self, galaxy: Galaxy):
        """Process a galaxy and write it immediately as a new row to the Parquet file."""
        galaxy_data = {}

        for column in self.columns:
            parts = column.split("_", 1)
            if len(parts) == 2:
                particle_type, attribute = parts
                values = []
                for particle in getattr(galaxy, particle_type):
                    if hasattr(particle, attribute):
                        values.append(getattr(particle, attribute))
                galaxy_data[column] = values
            else:
                if hasattr(galaxy, column):
                    galaxy_data[column] = getattr(galaxy, column)

        # Write this galaxy as a single row to the Parquet file
        table = pa.Table.from_pylist([galaxy_data], schema=self.schema)
        if self._writer is None:
            self._writer = pq.ParquetWriter(
                os.path.join(self.output_directory, f"part-{self._file_idx}.parquet"),
                self.schema,
                compression=self.compression,
            )
        self._writer.write_table(table)

        # Increment the index and file index
        self._idx += 1
        if self._idx >= self.chunk_size:
            self._idx = 0
            self._file_idx += 1
            self._writer.close()
            self._writer = None

    def close(self):
        """Finalize the output dataset by closing the Parquet writer."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
