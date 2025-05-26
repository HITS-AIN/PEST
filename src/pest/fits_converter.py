"""Convert FITS files to Parquet format."""

import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.io import fits
from skimage.transform import resize

from pest.converter import Converter
from pest.preprocessing import CreateNormalizedRGBColors


class FitsConverter(Converter):
    """Convert FITS files to Parquet format."""

    def __init__(
        self,
        image_size: int = 128,
        chunk_size: int = 1000,
        number_of_workers: int = 1,
    ):
        """Initialize the FitsConverter.

        Args:
            image_size (int): Size of the images to be converted (default: 128).
            chunk_size (int): Size of row chunks of parquet files (default: 1000).
            number_of_workers (int): Number of workers to use for conversion (default: 1).
        """
        self.image_size = image_size
        self.chunk_size = chunk_size
        self.number_of_workers = number_of_workers

        self.normalize_rgb = CreateNormalizedRGBColors(
            stretch=0.9,
            range=5,
            lower_limit=0.001,
            channel_combinations=[[2, 3], [1, 0], [0]],
            scalers=[0.7, 0.5, 1.3],
        )

    def convert(
        self,
        input_file: str,
        output_file: str,
    ):
        pass

    def convert_all(
        self,
        input_directories: str | list[str],
        output_directory: str,
    ):
        """Convert all FITS files in the input directory to Parquet format.

        Args:
            input_directories (str | list[str]): Path to the directory or list of directories containing FITS files.
            output_directory (str): Path to the directory where the Parquet files will be saved.
        """
        os.makedirs(output_directory, exist_ok=True)

        if isinstance(input_directories, str):
            input_directories = [input_directories]

        writer = None

        # Iterate over all input directories
        for input_directory in input_directories:
            for filename in sorted(os.listdir(input_directory)):
                if filename.endswith(".fits"):
                    filename = os.path.join(input_directory, filename)
                    splits = filename[: -len(".fits")].split("/")

                    data = fits.getdata(filename, 0)
                    data = np.array(data).astype(np.float32)
                    data = self.normalize_rgb(data)
                    data = resize(data, (3, self.image_size, self.image_size))

                    df = pd.DataFrame(
                        {
                            "data": [data.flatten()],
                            "simulation": splits[-5],
                            "snapshot": np.int32(splits[-3].split("_")[1].lstrip("0")),
                            "subhalo_id": np.int32(splits[-1].split("_")[1].lstrip("0")),
                        }
                    )

                    schema = pa.schema(
                        [
                            ("data", pa.list_(pa.float32())),
                            ("simulation", pa.string()),
                            ("snapshot", pa.int32()),
                            ("subhalo_id", pa.int32()),
                        ]
                    )

                    # Use pyarrow to write the data to a parquet file
                    table = pa.Table.from_pandas(df, schema=schema)

                    # Add shape metadata to the schema
                    table = table.replace_schema_metadata(metadata={"data_shape": str(data.shape)})

                    if writer is None:
                        writer = pq.ParquetWriter(
                            f"{output_directory}/0.parquet",
                            table.schema,
                            compression="snappy",
                        )

                    writer.write_table(table, row_group_size=self.chunk_size)

        if writer is not None:
            writer.close()
