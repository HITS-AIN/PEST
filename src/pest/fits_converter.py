"""Convert FITS files to Parquet format."""

import os

import numpy as np
import pandas as pd
import pyarrow as pa
from astropy.io import fits
from pyarrow import parquet
from skimage.transform import resize

from pest.converter import Converter
from pest.preprocessing import CreateNormalizedRGBColors


class FitsConverter(Converter):
    """Convert FITS files to Parquet format."""

    def __init__(
        self,
        image_size: int = 128,
        number_of_workers: int = 1,
    ):
        """Initialize the FitsConverter.

        Args:
            image_size (int): Size of the images to be converted (default: 128).
            number_of_workers (int): Number of workers to use for conversion (default: 1).
        """
        self.image_size = image_size
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

        series = []
        for input_directory in input_directories:
            for filename in sorted(os.listdir(input_directory)):
                if filename.endswith(".fits"):
                    filename = os.path.join(input_directory, filename)
                    splits = filename[: -len(".fits")].split("/")

                    data = fits.getdata(filename, 0)
                    data = np.array(data).astype(np.float32)
                    data = self.normalize_rgb(data)
                    data = resize(data, (3, self.image_size, self.image_size))

                    series.append(
                        pd.Series(
                            {
                                "data": data.flatten(),
                                "simulation": splits[-5],
                                "snapshot": splits[-3].split("_")[1],
                                "subhalo_id": splits[-1].split("_")[1],
                            }
                        )
                    )

        df = pd.DataFrame(series)

        # Use pyarrow to write the data to a parquet file
        table = pa.Table.from_pandas(df)

        # Add shape metadata to the schema
        table = table.replace_schema_metadata(metadata={"data_shape": str(data.shape)})

        parquet.write_table(
            table,
            os.path.join(output_directory, "0.parquet"),
            compression="snappy",
        )
