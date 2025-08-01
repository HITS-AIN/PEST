"""Convert FITS files to Parquet format."""

import io
import os
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.io import fits
from PIL import Image
from skimage.transform import resize

from pest.converter import Converter
from pest.preprocessing import CreateNormalizedRGBColors


class FitsConverter(Converter):
    """Convert FITS files to Parquet format."""

    def __init__(
        self,
        image_size: int = 128,
        datatype: str = "png",
        flatten: bool = False,
        chunk_size: int = 1000,
        compression: str = "snappy",
    ):
        """Initialize the FitsConverter.

        Args:
            image_size (int, optional): Size of the images to be converted (default: 128).
            datatype (str, optional): Data type for the output files ["png", "uint8", "float32"]
            (default: "png").
            flatten (bool, optional): Whether to flatten the data (default: False).
                The shape of the data will be preserved in the metadata.
            chunk_size (int, optional): Size of row chunks of parquet files (default: 1000).
            compression (str, optional): Compression algorithm for parquet files (default: "snappy").
        """

        if datatype not in ["png", "uint8", "float32"]:
            raise ValueError(f"Unsupported datatype: {datatype}. Supported types are 'png', 'uint8', and 'float32'.")

        if flatten and datatype == "png":
            raise ValueError("Flattening is not supported for PNG datatype. Set flatten=False.")

        self.image_size = image_size
        self.datatype = datatype
        self.flatten = flatten
        self.chunk_size = chunk_size

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

        batch = []
        file_idx = 0

        # Iterate over all input directories
        for input_directory in input_directories:
            for filename in sorted(os.listdir(input_directory)):
                if filename.endswith(".fits"):
                    filename = os.path.join(input_directory, filename)
                    splits = filename[: -len(".fits")].split("/")

                    data = fits.getdata(filename, 0)
                    data = np.array(data).astype(np.float32)
                    data = self.normalize_rgb(data)

                    # Skip unhealthy data
                    if np.isnan(data).any() or np.isinf(data).any() or np.all(data == data.flat[0]):
                        continue

                    data = resize(data, (3, self.image_size, self.image_size))
                    data_shape = data.shape

                    if self.datatype == "float32":
                        if self.flatten:
                            data = data.flatten()
                            data_schema = pa.list_(pa.float32())
                        else:
                            data = data.tolist()
                            data_schema = pa.list_(pa.list_(pa.list_(pa.float32())))
                    elif self.datatype == "uint8":
                        data = (data * 255).astype(np.uint8)
                        if self.flatten:
                            data = data.flatten()
                            data_schema = pa.list_(pa.uint8())
                        else:
                            data = data.tolist()
                            data_schema = pa.list_(pa.list_(pa.list_(pa.uint8())))
                    elif self.datatype == "png":
                        data = (data * 255).astype(np.uint8)
                        img = Image.fromarray(data.transpose(1, 2, 0))  # CHW to HWC
                        png_buffer = io.BytesIO()
                        img.save(png_buffer, format="PNG", optimize=True)
                        data = png_buffer.getvalue()
                        data_schema = pa.binary()

                    df = pd.DataFrame(
                        {
                            "data": [data],
                            "simulation": splits[-5],
                            "snapshot": np.int32(splits[-3].split("_")[1]),
                            "subhalo_id": np.int32(splits[-1].split("_")[1]),
                        }
                    )

                    schema = pa.schema(
                        [
                            ("data", data_schema),
                            ("simulation", pa.string()),
                            ("snapshot", pa.int32()),
                            ("subhalo_id", pa.int32()),
                        ]
                    )

                    # Use pyarrow to write the data to a parquet file
                    table = pa.Table.from_pandas(df, schema=schema)

                    # Add shape metadata to the schema
                    if self.flatten:
                        table = table.replace_schema_metadata(metadata={"data_shape": str(data_shape)})

                    batch.append(table)
                    # Write batch if chunk_size reached
                    if self.chunk_size and len(batch) >= self.chunk_size:
                        pq.write_table(
                            pa.concat_tables(batch),
                            f"{output_directory}/{file_idx}.parquet",
                            compression=self.compression,
                        )
                        file_idx += 1
                        batch = []

        # Write any remaining data
        if batch:
            pq.write_table(
                pa.concat_tables(batch),
                f"{output_directory}/{file_idx}.parquet",
                compression=self.compression,
            )
