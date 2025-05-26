"""Calibrate Gaia XP continuous to spectra and store to a Parquet file."""

import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow as pa
from gaiaxpy import calibrate
from pyarrow import parquet

from pest.converter import Converter


class GaiaConverter(Converter):
    """Calibrate Gaia XP continuous to spectra and store to a Parquet file."""

    input_file_suffix = ".csv.gz"

    # The following columns contain string representations of numpy arrays
    list_of_string_arrays = [
        "bp_coefficients",
        "bp_coefficient_errors",
        "bp_coefficient_correlations",
        "rp_coefficients",
        "rp_coefficient_errors",
        "rp_coefficient_correlations",
    ]

    def __init__(
        self,
        sampling: np.ndarray = np.arange(336, 1021, 2),
        with_flux_error: bool = True,
        number_of_workers: int = 1,
    ):
        """Initialize the GaiaConverter.

        Args:
            sampling (list[int]): Wavelength sampling to use for the calibration
                (default: np.arange(336, 1021, 2)).
            flux_error (bool): Whether to include the flux error in the output (default: True).
            number_of_workers (int): Number of workers to use for the conversion (default: 1).
        """
        self.sampling = sampling
        self.with_flux_error = with_flux_error
        self.number_of_workers = number_of_workers

    def convert(
        self,
        input_file: str,
        output_file: str,
    ):
        if os.path.exists(output_file):
            print(f"File {output_file} already exists, skipping")
            return
        else:
            print(f"Converting {input_file}")

        continuous_data = pd.read_csv(input_file, comment="#", compression="gzip")

        # Remove rows with missing or empty array data
        continuous_data.dropna(subset=self.list_of_string_arrays, inplace=True)

        # Convert string entries to numpy arrays
        for array in self.list_of_string_arrays:
            continuous_data[array] = continuous_data[array].apply(
                lambda x: np.fromstring(x[1:-1], dtype=np.float32, sep=",")
            )

        calibrated_data, _ = calibrate(continuous_data, sampling=self.sampling, save_file=False)

        if self.with_flux_error:
            # Convert 'flux' column to float32
            calibrated_data["flux_error"] = calibrated_data["flux_error"].apply(lambda x: np.array(x, dtype=np.float32))
        else:
            # Remove the 'flux_error' column from the calibrated data
            if "flux_error" in calibrated_data.columns:
                calibrated_data.drop(columns=["flux_error"], inplace=True)

        # Convert 'flux' column to float32
        calibrated_data["flux"] = calibrated_data["flux"].apply(lambda x: np.array(x, dtype=np.float32))

        # Use pyarrow to write the data to a parquet file
        table = pa.Table.from_pandas(calibrated_data)

        # Add shape metadata to the schema
        data_shape = f"(1, {len(calibrated_data['flux'][0])})"
        table = table.replace_schema_metadata(metadata={"flux_shape": data_shape, "flux_error_shape": data_shape})

        parquet.write_table(
            table,
            output_file,
            compression="snappy",
        )

    def convert_all(
        self,
        input_directories: str | list[str],
        output_directory: str,
    ):
        """Convert all Gaia XP continuous files in a directory to parquet files.

        Args:
            input_directories (str | list[str]): Path to the directory or list of directories containing files.
            output_directory (str): Path to the directory where the parquet files will be saved.
        """
        os.makedirs(output_directory, exist_ok=True)

        if isinstance(input_directories, str):
            input_directories = [input_directories]

        all_files = []
        for input_directory in input_directories:
            for root, _, files in os.walk(input_directory):
                for file in files:
                    if file.endswith(self.input_file_suffix):
                        input_file = os.path.join(root, file)
                        output_file = os.path.join(
                            output_directory,
                            f"{str(file).removesuffix(self.input_file_suffix)}.parquet",
                        )
                        all_files.append((input_file, output_file))

        print(f"Found {len(all_files)} files to convert")

        if self.number_of_workers == 1:
            for input_file, output_file in all_files:
                self.convert(input_file, output_file)
        else:
            with Pool(self.number_of_workers) as p:
                p.starmap(self.convert, all_files)
