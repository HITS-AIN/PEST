#!/usr/bin/env python3

"""Calibrate Gaia XP continuous to spectra and store to a Parquet file."""

import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow as pa
from gaiaxpy import calibrate
from pyarrow import parquet

list_of_arrays = [
    "bp_coefficients",
    "bp_coefficient_errors",
    "bp_coefficient_correlations",
    "rp_coefficients",
    "rp_coefficient_errors",
    "rp_coefficient_correlations",
]
sampling = np.arange(336, 1021, 2)
INPUT_FILES_SUFFIX = ".csv.gz"


def list_csv_gz_files(directory):
    return [file for file in os.listdir(directory) if file.endswith(INPUT_FILES_SUFFIX)]


def single_convert(input_file: str, input_path: str, output_path: str):
    output_file = f"{str(input_file).removesuffix(INPUT_FILES_SUFFIX)}.parquet"

    if os.path.exists(os.path.join(output_path, output_file)):
        print(f"File {output_file} already exists, skipping")
        return
    else:
        print(f"Converting {input_file}")

    continuous_data = pd.read_csv(
        os.path.join(input_path, input_file), comment="#", compression="gzip"
    )

    # Remove rows with missing or empty array data
    continuous_data.dropna(subset=list_of_arrays, inplace=True)

    # Convert string entries to numpy arrays
    for array in list_of_arrays:
        continuous_data[array] = continuous_data[array].apply(
            lambda x: np.fromstring(x[1:-1], dtype=np.float32, sep=",")
        )

    calibrated_data, _ = calibrate(continuous_data, sampling=sampling, save_file=False)

    # Remove the 'flux_error' column from the calibrated data
    if "flux_error" in calibrated_data.columns:
        calibrated_data.drop(columns=["flux_error"], inplace=True)

    # Convert 'flux' column to float32
    calibrated_data["flux"] = calibrated_data["flux"].apply(
        lambda x: np.array(x, dtype=np.float32)
    )

    # Use pyarrow to write the data to a parquet file
    table = pa.Table.from_pandas(calibrated_data)

    # Add shape metadata to the schema
    data_shape = f"(1, {len(calibrated_data['flux'][0])})"
    table = table.replace_schema_metadata(
        metadata={"flux_shape": data_shape, "flux_error_shape": data_shape}
    )

    parquet.write_table(
        table,
        os.path.join(output_path, output_file),
        compression="snappy",
    )


def gaia_calibrate(
    input_path: str,
    output_path: str,
    number_of_workers: int = 1,
):
    """Calibrate Gaia XP continuous to spectra and store to a Parquet file.

    Args:
        input_path (str): Path to the directory containing the Gaia XP continuous files (.csv.gz).
        output_path (str): Path to the directory where the parquet files will be saved.
        number_of_workers (int): Number of workers to use for the conversion.
    """
    list_of_files = list_csv_gz_files(input_path)
    print(f"Found {len(list_of_files)} files to convert")

    if number_of_workers == 1:
        for file in list_of_files:
            single_convert(file, input_path, output_path)
    else:
        with Pool(number_of_workers) as p:
            p.starmap(
                single_convert,
                [(file, input_path, output_path) for file in list_of_files],
            )
