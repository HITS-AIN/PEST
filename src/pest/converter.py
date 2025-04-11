from abc import ABC, abstractmethod


class Converter(ABC):
    """Base class for data converters."""

    def __init__(self):
        """Initialize the Converter."""

    @abstractmethod
    def convert(
        self,
        input_file: str,
        output_file: str,
    ):
        """Convert a single file to Parquet format.

        Args:
            input_file (str): Path to the input file.
            output_file (str): Path to the output file.
        """

    @abstractmethod
    def convert_all(
        self,
        input_directories: str | list[str],
        output_directory: str,
    ):
        """Convert all files in the input directory to Parquet format.

        Args:
            input_directories (str | list[str]): Path to the directory or list of directories containing files.
            output_directory (str): Path to the output directory.
        """
