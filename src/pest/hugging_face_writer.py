from collections.abc import Iterable


class HuggingFaceWriter:
    """Load a Parquet file into a Hugging Face ``datasets.Dataset``, optionally pushing it to the Hub.

    This is an optional load step: unlike ``ParquetWriter`` it does not consume
    the pipeline's records directly, but reads back the file written by an
    earlier ``ParquetWriter`` step. That keeps ``datasets``/``huggingface-hub``
    an optional dependency (``pip install astro-pest[hf]``) of the core
    pipeline, and avoids materializing the dataset a second time.

    Args:
        parquet_path (str): Path to the Parquet file written by ``ParquetWriter``.
        repo_id (str | None): If given, push the resulting dataset to this
            Hugging Face Hub repository.
    """

    def __init__(self, parquet_path: str, repo_id: str | None = None):
        self.parquet_path = parquet_path
        self.repo_id = repo_id

    def __call__(self, records: Iterable[dict]) -> None:
        from datasets import load_dataset

        dataset = load_dataset("parquet", data_files=self.parquet_path, split="train")
        if self.repo_id:
            dataset.push_to_hub(self.repo_id)
