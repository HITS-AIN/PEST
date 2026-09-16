from datasets import Dataset


class ParquetWriter:
    def __init__(self, output_path: str, chunk_size: int | None = None):
        self.output_path = output_path
        self.chunk_size = chunk_size

    def __call__(self, dataset: Dataset):
        dataset.to_parquet(self.output_path, batch_size=self.chunk_size)
