"""Generator that counts the number of images processed."""

from pest.generator import Generator


class Count(Generator):
    """Counts the number of images passed through the pipeline."""

    def __init__(self):
        self._count = 0

    def process(self, item) -> None:
        self._count += item

    def close(self) -> None:
        print(f"Total images: {self._count}")
