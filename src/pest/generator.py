from abc import ABC, abstractmethod


class Generator(ABC):
    """Base class for data generators."""

    @abstractmethod
    def process(self, item) -> None:
        """Process one item from the extractor."""

    @abstractmethod
    def close(self) -> None:
        """Finalize the generator, e.g. by closing files or writing metadata."""
