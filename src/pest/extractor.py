"""Base class for data extractors."""

from abc import ABC, abstractmethod
from typing import Any


class Extractor(ABC):
    """Base class for extracting data from simulations or surveys."""

    def __init__(self, **kwargs):
        """Initialize the Extractor.

        Args:
            **kwargs: Additional configuration parameters specific to each extractor.
        """

    @abstractmethod
    def extract(self) -> dict[str, Any]:
        """Extract data based on the configured parameters.

        Returns:
            Dict[str, Any]: Dictionary containing extracted data organized by components.
        """
        ...

    @abstractmethod
    def get_available_fields(self, component: str) -> list[str]:
        """Get available fields for a given component.

        Args:
            component (str): Name of the component (e.g., 'stars', 'gas', 'dark_matter').

        Returns:
            List[str]: List of available field names for the component.
        """
        ...

    @abstractmethod
    def validate_configuration(self) -> None:
        """Validate the current configuration.

        Returns:
            bool: True if configuration is valid, False otherwise.
        """

    def __repr__(self) -> str:
        """Return a string representation of the extractor."""
        return f"{self.__class__.__name__}()"
