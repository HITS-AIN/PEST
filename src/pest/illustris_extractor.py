"""Extractor for IllustrisTNG simulation data."""

from pathlib import Path
from typing import Any

import numpy as np

from .extractor import Extractor
from .illustris.groupcat import loadHeader, loadSubhalos
from .illustris.snapshot import loadSubhalo


class IllustrisExtractor(Extractor):
    """Extractor for IllustrisTNG simulation data."""

    # Available fields for each component type
    AVAILABLE_FIELDS = {
        "stars": [
            "masses",
            "positions",
            "velocities",
            "ages",
            "metallicities",
            "formation_times",
            "initial_masses",
            "gfm_stellar_photometrics",
        ],
        "gas": [
            "masses",
            "positions",
            "velocities",
            "densities",
            "temperatures",
            "internal_energies",
            "smoothing_lengths",
            "electron_abundances",
            "neutral_hydrogen_abundances",
            "star_formation_rates",
        ],
        "dark_matter": ["masses", "positions", "velocities", "particle_ids"],
        "black_holes": ["masses", "positions", "velocities", "mdot", "masses_bh"],
    }

    # Selector types
    SELECTOR_TYPES = ["stellar mass", "total mass", "gas mass", "dark_matter_mass"]

    # Object types
    OBJECT_TYPES = ["centrals", "satellites", "all"]

    def __init__(
        self,
        simulation_path: str,
        simulation: str = "TNG50-1",
        snapshot: int = 99,
        objects: str = "centrals",
        component: list[dict[str, Any]] | None = None,
    ):
        """Initialize the IllustrisExtractor.

        Args:
            simulation_path (str): Path to the Illustris simulation data.
            simulation (str, optional): Simulation name (e.g., 'TNG50', 'TNG100', 'TNG300'). Defaults to "TNG50".
            snapshot (int, optional): Snapshot number. Defaults to 99.
            objects (str, optional): Type of objects to extract ('centrals', 'satellites', 'all'). Defaults to "centrals".
            component (List[Dict[str, Any]], optional): List of component configurations. Each component should have:
                - name (str): Component name ('stars', 'gas', 'dark_matter', 'black_holes')
                - fields (List[str]): List of fields to extract
                - selector (Dict, optional): Selection criteria with 'type', 'min', and 'max' keys
        """
        super().__init__()

        self.simulation_path = Path(simulation_path)
        self.simulation = simulation
        self.snapshot = snapshot
        self.objects = objects
        self.components = component or []

        self.base_path = str(self.simulation_path / self.simulation)

        # Load simulation header
        header = loadHeader(self.base_path, self.snapshot)

        # Unit conversions
        self.mass_units_msun = 1e10 / header["HubbleParam"]
        self.dist_units_kpc = header["Time"] / header["HubbleParam"]

        # Apply object selection
        self.object_mask = self._get_object_mask()

        self.subhalos = loadSubhalos(self.base_path, self.snapshot)

        # Validate configuration on initialization
        self.validate_configuration()

    def extract(self) -> dict[str, Any]:
        """Extract data based on the configured parameters.

        Returns:
            Dict[str, Any]: Dictionary containing extracted data organized by components.
        """

        extracted_data = {}

        for component_config in self.components:
            component_name = component_config["name"]
            fields = component_config["fields"]
            selector = component_config.get("selector")

            # Apply selector if specified
            if selector:
                selection_mask = self._apply_selector(self.subhalos, selector)
                final_mask = self.object_mask & selection_mask
            else:
                final_mask = self.object_mask

            # Get particle type for this component
            ptype = self._get_particle_type(component_name)

            component_data = {}

            # Extract data for selected subhalos
            selected_subhalo_ids = np.where(final_mask)[0]

            for subhalo_id in selected_subhalo_ids:
                subhalo_data = {}

                # Load particle data for this subhalo
                for field in fields:
                    try:
                        field_data = self._load_particle_field(self.snapshot, subhalo_id, ptype, field)
                        subhalo_data[field] = field_data
                    except (AttributeError, KeyError, RuntimeError) as e:
                        print(f"Warning: Could not load field '{field}' for subhalo {subhalo_id}: {e}")
                        continue

                if subhalo_data:  # Only add if we have data
                    component_data[f"subhalo_{subhalo_id}"] = subhalo_data

            extracted_data[component_name] = component_data

        return extracted_data

    def get_available_fields(self, component: str) -> list[str]:
        """Get available fields for a given component.

        Args:
            component (str): Name of the component.

        Returns:
            List[str]: List of available field names for the component.
        """
        return self.AVAILABLE_FIELDS.get(component, [])

    def validate_configuration(self) -> None:
        """Validate the current configuration.

        Returns:
            bool: True if configuration is valid, False otherwise.
        """
        # Check if simulation path exists
        if not self.simulation_path.exists():
            raise ValueError(f"Error: Simulation path does not exist: {self.simulation_path}")

        # Check if objects type is valid
        if self.objects not in self.OBJECT_TYPES:
            raise ValueError(f"Error: Invalid objects type '{self.objects}'. Must be one of {self.OBJECT_TYPES}")

        # Validate components configuration
        for component_config in self.components:
            if not isinstance(component_config, dict):
                raise ValueError("Error: Each component must be a dictionary")

            if "name" not in component_config:
                raise ValueError("Error: Component must have a 'name' field")

            component_name = component_config["name"]
            if component_name not in self.AVAILABLE_FIELDS:
                raise ValueError(
                    f"Error: Unknown component '{component_name}'. Available: {list(self.AVAILABLE_FIELDS.keys())}"
                )

            if "fields" not in component_config:
                raise ValueError("Error: Component must have a 'fields' field")

            fields = component_config["fields"]
            available_fields = self.AVAILABLE_FIELDS[component_name]

            for field in fields:
                if field not in available_fields:
                    raise ValueError(
                        f"Error: Unknown field '{field}' for component '{component_name}'. Available: {available_fields}"
                    )

            # Validate selector if present
            if "selector" in component_config:
                selector = component_config["selector"]
                if not self._validate_selector(selector):
                    raise ValueError(f"Error: Invalid selector for component '{component_name}': {selector}")

    def _get_object_mask(self) -> np.ndarray:
        """Get mask for object selection (centrals, satellites, or all)."""
        subhalo_gr_nr = loadSubhalos(self.base_path, self.snapshot, ["SubhaloGrNr"])
        if self.objects == "centrals":
            return subhalo_gr_nr == 0  # Central subhalos
        elif self.objects == "satellites":
            return subhalo_gr_nr > 0  # Satellite subhalos
        else:  # "all"
            return np.ones(len(subhalo_gr_nr), dtype=bool)

    def _apply_selector(self, subhalos: dict[str, np.ndarray], selector: dict[str, Any]) -> np.ndarray:
        """Apply mass selection criteria."""
        selector_type = selector["type"]
        min_mass = selector["min"]
        max_mass = selector["max"]

        if selector_type == "stellar mass":
            masses = subhalos["SubhaloMassType"][:, 4] * self.mass_units_msun  # Stars are particle type 4
        elif selector_type == "total mass":
            masses = subhalos["SubhaloMass"] * self.mass_units_msun
        elif selector_type == "gas mass":
            masses = subhalos["SubhaloMassType"][:, 0] * self.mass_units_msun  # Gas is particle type 0
        elif selector_type == "dark_matter_mass":
            masses = subhalos["SubhaloMassType"][:, 1] * self.mass_units_msun  # DM is particle type 1
        else:
            raise ValueError(f"Unknown selector type: {selector_type}")

        return (masses >= min_mass) & (masses <= max_mass)

    def _get_particle_type(self, component_name: str) -> int:
        """Get particle type number for component."""
        type_mapping = {"gas": 0, "dark_matter": 1, "stars": 4, "black_holes": 5}
        return type_mapping[component_name]

    def _load_particle_field(
        self,
        snapshot: int,
        subhalo_id: int,
        ptype: int,
        field: str,
    ) -> np.ndarray:
        """Load a specific field for particles in a subhalo."""
        # Field name mapping to IllustrisTNG field names
        field_mapping = {
            "masses": "Masses",
            "positions": "Coordinates",
            "velocities": "Velocities",
            "ages": "GFM_StellarFormationTime",
            "metallicities": "GFM_Metallicity",
            "formation_times": "GFM_StellarFormationTime",
            "initial_masses": "GFM_InitialMass",
            "gfm_stellar_photometrics": "GFM_StellarPhotometrics",
            "densities": "Density",
            "temperatures": "Temperature",
            "internal_energies": "InternalEnergy",
            "smoothing_lengths": "SmoothingLength",
            "electron_abundances": "ElectronAbundance",
            "neutral_hydrogen_abundances": "NeutralHydrogenAbundance",
            "star_formation_rates": "StarFormationRate",
            "particle_ids": "ParticleIDs",
            "mdot": "BH_Mdot",
            "masses_bh": "BH_Mass",
        }

        illustris_field = field_mapping.get(field, field)

        # Load particle data for this subhalo
        particle_data = loadSubhalo(self.base_path, snapshot, subhalo_id, ptype, fields=[illustris_field])

        if particle_data is None or len(particle_data) == 0:
            return np.array([])

        data = particle_data[illustris_field]

        # Apply unit conversions
        if field == "masses":
            data = data * self.mass_units_msun
        elif field == "positions":
            data = data * self.dist_units_kpc

        return data

    def _validate_selector(self, selector: dict[str, Any]) -> bool:
        """Validate selector configuration."""
        required_keys = ["type", "min", "max"]
        for key in required_keys:
            if key not in selector:
                print(f"Error: Selector must have '{key}' field")
                return False

        if selector["type"] not in self.SELECTOR_TYPES:
            print(f"Error: Invalid selector type '{selector['type']}'. Must be one of {self.SELECTOR_TYPES}")
            return False

        if selector["min"] >= selector["max"]:
            print("Error: Selector 'min' must be less than 'max'")
            return False

        return True

    def __repr__(self) -> str:
        """Return a string representation of the extractor."""
        return (
            f"IllustrisExtractor(simulation_path='{self.simulation_path}', "
            f"simulation='{self.simulation}', snapshot={self.snapshot}, "
            f"objects='{self.objects}', components={len(self.components)})"
        )
