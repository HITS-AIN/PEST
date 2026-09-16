from dataclasses import dataclass, field
from enum import Enum


class OrientationType(Enum):
    ORIGINAL = "original"
    FACE_ON = "face_on"
    EDGE_ON = "edge_on"
    RANDOM = "random"


class PropertyType(Enum):
    STELLAR_MASS = "stellar_mass"
    STAR_FORMATION_RATE = "star_formation_rate"
    METALLICITY = "metallicity"
    GAS_DENSITY = "gas_density"
    DARK_MATTER_DENSITY = "dark_matter_density"
    DARK_MATTER_VELOCITY = "dark_matter_velocity"
    GAS_VELOCITY = "gas_velocity"
    STELLAR_VELOCITY = "stellar_velocity"
    STAR_AGE = "star_age"
    STAR_METALLICITY = "star_metallicity"


@dataclass
class Selector:
    property: PropertyType
    min_value: float
    max_value: float


@dataclass
class IllustrisGenerator:
    sim: str = "TNG50-1"
    selectors: list[Selector] = field(default_factory=list)
    # component = ("stars",)
    # objects = ("centrals",)
    # field = ("Masses",)
    # fov = ("scaled",)  # [kpc]
    # image_depth = (1.0,)  #  1 particles per pixel (min. S/N=sqrt(depth))
    # image_size = (128,)
    # smoothing = (0.0,)  # [kpc]
    # image_scale = ("log",)
    orientation: OrientationType = OrientationType.RANDOM
    # output_path = ("./images_test_local/",)


ig = IllustrisGenerator(selectors=[Selector(PropertyType.STELLAR_MASS, 5e10, 5.2e10)])
print(ig)
