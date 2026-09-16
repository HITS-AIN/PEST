from enum import Enum


class ComponentType(Enum):
    STARS = "stars"
    GAS = "gas"
    DARK_MATTER = "dark_matter"
    BLACK_HOLES = "black_holes"


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


class OrientationType(Enum):
    ORIGINAL = "original"
    FACE_ON = "face_on"
    EDGE_ON = "edge_on"
    RANDOM = "random"
