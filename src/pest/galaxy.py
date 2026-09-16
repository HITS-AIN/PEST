from dataclasses import dataclass, field


@dataclass
class Particle:
    """Particle properties"""

    id: int
    mass: float
    position: list[float]
    velocity: list[float]

    def __post_init__(self):
        if len(self.position) != 3:
            raise ValueError("Position must have exactly 3 coordinates")
        if len(self.velocity) != 3:
            raise ValueError("Velocity must have exactly 3 coordinates")


@dataclass
class Star(Particle):
    """Star-specific properties"""

    luminosity: float = 1.0


@dataclass
class Gas(Particle):
    """Gas-specific properties"""

    temperature: float = 100.0


@dataclass
class Galaxy:
    id: int
    central: bool
    mass: float
    position: list[float]
    velocity: list[float]
    stars: list[Star] = field(default_factory=list)
    gas: list[Gas] = field(default_factory=list)

    def __post_init__(self):
        if len(self.position) != 3:
            raise ValueError("Position must have exactly 3 coordinates")
        if len(self.velocity) != 3:
            raise ValueError("Velocity must have exactly 3 coordinates")
