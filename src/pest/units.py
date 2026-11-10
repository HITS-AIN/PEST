import numpy as np
import pynbody

# Units of physical_units(), as for pynbody's `physical_units()`, and the cosmological factors resolved there.
PHYSICAL_BASIS = ["kpc", "km s^-1", "Msol"]
COSMOLOGICAL_FACTORS = ["a", "h"]


def _is_unit(units) -> bool:
    return units is not None and not isinstance(units, pynbody.units.NoUnit)


def has_units(array) -> bool:
    """Whether `array` carries units that it can be converted from, as a pynbody `SimArray` does."""
    return _is_unit(getattr(array, "units", None)) and hasattr(array, "in_units")


def in_units(array, units) -> np.ndarray:
    """Convert a unit-carrying array (e.g. a pynbody `SimArray`) to `units` and return a plain numpy array.

    Arrays without units, or a `units` of `None`, are returned unchanged as numpy arrays,
    so this also works for records that do not come from pynbody.
    """
    if has_units(array) and _is_unit(units):
        array = array.in_units(units)
    return np.asarray(array)


def physical_units(array):
    """Convert a pynbody `SimArray` to physical kpc, km/s and Msol, resolving the scale factor `a` and `h`.

    The conversion uses the cosmology of the simulation the array belongs to, so
    the result no longer needs it, e.g. to concatenate arrays of different
    snapshots. Arrays without units, or with units that cannot be expressed in
    this basis, are returned unchanged.
    """
    if not has_units(array):
        return array
    try:
        powers = array.units.dimensional_project(PHYSICAL_BASIS + COSMOLOGICAL_FACTORS)
    except pynbody.units.UnitsException:
        return array
    target = pynbody.units.Unit("1")
    for basis, power in zip(PHYSICAL_BASIS, powers):
        if power != 0:
            target *= pynbody.units.Unit(basis) ** power
    return array.in_units(target)
