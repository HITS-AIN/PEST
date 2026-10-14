import numpy as np


def in_units(array, units) -> np.ndarray:
    """Convert a unit-carrying array (e.g. a pynbody `SimArray`) to `units` and return a plain numpy array.

    Arrays without units, or a `units` of `None`, are returned unchanged as numpy arrays,
    so this also works for records that do not come from pynbody.
    """
    array_units = getattr(array, "units", None)
    has_units = array_units is not None and type(array_units).__name__ != "NoUnit"
    target_is_unit = units is not None and type(units).__name__ != "NoUnit"
    if has_units and target_is_unit and hasattr(array, "in_units"):
        array = array.in_units(units)
    return np.asarray(array)
