import numpy as np

from .units import in_units


class FilterRange:
    """Keep records whose `column` value lies within `[min, max]`.

    Works both per record (`__call__`) and vectorized over a whole table of
    per-record columns (`mask`). The pipeline uses `mask` to drop records
    before they are extracted, if the dataset provides a `table()`.

    Args:
        column (str): Name of the column to filter on, e.g. "SubhaloMassType".
        min (float): Lower limit (default -inf).
        max (float): Upper limit (default inf).
        index (int | None): Index into the last axis of a multi-valued column,
            e.g. `4` for the stellar mass in `SubhaloMassType`.
        units (str | None): Convert unit-carrying values (pynbody `SimArray`)
            to these units before comparing, e.g. "Msol".
        inclusive (bool): Whether the limits themselves are kept (default True).
    """

    is_filter = True

    def __init__(
        self,
        column: str,
        min: float = -np.inf,
        max: float = np.inf,
        index: int | None = None,
        units: str | None = None,
        inclusive: bool = True,
    ):
        self.column = column
        self.min = min
        self.max = max
        self.index = index
        self.units = units
        self.inclusive = inclusive

    def mask(self, table: dict) -> np.ndarray:
        values = in_units(table[self.column], self.units)
        if self.index is not None:
            values = values[..., self.index]
        if self.inclusive:
            return (values >= self.min) & (values <= self.max)
        return (values > self.min) & (values < self.max)

    def __call__(self, record: dict) -> bool:
        value = record[self.column]
        # Keep the units of a SimArray scalar/row by indexing instead of wrapping it.
        return bool(self.mask({self.column: value[None] if hasattr(value, "units") else np.asarray(value)[None]})[0])
