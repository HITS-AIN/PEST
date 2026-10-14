import numpy as np
import pynbody
import pytest

from pest import FilterRange

VALUES = np.array([1.0, 2.0, 3.0, 4.0])


def test_mask_and_call_agree():
    f = FilterRange("x", min=2.0, max=3.0)
    mask = f.mask({"x": VALUES})
    assert list(mask) == [False, True, True, False]
    assert [f({"x": v}) for v in VALUES] == list(mask)


def test_exclusive_and_open_bounds():
    assert list(FilterRange("x", min=2.0, max=3.0, inclusive=False).mask({"x": VALUES})) == [False] * 4
    assert list(FilterRange("x", min=3.0).mask({"x": VALUES})) == [False, False, True, True]
    assert list(FilterRange("x", max=1.0).mask({"x": VALUES})) == [True, False, False, False]


def test_index():
    table = {"x": np.stack([VALUES, 10 * VALUES], axis=-1)}
    f = FilterRange("x", index=1, min=25.0)
    assert list(f.mask(table)) == [False, False, True, True]
    assert f({"x": table["x"][2]})
    assert not f({"x": table["x"][1]})


@pytest.mark.parametrize("units, expected", [("Msol", [False, True, False, False]), (None, [False] * 4)])
def test_units(units, expected):
    table = {"m": pynbody.array.SimArray(VALUES, "1e10 Msol")}
    f = FilterRange("m", min=1.5e10, max=2.5e10, units=units)
    assert list(f.mask(table)) == expected
