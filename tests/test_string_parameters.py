import pytest

from pest import (
    AddCircularMask,
    CreateNormalizedRGBColors,
    Crop,
    FilterInclinationAngle,
    FilterTruncatedGalaxies,
    MinMaxNormalize,
)


@pytest.mark.parametrize(
    "cls, kwargs, attributes",
    [
        (AddCircularMask, {"radius_fraction": "0.8"}, {"radius_fraction": 0.8}),
        (Crop, {"relative_size": "5e-1"}, {"relative_size": 0.5}),
        (FilterInclinationAngle, {"max_inclination": "60"}, {"max_inclination": 60.0}),
        (FilterTruncatedGalaxies, {"threshold": "0.5"}, {"threshold": 0.5}),
        (MinMaxNormalize, {"feature_range": ["-1", "1e0"]}, {"min_val": -1.0, "max_val": 1.0}),
        (
            CreateNormalizedRGBColors,
            {"stretch": "0.9", "lower_limit": "1e-3", "scalers": ["0.7", "5e-1", "1.3"]},
            {"stretch": 0.9, "lower_limit": 1e-3, "scalers": [0.7, 0.5, 1.3]},
        ),
    ],
)
def test_string_parameters_are_converted_to_float(cls, kwargs, attributes):
    transform = cls(**kwargs)
    for name, expected in attributes.items():
        assert getattr(transform, name) == expected
