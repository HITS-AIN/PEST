import make_illustris_tng_testdata
import pytest


@pytest.fixture(scope="session", autouse=True)
def illustris_tng_testdata():
    """The synthetic IllustrisTNG snapshot and catalogue, generated before the tests run."""
    make_illustris_tng_testdata.generate()
