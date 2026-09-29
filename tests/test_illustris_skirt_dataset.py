from pest import IllustrisSkirtDataset

EXPECTED_SERSIC_N = {90: 2.070126}


def test_illustris_skirt_dataset_sersic_n():
    dataset = IllustrisSkirtDataset(
        path="tests/data/illustris_tng_skirt",
        columns=["image", "simulation", "snapshot", "subhalo_id", ("sersic_n", "r")],
    )

    assert len(dataset) == 1

    for item in dataset:
        assert item["simulation"] == "TNG50"
        assert item["snapshot"] == 95
        assert item[("sersic_n", "r")] == EXPECTED_SERSIC_N[int(item["subhalo_id"])]


def test_illustris_skirt_dataset_empty_columns():
    dataset = IllustrisSkirtDataset(
        path="tests/data/illustris_tng_skirt",
        columns=[],
    )

    assert len(dataset) == 1

    for item in dataset:
        assert item == {}
