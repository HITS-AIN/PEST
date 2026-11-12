import os
from pathlib import Path

import pandas as pd
import pynbody
import pytest
import yaml

from pest import Pipeline
from pest.pipeline import load_config


def test_illustris_skirt_pipeline_1(tmp_path):
    output_path = tmp_path / "illustris_skirt.parquet"

    with open(Path(__file__).parent / "data" / "illustris_skirt_pipeline_1.yaml") as fh:
        config = yaml.safe_load(fh)
    config["load"][0]["init_args"]["output_path"] = str(output_path)

    Pipeline(config).run()

    assert output_path.exists(), "Parquet file was not created."

    df = pd.read_parquet(output_path)
    assert len(df) == 1
    assert set(df.columns) == {"image", "simulation", "snapshot", "subhalo_id"}
    assert df.iloc[0]["simulation"] == "TNG50"
    assert df.iloc[0]["snapshot"] == 95

    image = df.iloc[0]["image"]
    assert len(image) == 3
    assert len(image[0]) == 32
    assert len(image[0][0]) == 32


def test_illustris_skirt_pipeline_hugging_face_writer(tmp_path):
    output_path = tmp_path / "illustris_skirt.parquet"

    with open(Path(__file__).parent / "data" / "illustris_skirt_pipeline_1.yaml") as fh:
        config = yaml.safe_load(fh)
    config["load"][0]["init_args"]["output_path"] = str(output_path)
    config["load"].append(
        {
            "class_path": "pest.HuggingFaceWriter",
            "init_args": {"parquet_path": str(output_path)},
        }
    )

    Pipeline(config).run()

    from datasets import load_dataset

    dataset = load_dataset("parquet", data_files=str(output_path), split="train")
    assert len(dataset) == 1
    assert set(dataset.column_names) == {"image", "simulation", "snapshot", "subhalo_id"}
    assert dataset[0]["simulation"] == "TNG50"


def test_illustris_skirt_pipeline_2(tmp_path):
    output_path = tmp_path / "illustris_skirt.parquet"

    with open(Path(__file__).parent / "data" / "illustris_skirt_pipeline_2.yaml") as fh:
        config = yaml.safe_load(fh)
    config["load"][0]["init_args"]["output_path"] = str(output_path)

    Pipeline(config).run()

    assert output_path.exists(), "Parquet file was not created."

    df = pd.read_parquet(output_path)
    assert len(df) == 1
    assert set(df.columns) == {"simulation", "snapshot", "subhalo_id"}
    assert df.iloc[0]["simulation"] == "TNG50"
    assert df.iloc[0]["snapshot"] == 95


def test_illustris_skirt_pipeline_3(tmp_path):
    output_path = tmp_path / "illustris_skirt.parquet"

    with open(Path(__file__).parent / "data" / "illustris_skirt_pipeline_3.yaml") as fh:
        config = yaml.safe_load(fh)
    config["load"][0]["init_args"]["output_path"] = str(output_path)

    Pipeline(config).run()

    assert output_path.exists(), "Parquet file was not created."

    df = pd.read_parquet(output_path)
    assert len(df) == 1
    assert set(df.columns) == {"simulation", "snapshot", "subhalo_id", "sersic_n_r"}
    assert df.iloc[0]["simulation"] == "TNG50"
    assert df.iloc[0]["snapshot"] == 95
    assert df.iloc[0]["sersic_n_r"] == pytest.approx(2.070126)


def test_threads_per_worker(monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(16)), raising=False)

    # By default, the workers share the CPUs.
    assert Pipeline({"num_workers": 4}).threads_per_worker == 4
    assert Pipeline({"num_workers": 32}).threads_per_worker == 1
    assert Pipeline({"num_workers": 4, "threads_per_worker": 2}).threads_per_worker == 2
    with pytest.raises(ValueError):
        Pipeline({"threads_per_worker": 0})


def test_pipeline_sets_pynbody_threads(tmp_path, monkeypatch):
    monkeypatch.setitem(pynbody.config, "number_of_threads", pynbody.config["number_of_threads"])
    with open(Path(__file__).parent / "data" / "illustris_skirt_pipeline_2.yaml") as fh:
        config = yaml.safe_load(fh)
    config["load"][0]["init_args"]["output_path"] = str(tmp_path / "illustris_skirt.parquet")
    config["threads_per_worker"] = 3

    Pipeline(config).run()

    assert pynbody.config["number_of_threads"] == 3


def test_load_config_reads_exponent_notation_as_float(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "a: 1.0e10\nb: 1e10\nc: 1e-3\nd: -2.5E+3\ne: .5e1\nf: 5\ng: '1e10'\nh: 50 kpc\ni: abc1e10\nj: 0.001\n"
    )
    config = load_config(path)
    assert config == {
        "a": 1e10,
        "b": 1e10,
        "c": 1e-3,
        "d": -2.5e3,
        "e": 5.0,
        "f": 5,
        "g": "1e10",
        "h": "50 kpc",
        "i": "abc1e10",
        "j": 0.001,
    }
    assert all(isinstance(config[key], float) for key in "abcdej")
    assert isinstance(config["f"], int)
