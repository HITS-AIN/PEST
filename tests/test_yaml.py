import yaml


def test_yaml(tmp_path):
    with open("tests/config.yaml", "rb") as f:
        conf = yaml.safe_load(f.read())

    assert conf["field"] == "masses"
