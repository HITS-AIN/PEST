import yaml


def test_yaml():
    with open("tests/config.yaml", "rb") as f:
        conf = yaml.safe_load(f.read())

    assert conf["field"] == "masses"
