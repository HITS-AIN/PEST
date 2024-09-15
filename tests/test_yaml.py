from pest import data_preprocess_api
import yaml

def test_yaml(tmp_path):
    with open('tests/config.yaml', 'rb') as f:
        conf = yaml.safe_load(f.read())

    data_preprocess_api(
        output_path=tmp_path,
        **conf,
    )
