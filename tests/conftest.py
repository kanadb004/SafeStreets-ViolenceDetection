import pytest


@pytest.fixture
def data_yaml_path(tmp_path):
    p = tmp_path / "data.yaml"
    p.write_text(
        "data_root: data\n"
        "manifests_dir: data/manifests\n"
        "cache_dir: data/cache\n"
        "raw_dir: data/raw\n"
        "seed: 1265\n"
    )
    return p
