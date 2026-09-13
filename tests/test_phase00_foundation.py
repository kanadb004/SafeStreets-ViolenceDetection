import os
import subprocess
import sys

import numpy as np
import pytest

from safestreets.config import load_config
from safestreets.utils.seed import set_global_seed


@pytest.mark.phase0
def test_load_config_returns_typed_object(data_yaml_path):
    cfg = load_config(data_yaml_path)
    assert cfg.data_root == "data"
    assert cfg.seed == 1265


@pytest.mark.phase0
def test_changing_config_file_changes_returned_value(tmp_path):
    p = tmp_path / "data.yaml"
    p.write_text("data_root: data\nmanifests_dir: m\ncache_dir: c\nraw_dir: r\nseed: 1\n")
    cfg_before = load_config(p)
    assert cfg_before.data_root == "data"

    p.write_text("data_root: changed\nmanifests_dir: m\ncache_dir: c\nraw_dir: r\nseed: 1\n")
    cfg_after = load_config(p)
    assert cfg_after.data_root == "changed"


@pytest.mark.phase0
def test_env_var_overrides_data_root(data_yaml_path, monkeypatch):
    monkeypatch.setenv("SS_DATA_ROOT", "/tmp/x")
    cfg = load_config(data_yaml_path)
    assert cfg.data_root == "/tmp/x"


@pytest.mark.phase0
def test_set_global_seed_reproducible_across_processes():
    set_global_seed(1265)
    draw_here = np.random.rand(5).tolist()

    script = (
        "import numpy as np; "
        "from safestreets.utils.seed import set_global_seed; "
        "set_global_seed(1265); "
        "print(list(np.random.rand(5)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ},
    )
    draw_other_process = eval(result.stdout.strip())

    assert draw_here == pytest.approx(draw_other_process)
