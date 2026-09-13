"""Typed config loader for safestreets. Reads configs/*.yaml with env-var overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"

# Maps a dataclass field name to the env var that overrides it.
_ENV_OVERRIDES = {
    "data_root": "SS_DATA_ROOT",
    "manifests_dir": "SS_MANIFESTS_DIR",
    "cache_dir": "SS_CACHE_DIR",
    "raw_dir": "SS_RAW_DIR",
}


@dataclass(frozen=True)
class DataConfig:
    data_root: str
    manifests_dir: str
    cache_dir: str
    raw_dir: str
    seed: int


def load_config(config_path: Path | str | None = None) -> DataConfig:
    """Load DataConfig from configs/data.yaml, applying SS_* env-var overrides."""
    path = Path(config_path) if config_path is not None else CONFIGS_DIR / "data.yaml"
    with open(path) as f:
        raw = yaml.safe_load(f)

    for field, env_var in _ENV_OVERRIDES.items():
        if env_var in os.environ:
            raw[field] = os.environ[env_var]

    return DataConfig(**raw)
