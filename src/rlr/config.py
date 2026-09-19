"""Config loading helpers."""

from pathlib import Path
from typing import Any

import yaml

from rlr.env import CONFIGS


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = CONFIGS / path
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def codes_config() -> dict[str, Any]:
    return load_yaml(CONFIGS / "codes.yaml")
