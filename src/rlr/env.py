"""Environment bootstrap.

Imported first by every entrypoint (via ``rlr/__init__.py``) so cache variables
point to drive D before transformers / sentence-transformers are imported.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_DEFAULTS = {
    "HF_HOME": r"D:\VScode_projects\hf-cache",
    "PIP_CACHE_DIR": r"D:\VScode_projects\pip-cache",
    "UV_CACHE_DIR": r"D:\VScode_projects\uv-cache",
    "PYTHONIOENCODING": "utf-8",
    "TOKENIZERS_PARALLELISM": "false",
    # HF_HOME is on D, but `hf auth login` stored the token in the default location
    "HF_TOKEN_PATH": str(Path.home() / ".cache" / "huggingface" / "token"),
}


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if value.strip():
            os.environ.setdefault(key.strip(), value.strip())


_load_dotenv(ROOT / ".env")
for _key, _value in _DEFAULTS.items():
    os.environ.setdefault(_key, _value)

DATA = ROOT / "data"
RESULTS = ROOT / "results"
CONFIGS = ROOT / "configs"
MODELS = ROOT / "models"
