"""Environment bootstrap.

Imported first by every entrypoint (via ``rlr/__init__.py``) so that cache variables are set
before transformers / sentence-transformers are imported.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_DEFAULTS = {
    "PYTHONIOENCODING": "utf-8",
    "TOKENIZERS_PARALLELISM": "false",
    # HF_HOME may be moved (see below), but `hf auth login` stores the token in the default location
    "HF_TOKEN_PATH": str(Path.home() / ".cache" / "huggingface" / "token"),
}

# Heavy caches on the development machine live on drive D (drive C is nearly full). These defaults
# are applied only when that location exists, so a clone elsewhere keeps the standard cache paths.
_CACHE_ROOT = Path(r"D:\VScode_projects")
_CACHE_DEFAULTS = {
    "HF_HOME": _CACHE_ROOT / "hf-cache",
    "PIP_CACHE_DIR": _CACHE_ROOT / "pip-cache",
    "UV_CACHE_DIR": _CACHE_ROOT / "uv-cache",
    "MTEB_CACHE": _CACHE_ROOT / "mteb-cache",
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
if _CACHE_ROOT.exists():
    for _key, _path in _CACHE_DEFAULTS.items():
        os.environ.setdefault(_key, str(_path))

# PYTHONIOENCODING is read by the interpreter at start-up, so setting it here is too late for the
# already-configured streams: reconfigure them directly (Russian text on a cp1251 console).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

DATA = ROOT / "data"
RESULTS = ROOT / "results"
CONFIGS = ROOT / "configs"
MODELS = ROOT / "models"
