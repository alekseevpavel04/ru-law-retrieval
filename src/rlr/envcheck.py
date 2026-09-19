"""Print environment info: GPU, cache dirs, library versions."""

import os


def main(argv: list[str] | None = None) -> None:
    import sentence_transformers
    import torch
    import transformers

    for key in ("HF_HOME", "PIP_CACHE_DIR", "UV_CACHE_DIR"):
        print(f"{key}={os.environ.get(key)}")
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)} bf16={torch.cuda.is_bf16_supported()}")
    print(f"transformers={transformers.__version__} sentence-transformers={sentence_transformers.__version__}")
