"""Export a fine-tuned checkpoint in the classic sentence-transformers layout (same as intfloat/multilingual-e5-small).

sentence-transformers 6 saves modules with new class paths that older versions (e.g. 3.x used in tk-rf-rag)
cannot load. The base model repo layout is copied and only the weights are replaced; prompts and
max_seq_length=512 are set. Embeddings of the exported model are checked against the checkpoint.

Usage: python scripts/export_model.py models/e1_small_llm_hn/best data/export/hf_model intfloat/multilingual-e5-small
"""

import json
import shutil
import sys
from pathlib import Path

import rlr  # noqa: F401  (cache dirs on D)
from huggingface_hub import snapshot_download
from safetensors.torch import load_file


def main() -> None:
    ckpt, out, base = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    base_dir = Path(snapshot_download(base, allow_patterns=["*.json", "*.model", "1_Pooling/*", "tokenizer*"]))
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(
        base_dir,
        out,
        ignore=shutil.ignore_patterns("onnx", "*.onnx", "openvino", ".cache", "pytorch_model.bin", "README.md"),
    )
    (out / "2_Normalize").mkdir(exist_ok=True)
    shutil.copy(ckpt / "model.safetensors", out / "model.safetensors")
    # tokenizer files stay those of the base model (the tokenizer is not trained; older transformers can read them)

    base_keys = set(
        load_file(Path(snapshot_download(base, allow_patterns=["model.safetensors"])) / "model.safetensors")
    )
    ft_keys = set(load_file(out / "model.safetensors"))
    diff = (base_keys ^ ft_keys) - {"embeddings.position_ids"}  # a buffer, not a weight
    assert not diff, f"weight keys differ: {sorted(diff)[:5]}"

    (out / "sentence_bert_config.json").write_text(
        json.dumps({"max_seq_length": 512, "do_lower_case": False}, indent=2)
    )
    (out / "config_sentence_transformers.json").write_text(
        json.dumps(
            {
                "prompts": {"query": "query: ", "passage": "passage: ", "document": "passage: "},
                "default_prompt_name": None,
                "similarity_fn_name": "cosine",
            },
            indent=2,
        )
    )

    from sentence_transformers import SentenceTransformer

    texts = ["query: Сколько дней ежегодного отпуска?", "passage: Ежегодный основной оплачиваемый отпуск 28 дней."]
    a = SentenceTransformer(str(ckpt), device="cpu").encode(texts, normalize_embeddings=True)
    b = SentenceTransformer(str(out), device="cpu").encode(texts, normalize_embeddings=True)
    cos = (a * b).sum(axis=1)
    print("cosine checkpoint vs export:", cos)
    assert cos.min() > 0.9999
    print(f"exported to {out}")


if __name__ == "__main__":
    main()
