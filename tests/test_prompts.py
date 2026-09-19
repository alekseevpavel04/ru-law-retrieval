"""Prompt sanity tests for every baseline model.

Fast tests check the config against the model cards; slow tests (``-m slow``, need
downloads + GPU) check that with the configured prompts a relevant passage beats
an irrelevant one, and that the prompts match ``config_sentence_transformers.json``
when the model ships one.
"""

import json
from pathlib import Path

import pytest

from rlr.config import load_yaml

CFG = load_yaml("baselines.yaml")
MODELS = CFG["models"]

EXPECTED = {  # from model cards (checked 2026-09-19)
    "intfloat/multilingual-e5-small": ("query: ", "passage: "),
    "intfloat/multilingual-e5-base": ("query: ", "passage: "),
    "intfloat/multilingual-e5-large": ("query: ", "passage: "),
    "deepvk/USER-base": ("query: ", "passage: "),
    "ai-forever/FRIDA": ("search_query: ", "search_document: "),
    "ai-forever/ru-en-RoSBERTa": ("search_query: ", "search_document: "),
    "deepvk/USER2-base": ("search_query: ", "search_document: "),
    "BAAI/bge-m3": ("", ""),
    "deepvk/USER-bge-m3": ("", ""),
    "cointegrated/rubert-tiny2": ("", ""),
}


@pytest.mark.parametrize("m", MODELS, ids=[m["name"] for m in MODELS])
def test_prompts_match_model_card(m):
    if m["path"] in EXPECTED:
        assert (m["query_prompt"], m["doc_prompt"]) == EXPECTED[m["path"]]
    else:  # instruction models: instruction on the query side only
        assert m["query_prompt"].startswith("Instruct: ")
        assert m["doc_prompt"] == ""


def test_names_unique():
    names = [m["name"] for m in MODELS]
    assert len(names) == len(set(names))


QUERY = "Сколько дней ежегодного оплачиваемого отпуска положено работнику?"
RELEVANT = "Ежегодный основной оплачиваемый отпуск предоставляется работникам продолжительностью 28 календарных дней."
IRRELEVANT = "Собственник вправе истребовать свое имущество из чужого незаконного владения."


@pytest.mark.slow
@pytest.mark.parametrize("m", MODELS, ids=[m["name"] for m in MODELS])
def test_relevant_beats_irrelevant_and_st_prompts(m):
    from huggingface_hub import hf_hub_download

    from rlr.eval.retrieve import DenseEncoder

    enc = DenseEncoder(
        m["path"],
        m["query_prompt"],
        m["doc_prompt"],
        dtype=m.get("dtype", CFG["dtype"]),
        padding_side=m.get("padding_side"),
        name=m["name"],
    )
    q = enc.encode_queries([QUERY])
    d = enc.encode([RELEVANT, IRRELEVANT], m["doc_prompt"])
    s = (q @ d.T)[0]
    assert s[0] > s[1], (m["name"], s)
    try:
        cfg_path = Path(hf_hub_download(m["path"], "config_sentence_transformers.json"))
    except Exception:
        return
    st_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    prompts = st_cfg.get("prompts") or {}
    for key, ours in (("query", m["query_prompt"]), ("search_query", m["query_prompt"])):
        if key in prompts:
            assert prompts[key] == ours
    for key, ours in (
        ("passage", m["doc_prompt"]),
        ("document", m["doc_prompt"]),
        ("search_document", m["doc_prompt"]),
    ):
        if key in prompts:
            assert prompts[key] == ours
