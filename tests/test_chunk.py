import pytest

from rlr.data.chunk import chunk_article, doc_header, split_text


def test_short_text_is_one_chunk():
    assert split_text("один абзац", 100, 10) == ["один абзац"]


def test_chunks_respect_size_and_keep_order():
    paragraphs = [f"абзац номер {i} " + "слово " * 10 for i in range(20)]
    pieces = split_text("\n".join(paragraphs), 200, 40)
    assert all(len(p) <= 200 for p in pieces)
    joined = "\n".join(pieces)
    for i in range(20):
        assert f"абзац номер {i} " in joined
    assert len(pieces) > 1


def test_overlap_repeats_tail_paragraph():
    text = "\n".join(["a" * 50, "b" * 50, "c" * 50, "d" * 50])
    pieces = split_text(text, 110, 60)
    assert pieces[0] == "a" * 50 + "\n" + "b" * 50
    assert pieces[1].startswith("b" * 50)  # tail of the previous chunk


def test_long_paragraph_split_by_sentences_then_words():
    sentence = "Слово " * 60  # ~360 chars without sentence breaks
    pieces = split_text(sentence.strip(), 100, 0)
    assert all(len(p) <= 100 for p in pieces)
    assert " ".join(pieces).split() == sentence.split()


def test_no_empty_trailing_chunk_from_overlap():
    text = "\n".join(["a" * 60, "b" * 30])
    assert split_text(text, 100, 50) == ["a" * 60 + "\n" + "b" * 30]


def test_bad_params():
    with pytest.raises(ValueError):
        split_text("x", 0, 0)
    with pytest.raises(ValueError):
        split_text("x", 10, 10)


def test_chunk_article_ids_and_header():
    header = doc_header("Трудовой кодекс РФ", "81", "Расторжение")
    chunks = chunk_article("tk-81", header, "x" * 50 + "\n" + "y" * 50, 60, 0)
    assert [c.chunk_id for c in chunks] == ["tk-81#0", "tk-81#1"]
    assert all(c.text.startswith("Трудовой кодекс РФ, Статья 81. Расторжение\n") for c in chunks)
    assert chunks[0].body == "x" * 50


def test_empty_body_falls_back_to_header():
    chunks = chunk_article("tk-1", "H", "", 100, 10)
    assert len(chunks) == 1
