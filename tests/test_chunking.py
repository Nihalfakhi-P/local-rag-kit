from itertools import pairwise

import pytest

from ragkit.chunking import chunk_document, split_text


def test_short_text_is_one_chunk():
    assert split_text("Hello world.", 100, 10) == ["Hello world."]


def test_empty_text_has_no_chunks():
    assert split_text("  \n\n  ", 100, 10) == []


def test_chunks_respect_size_and_are_non_empty():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 40 for i in range(20))
    chunks = split_text(text, 300, 50)
    assert len(chunks) > 1
    assert all(0 < len(c) <= 300 for c in chunks)


def test_consecutive_chunks_overlap():
    text = " ".join(f"token{i}" for i in range(400))
    chunks = split_text(text, 200, 60)
    for prev, nxt in pairwise(chunks):
        assert prev.split()[-1] in nxt.split()[:15]


def test_prefers_paragraph_boundaries():
    a, b = "A" * 50 + ".", "B" * 50 + "."
    assert split_text(f"{a}\n\n{b}", 60, 0) == [a, b]


def test_hard_cuts_unbreakable_tokens():
    chunks = split_text("x" * 1000, 300, 0)
    assert [len(c) for c in chunks] == [300, 300, 300, 100]


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_rejects_invalid_parameters(size, overlap):
    with pytest.raises(ValueError):
        split_text("text", size, overlap)


def test_chunk_ids_are_deterministic_and_unique():
    text = "\n\n".join("para " * 30 for _ in range(10))
    first = chunk_document("doc.md", text, 200, 20)
    second = chunk_document("doc.md", text, 200, 20)
    assert [c.id for c in first] == [c.id for c in second]
    assert len({c.id for c in first}) == len(first)
    assert [c.index for c in first] == list(range(len(first)))
