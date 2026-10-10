import pytest

from ragkit.bm25 import BM25, tokenize
from ragkit.retrieval import reciprocal_rank_fusion


def test_tokenize_keeps_codes_and_drops_stopwords():
    assert tokenize("What does error E-4021 mean for the VPN?") == [
        "error",
        "e-4021",
        "mean",
        "vpn",
    ]


def test_bm25_ranks_exact_term_matches_first():
    bm25 = BM25()
    bm25.index(
        [
            ("a", "The VPN gateway is under maintenance on Sundays."),
            ("b", "Error E-4021 means your device certificate expired."),
            ("c", "Annual leave is 24 days per year."),
        ]
    )
    results = bm25.search("E-4021", 3)
    assert [doc_id for doc_id, _ in results] == ["b"]


def test_bm25_rare_terms_outweigh_common_ones():
    bm25 = BM25()
    bm25.index([("a", "leave leave policy"), ("b", "policy parental"), ("c", "policy expenses")])
    top, _ = bm25.search("parental policy", 3)[0]
    assert top == "b"


def test_bm25_handles_empty_index_and_unknown_terms():
    bm25 = BM25()
    assert bm25.search("anything", 5) == []
    bm25.index([("a", "hello world")])
    assert bm25.search("zebra", 5) == []


def test_rrf_rewards_agreement_between_rankers():
    scores = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["a"] == scores["b"]
    assert scores["a"] > scores["c"] > 0
    assert scores["d"] == pytest.approx(1 / 63)


def test_hybrid_retrieval_finds_the_right_document(loaded):
    hits = loaded.retrieve("How many days of paid annual leave do full-time employees get?").hits
    assert hits[0].chunk.source == "leave-policy.md"
    assert "24 days" in hits[0].chunk.text


def test_sparse_side_rescues_exact_codes(loaded):
    hits = loaded.retrieve("E-4033").hits
    assert hits[0].chunk.source == "it-security.md"
    assert "E-4033" in hits[0].chunk.text


def test_bm25_index_tracks_store_writes(pipeline):
    pipeline.ingest_text("a.md", "Quokkas live on Rottnest Island.")
    assert pipeline.retrieve("quokkas").hits[0].chunk.source == "a.md"
    pipeline.ingest_text("b.md", "Wombats produce cube-shaped droppings.")
    assert pipeline.retrieve("wombats").hits[0].chunk.source == "b.md"
