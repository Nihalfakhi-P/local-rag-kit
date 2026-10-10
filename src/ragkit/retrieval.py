from ragkit.bm25 import BM25
from ragkit.models import Chunk, Hit, Retrieval
from ragkit.ollama import Embedder
from ragkit.store import ChromaStore


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Combine ranked id lists: score(d) = sum over lists of 1 / (k + rank).

    Rank-based, so it needs no calibration between cosine similarities and BM25 scores.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


class HybridRetriever:
    """Dense (embeddings) + sparse (BM25) retrieval fused with RRF."""

    def __init__(self, store: ChromaStore, embedder: Embedder, candidate_k: int = 20) -> None:
        self.store = store
        self.embedder = embedder
        self.candidate_k = candidate_k
        self._bm25 = BM25()
        self._chunks: dict[str, Chunk] = {}
        self._indexed_version: int | None = None

    def _refresh_sparse_index(self) -> None:
        if self._indexed_version == self.store.version:
            return
        chunks = self.store.all_chunks()
        self._chunks = {c.id: c for c in chunks}
        self._bm25.index([(c.id, c.text) for c in chunks])
        self._indexed_version = self.store.version

    def retrieve(self, query: str, top_k: int) -> Retrieval:
        self._refresh_sparse_index()
        dense = self.store.query(self.embedder.embed_query(query), self.candidate_k)
        sparse = self._bm25.search(query, self.candidate_k)

        similarity = {chunk.id: sim for chunk, sim in dense}
        chunks = self._chunks | {chunk.id: chunk for chunk, _ in dense}
        fused = reciprocal_rank_fusion([[c.id for c, _ in dense], [cid for cid, _ in sparse]])

        ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:top_k]
        hits = [Hit(chunks[cid], score, similarity.get(cid)) for cid, score in ranked]
        return Retrieval(hits=hits, best_similarity=max(similarity.values(), default=0.0))
