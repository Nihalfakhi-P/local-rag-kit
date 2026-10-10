from collections import Counter

import chromadb

from ragkit.models import Chunk


class ChromaStore:
    """Thin wrapper over a ChromaDB collection using cosine distance."""

    def __init__(self, persist_dir: str | None, collection: str) -> None:
        client = (
            chromadb.PersistentClient(path=persist_dir)
            if persist_dir
            else chromadb.EphemeralClient()
        )
        self._col = client.get_or_create_collection(
            collection, configuration={"hnsw": {"space": "cosine"}}
        )
        # Bumped on every write so derived indexes (BM25) know when to rebuild.
        self.version = 0

    def count(self) -> int:
        return self._col.count()

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._col.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[{"source": c.source, "index": c.index} for c in chunks],
        )
        self.version += 1

    def delete_source(self, source: str) -> int:
        existing = self._col.get(where={"source": source}, include=[])["ids"]
        if existing:
            self._col.delete(ids=existing)
            self.version += 1
        return len(existing)

    def query(self, embedding: list[float], n: int) -> list[tuple[Chunk, float]]:
        """Nearest chunks with cosine similarity (1 - cosine distance)."""
        total = self.count()
        if total == 0:
            return []
        res = self._col.query(
            query_embeddings=[embedding],
            n_results=min(n, total),
            include=["documents", "metadatas", "distances"],
        )
        return [
            (
                Chunk(id=cid, text=doc, source=str(meta["source"]), index=int(meta["index"])),
                1.0 - float(dist),
            )
            for cid, doc, meta, dist in zip(
                res["ids"][0],
                res["documents"][0],
                res["metadatas"][0],
                res["distances"][0],
                strict=True,
            )
        ]

    def all_chunks(self) -> list[Chunk]:
        res = self._col.get(include=["documents", "metadatas"])
        return [
            Chunk(id=cid, text=doc, source=str(meta["source"]), index=int(meta["index"]))
            for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"], strict=True)
        ]

    def sources(self) -> dict[str, int]:
        metas = self._col.get(include=["metadatas"])["metadatas"] or []
        return dict(sorted(Counter(str(m["source"]) for m in metas).items()))
