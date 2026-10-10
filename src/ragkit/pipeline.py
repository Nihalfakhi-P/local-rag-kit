from pathlib import Path

import httpx

from ragkit.chunking import chunk_document
from ragkit.config import Settings
from ragkit.guardrails import REFUSAL, check_grounding
from ragkit.loaders import iter_documents
from ragkit.models import Answer, Retrieval
from ragkit.ollama import LLM, Embedder, OllamaEmbedder, OllamaLLM
from ragkit.prompts import SYSTEM_PROMPT, build_user_prompt
from ragkit.retrieval import HybridRetriever
from ragkit.store import ChromaStore


class RAGPipeline:
    def __init__(
        self, settings: Settings, store: ChromaStore, embedder: Embedder, llm: LLM
    ) -> None:
        self.settings = settings
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.retriever = HybridRetriever(store, embedder, settings.candidate_k)

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "RAGPipeline":
        settings = settings or Settings()
        client = httpx.Client(base_url=settings.ollama_url, timeout=settings.request_timeout)
        return cls(
            settings,
            ChromaStore(settings.persist_dir, settings.collection),
            OllamaEmbedder(
                client, settings.embed_model, settings.document_prefix, settings.query_prefix
            ),
            OllamaLLM(client, settings.llm_model, settings.temperature),
        )

    # ---- ingestion -------------------------------------------------------

    def ingest_text(self, source: str, text: str) -> int:
        """(Re)index one document. Re-ingesting a source replaces its old chunks."""
        chunks = chunk_document(source, text, self.settings.chunk_size, self.settings.chunk_overlap)
        embeddings = self.embedder.embed_documents([c.text for c in chunks]) if chunks else []
        self.store.delete_source(source)
        self.store.upsert(chunks, embeddings)
        return len(chunks)

    def ingest_path(self, path: Path) -> dict[str, int]:
        return {source: self.ingest_text(source, text) for source, text in iter_documents(path)}

    # ---- querying --------------------------------------------------------

    def retrieve(self, question: str, top_k: int | None = None) -> Retrieval:
        return self.retriever.retrieve(question, top_k or self.settings.top_k)

    def ask(self, question: str, top_k: int | None = None) -> Answer:
        retrieval = self.retrieve(question, top_k)

        # Relevance gate: nothing close enough means the LLM would only be guessing.
        if not retrieval.hits or retrieval.best_similarity < self.settings.min_similarity:
            return Answer(
                text=REFUSAL,
                sources=[],
                grounded=True,
                refused=True,
                reason="no sufficiently relevant context",
            )

        raw = self.llm.generate(SYSTEM_PROMPT, build_user_prompt(question, retrieval.hits))
        report = check_grounding(raw, len(retrieval.hits))

        if report.refused:
            return Answer(
                text=REFUSAL,
                sources=[],
                grounded=True,
                refused=True,
                reason="model found no answer in context",
            )
        if not report.grounded and self.settings.strict_grounding:
            return Answer(
                text=REFUSAL,
                sources=[],
                grounded=True,
                refused=True,
                reason="answer failed citation check",
            )
        return Answer(
            text=raw.strip(),
            sources=retrieval.hits,
            grounded=report.grounded,
            refused=False,
            citations=report.citations,
            reason=None if report.grounded else "missing or invalid citations",
        )
