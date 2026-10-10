import hashlib
import math
import uuid
from collections.abc import Callable
from pathlib import Path

import pytest

from ragkit.bm25 import tokenize
from ragkit.config import Settings
from ragkit.pipeline import RAGPipeline
from ragkit.store import ChromaStore

EXAMPLES = Path(__file__).parent.parent / "examples"
DIM = 256


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder: cosine similarity tracks word overlap.

    Good enough to exercise real ranking logic without a model server.
    """

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * DIM
        for tok in tokenize(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % DIM] += 1.0 if (h >> 8) % 2 else -1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class FakeLLM:
    """Records prompts; replies with a fixed string or a function of the user prompt."""

    def __init__(self, reply: str | Callable[[str], str] = "It is in the policy [1].") -> None:
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.reply(user) if callable(self.reply) else self.reply


def make_pipeline(llm: FakeLLM | None = None, **overrides) -> RAGPipeline:
    values = {
        "persist_dir": "",
        "collection": f"test-{uuid.uuid4().hex[:12]}",
        "chunk_size": 400,
        "chunk_overlap": 60,
        "min_similarity": 0.15,
    } | overrides
    settings = Settings(_env_file=None, **values)
    store = ChromaStore(None, settings.collection)
    return RAGPipeline(settings, store, FakeEmbedder(), llm or FakeLLM())


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def pipeline(llm: FakeLLM) -> RAGPipeline:
    return make_pipeline(llm)


@pytest.fixture
def loaded(pipeline: RAGPipeline) -> RAGPipeline:
    pipeline.ingest_path(EXAMPLES / "docs")
    return pipeline
