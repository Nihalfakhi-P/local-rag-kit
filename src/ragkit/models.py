from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    index: int


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # fused (RRF) score used for ranking
    similarity: float | None  # cosine similarity from the dense search, if it found this chunk


@dataclass(frozen=True)
class Retrieval:
    hits: list[Hit]
    best_similarity: float  # best dense match, drives the relevance gate


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[Hit]
    grounded: bool
    refused: bool
    citations: list[int] = field(default_factory=list)
    reason: str | None = None
