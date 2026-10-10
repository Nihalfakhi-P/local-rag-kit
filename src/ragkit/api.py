from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ragkit import __version__
from ragkit.loaders import UnsupportedFileError, load_bytes
from ragkit.models import Answer
from ragkit.ollama import OllamaError
from ragkit.pipeline import RAGPipeline

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceOut(BaseModel):
    number: int
    source: str
    chunk_index: int
    score: float
    similarity: float | None
    text: str


class QueryResponse(BaseModel):
    answer: str
    grounded: bool
    refused: bool
    citations: list[int]
    reason: str | None
    sources: list[SourceOut]

    @classmethod
    def from_answer(cls, answer: Answer) -> "QueryResponse":
        return cls(
            answer=answer.text,
            grounded=answer.grounded,
            refused=answer.refused,
            citations=answer.citations,
            reason=answer.reason,
            sources=[
                SourceOut(
                    number=i,
                    source=h.chunk.source,
                    chunk_index=h.chunk.index,
                    score=round(h.score, 5),
                    similarity=None if h.similarity is None else round(h.similarity, 4),
                    text=h.chunk.text,
                )
                for i, h in enumerate(answer.sources, start=1)
            ],
        )


def create_app(pipeline: RAGPipeline | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.pipeline = pipeline or RAGPipeline.from_settings()
        yield

    app = FastAPI(
        title="local-rag-kit",
        version=__version__,
        description="Private, fully local RAG: hybrid retrieval, citations, grounding guardrails.",
        lifespan=lifespan,
    )

    @app.exception_handler(OllamaError)
    async def ollama_unavailable(_: Request, exc: OllamaError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    def rag(request: Request) -> RAGPipeline:
        return request.app.state.pipeline

    @app.get("/health")
    def health(request: Request) -> dict:
        return {"status": "ok", "version": __version__, "chunks": rag(request).store.count()}

    @app.get("/documents")
    def documents(request: Request) -> dict[str, int]:
        """Indexed sources and their chunk counts."""
        return rag(request).store.sources()

    @app.post("/ingest")
    async def ingest(request: Request, files: Annotated[list[UploadFile], File()]) -> dict:
        ingested: dict[str, int] = {}
        for upload in files:
            data = await upload.read(MAX_UPLOAD_BYTES + 1)
            if len(data) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"{upload.filename} exceeds {MAX_UPLOAD_BYTES} bytes")
            name = upload.filename or "upload.txt"
            try:
                text = load_bytes(name, data)
            except UnsupportedFileError as exc:
                raise HTTPException(415, str(exc)) from exc
            ingested[name] = rag(request).ingest_text(name, text)
        return {"ingested": ingested}

    @app.delete("/documents/{source:path}")
    def delete_document(request: Request, source: str) -> dict:
        removed = rag(request).store.delete_source(source)
        if not removed:
            raise HTTPException(404, f"No indexed document named {source!r}")
        return {"deleted": source, "chunks": removed}

    @app.post("/query", response_model=QueryResponse)
    def query(request: Request, body: QueryRequest) -> QueryResponse:
        return QueryResponse.from_answer(rag(request).ask(body.question, body.top_k))

    return app
