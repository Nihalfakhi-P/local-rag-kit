from typing import Protocol

import httpx


class OllamaError(RuntimeError):
    """Ollama is unreachable or rejected the request (e.g. model not pulled)."""


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class LLM(Protocol):
    def generate(self, system: str, user: str) -> str: ...


def _post(client: httpx.Client, path: str, payload: dict, model: str) -> dict:
    try:
        response = client.post(path, json=payload)
    except httpx.HTTPError as exc:
        raise OllamaError(
            f"Cannot reach Ollama at {client.base_url} ({exc.__class__.__name__}). "
            "Is it running? Start it with `ollama serve`."
        ) from exc
    if response.status_code == 404:
        raise OllamaError(f"Model {model!r} not found. Pull it with `ollama pull {model}`.")
    if response.is_error:
        raise OllamaError(f"Ollama {path} failed ({response.status_code}): {response.text[:300]}")
    return response.json()


class OllamaEmbedder:
    def __init__(
        self,
        client: httpx.Client,
        model: str,
        document_prefix: str = "",
        query_prefix: str = "",
        batch_size: int = 32,
    ) -> None:
        self.client = client
        self.model = model
        self.document_prefix = document_prefix
        self.query_prefix = query_prefix
        self.batch_size = batch_size

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            payload = {"model": self.model, "input": batch}
            data = _post(self.client, "/api/embed", payload, self.model)
            vectors.extend(data["embeddings"])
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed([self.document_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._embed([self.query_prefix + text])[0]


class OllamaLLM:
    def __init__(self, client: httpx.Client, model: str, temperature: float = 0.1) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature

    def generate(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        data = _post(self.client, "/api/chat", payload, self.model)
        return data["message"]["content"]
