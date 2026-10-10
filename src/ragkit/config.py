from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All tunables, overridable via `RAG_*` environment variables or a `.env` file."""

    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    # Ollama
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2"
    embed_model: str = "nomic-embed-text"
    # nomic-embed-text is trained with task prefixes; set both to "" for other models.
    query_prefix: str = "search_query: "
    document_prefix: str = "search_document: "
    temperature: float = 0.1
    request_timeout: float = 120.0

    # Vector store
    persist_dir: str = ".ragkit/chroma"
    collection: str = "docs"

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 120

    # Retrieval
    top_k: int = 5
    candidate_k: int = 20
    # Relevance gate: refuse without calling the LLM when the best dense match is below this.
    min_similarity: float = 0.45

    # Guardrails: replace answers with invalid or missing citations by a refusal.
    strict_grounding: bool = False
