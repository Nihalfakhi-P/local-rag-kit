import pytest
from fastapi.testclient import TestClient

from ragkit.api import create_app
from ragkit.ollama import OllamaError
from tests.conftest import FakeLLM, make_pipeline


@pytest.fixture
def client():
    rag = make_pipeline(FakeLLM("Full-time employees get 24 days [1]."))
    with TestClient(create_app(rag)) as c:
        yield c


def upload(client, name: str, body: bytes):
    return client.post("/ingest", files=[("files", (name, body, "text/plain"))])


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["chunks"] == 0


def test_ingest_query_and_delete_flow(client):
    res = upload(client, "leave.md", b"# Leave\n\nFull-time employees get 24 days of annual leave.")
    assert res.status_code == 200
    assert res.json() == {"ingested": {"leave.md": 1}}
    assert client.get("/documents").json() == {"leave.md": 1}

    res = client.post("/query", json={"question": "How many days of annual leave?"})
    assert res.status_code == 200
    body = res.json()
    assert body["answer"] == "Full-time employees get 24 days [1]."
    assert body["grounded"] is True and body["refused"] is False
    assert body["citations"] == [1]
    assert body["sources"][0]["source"] == "leave.md"
    assert body["sources"][0]["number"] == 1

    assert client.delete("/documents/leave.md").json() == {"deleted": "leave.md", "chunks": 1}
    assert client.get("/documents").json() == {}
    assert client.delete("/documents/leave.md").status_code == 404


def test_unsupported_file_type_is_rejected(client):
    res = upload(client, "malware.exe", b"MZ...")
    assert res.status_code == 415
    assert ".exe" in res.json()["detail"]


@pytest.mark.parametrize(
    "payload", [{"question": ""}, {"question": "x" * 2001}, {"question": "ok", "top_k": 0}, {}]
)
def test_query_validation(client, payload):
    assert client.post("/query", json=payload).status_code == 422


def test_ollama_outage_maps_to_503():
    class DownLLM:
        def generate(self, system: str, user: str) -> str:
            raise OllamaError("Cannot reach Ollama")

    rag = make_pipeline(DownLLM())
    rag.ingest_text("leave.md", "Full-time employees get 24 days of annual leave.")
    with TestClient(create_app(rag)) as c:
        res = c.post("/query", json={"question": "How many days of annual leave?"})
    assert res.status_code == 503
    assert "Cannot reach Ollama" in res.json()["detail"]
