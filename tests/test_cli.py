import pytest

from ragkit import cli
from ragkit.ollama import OllamaError
from ragkit.pipeline import RAGPipeline
from tests.conftest import EXAMPLES, FakeLLM, make_pipeline


@pytest.fixture
def rag(monkeypatch) -> RAGPipeline:
    pipeline = make_pipeline(FakeLLM("Full-time employees get 24 days [1]."))
    monkeypatch.setattr(RAGPipeline, "from_settings", classmethod(lambda cls, s=None: pipeline))
    return pipeline


def test_ingest_then_ask(rag, capsys):
    assert cli.main(["ingest", str(EXAMPLES / "docs")]) == 0
    out = capsys.readouterr().out
    assert "leave-policy.md" in out and "Indexed 4 documents" in out

    assert cli.main(["ask", "How many days of annual leave do full-time employees get?"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Full-time employees get 24 days [1].")
    assert " *[1] leave-policy.md" in out


def test_ingest_missing_path(rag, capsys):
    assert cli.main(["ingest", "does/not/exist"]) == 2
    assert "does not exist" in capsys.readouterr().err


def test_eval_retrieval_only(rag, capsys):
    rag.ingest_path(EXAMPLES / "docs")
    assert cli.main(["eval", str(EXAMPLES / "eval.jsonl"), "--retrieval-only"]) == 0
    out = capsys.readouterr().out
    assert "hit rate@5" in out and "Answers" not in out


def test_ollama_errors_exit_cleanly(monkeypatch, capsys):
    class Down:
        def ingest_path(self, path):
            raise OllamaError("Cannot reach Ollama")

    monkeypatch.setattr(RAGPipeline, "from_settings", classmethod(lambda cls, s=None: Down()))
    assert cli.main(["ingest", str(EXAMPLES / "docs")]) == 1
    assert "Cannot reach Ollama" in capsys.readouterr().err
