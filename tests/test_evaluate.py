import json

import httpx
import pytest

from ragkit.evaluate import evaluate_answers, evaluate_retrieval, format_report, load_cases
from ragkit.ollama import OllamaEmbedder, OllamaError, OllamaLLM
from tests.conftest import EXAMPLES, FakeLLM, make_pipeline


def test_load_cases_defaults_to_answerable():
    cases = load_cases(EXAMPLES / "eval.jsonl")
    assert len(cases) == 19
    assert sum(not c.answerable for c in cases) == 4
    assert cases[0].expected_sources == ["leave-policy.md"]
    assert cases[0].must_include == ["24 days"]
    assert all(c.must_include for c in cases if c.answerable)
    assert all(not c.must_include for c in cases if not c.answerable)


def test_retrieval_metrics_on_sample_corpus(loaded):
    report = evaluate_retrieval(loaded, load_cases(EXAMPLES / "eval.jsonl"), k=5)
    assert report["cases"] == 15
    # Even a bag-of-words fake embedder + BM25 should retrieve the right doc almost always.
    assert report["hit_rate"] >= 0.9
    assert 0 < report["mrr"] <= 1


def test_answer_metrics_and_report():
    # Gate disabled: this test checks metric arithmetic, not the fake embedder's similarity scale.
    rag = make_pipeline(min_similarity=0.0)
    rag.ingest_path(EXAMPLES / "docs")
    cases = load_cases(EXAMPLES / "eval.jsonl")
    answers = evaluate_answers(rag, cases)
    assert answers["cases"] == 19
    # FakeLLM always answers with [1]: every answerable case is answered and validly cited,
    # and every unanswerable one is (wrongly) answered too.
    assert answers["answerable_answered"] == 1.0
    assert answers["unanswerable_refused"] == 0.0
    assert answers["citation_validity"] == 1.0
    assert answers["behaviour_accuracy"] == 15 / 19
    # ...but its canned reply contains none of the expected facts.
    assert answers["accuracy_cases"] == 15
    assert answers["answer_accuracy"] == 0.0
    text = format_report(evaluate_retrieval(rag, cases, 5), answers)
    assert "hit rate@5" in text and "valid citations" in text
    assert "wrong: What is the capital of Australia? -> answered" in text
    assert "correct content         0%  (15 checked)" in text
    assert "incorrect: What is the minimum password length?" in text


def test_answer_accuracy_checks_content_with_alternatives():
    def reply(user_prompt: str) -> str:
        if "password" in user_prompt.lower().split("question:")[-1]:
            return "Passwords must be at least 14 characters [1]."
        return "Pension is matched up to 5% of base salary [1]."

    rag = make_pipeline(FakeLLM(reply), min_similarity=0.0)
    rag.ingest_path(EXAMPLES / "docs")
    cases = [
        c
        for c in load_cases(EXAMPLES / "eval.jsonl")
        if "password" in c.question or "pension" in c.question
    ]
    answers = evaluate_answers(rag, cases)
    # "5%" satisfies the "5 percent|5%|five percent" alternatives.
    assert answers["answer_accuracy"] == 1.0
    assert answers["accuracy_cases"] == 2


def _client(handler) -> httpx.Client:
    return httpx.Client(base_url="http://ollama.test", transport=httpx.MockTransport(handler))


def test_ollama_embedder_batches_and_prefixes():
    seen: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        inputs = json.loads(request.content)["input"]
        seen.append(inputs)
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2]] * len(inputs)})

    emb = OllamaEmbedder(_client(handler), "nomic", "doc: ", "q: ", batch_size=2)
    assert len(emb.embed_documents(["a", "b", "c"])) == 3
    assert emb.embed_query("hi") == [0.1, 0.2]
    assert seen == [["doc: a", "doc: b"], ["doc: c"], ["q: hi"]]


def test_ollama_errors_are_actionable():
    def missing_model(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model not found"})

    with pytest.raises(OllamaError, match=r"ollama pull llama3\.2"):
        OllamaLLM(_client(missing_model), "llama3.2").generate("s", "u")

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(OllamaError, match="ollama serve"):
        OllamaLLM(_client(down), "llama3.2").generate("s", "u")
