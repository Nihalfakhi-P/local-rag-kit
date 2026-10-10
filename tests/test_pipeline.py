from ragkit.guardrails import REFUSAL
from tests.conftest import EXAMPLES, FakeLLM, make_pipeline


def test_ingest_directory_indexes_every_document(pipeline):
    counts = pipeline.ingest_path(EXAMPLES / "docs")
    assert set(counts) == {"expenses.md", "it-security.md", "leave-policy.md", "onboarding.md"}
    assert pipeline.store.count() == sum(counts.values())
    assert pipeline.store.sources() == dict(sorted(counts.items()))


def test_reingesting_a_source_replaces_its_chunks(pipeline):
    pipeline.ingest_text("note.md", "Version one. " * 100)
    pipeline.ingest_text("note.md", "Version two is short.")
    assert pipeline.store.sources() == {"note.md": 1}
    assert pipeline.retrieve("version").hits[0].chunk.text == "Version two is short."


def test_answer_is_built_from_retrieved_context(loaded, llm):
    answer = loaded.ask("How many days of paid annual leave do full-time employees get?")
    assert not answer.refused
    assert answer.grounded
    assert answer.citations == [1]
    system, user = llm.calls[0]
    assert "<context>" in user and "24 days of paid annual leave" in user
    assert "ONLY the numbered sources" in system


def test_off_topic_question_is_refused_without_calling_the_llm(loaded, llm):
    answer = loaded.ask("Who painted the Mona Lisa?")
    assert answer.refused and answer.text == REFUSAL
    assert answer.reason == "no sufficiently relevant context"
    assert llm.calls == []


def test_model_refusal_is_normalised():
    rag = make_pipeline(FakeLLM("I don\u2019t know based on the provided documents."))
    rag.ingest_path(EXAMPLES / "docs")
    answer = rag.ask("Does the expense policy cover gym memberships?")
    assert answer.refused and answer.text == REFUSAL and answer.sources == []


def test_hallucinated_citation_is_flagged():
    rag = make_pipeline(FakeLLM("You get 24 days [1] and a free car [9]."))
    rag.ingest_path(EXAMPLES / "docs")
    answer = rag.ask("How much annual leave do employees get?")
    assert not answer.refused
    assert not answer.grounded
    assert answer.reason == "missing or invalid citations"


def test_strict_mode_replaces_ungrounded_answers_with_refusal():
    rag = make_pipeline(FakeLLM("You get 24 days."), strict_grounding=True)
    rag.ingest_path(EXAMPLES / "docs")
    answer = rag.ask("How much annual leave do employees get?")
    assert answer.refused and answer.text == REFUSAL
    assert answer.reason == "answer failed citation check"


def test_empty_index_refuses(pipeline, llm):
    assert pipeline.ask("Anything?").refused
    assert llm.calls == []
