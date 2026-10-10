"""Offline evaluation: retrieval quality and answer-level guardrail behaviour."""

import json
from dataclasses import dataclass, field
from pathlib import Path

from ragkit.pipeline import RAGPipeline


@dataclass(frozen=True)
class EvalCase:
    question: str
    expected_sources: list[str] = field(default_factory=list)
    answerable: bool = True
    # Case-insensitive substrings a correct answer must contain, e.g. ["24 days"].
    # "a|b" accepts either alternative, e.g. ["5 percent|5%"].
    must_include: list[str] = field(default_factory=list)


def load_cases(path: Path) -> list[EvalCase]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            cases.append(
                EvalCase(
                    question=row["question"],
                    expected_sources=row.get("expected_sources", []),
                    answerable=row.get("answerable", True),
                    must_include=row.get("must_include", []),
                )
            )
    return cases


def evaluate_retrieval(pipeline: RAGPipeline, cases: list[EvalCase], k: int) -> dict:
    """Hit rate@k and MRR over answerable cases (first chunk from an expected source)."""
    rows = []
    for case in (c for c in cases if c.answerable):
        sources = [h.chunk.source for h in pipeline.retrieve(case.question, k).hits]
        rank = next((i for i, s in enumerate(sources, 1) if s in case.expected_sources), None)
        rows.append({"question": case.question, "rank": rank, "retrieved": sources})
    n = len(rows) or 1
    return {
        "k": k,
        "cases": len(rows),
        "hit_rate": sum(r["rank"] is not None for r in rows) / n,
        "mrr": sum(1 / r["rank"] for r in rows if r["rank"]) / n,
        "rows": rows,
    }


def evaluate_answers(pipeline: RAGPipeline, cases: list[EvalCase]) -> dict:
    """Does the system answer what it should, refuse what it can't, cite, and get it right?"""
    rows = []
    for case in cases:
        answer = pipeline.ask(case.question)
        text = answer.text.lower()
        rows.append(
            {
                "question": case.question,
                "answerable": case.answerable,
                "refused": answer.refused,
                "grounded": answer.grounded,
                "correct_behaviour": answer.refused != case.answerable,
                # None when the case defines no expected content.
                "correct": all(_contains_any(text, s) for s in case.must_include)
                if case.must_include and not answer.refused
                else None,
                "answer": answer.text,
            }
        )
    answered = [r for r in rows if not r["refused"]]
    checked = [r for r in answered if r["correct"] is not None]
    n = len(rows) or 1
    return {
        "cases": len(rows),
        "behaviour_accuracy": sum(r["correct_behaviour"] for r in rows) / n,
        "unanswerable_refused": _ratio(rows, lambda r: not r["answerable"], lambda r: r["refused"]),
        "answerable_answered": _ratio(rows, lambda r: r["answerable"], lambda r: not r["refused"]),
        "citation_validity": (sum(r["grounded"] for r in answered) / len(answered))
        if answered
        else 1.0,
        "answer_accuracy": (sum(r["correct"] for r in checked) / len(checked)) if checked else None,
        "accuracy_cases": len(checked),
        "rows": rows,
    }


def _contains_any(text: str, alternatives: str) -> bool:
    return any(alt.strip().lower() in text for alt in alternatives.split("|"))


def _ratio(rows: list[dict], where, metric) -> float:
    subset = [r for r in rows if where(r)]
    return sum(bool(metric(r)) for r in subset) / len(subset) if subset else 1.0


def format_report(retrieval: dict, answers: dict | None) -> str:
    lines = [
        f"Retrieval  ({retrieval['cases']} answerable questions, k={retrieval['k']})",
        f"  hit rate@{retrieval['k']}   {retrieval['hit_rate']:.0%}",
        f"  MRR          {retrieval['mrr']:.3f}",
    ]
    misses = [r["question"] for r in retrieval["rows"] if r["rank"] is None]
    lines += [f"  miss: {q}" for q in misses]
    if answers:
        lines += [
            "",
            f"Answers    ({answers['cases']} questions)",
            f"  correct answer/refuse   {answers['behaviour_accuracy']:.0%}",
            f"  answerable answered     {answers['answerable_answered']:.0%}",
            f"  unanswerable refused    {answers['unanswerable_refused']:.0%}",
            f"  valid citations         {answers['citation_validity']:.0%}",
        ]
        if answers["answer_accuracy"] is not None:
            lines.append(
                f"  correct content         {answers['answer_accuracy']:.0%}"
                f"  ({answers['accuracy_cases']} checked)"
            )
        lines += [
            f"  wrong: {r['question']} -> {'refused' if r['refused'] else 'answered'}"
            for r in answers["rows"]
            if not r["correct_behaviour"]
        ]
        lines += [
            f"  incorrect: {r['question']} -> {r['answer'][:80]}"
            for r in answers["rows"]
            if r["correct"] is False
        ]
    return "\n".join(lines)
