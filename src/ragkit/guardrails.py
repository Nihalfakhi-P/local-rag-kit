import re
from dataclasses import dataclass

REFUSAL = "I don't know based on the provided documents."

# "[2]" as instructed, plus prose forms small models fall back to: "source 2", "source #2".
_CITATION = re.compile(r"\[(\d+)\]|\bsources?\s*#?\s*(\d+)\b", re.IGNORECASE)
# Tags that delimit sources in the prompt; a document must not be able to forge or close them.
_PROMPT_TAGS = re.compile(r"</?\s*(?:context|source)\b[^>]*>", re.IGNORECASE)


def sanitize_context(text: str) -> str:
    """Strip prompt-structure tags from retrieved text (basic prompt-injection hardening)."""
    return _PROMPT_TAGS.sub("", text)


def extract_citations(answer: str) -> list[int]:
    """Unique citation numbers in order of first appearance."""
    seen: dict[int, None] = {}
    for match in _CITATION.finditer(answer):
        seen.setdefault(int(match.group(1) or match.group(2)), None)
    return list(seen)


def is_refusal(answer: str) -> bool:
    normalized = answer.replace("\u2019", "'").lower()
    return "i don't know based on the provided documents" in normalized


@dataclass(frozen=True)
class GroundingReport:
    refused: bool
    citations: list[int]
    invalid_citations: list[int]

    @property
    def grounded(self) -> bool:
        """A refusal is grounded; an answer is grounded if it cites only real sources."""
        return self.refused or (bool(self.citations) and not self.invalid_citations)


def check_grounding(answer: str, n_sources: int) -> GroundingReport:
    citations = extract_citations(answer)
    return GroundingReport(
        refused=is_refusal(answer),
        citations=citations,
        invalid_citations=[c for c in citations if not 1 <= c <= n_sources],
    )
