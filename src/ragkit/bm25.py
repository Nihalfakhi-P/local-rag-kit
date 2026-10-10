import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")
_STOPWORDS = frozenset(
    (
        "a an and are as at be by can do does for from has have how i if in is it its of on or "
        "our so that the their there this to was we what when where which who why will with "
        "you your"
    ).split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, keeping codes like `e-4021` intact, minus common stopwords."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


class BM25:
    """Okapi BM25 keyword index.

    Complements dense retrieval on exact terms that embeddings blur: error codes,
    product names, IDs, acronyms.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._ids: list[str] = []
        self._tfs: list[Counter[str]] = []
        self._lengths: list[int] = []
        self._idf: dict[str, float] = {}
        self._avg_len = 0.0

    def index(self, docs: list[tuple[str, str]]) -> None:
        self._ids = [doc_id for doc_id, _ in docs]
        self._tfs = [Counter(tokenize(text)) for _, text in docs]
        self._lengths = [sum(tf.values()) for tf in self._tfs]
        n = len(docs)
        self._avg_len = (sum(self._lengths) / n) if n else 0.0
        df: Counter[str] = Counter()
        for tf in self._tfs:
            df.update(tf.keys())
        self._idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        terms = [t for t in tokenize(query) if t in self._idf]
        if not terms or not self._ids:
            return []
        scores: list[tuple[str, float]] = []
        for doc_id, tf, length in zip(self._ids, self._tfs, self._lengths, strict=True):
            score = 0.0
            for term in terms:
                f = tf.get(term, 0)
                if f:
                    norm = 1 - self.b + self.b * length / (self._avg_len or 1)
                    score += self._idf[term] * f * (self.k1 + 1) / (f + self.k1 * norm)
            if score > 0:
                scores.append((doc_id, score))
        scores.sort(key=lambda s: s[1], reverse=True)
        return scores[:k]
