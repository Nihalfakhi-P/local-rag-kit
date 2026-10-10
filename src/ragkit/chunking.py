import hashlib
import re

from ragkit.models import Chunk

# Split on the largest natural boundary that fits: paragraph, line, sentence, word.
SEPARATORS = ("\n\n", "\n", ". ", " ")


def _split(text: str, size: int, separators: tuple[str, ...]) -> list[str]:
    if len(text) <= size:
        return [text]
    for i, sep in enumerate(separators):
        if sep not in text:
            continue
        parts = text.split(sep)
        pieces: list[str] = []
        for j, part in enumerate(parts):
            piece = part + (sep if j < len(parts) - 1 else "")
            if len(piece) > size:
                pieces.extend(_split(piece, size, separators[i + 1 :]))
            elif piece:
                pieces.append(piece)
        return pieces
    # No boundary left (e.g. a very long token): hard cut.
    return [text[k : k + size] for k in range(0, len(text), size)]


def split_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """Recursively split text into chunks of at most `chunk_size` characters.

    Consecutive chunks share up to `overlap` characters (cut at a word boundary),
    so facts that straddle a boundary stay retrievable.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    text = re.sub(r"\n{3,}", "\n\n", text.replace("\r\n", "\n")).strip()
    if not text:
        return []

    chunks: list[str] = []
    current = ""
    for piece in _split(text, chunk_size, SEPARATORS):
        if len(current) + len(piece) <= chunk_size:
            current += piece
            continue
        if current.strip():
            chunks.append(current.strip())
        tail = current[-overlap:] if overlap else ""
        if " " in tail:
            tail = tail[tail.index(" ") + 1 :]
        current = tail + piece if len(tail) + len(piece) <= chunk_size else piece
    if current.strip():
        chunks.append(current.strip())
    return chunks


def chunk_id(source: str, index: int, text: str) -> str:
    return hashlib.sha1(f"{source}\x00{index}\x00{text}".encode()).hexdigest()[:20]


def chunk_document(source: str, text: str, chunk_size: int, overlap: int) -> list[Chunk]:
    return [
        Chunk(id=chunk_id(source, i, piece), text=piece, source=source, index=i)
        for i, piece in enumerate(split_text(text, chunk_size, overlap))
    ]
