from collections.abc import Iterator
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = frozenset({".md", ".markdown", ".txt", ".pdf"})


class UnsupportedFileError(ValueError):
    pass


def load_bytes(filename: str, data: bytes) -> str:
    """Extract plain text from an uploaded or on-disk file."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileError(
            f"Unsupported file type {suffix or '(none)'!r}; "
            f"expected one of {sorted(SUPPORTED_EXTENSIONS)}"
        )
    if suffix == ".pdf":
        reader = PdfReader(BytesIO(data))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return data.decode("utf-8", errors="replace")


def iter_documents(path: Path) -> Iterator[tuple[str, str]]:
    """Yield (source, text) for a file, or for every supported file under a directory.

    `source` is the path relative to the given directory (POSIX style), so ids stay
    stable no matter where the corpus is checked out.
    """
    if path.is_file():
        yield path.name, load_bytes(path.name, path.read_bytes())
        return
    for file in sorted(path.rglob("*")):
        if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield file.relative_to(path).as_posix(), load_bytes(file.name, file.read_bytes())
