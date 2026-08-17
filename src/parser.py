"""Parse vector-podcast markdown files into structured chunks for indexing."""
import re
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class PodcastChunk:
    episode_title: str
    url: str
    pub_date: str
    description: str
    chunk_text: str
    chunk_index: int
    total_chunks: int = 0
    doc_id: str = ""


def _parse_frontmatter(content: str) -> tuple[dict, str]:
    """Split YAML frontmatter from body text."""
    if not content.startswith("---"):
        return {}, content
    end = content.find("\n---\n", 4)
    if end == -1:
        return {}, content
    try:
        meta = yaml.safe_load(content[4:end])
    except yaml.YAMLError:
        meta = {}
    body = content[end + 5:].strip()
    return meta or {}, body


def _clean_text(text: str) -> str:
    """Remove HTML tags and normalize whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def load_podcast_chunks(
    data_dir: str | Path,
    chunk_size: int = 400,
    overlap: int = 50,
) -> list[PodcastChunk]:
    """Load all podcast markdown files and return a flat list of text chunks."""
    data_dir = Path(data_dir)
    all_chunks: list[PodcastChunk] = []

    for md_file in sorted(data_dir.glob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(content)

        title = meta.get("title", md_file.stem)
        url = meta.get("url", "")
        pub_date = meta.get("pub_date", "")
        description = _clean_text(meta.get("description", ""))

        body_clean = _clean_text(body)
        if not body_clean:
            continue

        text_chunks = _chunk_text(body_clean, chunk_size, overlap)
        for i, chunk in enumerate(text_chunks):
            all_chunks.append(
                PodcastChunk(
                    episode_title=title,
                    url=url,
                    pub_date=pub_date,
                    description=description[:500],
                    chunk_text=chunk,
                    chunk_index=i,
                    total_chunks=len(text_chunks),
                    doc_id=f"{md_file.stem}_{i}",
                )
            )

    return all_chunks
