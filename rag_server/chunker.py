from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ChunkedText:
    """Text plus chunk-local metadata produced by a chunking strategy."""

    content: str
    metadata: dict[str, object] = field(default_factory=dict)


class Chunker(Protocol):
    def split(self, text: str) -> list[str]:
        ...

    def split_with_metadata(self, text: str) -> list[ChunkedText]:
        ...


class TextChunker:
    """Small dependency-free chunker for the first RAG iteration.

    It prefers paragraph and sentence boundaries, then falls back to hard
    slicing for very long units. A production parser can replace this class
    without changing the ingestion service.
    """

    _paragraph_pattern = re.compile(r"\n\s*\n+")
    _sentence_pattern = re.compile(r"(?<=[.!?。！？])\s*")

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be between zero and chunk_size - 1",
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> list[str]:
        normalized = self._normalize(text)
        if not normalized:
            return []
        if len(normalized) <= self.chunk_size:
            return [normalized]

        units = self._units(normalized)
        chunks: list[str] = []
        current = ""

        for unit in units:
            if len(unit) > self.chunk_size:
                if current:
                    chunks.append(current)
                    current = self._overlap_tail(current)
                chunks.extend(self._split_long_unit(unit))
                current = ""
                continue

            candidate = self._join(current, unit)
            if current and len(candidate) > self.chunk_size:
                chunks.append(current)
                current = self._join(self._overlap_tail(current), unit)
            else:
                current = candidate

        if current:
            chunks.append(current)

        return [chunk for chunk in chunks if chunk]

    def split_with_metadata(self, text: str) -> list[ChunkedText]:
        return [ChunkedText(content=chunk) for chunk in self.split(text)]

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n")).strip()

    def _units(self, text: str) -> list[str]:
        units: list[str] = []
        for paragraph in self._paragraph_pattern.split(text):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            sentences = [
                sentence.strip()
                for sentence in self._sentence_pattern.split(paragraph)
                if sentence.strip()
            ]
            units.extend(sentences or [paragraph])
        return units

    def _split_long_unit(self, unit: str) -> list[str]:
        step = self.chunk_size - self.chunk_overlap
        chunks: list[str] = []
        start = 0
        while start < len(unit):
            chunk = unit[start:start + self.chunk_size].strip()
            if chunk:
                chunks.append(chunk)
            if start + self.chunk_size >= len(unit):
                break
            start += step
        return chunks

    def _overlap_tail(self, text: str) -> str:
        if self.chunk_overlap == 0:
            return ""
        return text[-self.chunk_overlap:]

    @staticmethod
    def _join(left: str, right: str) -> str:
        if not left:
            return right
        if not right:
            return left
        return f"{left} {right}".strip()


class MarkdownChunker(TextChunker):
    """Chunk Markdown by heading sections while retaining heading paths."""

    _heading_pattern = re.compile(
        r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$",
        flags=re.MULTILINE,
    )

    def split_with_metadata(self, text: str) -> list[ChunkedText]:
        normalized = self._normalize(text)
        if not normalized:
            return []

        sections: list[tuple[str, str]] = []
        headings: list[str] = []
        matches = list(self._heading_pattern.finditer(normalized))
        if not matches:
            sections = [(normalized, "")]
        else:
            prefix = normalized[: matches[0].start()].strip()
            if prefix:
                sections.append((prefix, ""))
            for index, match in enumerate(matches):
                level = len(match.group(1))
                heading = match.group(2).strip()
                headings = headings[: level - 1] + [heading]
                end = (
                    matches[index + 1].start()
                    if index + 1 < len(matches)
                    else len(normalized)
                )
                section = normalized[match.start():end].strip()
                if section:
                    sections.append((section, " > ".join(headings)))

        result: list[ChunkedText] = []
        for section, heading_path in sections:
            for chunk in TextChunker.split(self, section):
                metadata: dict[str, object] = {}
                if heading_path:
                    metadata["heading_path"] = heading_path
                result.append(ChunkedText(chunk, metadata))
        return result

    def split(self, text: str) -> list[str]:
        return [item.content for item in self.split_with_metadata(text)]
