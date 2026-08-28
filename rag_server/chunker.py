from __future__ import annotations

import re


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
