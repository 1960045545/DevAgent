from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from collections.abc import Iterator, Sequence
from html.parser import HTMLParser
from pathlib import Path
from typing import Protocol

from rag_server.schemas import DocumentRecord


class DocumentLoader(Protocol):
    def load(self, path: str | Path) -> DocumentRecord:
        ...

    def iter_documents(
        self,
        root: str | Path | None = None,
    ) -> Iterator[DocumentRecord]:
        ...


class _HTMLTextParser(HTMLParser):
    _ignored_tags = {"script", "style", "noscript", "template"}

    def __init__(self) -> None:
        super().__init__()
        self._ignored_depth = 0
        self.parts: list[str] = []
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self._ignored_tags:
            self._ignored_depth += 1
        if tag == "title":
            self._in_title = True
        if tag in {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._ignored_tags and self._ignored_depth:
            self._ignored_depth -= 1
        if tag == "title":
            self._in_title = False
        if tag in {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        text = " ".join(data.split())
        if not text:
            return
        if self._in_title:
            self.title += text
        self.parts.append(text)


class LocalDocumentLoader:
    """Load supported local files into the common document contract.

    This loader intentionally performs no indexing and ships no sample data.
    It is suitable for a later ingestion worker as well as local imports.
    """

    DEFAULT_EXTENSIONS = (
        ".txt",
        ".md",
        ".markdown",
        ".html",
        ".htm",
        ".json",
        ".csv",
    )
    DEFAULT_IGNORED_DIRECTORIES = frozenset({
        ".git",
        ".worktrees",
        ".transcripts",
        "__pycache__",
    })

    def __init__(
        self,
        root: str | Path,
        *,
        extensions: Sequence[str] | None = None,
        ignored_directories: Sequence[str] | None = None,
        tenant_id: str | None = None,
        permission_ids: Sequence[str] = (),
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.exists() or not self.root.is_dir():
            raise ValueError(f"document root is not a directory: {self.root}")
        self.extensions = frozenset(
            self._normalize_extension(item)
            for item in (extensions or self.DEFAULT_EXTENSIONS)
        )
        self.ignored_directories = frozenset(
            ignored_directories or self.DEFAULT_IGNORED_DIRECTORIES
        )
        self.tenant_id = tenant_id
        self.permission_ids = tuple(str(item) for item in permission_ids)

    def load(self, path: str | Path) -> DocumentRecord:
        resolved = self._safe_path(path)
        if not resolved.is_file():
            raise ValueError(f"document path is not a file: {resolved}")
        suffix = resolved.suffix.lower()
        if suffix not in self.extensions:
            raise ValueError(
                f"unsupported document extension: {suffix or '<none>'}"
            )

        content, title = self._read_content(resolved, suffix)
        if not content.strip():
            raise ValueError(f"document is empty: {resolved}")
        relative = resolved.relative_to(self.root).as_posix()
        title = title.strip() or resolved.stem
        return DocumentRecord(
            doc_id=self._document_id(relative),
            title=title,
            content=content,
            source_uri=relative,
            tenant_id=self.tenant_id,
            permission_ids=self.permission_ids,
            metadata={
                "relative_path": relative,
                "file_name": resolved.name,
                "extension": suffix,
            },
            source_type=suffix.lstrip(".") or "text",
        )

    def iter_documents(self, root: str | Path | None = None) -> Iterator[DocumentRecord]:
        scan_root = self.root if root is None else self._safe_path(root)
        if not scan_root.is_dir():
            raise ValueError(f"document scan root is not a directory: {scan_root}")
        for path in sorted(scan_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in self.extensions:
                continue
            if any(part in self.ignored_directories for part in path.parts):
                continue
            yield self.load(path)

    def _safe_path(self, path: str | Path) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        resolved = candidate.expanduser().resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("document path must stay within the loader root") from exc
        return resolved

    @staticmethod
    def _normalize_extension(value: str) -> str:
        normalized = value.strip().lower()
        return normalized if normalized.startswith(".") else f".{normalized}"

    @staticmethod
    def _document_id(relative_path: str) -> str:
        digest = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:20]
        return f"file-{digest}"

    @staticmethod
    def _read_content(path: Path, suffix: str) -> tuple[str, str]:
        raw = path.read_text(encoding="utf-8", errors="replace")
        if suffix == ".json":
            value = json.loads(raw)
            if isinstance(value, dict) and isinstance(value.get("content"), str):
                return value["content"], str(value.get("title") or "")
            return json.dumps(value, ensure_ascii=False, indent=2, default=str), ""
        if suffix == ".csv":
            rows = list(csv.reader(io.StringIO(raw)))
            return "\n".join(" | ".join(row) for row in rows), ""
        if suffix in {".html", ".htm"}:
            parser = _HTMLTextParser()
            parser.feed(raw)
            content = re.sub(r"\n{3,}", "\n\n", "\n".join(parser.parts)).strip()
            return content, parser.title
        return raw, ""
