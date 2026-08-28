from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class QwenEmbeddingProvider:
    """Qwen embedding adapter backed by sentence-transformers."""

    def __init__(
        self,
        *,
        model_name: str,
        device: str = "auto",
        batch_size: int = 8,
        max_length: int = 8192,
        dimension: int = 0,
    ) -> None:
        if not model_name:
            raise ValueError("embedding model name must not be empty")
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.batch_size = batch_size
        self.max_length = max_length
        self._model: Any | None = None
        self._dimension: int | None = dimension or None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._ensure_loaded()
        if self._dimension is None:
            raise RuntimeError("embedding dimension is unavailable")
        return self._dimension

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        if not texts:
            return []
        model = self._ensure_loaded()
        embeddings = model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        model = self._ensure_loaded()
        embedding = model.encode(
            text,
            prompt_name="query",
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embedding.tolist()

    def _ensure_loaded(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for Qwen embeddings",
            ) from exc

        self._model = SentenceTransformer(
            self.model_name,
            device=self.device,
        )
        self._model.max_seq_length = self.max_length
        self._dimension = int(
            self._model.get_sentence_embedding_dimension(),
        )
        return self._model

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
