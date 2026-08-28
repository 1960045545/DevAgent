from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rag_server.config import RagSettings
from rag_server.schemas import SearchHit


class Qwen3Reranker:
    """Qwen3 causal-LM reranker using the yes/no next-token probability."""

    _system_prompt = (
        "<|im_start|>system\n"
        "Judge whether the Document meets the requirements based on the "
        "Query and the Instruct provided. Note that the answer can only be "
        '"yes" or "no".'
        "<|im_end|>\n"
        "<|im_start|>user\n"
    )
    _assistant_suffix = (
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )

    def __init__(
        self,
        settings: RagSettings,
    ) -> None:
        if not settings.reranker_model:
            raise ValueError("reranker model name must not be empty")
        self.settings = settings
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._prefix_tokens: list[int] = []
        self._suffix_tokens: list[int] = []
        self._true_token_id: int | None = None
        self._false_token_id: int | None = None

    def rank(
        self,
        query: str,
        candidates: Sequence[SearchHit],
        limit: int,
    ) -> list[SearchHit]:
        if not candidates:
            return []
        tokenizer, model = self._ensure_loaded()

        pairs = [
            self._format_instruction(
                query=query,
                document=hit.content,
            )
            for hit in candidates
        ]
        scores: list[float] = []
        for start in range(0, len(pairs), self.settings.reranker_batch_size):
            batch = pairs[start:start + self.settings.reranker_batch_size]
            inputs = self._prepare_inputs(tokenizer, batch)
            with self._torch.inference_mode():
                outputs = model(**inputs)
                logits = outputs.logits[:, -1, :]
                true_logits = logits[:, self._true_token_id]
                false_logits = logits[:, self._false_token_id]
                probabilities = self._torch.softmax(
                    self._torch.stack(
                        [false_logits, true_logits],
                        dim=1,
                    ),
                    dim=1,
                )
                scores.extend(
                    probabilities[:, 1].detach().float().cpu().tolist(),
                )

        ranked = []
        for hit, score in zip(candidates, scores):
            hit.rerank_score = float(score)
            ranked.append(hit)
        ranked.sort(
            key=lambda item: item.rerank_score or 0.0,
            reverse=True,
        )
        return ranked[:limit]

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "torch and transformers are required for Qwen reranking",
            ) from exc

        device = self._resolve_device(self.settings.reranker_device)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(
            self.settings.reranker_model,
            padding_side="left",
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.settings.reranker_model,
            torch_dtype=dtype,
        )
        model.to(device)
        model.eval()

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._prefix_tokens = tokenizer.encode(
            self._system_prompt,
            add_special_tokens=False,
        )
        self._suffix_tokens = tokenizer.encode(
            self._assistant_suffix,
            add_special_tokens=False,
        )
        self._true_token_id = tokenizer.convert_tokens_to_ids("yes")
        self._false_token_id = tokenizer.convert_tokens_to_ids("no")
        if (
            self._true_token_id is None
            or self._false_token_id is None
            or self._true_token_id < 0
            or self._false_token_id < 0
        ):
            raise RuntimeError("Qwen reranker yes/no token ids are unavailable")
        return tokenizer, model

    def _prepare_inputs(
        self,
        tokenizer: Any,
        pairs: Sequence[str],
    ) -> Any:
        available_length = (
            self.settings.reranker_max_length
            - len(self._prefix_tokens)
            - len(self._suffix_tokens)
        )
        if available_length <= 0:
            raise ValueError("reranker_max_length is too small")

        encoded = tokenizer(
            list(pairs),
            add_special_tokens=False,
            padding=False,
            truncation=True,
            max_length=available_length,
        )
        input_ids = [
            self._prefix_tokens + ids + self._suffix_tokens
            for ids in encoded["input_ids"]
        ]
        inputs = tokenizer.pad(
            {"input_ids": input_ids},
            padding=True,
            return_tensors="pt",
        )
        return {
            key: value.to(self._model.device)
            for key, value in inputs.items()
        }

    def _format_instruction(
        self,
        *,
        query: str,
        document: str,
    ) -> str:
        return (
            f"<Instruct>: {self.settings.reranker_instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}"
        )

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
