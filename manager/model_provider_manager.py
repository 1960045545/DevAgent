from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Literal


ModelPurpose = Literal["chat", "history", "profile"]


@dataclass
class ModelProviderConfig:
    name: str
    base_url: str | None
    api_key: str | None = None
    chat_model_id: str | None = None
    history_model_id: str | None = None
    profile_model_id: str | None = None

    def model_for(
        self,
        purpose: ModelPurpose,
        override_model_id: str | None = None,
    ) -> str | None:
        if override_model_id:
            return override_model_id

        if purpose == "history":
            return self.history_model_id or self.chat_model_id

        if purpose == "profile":
            return self.profile_model_id or self.chat_model_id

        return self.chat_model_id


@dataclass
class ModelProviderState:
    failures: int = 0
    unavailable_until: float = 0.0
    last_error: str | None = None

    def in_cooldown(self, now: float | None = None) -> bool:
        current_time = time.monotonic() if now is None else now
        return self.unavailable_until > current_time

    def needs_probe(self, now: float | None = None) -> bool:
        current_time = time.monotonic() if now is None else now
        return (
            self.unavailable_until > 0
            and self.unavailable_until <= current_time
        )


class ModelProviderHealth:
    _states: dict[str, ModelProviderState] = {}
    _lock = threading.RLock()

    @classmethod
    def get_state(cls, name: str) -> ModelProviderState:
        with cls._lock:
            if name not in cls._states:
                cls._states[name] = ModelProviderState()

            return cls._states[name]

    @classmethod
    def can_try(cls, name: str) -> bool:
        with cls._lock:
            return not cls.get_state(name).in_cooldown()

    @classmethod
    def needs_probe(cls, name: str) -> bool:
        with cls._lock:
            return cls.get_state(name).needs_probe()

    @classmethod
    def record_failure(
        cls,
        name: str,
        *,
        attempt: int,
        error: BaseException,
    ) -> None:
        with cls._lock:
            state = cls.get_state(name)
            state.failures = attempt
            state.last_error = str(error)

    @classmethod
    def mark_unavailable(
        cls,
        name: str,
        *,
        error: BaseException,
        cooldown_seconds: float,
    ) -> None:
        with cls._lock:
            state = cls.get_state(name)
            state.unavailable_until = (
                time.monotonic() + cooldown_seconds
            )
            state.last_error = str(error)

    @classmethod
    def record_success(cls, name: str) -> None:
        with cls._lock:
            state = cls.get_state(name)
            state.failures = 0
            state.unavailable_until = 0.0
            state.last_error = None
