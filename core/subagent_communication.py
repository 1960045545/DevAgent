from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from core.tool_space import ToolSpec


_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MESSAGE_TYPES = {"context", "request", "response", "artifact", "status"}
_PROTOCOL_KINDS = {
    "task",
    "context",
    "artifact",
    "plan_approval",
    "shutdown",
}
_PROTOCOL_REQUEST_TYPES = {
    f"{kind}_request" for kind in _PROTOCOL_KINDS
}
_PROTOCOL_RESPONSE_TYPES = {
    f"{kind}_response" for kind in _PROTOCOL_KINDS
}
_PROTOCOL_MESSAGE_TYPES = (
    _PROTOCOL_REQUEST_TYPES | _PROTOCOL_RESPONSE_TYPES
)
_ALL_MESSAGE_TYPES = _MESSAGE_TYPES | _PROTOCOL_MESSAGE_TYPES
_RESPONSE_FOR_REQUEST = {
    f"{kind}_request": f"{kind}_response"
    for kind in _PROTOCOL_KINDS
}
_REQUEST_FOR_RESPONSE = {
    response: request
    for request, response in _RESPONSE_FOR_REQUEST.items()
}
_PROTOCOL_STATUS = {"pending", "approved", "rejected"}
_SENSITIVE_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    ".git",
    ".ssh",
    "id_rsa",
    "id_ed25519",
}
_SENSITIVE_SUFFIXES = {".key", ".pem", ".p12", ".pfx"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


@dataclass(frozen=True, slots=True)
class CommunicationPaths:
    """Stable paths for one isolated parent request."""

    request_root: Path
    protocols: Path
    shared_context: Path
    shared_artifacts: Path


@dataclass
class ProtocolState:
    """State of one request-response protocol card."""

    request_id: str
    protocol_type: str
    sender: str
    target: str
    status: str
    payload: dict[str, Any]
    created_at: str
    updated_at: str
    request_message_id: str | None = None
    response_message_id: str | None = None
    response_payload: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "protocol_request_id": self.request_id,
            "type": self.protocol_type,
            "protocol_type": self.protocol_type,
            "sender": self.sender,
            "target": self.target,
            "status": self.status,
            "payload": dict(self.payload),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "request_message_id": self.request_message_id,
            "response_message_id": self.response_message_id,
            "response_payload": (
                dict(self.response_payload)
                if self.response_payload is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ProtocolState":
        required = {
            "request_id",
            "type",
            "sender",
            "target",
            "status",
            "payload",
            "created_at",
            "updated_at",
        }
        missing = required - set(value)
        if missing:
            raise ValueError(
                "protocol state is missing fields: "
                + ", ".join(sorted(missing))
            )
        payload = value["payload"]
        response_payload = value.get("response_payload")
        if not isinstance(payload, dict):
            raise ValueError("protocol state payload must be an object")
        if response_payload is not None and not isinstance(response_payload, dict):
            raise ValueError("protocol response payload must be an object")
        return cls(
            request_id=str(value["request_id"]),
            protocol_type=str(value["type"]),
            sender=str(value["sender"]),
            target=str(value["target"]),
            status=str(value["status"]),
            payload=dict(payload),
            created_at=str(value["created_at"]),
            updated_at=str(value["updated_at"]),
            request_message_id=(
                str(value["request_message_id"])
                if value.get("request_message_id")
                else None
            ),
            response_message_id=(
                str(value["response_message_id"])
                if value.get("response_message_id")
                else None
            ),
            response_payload=(
                dict(response_payload)
                if response_payload is not None
                else None
            ),
        )


class SubagentCommunication:
    """Request-scoped, file-backed communication for parent and child agents.

    The store owns all path construction. Callers only provide logical task,
    message, or artifact names, so a child agent cannot use these APIs as a
    general filesystem escape hatch.
    """

    def __init__(
        self,
        workspace_root: str | Path,
        request_id: str | None = None,
        *,
        max_content_chars: int = 100_000,
        max_artifact_chars: int = 1_000_000,
    ) -> None:
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        if not self.workspace_root.exists() or not self.workspace_root.is_dir():
            raise ValueError("workspace_root must be an existing directory")

        self.request_id = self._validate_id(
            request_id or uuid4().hex,
            "request_id",
        )
        self.max_content_chars = max(1, max_content_chars)
        self.max_artifact_chars = max(1, max_artifact_chars)
        self._lock = RLock()

        sandbox_root = self.workspace_root / ".agent_sandbox"
        self._assert_inside_workspace(sandbox_root, allow_missing=True)
        request_root = sandbox_root / "subtasks" / self.request_id
        self._assert_inside_workspace(request_root, allow_missing=True)
        self.paths = CommunicationPaths(
            request_root=request_root,
            protocols=request_root / "protocols",
            shared_context=request_root / "shared" / "context",
            shared_artifacts=request_root / "shared" / "artifacts",
        )
        self._initialize()

    @property
    def root(self) -> Path:
        return self.paths.request_root

    @property
    def relative_root(self) -> str:
        return self.root.relative_to(self.workspace_root).as_posix()

    def prompt_context(self, task_id: str) -> str:
        task_id = self._validate_id(task_id, "task_id")
        return (
            "## File-backed sub-agent communication\n"
            f"request_id: {self.request_id}\n"
            f"current_task_id: {task_id}\n"
            f"communication_root: {self.relative_root}\n"
            "The communication tools are scoped to this request. Use them "
            "for messages, dependency results, and artifacts. Do not invent "
            "absolute paths or access another request."
        )

    def register_task(
        self,
        task_id: str,
        *,
        task: str = "",
        summary: str = "",
        dependencies: list[str] | None = None,
    ) -> dict[str, Any]:
        task_id = self._validate_id(task_id, "task_id")
        task_root = self._task_root(task_id)
        task_root.mkdir(parents=True, exist_ok=True)
        (task_root / "messages").mkdir(exist_ok=True)
        (task_root / "artifacts").mkdir(exist_ok=True)
        inbox_state_path = task_root / "inbox_state.json"
        if not inbox_state_path.exists():
            self._atomic_write_json(
                inbox_state_path,
                {
                    "request_id": self.request_id,
                    "task_id": task_id,
                    "consumed_message_ids": [],
                    "last_consumed_message_id": None,
                    "updated_at": _utc_now(),
                },
            )
        input_path = task_root / "input.json"
        if not input_path.exists():
            self._atomic_write_json(
                input_path,
                {
                    "request_id": self.request_id,
                    "task_id": task_id,
                    "task": task,
                    "summary": summary,
                    "dependencies": list(dependencies or []),
                    "created_at": _utc_now(),
                },
            )

        with self._lock:
            manifest = self._read_manifest()
            tasks = manifest.setdefault("tasks", {})
            existing = tasks.get(task_id, {})
            tasks[task_id] = {
                **existing,
                "task_id": task_id,
                "task": task or existing.get("task", ""),
                "summary": summary or existing.get("summary", ""),
                "dependencies": list(
                    dependencies
                    if dependencies is not None
                    else existing.get("dependencies", [])
                ),
            }
            manifest["updated_at"] = _utc_now()
            self._atomic_write_json(self.root / "manifest.json", manifest)
        return {
            "task_id": task_id,
            "path": self._relative_to_workspace(task_root),
        }

    def write_task_input(
        self,
        task_id: str,
        *,
        task: str,
        instructions: str,
        summary: str = "",
        dependencies: list[str] | None = None,
        tool_names: list[str] | None = None,
    ) -> dict[str, Any]:
        task_id = self._validate_id(task_id, "task_id")
        if not instructions.strip():
            raise ValueError("instructions cannot be empty")
        if len(instructions) > self.max_content_chars:
            raise ValueError("instructions exceed communication limit")
        self.register_task(
            task_id,
            task=task,
            summary=summary,
            dependencies=dependencies,
        )
        payload = {
            "request_id": self.request_id,
            "task_id": task_id,
            "task": task,
            "instructions": instructions,
            "summary": summary,
            "dependencies": list(dependencies or []),
            "tool_names": list(tool_names or []),
            "created_at": _utc_now(),
        }
        path = self._task_root(task_id) / "input.json"
        self._atomic_write_json(path, payload)
        return {
            "task_id": task_id,
            "path": self._relative_to_workspace(path),
        }

    def write_result(
        self,
        task_id: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        task_id = self._validate_id(task_id, "task_id")
        if not isinstance(result, dict):
            raise ValueError("task result must be a JSON object")
        self.register_task(task_id)
        result = dict(result)
        artifacts = result.get("artifacts")
        if artifacts is not None:
            if not isinstance(artifacts, list):
                raise ValueError("task result artifacts must be an array")
            result["artifacts"] = [
                self._normalize_declared_artifact_reference(str(path))
                for path in artifacts
                if path
            ]
        payload = {
            "request_id": self.request_id,
            "task_id": task_id,
            "updated_at": _utc_now(),
            **result,
        }
        path = self._task_root(task_id) / "result.json"
        self._atomic_write_json(path, payload)
        return {
            "task_id": task_id,
            "path": self._relative_to_workspace(path),
        }

    def read_task_result(self, task_id: str) -> dict[str, Any]:
        task_id = self._validate_id(task_id, "task_id")
        path = self._task_root(task_id) / "result.json"
        self._assert_inside_request(path, allow_missing=True)
        if not path.exists():
            return {
                "task_id": task_id,
                "status": "not_available",
                "path": self._relative_to_workspace(path),
            }
        value = self._read_json(path)
        if not isinstance(value, dict):
            raise ValueError("task result must be a JSON object")
        return value

    def send_message(
        self,
        *,
        from_task_id: str,
        to_task_id: str,
        content: str,
        message_type: str = "context",
        artifact_paths: list[str] | None = None,
        reply_to: str | None = None,
        protocol_request_id: str | None = None,
        protocol_payload: dict[str, Any] | None = None,
        _allow_protocol: bool = False,
    ) -> dict[str, Any]:
        from_task_id = self._validate_id(from_task_id, "from_task_id")
        to_task_id = self._validate_id(to_task_id, "to_task_id")
        if message_type not in _ALL_MESSAGE_TYPES:
            raise ValueError(f"unsupported message type: {message_type}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("message content cannot be empty")
        if len(content) > self.max_content_chars:
            raise ValueError("message content exceeds communication limit")
        normalized_artifacts = [
            self.validate_artifact_reference(path)
            for path in (artifact_paths or [])
        ]
        if reply_to is not None:
            reply_to = self._validate_message_id(reply_to)

        is_protocol = message_type in _PROTOCOL_MESSAGE_TYPES
        if is_protocol:
            if not _allow_protocol:
                raise PermissionError(
                    "protocol messages must be created through the protocol API"
                )
            if protocol_request_id is None:
                raise ValueError("protocol messages require protocol_request_id")
            protocol_request_id = self._validate_id(
                protocol_request_id,
                "protocol_request_id",
            )
            protocol_kind = self._protocol_kind_for_message_type(message_type)
            if protocol_payload is None:
                raise ValueError("protocol messages require a payload object")
            self._validate_protocol_payload(
                protocol_kind,
                protocol_payload,
                response=message_type in _PROTOCOL_RESPONSE_TYPES,
            )
        elif protocol_request_id is not None or protocol_payload is not None:
            raise ValueError(
                "protocol_request_id and payload require a protocol message type"
            )

        self.register_task(from_task_id)
        self.register_task(to_task_id)
        message_id = f"msg-{uuid4().hex}"
        message = {
            "message_id": message_id,
            "request_id": (
                protocol_request_id
                if is_protocol
                else self.request_id
            ),
            "scope_request_id": self.request_id,
            "from_task_id": from_task_id,
            "to_task_id": to_task_id,
            "type": message_type,
            "content": content,
            "artifact_paths": normalized_artifacts,
            "created_at": _utc_now(),
            "reply_to": reply_to,
        }
        if is_protocol:
            message.update(
                {
                    "protocol_request_id": protocol_request_id,
                    "protocol_type": self._protocol_kind_for_message_type(
                        message_type
                    ),
                    "payload": dict(protocol_payload or {}),
                }
            )
        self._validate_message_envelope(message)
        destination = self._task_root(to_task_id) / "messages" / f"{message_id}.json"
        self._atomic_write_json(destination, message)
        return {
            "message_id": message_id,
            "path": self._relative_to_workspace(destination),
            "to_task_id": to_task_id,
        }

    def create_protocol_request(
        self,
        *,
        protocol_type: str,
        sender: str,
        target: str,
        payload: dict[str, Any],
        content: str = "",
    ) -> dict[str, Any]:
        """Create and persist a pending request card plus its inbox message."""
        protocol_type = self._validate_protocol_kind(protocol_type)
        sender = self._validate_id(sender, "sender")
        target = self._validate_id(target, "target")
        self._validate_protocol_payload(
            protocol_type,
            payload,
            response=False,
        )
        self.register_task(sender)
        self.register_task(target)

        request_id = f"req-{uuid4().hex}"
        now = _utc_now()
        state = ProtocolState(
            request_id=request_id,
            protocol_type=protocol_type,
            sender=sender,
            target=target,
            status="pending",
            payload=dict(payload),
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            if self._get_protocol_state_locked(request_id) is not None:
                raise ValueError("protocol request id collision")
            self._save_protocol_state_locked(state)

            message = self.send_message(
                from_task_id=sender,
                to_task_id=target,
                content=content.strip() or json.dumps(
                    payload,
                    ensure_ascii=False,
                ),
                message_type=f"{protocol_type}_request",
                protocol_request_id=request_id,
                protocol_payload=payload,
                _allow_protocol=True,
            )
            state.request_message_id = str(message["message_id"])
            state.updated_at = _utc_now()
            self._save_protocol_state_locked(state)

        return {
            "request_id": request_id,
            "protocol_request_id": request_id,
            "type": f"{protocol_type}_request",
            "status": state.status,
            "message_id": message["message_id"],
            "protocol_state": state.to_dict(),
        }

    def respond_protocol(
        self,
        request_id: str,
        *,
        responder: str,
        payload: dict[str, Any],
        content: str = "",
    ) -> dict[str, Any]:
        """Respond once to a pending card after validating its contract."""
        request_id = self._validate_id(request_id, "protocol_request_id")
        responder = self._validate_id(responder, "responder")
        with self._lock:
            state = self._get_protocol_state_locked(request_id)
            if state is None:
                raise ValueError(f"unknown protocol request: {request_id}")
            if responder != state.target:
                raise PermissionError(
                    "only the protocol target can send its response"
                )
            if state.status != "pending":
                return {
                    "request_id": request_id,
                    "protocol_request_id": request_id,
                    "status": state.status,
                    "sent": False,
                    "matched": False,
                    "reason": "protocol request already resolved",
                    "protocol_state": state.to_dict(),
                }
            if state.response_message_id is not None:
                return {
                    "request_id": request_id,
                    "protocol_request_id": request_id,
                    "status": state.status,
                    "sent": False,
                    "matched": False,
                    "reason": "a response has already been sent",
                    "protocol_state": state.to_dict(),
                }
            self._validate_protocol_response_for_state(state, payload)
            response_type = f"{state.protocol_type}_response"
            message = self.send_message(
                from_task_id=responder,
                to_task_id=state.sender,
                content=content.strip() or json.dumps(
                    payload,
                    ensure_ascii=False,
                ),
                message_type=response_type,
                protocol_request_id=request_id,
                protocol_payload=payload,
                reply_to=state.request_message_id,
                _allow_protocol=True,
            )
            state.response_message_id = str(message["message_id"])
            state.response_payload = dict(payload)
            # The response is pending until the sender's inbox consumes it.
            # This mirrors the s16 request -> inbox -> match lifecycle.
            state.status = "pending"
            state.updated_at = _utc_now()
            self._save_protocol_state_locked(state)
            return {
                "request_id": request_id,
                "protocol_request_id": request_id,
                "type": response_type,
                "status": state.status,
                "sent": True,
                "matched": False,
                "message_id": message["message_id"],
                "protocol_state": state.to_dict(),
            }

    def get_protocol_state(self, request_id: str) -> dict[str, Any]:
        request_id = self._validate_id(request_id, "protocol_request_id")
        with self._lock:
            state = self._get_protocol_state_locked(request_id)
            if state is None:
                raise ValueError(f"unknown protocol request: {request_id}")
            return state.to_dict()

    def match_response(self, message: dict[str, Any]) -> dict[str, Any]:
        """Match one response to its card; mismatches never mutate state."""
        if not isinstance(message, dict):
            raise ValueError("protocol message must be an object")
        try:
            self._validate_message_envelope(message)
        except (TypeError, ValueError, PermissionError) as exc:
            return {
                "matched": False,
                "reason": str(exc),
            }
        message_type = str(message.get("type", ""))
        if message_type not in _PROTOCOL_RESPONSE_TYPES:
            raise ValueError("message is not a protocol response")
        if message.get("scope_request_id") != self.request_id:
            return {
                "matched": False,
                "reason": "message belongs to another communication scope",
            }
        request_id = message.get("protocol_request_id", message.get("request_id"))
        request_id = self._validate_id(str(request_id or ""), "protocol_request_id")
        if message.get("request_id") != request_id:
            return {
                "request_id": request_id,
                "matched": False,
                "reason": "protocol request id fields do not match",
            }
        with self._lock:
            state = self._get_protocol_state_locked(request_id)
            if state is None:
                return {
                    "request_id": request_id,
                    "matched": False,
                    "reason": "unknown protocol request",
                }
            expected_type = f"{state.protocol_type}_response"
            if message_type != expected_type:
                return {
                    "request_id": request_id,
                    "matched": False,
                    "reason": "protocol response type does not match request",
                    "protocol_state": state.to_dict(),
                }
            if message.get("reply_to") != state.request_message_id:
                return {
                    "request_id": request_id,
                    "matched": False,
                    "reason": "protocol response does not reply to request message",
                    "protocol_state": state.to_dict(),
                }
            if (
                message.get("from_task_id") != state.target
                or message.get("to_task_id") != state.sender
            ):
                return {
                    "request_id": request_id,
                    "matched": False,
                    "reason": "protocol response participants do not match",
                    "protocol_state": state.to_dict(),
                }
            if state.status != "pending":
                return {
                    "request_id": request_id,
                    "matched": False,
                    "reason": "protocol request already resolved",
                    "protocol_state": state.to_dict(),
                }
            known_response_id = state.response_message_id
            message_id = str(message.get("message_id", ""))
            if known_response_id is None:
                return {
                    "request_id": request_id,
                    "matched": False,
                    "reason": "no response has been sent for protocol request",
                    "protocol_state": state.to_dict(),
                }
            if message_id != known_response_id:
                return {
                    "request_id": request_id,
                    "matched": False,
                    "reason": "a different response has already been sent",
                    "protocol_state": state.to_dict(),
                }
            payload = message.get("payload")
            if not isinstance(payload, dict):
                return {
                    "request_id": request_id,
                    "matched": False,
                    "reason": "protocol response payload is missing",
                    "protocol_state": state.to_dict(),
                }
            try:
                self._validate_protocol_response_for_state(state, payload)
            except (TypeError, ValueError, PermissionError) as exc:
                return {
                    "request_id": request_id,
                    "matched": False,
                    "reason": str(exc),
                    "protocol_state": state.to_dict(),
                }
            if state.response_payload != payload:
                return {
                    "request_id": request_id,
                    "matched": False,
                    "reason": "protocol response payload does not match sent response",
                    "protocol_state": state.to_dict(),
                }
            state.status = self._protocol_decision(
                state.protocol_type,
                payload,
            )
            state.updated_at = _utc_now()
            state.response_message_id = message_id or None
            state.response_payload = dict(payload)
            self._save_protocol_state_locked(state)
            return {
                "request_id": request_id,
                "matched": True,
                "status": state.status,
                "protocol_state": state.to_dict(),
            }

    def dispatch_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """Route ordinary messages and protocol messages through one entrypoint."""
        if not isinstance(message, dict):
            raise ValueError("message must be an object")
        self._validate_message_envelope(message)
        message_type = str(message.get("type", ""))
        if message_type in _PROTOCOL_RESPONSE_TYPES:
            matched = self.match_response(message)
            return {
                "route": "protocol_response",
                "message": message,
                "match": matched,
            }
        if message_type in _PROTOCOL_REQUEST_TYPES:
            request_id = message["protocol_request_id"]
            request_id = self._validate_id(str(request_id or ""), "protocol_request_id")
            protocol_kind = self._protocol_kind_for_message_type(message_type)
            payload = message.get("payload")
            if not isinstance(payload, dict):
                raise ValueError("protocol request payload is missing")
            self._validate_protocol_payload(protocol_kind, payload, response=False)
            sender = str(message["from_task_id"])
            target = str(message["to_task_id"])
            with self._lock:
                if not self._task_registered_locked(sender):
                    return {
                        "route": "rejected",
                        "message": message,
                        "reason": "protocol request sender is not registered",
                    }
                if not self._task_registered_locked(target):
                    return {
                        "route": "rejected",
                        "message": message,
                        "reason": "protocol request target is not registered",
                    }
            with self._lock:
                existing = self._get_protocol_state_locked(request_id)
                if existing is not None:
                    if (
                        existing.protocol_type != protocol_kind
                        or existing.sender != sender
                        or existing.target != target
                        or existing.payload != payload
                    ):
                        return {
                            "route": "rejected",
                            "message": message,
                            "reason": "protocol request does not match existing request card",
                            "protocol_state": existing.to_dict(),
                        }
                else:
                    now = _utc_now()
                    self._save_protocol_state_locked(
                        ProtocolState(
                            request_id=request_id,
                            protocol_type=protocol_kind,
                            sender=sender,
                            target=target,
                            status="pending",
                            payload=dict(payload),
                            created_at=str(message.get("created_at", now)),
                            updated_at=now,
                            request_message_id=str(message.get("message_id", "")) or None,
                        )
                    )
            return {
                "route": "protocol_request",
                "message": message,
                "protocol_state": self.get_protocol_state(request_id),
            }
        if message_type not in _MESSAGE_TYPES:
            return {
                "route": "rejected",
                "message": message,
                "reason": "unknown message type",
            }
        return {"route": "message", "message": message}

    def consume_inbox(
        self,
        task_id: str,
        *,
        after_message_id: str | None = None,
        max_messages: int = 100,
        message_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Read and route one inbox using the same protocol dispatcher."""
        task_id = self._validate_id(task_id, "task_id")
        if max_messages <= 0:
            raise ValueError("max_messages must be greater than zero")
        if after_message_id is not None:
            after_message_id = self._validate_message_id(after_message_id)
        selected_message_ids: set[str] | None = None
        if message_ids is not None:
            selected_message_ids = {
                self._validate_message_id(message_id)
                for message_id in message_ids
            }

        with self._lock:
            messages = self._read_messages(
                task_id,
                after_message_id=after_message_id,
            )
            inbox_state = self._read_inbox_state_locked(task_id)
            consumed_ids = set(inbox_state["consumed_message_ids"])
            unread = [
                message
                for message in messages
                if message.get("message_id") not in consumed_ids
            ]
            if selected_message_ids is not None:
                unread = [
                    message
                    for message in unread
                    if message.get("message_id") in selected_message_ids
                ]
            batch = unread[:max_messages]
            routed: list[dict[str, Any]] = []
            for message in batch:
                try:
                    if message.get("to_task_id") != task_id:
                        routed.append(
                            {
                                "route": "rejected",
                                "message": message,
                                "reason": "message recipient does not match inbox",
                            }
                        )
                    else:
                        routed.append(self.dispatch_message(message))
                except (TypeError, ValueError, PermissionError) as exc:
                    routed.append(
                        {
                            "route": "rejected",
                            "message": message,
                            "reason": str(exc),
                        }
                    )

            batch_ids = [str(message["message_id"]) for message in batch]
            if batch_ids:
                inbox_state["consumed_message_ids"].extend(
                    message_id
                    for message_id in batch_ids
                    if message_id not in consumed_ids
                )
                inbox_state["last_consumed_message_id"] = batch_ids[-1]
                inbox_state["updated_at"] = _utc_now()
                self._save_inbox_state_locked(task_id, inbox_state)

            return {
                "task_id": task_id,
                "messages": routed,
                "count": len(routed),
                "has_more": len(unread) > max_messages,
                "new_message_ids": batch_ids,
                "last_message_id": batch_ids[-1] if batch_ids else None,
            }

    def peek_inbox(
        self,
        task_id: str,
        *,
        after_message_id: str | None = None,
        max_messages: int = 100,
    ) -> dict[str, Any]:
        """Inspect unread inbox messages without consuming them."""
        task_id = self._validate_id(task_id, "task_id")
        if max_messages <= 0:
            raise ValueError("max_messages must be greater than zero")
        if after_message_id is not None:
            after_message_id = self._validate_message_id(after_message_id)

        with self._lock:
            messages = self._read_messages(
                task_id,
                after_message_id=after_message_id,
            )
            inbox_state = self._read_inbox_state_locked(task_id)
            consumed_ids = set(inbox_state["consumed_message_ids"])
            unread = [
                message
                for message in messages
                if message.get("message_id") not in consumed_ids
            ]
            batch = unread[:max_messages]
            return {
                "task_id": task_id,
                "messages": batch,
                "count": len(batch),
                "has_more": len(unread) > max_messages,
            }

    def list_messages(
        self,
        task_id: str,
        *,
        after_message_id: str | None = None,
        max_messages: int = 100,
    ) -> dict[str, Any]:
        task_id = self._validate_id(task_id, "task_id")
        if max_messages <= 0:
            raise ValueError("max_messages must be greater than zero")
        if after_message_id is not None:
            after_message_id = self._validate_message_id(after_message_id)
        messages = self._read_messages(
            task_id,
            after_message_id=after_message_id,
        )
        has_more = len(messages) > max_messages
        messages = messages[:max_messages]
        return {
            "task_id": task_id,
            "messages": messages,
            "count": len(messages),
            "has_more": has_more,
        }

    def _read_messages(
        self,
        task_id: str,
        *,
        after_message_id: str | None = None,
    ) -> list[dict[str, Any]]:
        task_id = self._validate_id(task_id, "task_id")
        if after_message_id is not None:
            after_message_id = self._validate_message_id(after_message_id)
        messages_dir = self._task_root(task_id) / "messages"
        self._assert_inside_request(messages_dir, allow_missing=True)
        messages: list[dict[str, Any]] = []
        if messages_dir.exists():
            for path in messages_dir.glob("msg-*.json"):
                try:
                    value = self._read_json(path)
                except (OSError, json.JSONDecodeError, ValueError):
                    # Atomic writes prevent this in normal operation. Ignore a
                    # manually-created partial file instead of breaking a run.
                    continue
                if not isinstance(value, dict):
                    continue
                messages.append(value)
        messages.sort(
            key=lambda value: (
                str(value.get("created_at", "")),
                str(value.get("message_id", "")),
            )
        )
        if after_message_id is not None:
            cursor_index = next(
                (
                    index
                    for index, value in enumerate(messages)
                    if value.get("message_id") == after_message_id
                ),
                None,
            )
            if cursor_index is not None:
                messages = messages[cursor_index + 1:]
        return messages

    def write_artifact(
        self,
        task_id: str,
        path: str,
        content: str,
        *,
        shared: bool = False,
    ) -> dict[str, Any]:
        task_id = self._validate_id(task_id, "task_id")
        if not isinstance(content, str):
            raise ValueError("artifact content must be text")
        if len(content) > self.max_artifact_chars:
            raise ValueError("artifact exceeds communication limit")
        base = self.paths.shared_artifacts if shared else self._task_root(task_id) / "artifacts"
        artifact = self._safe_relative_path(base, path)
        self._check_sensitive(artifact)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write_text(artifact, content)
        return {
            "task_id": task_id,
            "shared": shared,
            "path": artifact.relative_to(self.root).as_posix(),
            "workspace_path": self._relative_to_workspace(artifact),
            "bytes_written": len(content.encode("utf-8")),
        }

    def read_artifact(
        self,
        path: str,
        *,
        max_chars: int = 100_000,
    ) -> dict[str, Any]:
        if max_chars <= 0:
            raise ValueError("max_chars must be greater than zero")
        artifact = self._artifact_reference_path(path)
        self._check_sensitive(artifact)
        if not artifact.is_file():
            raise ValueError(f"artifact does not exist: {path}")
        content = artifact.read_text(encoding="utf-8", errors="replace")
        limit = min(max_chars, self.max_artifact_chars)
        return {
            "path": artifact.relative_to(self.root).as_posix(),
            "workspace_path": self._relative_to_workspace(artifact),
            "content": content[:limit],
            "truncated": len(content) > limit,
        }

    def write_shared_context(
        self,
        path: str,
        content: str,
    ) -> dict[str, Any]:
        if not isinstance(content, str):
            raise ValueError("context content must be text")
        if len(content) > self.max_content_chars:
            raise ValueError("context exceeds communication limit")
        context_path = self._safe_relative_path(
            self.paths.shared_context,
            path,
        )
        self._check_sensitive(context_path)
        context_path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write_text(context_path, content)
        return {
            "path": context_path.relative_to(self.root).as_posix(),
            "workspace_path": self._relative_to_workspace(context_path),
            "bytes_written": len(content.encode("utf-8")),
        }

    def read_shared_context(
        self,
        path: str,
        *,
        max_chars: int = 100_000,
    ) -> dict[str, Any]:
        if max_chars <= 0:
            raise ValueError("max_chars must be greater than zero")
        context_path = self._safe_relative_path(
            self.paths.shared_context,
            path,
        )
        self._check_sensitive(context_path)
        if not context_path.is_file():
            raise ValueError(f"shared context does not exist: {path}")
        content = context_path.read_text(
            encoding="utf-8",
            errors="replace",
        )
        limit = min(max_chars, self.max_content_chars)
        return {
            "path": context_path.relative_to(self.root).as_posix(),
            "workspace_path": self._relative_to_workspace(context_path),
            "content": content[:limit],
            "truncated": len(content) > limit,
        }

    def validate_artifact_reference(self, path: str) -> str:
        artifact = self._artifact_reference_path(path)
        self._check_sensitive(artifact)
        return artifact.relative_to(self.root).as_posix()

    def _normalize_declared_artifact_reference(self, path: str) -> str:
        if not isinstance(path, str) or not path.strip():
            raise ValueError("artifact reference cannot be empty")
        normalized = path.replace("\\", "/")
        candidate = Path(path)
        if candidate.is_absolute() or normalized.startswith("/"):
            raise PermissionError("absolute artifact paths are not allowed")
        if any(part in {"", ".", ".."} for part in normalized.split("/")):
            raise PermissionError("path traversal is not allowed")
        if normalized.startswith("tasks/") or normalized.startswith("shared/"):
            return self.validate_artifact_reference(normalized)
        self._check_sensitive(Path(normalized))
        return normalized

    def _validate_message_envelope(self, message: dict[str, Any]) -> None:
        required = {
            "message_id",
            "request_id",
            "scope_request_id",
            "from_task_id",
            "to_task_id",
            "type",
            "content",
            "artifact_paths",
            "created_at",
            "reply_to",
        }
        missing = required - set(message)
        if missing:
            raise ValueError(
                "message is missing fields: " + ", ".join(sorted(missing))
            )
        for field in (
            "message_id",
            "request_id",
            "scope_request_id",
            "from_task_id",
            "to_task_id",
        ):
            if not isinstance(message[field], str):
                raise ValueError(f"message {field} must be a string")
        self._validate_message_id(message["message_id"])
        self._validate_id(message["request_id"], "request_id")
        self._validate_id(message["scope_request_id"], "scope_request_id")
        self._validate_id(message["from_task_id"], "from_task_id")
        self._validate_id(message["to_task_id"], "to_task_id")
        if message["scope_request_id"] != self.request_id:
            raise ValueError("message belongs to another communication scope")
        if not isinstance(message["content"], str) or not message["content"].strip():
            raise ValueError("message content cannot be empty")
        if not isinstance(message["created_at"], str) or not message["created_at"].strip():
            raise ValueError("message created_at must be a non-empty string")
        artifacts = message["artifact_paths"]
        if not isinstance(artifacts, list):
            raise ValueError("message artifact_paths must be an array")
        for artifact in artifacts:
            if not isinstance(artifact, str) or not artifact.strip():
                raise ValueError("message artifact_paths must contain paths")
            self.validate_artifact_reference(artifact)
        reply_to = message.get("reply_to")
        if reply_to is not None:
            if not isinstance(reply_to, str):
                raise ValueError("message reply_to must be a message id")
            self._validate_message_id(reply_to)
        message_type = str(message["type"])
        if message_type in _PROTOCOL_MESSAGE_TYPES:
            protocol_request_id = message.get("protocol_request_id")
            protocol_type = message.get("protocol_type")
            payload = message.get("payload")
            if protocol_request_id is None:
                raise ValueError("protocol message requires protocol_request_id")
            self._validate_id(
                str(protocol_request_id),
                "protocol_request_id",
            )
            if message["request_id"] != protocol_request_id:
                raise ValueError(
                    "protocol request id fields do not match"
                )
            expected_protocol_type = self._protocol_kind_for_message_type(
                message_type
            )
            if protocol_type != expected_protocol_type:
                raise ValueError("protocol type does not match message type")
            if not isinstance(payload, dict):
                raise ValueError("protocol message payload must be an object")
            self._validate_protocol_payload(
                expected_protocol_type,
                payload,
                response=message_type in _PROTOCOL_RESPONSE_TYPES,
            )
            if (
                message_type in _PROTOCOL_REQUEST_TYPES
                and reply_to is not None
            ):
                raise ValueError("protocol request reply_to must be null")

    def _get_protocol_state_locked(
        self,
        request_id: str,
    ) -> ProtocolState | None:
        protocol_path = self.paths.protocols / f"{request_id}.json"
        self._assert_inside_request(protocol_path, allow_missing=True)
        value = None
        if protocol_path.is_file():
            value = self._read_json(protocol_path)
        if value is None:
            manifest = self._read_manifest()
            value = manifest.get("protocols", {}).get(request_id)
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError("protocol state must be an object")
        state = ProtocolState.from_dict(value)
        self._validate_id(state.request_id, "protocol_request_id")
        if state.request_id != request_id:
            raise ValueError("protocol state request id does not match key")
        self._validate_protocol_kind(state.protocol_type)
        self._validate_id(state.sender, "protocol sender")
        self._validate_id(state.target, "protocol target")
        if state.status not in _PROTOCOL_STATUS:
            raise ValueError("invalid protocol state status")
        if state.request_message_id is not None:
            self._validate_message_id(state.request_message_id)
        if state.response_message_id is not None:
            self._validate_message_id(state.response_message_id)
        return state

    def _save_protocol_state_locked(self, state: ProtocolState) -> None:
        self.paths.protocols.mkdir(parents=True, exist_ok=True)
        protocol_path = self.paths.protocols / f"{state.request_id}.json"
        self._assert_inside_request(protocol_path, allow_missing=True)
        self._atomic_write_json(protocol_path, state.to_dict())
        manifest = self._read_manifest()
        protocols = manifest.setdefault("protocols", {})
        protocols[state.request_id] = state.to_dict()
        manifest["updated_at"] = state.updated_at
        self._atomic_write_json(self.root / "manifest.json", manifest)

    def _task_registered_locked(self, task_id: str) -> bool:
        manifest = self._read_manifest()
        tasks = manifest.get("tasks", {})
        return isinstance(tasks, dict) and task_id in tasks

    def _read_inbox_state_locked(self, task_id: str) -> dict[str, Any]:
        task_id = self._validate_id(task_id, "task_id")
        path = self._task_root(task_id) / "inbox_state.json"
        self._assert_inside_request(path, allow_missing=True)
        if not path.exists():
            return {
                "request_id": self.request_id,
                "task_id": task_id,
                "consumed_message_ids": [],
                "last_consumed_message_id": None,
                "updated_at": _utc_now(),
            }
        value = self._read_json(path)
        if not isinstance(value, dict):
            raise ValueError("inbox state must be an object")
        if value.get("request_id") != self.request_id:
            raise ValueError("inbox state belongs to another communication scope")
        if value.get("task_id") != task_id:
            raise ValueError("inbox state task id does not match path")
        consumed = value.get("consumed_message_ids", [])
        if not isinstance(consumed, list) or not all(
            isinstance(message_id, str)
            and re.fullmatch(r"msg-[A-Za-z0-9]+", message_id)
            for message_id in consumed
        ):
            raise ValueError("inbox consumed_message_ids must be message ids")
        last_consumed = value.get("last_consumed_message_id")
        if last_consumed is not None:
            self._validate_message_id(str(last_consumed))
        return {
            "request_id": self.request_id,
            "task_id": task_id,
            "consumed_message_ids": list(dict.fromkeys(consumed)),
            "last_consumed_message_id": last_consumed,
            "updated_at": str(value.get("updated_at", _utc_now())),
        }

    def _save_inbox_state_locked(
        self,
        task_id: str,
        state: dict[str, Any],
    ) -> None:
        task_id = self._validate_id(task_id, "task_id")
        if state.get("request_id") != self.request_id:
            raise ValueError("inbox state belongs to another communication scope")
        if state.get("task_id") != task_id:
            raise ValueError("inbox state task id does not match path")
        path = self._task_root(task_id) / "inbox_state.json"
        self._assert_inside_request(path, allow_missing=True)
        self._atomic_write_json(path, state)

    @staticmethod
    def _validate_protocol_kind(protocol_type: str) -> str:
        normalized = str(protocol_type or "").strip().lower()
        if normalized not in _PROTOCOL_KINDS:
            raise ValueError(f"unsupported protocol type: {protocol_type}")
        return normalized

    @classmethod
    def _protocol_kind_for_message_type(cls, message_type: str) -> str:
        if message_type in _PROTOCOL_REQUEST_TYPES:
            return message_type.removesuffix("_request")
        if message_type in _PROTOCOL_RESPONSE_TYPES:
            return message_type.removesuffix("_response")
        raise ValueError(f"not a protocol message type: {message_type}")

    def _validate_protocol_payload(
        self,
        protocol_type: str,
        payload: dict[str, Any],
        *,
        response: bool,
    ) -> None:
        protocol_type = self._validate_protocol_kind(protocol_type)
        if not isinstance(payload, dict):
            raise ValueError("protocol payload must be an object")
        try:
            encoded = json.dumps(payload, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("protocol payload must be JSON serializable") from exc
        if len(encoded) > self.max_content_chars:
            raise ValueError("protocol payload exceeds communication limit")

        if response:
            self._validate_protocol_response_payload(protocol_type, payload)
        else:
            self._validate_protocol_request_payload(protocol_type, payload)

    def _validate_protocol_request_payload(
        self,
        protocol_type: str,
        payload: dict[str, Any],
    ) -> None:
        if protocol_type == "task":
            self._required_text(payload, "task_id")
            self._required_text(payload, "instructions")
            dependencies = payload.get("dependencies", [])
            tool_names = payload.get("tool_names", [])
            if not isinstance(dependencies, list) or not all(
                isinstance(value, str) and value.strip()
                for value in dependencies
            ):
                raise ValueError("task dependencies must be an array of task ids")
            if not isinstance(tool_names, list) or not all(
                isinstance(value, str) and value.strip()
                for value in tool_names
            ):
                raise ValueError("task tool_names must be an array of names")
            return
        if protocol_type == "context":
            self._required_text(payload, "question")
            return
        if protocol_type == "artifact":
            self._normalize_declared_artifact_reference(
                self._required_text(payload, "path")
            )
            operation = payload.get("operation", "read")
            if operation not in {"read", "write"}:
                raise ValueError("artifact operation must be read or write")
            return
        if protocol_type == "plan_approval":
            self._required_text(payload, "plan")
            risk = payload.get("risk", "medium")
            if risk not in {"low", "medium", "high"}:
                raise ValueError("plan risk must be low, medium, or high")
            return
        if protocol_type == "shutdown":
            self._required_text(payload, "reason")

    def _validate_protocol_response_payload(
        self,
        protocol_type: str,
        payload: dict[str, Any],
    ) -> None:
        if protocol_type == "task":
            self._required_text(payload, "task_id")
            self._required_text(payload, "summary")
            status = payload.get("status")
            if status not in {"completed", "blocked", "failed"}:
                raise ValueError(
                    "task response status must be completed, blocked, or failed"
                )
        elif protocol_type in {"context", "artifact"}:
            self._required_text(payload, "summary")
            status = payload.get("status")
            if status not in {"completed", "rejected", "failed"}:
                raise ValueError(
                    f"{protocol_type} response status must be completed, "
                    "rejected, or failed"
                )
        else:
            if not isinstance(payload.get("approve"), bool):
                raise ValueError(
                    f"{protocol_type} response requires boolean approve"
                )

        artifacts = payload.get("artifacts", [])
        if not isinstance(artifacts, list) or not all(
            isinstance(path, str) and path.strip()
            for path in artifacts
        ):
            raise ValueError(
                "protocol response artifacts must be an array of paths"
            )
        for path in artifacts:
            self._normalize_declared_artifact_reference(path)

    def _validate_protocol_response_for_state(
        self,
        state: ProtocolState,
        payload: dict[str, Any],
    ) -> None:
        self._validate_protocol_payload(
            state.protocol_type,
            payload,
            response=True,
        )
        if (
            state.protocol_type == "task"
            and payload.get("task_id") != state.payload.get("task_id")
        ):
            raise ValueError("task response task_id does not match target")

    @staticmethod
    def _required_text(payload: dict[str, Any], field: str) -> str:
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"protocol payload requires non-empty {field}")
        return value.strip()

    @staticmethod
    def _protocol_decision(
        protocol_type: str,
        payload: dict[str, Any],
    ) -> str:
        if protocol_type == "task":
            return "approved" if payload.get("status") == "completed" else "rejected"
        if protocol_type in {"context", "artifact"}:
            return "approved" if payload.get("status") == "completed" else "rejected"
        return "approved" if payload.get("approve") is True else "rejected"

    def _initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.paths.protocols.mkdir(parents=True, exist_ok=True)
        self.paths.shared_context.mkdir(parents=True, exist_ok=True)
        self.paths.shared_artifacts.mkdir(parents=True, exist_ok=True)
        manifest = self.root / "manifest.json"
        if not manifest.exists():
            self._atomic_write_json(
                manifest,
                {
                    "version": 1,
                    "request_id": self.request_id,
                    "created_at": _utc_now(),
                    "updated_at": _utc_now(),
                    "tasks": {},
                    "protocols": {},
                },
            )
        self.register_task("main", task="Parent agent coordination")

    def _read_manifest(self) -> dict[str, Any]:
        path = self.root / "manifest.json"
        if not path.exists():
            return {
                "version": 1,
                "request_id": self.request_id,
                "created_at": _utc_now(),
                "tasks": {},
                "protocols": {},
            }
        value = self._read_json(path)
        if not isinstance(value, dict):
            raise ValueError("communication manifest must be an object")
        return value

    def _task_root(self, task_id: str) -> Path:
        task_id = self._validate_id(task_id, "task_id")
        path = self.root / "tasks" / task_id
        self._assert_inside_request(path, allow_missing=True)
        return path

    def _artifact_reference_path(self, path: str) -> Path:
        if not isinstance(path, str) or not path.strip():
            raise ValueError("artifact path cannot be empty")
        candidate = Path(path)
        if candidate.is_absolute():
            raise PermissionError("absolute artifact paths are not allowed")
        normalized = path.replace("\\", "/")
        if normalized.startswith("/"):
            raise PermissionError("absolute artifact paths are not allowed")
        if any(part in {"", ".", ".."} for part in normalized.split("/")):
            raise PermissionError("path traversal is not allowed")
        if normalized.startswith("tasks/") or normalized.startswith("shared/"):
            artifact = self.root / normalized
        else:
            # Tool results return workspace-relative paths. Accept only the
            # artifact directories, never arbitrary files in the request root.
            artifact = self.root / "shared" / "artifacts" / normalized
        self._assert_inside_request(artifact, allow_missing=True)
        relative = artifact.relative_to(self.root).parts
        allowed = (
            len(relative) >= 3
            and relative[0] == "tasks"
            and relative[2] == "artifacts"
        ) or (
            len(relative) >= 3
            and relative[:2] == ("shared", "artifacts")
        )
        if not allowed:
            raise PermissionError("artifact must be inside an artifact directory")
        return artifact

    def _safe_relative_path(self, base: Path, path: str) -> Path:
        if not isinstance(path, str) or not path.strip():
            raise ValueError("path cannot be empty")
        candidate = Path(path)
        if candidate.is_absolute():
            raise PermissionError("absolute paths are not allowed")
        normalized = path.replace("\\", "/")
        if any(part in {"", ".", ".."} for part in normalized.split("/")):
            raise PermissionError("path traversal is not allowed")
        target = base.joinpath(*normalized.split("/"))
        self._assert_inside_request(target, allow_missing=True)
        return target

    def _assert_inside_workspace(self, path: Path, *, allow_missing: bool) -> None:
        resolved = path.resolve(strict=not allow_missing)
        try:
            resolved.relative_to(self.workspace_root)
        except ValueError as exc:
            raise PermissionError("communication path escapes workspace") from exc

    def _assert_inside_request(self, path: Path, *, allow_missing: bool) -> None:
        resolved = path.resolve(strict=not allow_missing)
        try:
            resolved.relative_to(self.root.resolve())
        except ValueError as exc:
            raise PermissionError("communication path escapes request scope") from exc

    def _relative_to_workspace(self, path: Path) -> str:
        self._assert_inside_workspace(path, allow_missing=True)
        return path.resolve(strict=False).relative_to(self.workspace_root).as_posix()

    def _check_sensitive(self, path: Path) -> None:
        for part in path.parts:
            lowered = part.lower()
            if lowered in _SENSITIVE_NAMES or any(
                lowered.endswith(suffix) for suffix in _SENSITIVE_SUFFIXES
            ):
                raise PermissionError("sensitive communication paths are blocked")

    @staticmethod
    def _validate_id(value: str, field: str) -> str:
        if not isinstance(value, str) or not _REQUEST_ID_PATTERN.fullmatch(value):
            raise ValueError(f"invalid {field}")
        return value

    @staticmethod
    def _validate_message_id(value: str) -> str:
        if not isinstance(value, str) or not re.fullmatch(r"msg-[A-Za-z0-9]+", value):
            raise ValueError("invalid message_id")
        return value

    @staticmethod
    def _read_json(path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _atomic_write_text(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    @classmethod
    def _atomic_write_json(cls, path: Path, value: Any) -> None:
        cls._atomic_write_text(path, _json_text(value))


SUBAGENT_SEND_MESSAGE_SPEC = ToolSpec(
    name="subagent_send_message",
    description=(
        "Send a structured message from the current sub-agent to the parent "
        "agent or another task in this request. Include artifact paths when "
        "the message refers to a file."
    ),
    parameters={
        "type": "object",
        "properties": {
            "to_task_id": {"type": "string", "minLength": 1},
            "content": {"type": "string", "minLength": 1},
            "message_type": {
                "type": "string",
                "enum": sorted(_MESSAGE_TYPES),
                "default": "context",
            },
            "artifact_paths": {
                "type": "array",
                "items": {"type": "string"},
                "default": [],
            },
            "reply_to": {"type": "string"},
        },
        "required": ["to_task_id", "content"],
        "additionalProperties": False,
    },
    category="communication",
)

SUBAGENT_REQUEST_PROTOCOL_SPEC = ToolSpec(
    name="subagent_request_protocol",
    description=(
        "Create a correlated protocol request. The payload must follow the "
        "schema for the selected protocol_type and remains pending until a "
        "validated response is matched."
    ),
    parameters={
        "type": "object",
        "properties": {
            "protocol_type": {
                "type": "string",
                "enum": sorted(_PROTOCOL_KINDS),
            },
            "to_task_id": {"type": "string", "minLength": 1},
            "payload": {"type": "object"},
            "content": {"type": "string", "default": ""},
        },
        "required": ["protocol_type", "to_task_id", "payload"],
        "additionalProperties": False,
    },
    category="communication",
)

SUBAGENT_RESPOND_PROTOCOL_SPEC = ToolSpec(
    name="subagent_respond_protocol",
    description=(
        "Send exactly one response to a pending protocol request. The "
        "request_id and payload type must match the received request."
    ),
    parameters={
        "type": "object",
        "properties": {
            "request_id": {"type": "string", "minLength": 1},
            "payload": {"type": "object"},
            "content": {"type": "string", "default": ""},
        },
        "required": ["request_id", "payload"],
        "additionalProperties": False,
    },
    category="communication",
)

SUBAGENT_GET_PROTOCOL_SPEC = ToolSpec(
    name="subagent_get_protocol",
    description="Read the current state of a correlated protocol request.",
    parameters={
        "type": "object",
        "properties": {
            "request_id": {"type": "string", "minLength": 1},
        },
        "required": ["request_id"],
        "additionalProperties": False,
    },
    category="communication",
)

SUBAGENT_CONSUME_INBOX_SPEC = ToolSpec(
    name="subagent_consume_inbox",
    description=(
        "Read and route the current task inbox. Protocol responses are "
        "matched against their request card before being returned."
    ),
    parameters={
        "type": "object",
        "properties": {
            "after_message_id": {"type": "string"},
            "max_messages": {
                "type": "integer",
                "minimum": 1,
                "maximum": 200,
                "default": 100,
            },
        },
        "additionalProperties": False,
    },
    category="communication",
)

SUBAGENT_READ_MESSAGES_SPEC = ToolSpec(
    name="subagent_read_messages",
    description="Read messages addressed to the current agent in this request.",
    parameters={
        "type": "object",
        "properties": {
            "after_message_id": {"type": "string"},
            "max_messages": {"type": "integer", "minimum": 1, "maximum": 200, "default": 100},
        },
        "additionalProperties": False,
    },
    category="communication",
)

SUBAGENT_LIST_MESSAGES_SPEC = ToolSpec(
    name="subagent_list_messages",
    description="List available messages addressed to a task in this request.",
    parameters={
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "minLength": 1},
            "after_message_id": {"type": "string"},
            "max_messages": {"type": "integer", "minimum": 1, "maximum": 200, "default": 100},
        },
        "required": ["task_id"],
        "additionalProperties": False,
    },
    category="communication",
)

SUBAGENT_WRITE_ARTIFACT_SPEC = ToolSpec(
    name="subagent_write_artifact",
    description=(
        "Write a text artifact for the current task. Set shared=true only "
        "when another task should consume the artifact."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "minLength": 1},
            "content": {"type": "string"},
            "shared": {"type": "boolean", "default": False},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    },
    category="communication",
)

SUBAGENT_READ_ARTIFACT_SPEC = ToolSpec(
    name="subagent_read_artifact",
    description="Read a task or shared artifact by its request-relative path.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "minLength": 1},
            "max_chars": {"type": "integer", "minimum": 1, "maximum": 500000, "default": 100000},
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    category="communication",
)

SUBAGENT_READ_TASK_RESULT_SPEC = ToolSpec(
    name="subagent_read_task_result",
    description="Read the structured result of a predecessor task, if available.",
    parameters={
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "minLength": 1},
        },
        "required": ["task_id"],
        "additionalProperties": False,
    },
    category="communication",
)

SUBAGENT_WRITE_CONTEXT_SPEC = ToolSpec(
    name="subagent_write_shared_context",
    description="Write a request-scoped shared context file for other tasks.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "minLength": 1},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    },
    category="communication",
)

SUBAGENT_READ_CONTEXT_SPEC = ToolSpec(
    name="subagent_read_shared_context",
    description="Read a request-scoped shared context file.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "minLength": 1},
            "max_chars": {"type": "integer", "minimum": 1, "maximum": 100000, "default": 100000},
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    category="communication",
)


@dataclass
class SubagentCommunicationToolset:
    """Tools bound to one request and one current task identity."""

    communication: SubagentCommunication
    current_task_id: str

    def __post_init__(self) -> None:
        self.current_task_id = self.communication._validate_id(
            self.current_task_id,
            "current_task_id",
        )
        self.communication.register_task(self.current_task_id)

    @property
    def specs(self) -> list[ToolSpec]:
        return [
            SUBAGENT_REQUEST_PROTOCOL_SPEC,
            SUBAGENT_RESPOND_PROTOCOL_SPEC,
            SUBAGENT_GET_PROTOCOL_SPEC,
            SUBAGENT_CONSUME_INBOX_SPEC,
            SUBAGENT_WRITE_ARTIFACT_SPEC,
            SUBAGENT_READ_ARTIFACT_SPEC,
            SUBAGENT_READ_TASK_RESULT_SPEC,
            SUBAGENT_WRITE_CONTEXT_SPEC,
            SUBAGENT_READ_CONTEXT_SPEC,
        ]

    @property
    def handlers(self) -> dict[str, Any]:
        return {
            "subagent_request_protocol": self.request_protocol,
            "subagent_respond_protocol": self.respond_protocol,
            "subagent_get_protocol": self.get_protocol,
            "subagent_consume_inbox": self.consume_inbox,
            "subagent_write_artifact": self.write_artifact,
            "subagent_read_artifact": self.read_artifact,
            "subagent_read_task_result": self.read_task_result,
            "subagent_write_shared_context": self.write_shared_context,
            "subagent_read_shared_context": self.read_shared_context,
        }

    def send_message(
        self,
        to_task_id: str,
        content: str,
        message_type: str = "context",
        artifact_paths: list[str] | None = None,
        reply_to: str | None = None,
    ) -> dict[str, Any]:
        if message_type in _PROTOCOL_MESSAGE_TYPES:
            raise ValueError(
                "protocol messages must use subagent_request_protocol or "
                "subagent_respond_protocol"
            )
        return self.communication.send_message(
            from_task_id=self.current_task_id,
            to_task_id=to_task_id,
            content=content,
            message_type=message_type,
            artifact_paths=artifact_paths,
            reply_to=reply_to,
        )

    def request_protocol(
        self,
        protocol_type: str,
        to_task_id: str,
        payload: dict[str, Any],
        content: str = "",
    ) -> dict[str, Any]:
        return self.communication.create_protocol_request(
            protocol_type=protocol_type,
            sender=self.current_task_id,
            target=to_task_id,
            payload=payload,
            content=content,
        )

    def respond_protocol(
        self,
        request_id: str,
        payload: dict[str, Any],
        content: str = "",
    ) -> dict[str, Any]:
        return self.communication.respond_protocol(
            request_id,
            responder=self.current_task_id,
            payload=payload,
            content=content,
        )

    def get_protocol(self, request_id: str) -> dict[str, Any]:
        return self.communication.get_protocol_state(request_id)

    def consume_inbox(
        self,
        after_message_id: str | None = None,
        max_messages: int = 100,
    ) -> dict[str, Any]:
        return self.communication.consume_inbox(
            self.current_task_id,
            after_message_id=after_message_id,
            max_messages=max_messages,
        )

    def read_messages(
        self,
        after_message_id: str | None = None,
        max_messages: int = 100,
    ) -> dict[str, Any]:
        return self.communication.list_messages(
            self.current_task_id,
            after_message_id=after_message_id,
            max_messages=max_messages,
        )

    def list_messages(
        self,
        task_id: str,
        after_message_id: str | None = None,
        max_messages: int = 100,
    ) -> dict[str, Any]:
        return self.communication.list_messages(
            task_id,
            after_message_id=after_message_id,
            max_messages=max_messages,
        )

    def write_artifact(
        self,
        path: str,
        content: str,
        shared: bool = False,
    ) -> dict[str, Any]:
        return self.communication.write_artifact(
            self.current_task_id,
            path,
            content,
            shared=shared,
        )

    def read_artifact(
        self,
        path: str,
        max_chars: int = 100_000,
    ) -> dict[str, Any]:
        return self.communication.read_artifact(path, max_chars=max_chars)

    def read_task_result(self, task_id: str) -> dict[str, Any]:
        return self.communication.read_task_result(task_id)

    def write_shared_context(
        self,
        path: str,
        content: str,
    ) -> dict[str, Any]:
        return self.communication.write_shared_context(path, content)

    def read_shared_context(
        self,
        path: str,
        max_chars: int = 100_000,
    ) -> dict[str, Any]:
        return self.communication.read_shared_context(
            path,
            max_chars=max_chars,
        )


# This alias makes the storage role explicit for callers that prefer the
# protocol name used in design documents.
SubagentFileStore = SubagentCommunication
