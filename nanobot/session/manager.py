"""Session management for conversation history."""

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.config.paths import get_legacy_sessions_dir
from nanobot.utils.helpers import ensure_dir, safe_filename


def _atomic_write_text(path: Path, content: str) -> None:
    """Write a file atomically using a same-directory temporary file."""
    ensure_dir(path.parent)
    tmp = path.with_name(f".{path.name}.{id(content)}.tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


@dataclass
class Session:
    """
    A conversation session.

    Stores messages in JSONL format for easy reading and persistence.

    Important: Messages are append-only for LLM cache efficiency.
    The consolidation process writes summaries to MEMORY.md/HISTORY.md
    but does NOT modify the messages list or get_history() output.
    """

    key: str  # channel:chat_id
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    last_consolidated: int = 0  # Number of messages already consolidated to files
    _SHORT_TERM_KEEP_RAW: int = 6
    _SHORT_TERM_SIGNAL_THRESHOLD: float = 0.24

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Add a message to the session."""
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        self.messages.append(msg)
        self.updated_at = datetime.now()

    @staticmethod
    def _find_legal_start(messages: list[dict[str, Any]]) -> int:
        """Find first index where every tool result has a matching assistant tool_call."""
        declared: set[str] = set()
        start = 0
        for i, msg in enumerate(messages):
            role = msg.get("role")
            if role == "assistant":
                for tc in msg.get("tool_calls") or []:
                    if isinstance(tc, dict) and tc.get("id"):
                        declared.add(str(tc["id"]))
            elif role == "tool":
                tid = msg.get("tool_call_id")
                if tid and str(tid) not in declared:
                    start = i + 1
                    declared.clear()
                    for prev in messages[start:i + 1]:
                        if prev.get("role") == "assistant":
                            for tc in prev.get("tool_calls") or []:
                                if isinstance(tc, dict) and tc.get("id"):
                                    declared.add(str(tc["id"]))
        return start

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return " ".join(parts)
        return ""

    @classmethod
    def _message_signal_score(cls, message: dict[str, Any]) -> float:
        text = " ".join(cls._content_to_text(message.get("content", "")).split())
        if not text:
            return 0.0
        words = text.lower().split()
        if len(text) <= 8 and len(words) <= 2:
            return 0.05
        unique_ratio = len(set(words)) / max(1, len(words))
        length_score = min(len(text) / 120, 1.0)
        role = str(message.get("role", ""))
        role_bonus = 0.12 if role == "user" else 0.08 if role == "assistant" else 0.0
        return (unique_ratio * 0.45) + (length_score * 0.55) + role_bonus

    @classmethod
    def _iter_context_units(cls, messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        """Group assistant tool-call messages with their tool results as one context unit."""
        units: list[list[dict[str, Any]]] = []
        i = 0
        while i < len(messages):
            message = messages[i]
            if message.get("role") == "assistant" and message.get("tool_calls"):
                tool_ids = {
                    str(tc.get("id"))
                    for tc in (message.get("tool_calls") or [])
                    if isinstance(tc, dict) and tc.get("id")
                }
                unit = [message]
                i += 1
                while i < len(messages):
                    current = messages[i]
                    if current.get("role") == "tool" and str(current.get("tool_call_id")) in tool_ids:
                        unit.append(current)
                        i += 1
                        continue
                    break
                units.append(unit)
                continue
            units.append([message])
            i += 1
        return units

    @classmethod
    def filter_short_term_context(
        cls,
        messages: list[dict[str, Any]],
        *,
        keep_recent_messages: int | None = None,
    ) -> list[dict[str, Any]]:
        """Filter low-value older short-term messages while preserving recent raw turns."""
        if not messages:
            return []

        keep_recent = cls._SHORT_TERM_KEEP_RAW if keep_recent_messages is None else max(0, keep_recent_messages)
        if len(messages) <= keep_recent:
            return messages

        units = cls._iter_context_units(messages)
        recent_units: list[list[dict[str, Any]]] = []
        recent_count = 0
        while units and recent_count < keep_recent:
            unit = units.pop()
            recent_units.append(unit)
            recent_count += len(unit)
        older_units = units
        recent = [message for unit in reversed(recent_units) for message in unit]
        filtered: list[dict[str, Any]] = []
        for unit in older_units:
            unit_text = " ".join(
                cls._content_to_text(message.get("content", ""))
                for message in unit
            ).strip()
            unit_score = max((cls._message_signal_score(message) for message in unit), default=0.0)
            is_tool_unit = any(message.get("role") == "tool" for message in unit) or any(
                message.get("role") == "assistant" and message.get("tool_calls")
                for message in unit
            )
            if unit_score < cls._SHORT_TERM_SIGNAL_THRESHOLD and (len(unit_text) < 24 or is_tool_unit):
                continue
            filtered.extend(unit)
        return filtered + recent

    def get_history(self, max_messages: int = 500) -> list[dict[str, Any]]:
        """Return unconsolidated messages for LLM input, aligned to a legal tool-call boundary."""
        unconsolidated = self.messages[self.last_consolidated:]
        sliced = unconsolidated[-max_messages:]

        # Drop leading non-user messages to avoid starting mid-turn when possible.
        for i, message in enumerate(sliced):
            if message.get("role") == "user":
                sliced = sliced[i:]
                break

        # Some providers reject orphan tool results if the matching assistant
        # tool_calls message fell outside the fixed-size history window.
        start = self._find_legal_start(sliced)
        if start:
            sliced = sliced[start:]
        if len(sliced) >= 20:
            sliced = self.filter_short_term_context(sliced)

        out: list[dict[str, Any]] = []
        for message in sliced:
            entry: dict[str, Any] = {"role": message["role"], "content": message.get("content", "")}
            for key in ("tool_calls", "tool_call_id", "name"):
                if key in message:
                    entry[key] = message[key]
            out.append(entry)
        return out

    def clear(self) -> None:
        """Clear all messages and reset session to initial state."""
        self.messages = []
        self.last_consolidated = 0
        self.updated_at = datetime.now()

    def remove_message_by_id(self, message_id: str) -> bool:
        """Remove a message from history by its `message_id` field or content match.

        Returns True if a message was removed.
        """
        before = len(self.messages)
        self.messages = [
            m for m in self.messages
            if not (
                m.get("message_id") == message_id
                or (isinstance(m.get("content"), str) and message_id in m["content"])
            )
        ]
        removed = len(self.messages) < before
        if removed:
            self.updated_at = datetime.now()
        return removed


class SessionManager:
    """
    Manages conversation sessions.

    Sessions are stored as JSONL files in the sessions directory.
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.sessions_dir = ensure_dir(self.workspace / "sessions")
        self.legacy_sessions_dir = get_legacy_sessions_dir()
        self._cache: dict[str, Session] = {}

    def _get_session_path(self, key: str) -> Path:
        """Get the file path for a session."""
        safe_key = safe_filename(key.replace(":", "_"))
        return self.sessions_dir / f"{safe_key}.jsonl"

    def _get_legacy_session_path(self, key: str) -> Path:
        """Legacy global session path (~/.nanobot/sessions/)."""
        safe_key = safe_filename(key.replace(":", "_"))
        return self.legacy_sessions_dir / f"{safe_key}.jsonl"

    def get_or_create(self, key: str) -> Session:
        """
        Get an existing session or create a new one.

        Args:
            key: Session key (usually channel:chat_id).

        Returns:
            The session.
        """
        if key in self._cache:
            return self._cache[key]

        session = self._load(key)
        if session is None:
            session = Session(key=key)

        self._cache[key] = session
        return session

    def _load(self, key: str) -> Session | None:
        """Load a session from disk."""
        path = self._get_session_path(key)
        if not path.exists():
            legacy_path = self._get_legacy_session_path(key)
            if legacy_path.exists():
                try:
                    shutil.move(str(legacy_path), str(path))
                    logger.info("Migrated session {} from legacy path", key)
                except Exception:
                    logger.exception("Failed to migrate session {}", key)

        if not path.exists():
            return None

        try:
            messages = []
            metadata = {}
            created_at = None
            last_consolidated = 0

            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    data = json.loads(line)

                    if data.get("_type") == "metadata":
                        metadata = data.get("metadata", {})
                        created_at = datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None
                        last_consolidated = data.get("last_consolidated", 0)
                    else:
                        messages.append(data)

            return Session(
                key=key,
                messages=messages,
                created_at=created_at or datetime.now(),
                metadata=metadata,
                last_consolidated=last_consolidated
            )
        except Exception as e:
            logger.warning("Failed to load session {}: {}", key, e)
            return None

    def save(self, session: Session) -> None:
        """Save a session to disk."""
        path = self._get_session_path(session.key)

        metadata_line = {
            "_type": "metadata",
            "key": session.key,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "metadata": session.metadata,
            "last_consolidated": session.last_consolidated
        }
        lines = [json.dumps(metadata_line, ensure_ascii=False)]
        lines.extend(json.dumps(msg, ensure_ascii=False) for msg in session.messages)
        _atomic_write_text(path, "\n".join(lines) + "\n")

        self._cache[session.key] = session

    def invalidate(self, key: str) -> None:
        """Remove a session from the in-memory cache."""
        self._cache.pop(key, None)

    def list_sessions(self) -> list[dict[str, Any]]:
        """
        List all sessions.

        Returns:
            List of session info dicts.
        """
        sessions = []

        for path in self.sessions_dir.glob("*.jsonl"):
            try:
                # Read just the metadata line
                with open(path, encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    if first_line:
                        data = json.loads(first_line)
                        if data.get("_type") == "metadata":
                            key = data.get("key") or path.stem.replace("_", ":", 1)
                            sessions.append({
                                "key": key,
                                "created_at": data.get("created_at"),
                                "updated_at": data.get("updated_at"),
                                "path": str(path)
                            })
            except Exception:
                continue

        return sorted(sessions, key=lambda x: x.get("updated_at", ""), reverse=True)
