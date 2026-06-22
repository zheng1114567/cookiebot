"""Sandbox mode — policy engine, approval workflow, and audit logging."""

from __future__ import annotations

import asyncio
import enum
import json
import re
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanobot.bus.queue import MessageBus
    from nanobot.config.schema import SandboxConfig

# ── shared deny patterns ────────────────────────────────────────────────────

BUILTIN_DENY_PATTERNS: list[str] = [
    r"\brm\s+-[rf]{1,2}\b",
    r"\bdel\s+/[fq]\b",
    r"\brmdir\s+/s\b",
    r"(?:^|[;&|]\s*)format\b",
    r"\b(mkfs|diskpart)\b",
    r"\bdd\s+if=",
    r">\s*/dev/sd",
    r"\b(shutdown|reboot|poweroff)\b",
    r":\(\)\s*\{.*\};\s*:",
]


# ── policy decision ─────────────────────────────────────────────────────────


class PolicyDecision(enum.Enum):
    SAFE = "safe"
    DANGEROUS = "dangerous"
    NEEDS_APPROVAL = "needs_approval"


# ── policy engine ───────────────────────────────────────────────────────────


class SandboxPolicy:
    """Classifies tool calls based on configurable rules."""

    def __init__(self, config: "SandboxConfig"):
        self.config = config

    @property
    def effective_deny_patterns(self) -> list[str]:
        patterns: list[str] = []
        if self.config.use_builtin_deny:
            patterns.extend(BUILTIN_DENY_PATTERNS)
        patterns.extend(self.config.deny_patterns)
        return patterns

    def classify_exec(self, command: str) -> PolicyDecision:
        """Classify an exec command."""
        if self.config.mode == "off":
            return PolicyDecision.SAFE

        cmd = command.strip()

        for pat in self.effective_deny_patterns:
            try:
                if re.search(pat, cmd):
                    return PolicyDecision.DANGEROUS
            except re.error:
                pass

        if self.config.require_approval:
            for pat in self.config.approve_patterns:
                try:
                    if re.search(pat, cmd):
                        return PolicyDecision.NEEDS_APPROVAL
                except re.error:
                    pass

        if self.config.mode == "strict":
            return PolicyDecision.NEEDS_APPROVAL

        return PolicyDecision.SAFE

    def classify_tool(self, tool_name: str, params: dict) -> PolicyDecision:
        """Classify any tool call."""
        if tool_name == "exec":
            return self.classify_exec(params.get("command", ""))

        if tool_name in ("write_file", "edit_file"):
            if self.config.mode == "strict":
                return PolicyDecision.NEEDS_APPROVAL
            return PolicyDecision.SAFE

        return PolicyDecision.SAFE


# ── approval manager ────────────────────────────────────────────────────────


class ApprovalManager:
    """Manages the async approval workflow: ask, wait, audit."""

    _APPROVE_WORDS = frozenset({"yes", "y", "approve", "ok", "true", "1"})
    _DENY_WORDS = frozenset({"no", "n", "deny", "reject", "false", "0", "cancel"})

    def __init__(
        self,
        policy: SandboxPolicy,
        audit_log_path: Path,
        bus: "MessageBus",
    ):
        self.policy = policy
        self._audit_path = audit_log_path
        self._bus = bus
        self._pending_events: dict[str, asyncio.Event] = {}
        self._pending_decisions: dict[str, bool] = {}
        self._session_key_var: ContextVar[str | None] = ContextVar("approval_session_key", default=None)
        self._channel_var: ContextVar[str | None] = ContextVar("approval_channel", default=None)
        self._chat_id_var: ContextVar[str | None] = ContextVar("approval_chat_id", default=None)
        self._session_key: str | None = None
        self._channel: str | None = None
        self._chat_id: str | None = None

    def bind(self, session_key: str, channel: str, chat_id: str) -> None:
        """Bind context for the current turn."""
        self._session_key = session_key
        self._channel = channel
        self._chat_id = chat_id
        self._session_key_var.set(session_key)
        self._channel_var.set(channel)
        self._chat_id_var.set(chat_id)

    async def check_and_request(self, tool_name: str, params: dict) -> str | None:
        """Check policy and request approval if needed.

        Returns None if OK to proceed, or an error message string if blocked.
        """
        decision = self.policy.classify_tool(tool_name, params)
        summary = self._summarize(tool_name, params)

        if decision == PolicyDecision.SAFE:
            self._write_audit(tool_name, params, "auto_ran")
            return None

        if decision == PolicyDecision.DANGEROUS:
            self._write_audit(tool_name, params, "blocked")
            return f"Error: Command blocked by safety policy: {summary}"

        # NEEDS_APPROVAL
        from nanobot.bus.events import OutboundMessage

        await self._bus.publish_outbound(
            OutboundMessage(
                channel=self._channel or self._channel_var.get() or "cli",
                chat_id=self._chat_id or self._chat_id_var.get() or "direct",
                content=(
                    f"⚠️  Sandbox: approve this action?\n\n"
                    f"```\n{summary}\n```\n\n"
                    f"Reply yes or no"
                ),
                metadata={"_approval_request": True},
                kind="approval_request",
            )
        )

        event = asyncio.Event()
        sk = self._session_key or self._session_key_var.get() or ""
        self._pending_events[sk] = event

        try:
            await asyncio.wait_for(
                event.wait(), timeout=self.policy.config.approval_timeout_s
            )
            approved = self._pending_decisions.pop(sk, False)
        except asyncio.TimeoutError:
            self._pending_events.pop(sk, None)
            self._pending_decisions.pop(sk, None)
            self._write_audit(tool_name, params, "denied", "timeout")
            return (
                f"Error: Approval request timed out "
                f"({self.policy.config.approval_timeout_s}s)"
            )

        if approved:
            self._write_audit(tool_name, params, "approved", "yes")
            return None
        else:
            self._write_audit(tool_name, params, "denied", "no")
            return f"Error: Command rejected by user: {summary}"

    def has_pending(self, session_key: str) -> bool:
        return session_key in self._pending_events

    def respond(self, session_key: str, approved: bool) -> None:
        self._pending_decisions[session_key] = approved
        if event := self._pending_events.pop(session_key, None):
            event.set()

    @staticmethod
    def is_approval_response(text: str) -> bool:
        t = text.strip().lower()
        return t in ApprovalManager._APPROVE_WORDS or t in ApprovalManager._DENY_WORDS

    @staticmethod
    def _summarize(tool_name: str, params: dict) -> str:
        if tool_name == "exec":
            return f"exec: {params.get('command', '?')}"
        if tool_name == "write_file":
            return f"write_file: {params.get('path', '?')} ({len(params.get('content', ''))} chars)"
        if tool_name == "edit_file":
            return f"edit_file: {params.get('path', '?')}"
        return f"{tool_name}: {json.dumps(params, default=str)[:200]}"

    def _write_audit(
        self,
        tool_name: str,
        params: dict,
        decision: str,
        user: str | None = None,
    ) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "session": self._session_key or self._session_key_var.get(),
            "tool": tool_name,
            "summary": self._summarize(tool_name, params),
            "decision": decision,
            "user": user,
        }
        try:
            self._audit_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._audit_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass
