"""Middleware primitives for cross-cutting agent behaviors."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Awaitable, Protocol

from nanobot.bus.events import OutboundMessage


class ToolMiddleware(Protocol):
    """Hook tool execution before and after the concrete tool runs."""

    async def before_execute(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> dict[str, Any] | str | None:
        """Return updated params, a terminal string result, or None."""

    async def after_execute(
        self,
        tool_name: str,
        params: dict[str, Any],
        result: str,
    ) -> str:
        """Return the final tool result string."""


class OutboundMiddleware(Protocol):
    """Hook outbound messages before they are emitted to channels."""

    async def before_send(self, message: OutboundMessage) -> OutboundMessage | None:
        """Return the updated message, or None to suppress it."""


class ApprovalToolMiddleware:
    """Adapter that exposes an approval callback as tool middleware."""

    def __init__(self, checker):
        self._checker = checker

    async def before_execute(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> dict[str, Any] | str | None:
        error = await self._checker(tool_name, params)
        return error if error is not None else None

    async def after_execute(
        self,
        tool_name: str,
        params: dict[str, Any],
        result: str,
    ) -> str:
        return result


class OutboundDefaultsMiddleware:
    """Normalize outbound payloads without changing their behavior."""

    async def before_send(self, message: OutboundMessage) -> OutboundMessage | None:
        metadata = dict(message.metadata or {})
        return replace(message, metadata=metadata)
