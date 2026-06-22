"""Tool registry for dynamic tool management."""

import asyncio
from contextvars import ContextVar
from typing import Any, Awaitable, Callable

from nanobot.agent.middleware import ApprovalToolMiddleware, ToolMiddleware
from nanobot.agent.tools.base import Tool


class ToolRegistry:
    """
    Registry for agent tools.

    Allows dynamic registration and execution of tools.
    """

    def __init__(self, default_timeout: float | None = 120):
        self._tools: dict[str, Tool] = {}
        self.default_timeout = default_timeout
        self._middlewares: list[ToolMiddleware] = []
        self._approval_handler_var: ContextVar[Callable[..., Awaitable[str | None]] | None] = ContextVar(
            "tool_approval_handler", default=None
        )

    def set_approval_handler(
        self, handler: Callable[..., Awaitable[str | None]] | None
    ) -> None:
        """Set an optional pre-execution approval handler."""
        self._approval_handler_var.set(handler)

    def register_middleware(self, middleware: ToolMiddleware) -> None:
        """Register a tool middleware."""
        self._middlewares.append(middleware)

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def get_definitions(self) -> list[dict[str, Any]]:
        """Get all tool definitions in OpenAI format."""
        return [tool.to_schema() for tool in self._tools.values()]

    async def aclose(self) -> None:
        """Close tools that expose an async close hook."""
        for tool in self._tools.values():
            close = getattr(tool, "aclose", None)
            if callable(close):
                try:
                    await close()
                except Exception:
                    pass

    async def execute(self, name: str, params: dict[str, Any]) -> str:
        """Execute a tool by name with given parameters."""
        _HINT = "\n\n[Analyze the error above and try a different approach.]"

        approval_handler = self._approval_handler_var.get()
        middlewares = list(self._middlewares)
        if approval_handler:
            middlewares.insert(0, ApprovalToolMiddleware(approval_handler))

        tool = self._tools.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found. Available: {', '.join(self.tool_names)}"

        try:
            for middleware in middlewares:
                updated = await middleware.before_execute(name, params)
                if isinstance(updated, str):
                    return updated
                if isinstance(updated, dict):
                    params = updated

            # Attempt to cast parameters to match schema types
            params = tool.cast_params(params)

            # Validate parameters
            errors = tool.validate_params(params)
            if errors:
                return f"Error: Invalid parameters for tool '{name}': " + "; ".join(errors) + _HINT
            coro = tool.execute(**params)
            if self.default_timeout and self.default_timeout > 0:
                result = await asyncio.wait_for(coro, timeout=self.default_timeout)
            else:
                result = await coro
            for middleware in middlewares:
                result = await middleware.after_execute(name, params, result)
            if isinstance(result, str) and result.startswith("Error"):
                return result + _HINT
            return result
        except asyncio.TimeoutError:
            return (
                f"Error: Tool '{name}' timed out after {self.default_timeout:g} seconds"
                + _HINT
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            return f"Error executing {name}: {str(e)}" + _HINT

    @property
    def tool_names(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
