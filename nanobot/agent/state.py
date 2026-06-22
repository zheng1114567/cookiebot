"""LangGraph agent state definition."""

from __future__ import annotations

import operator
from typing import Annotated, NotRequired, TypedDict


class AgentState(TypedDict):
    """State for the agent LangGraph.

    ``messages`` uses ``operator.add`` so each node's returned messages
    are appended to the running list rather than replacing it.
    """

    messages: Annotated[list[dict], operator.add]
    iterations: int
    tools_used: list[str]
    error: NotRequired[str]


def make_should_continue(max_iterations: int):
    """Return a conditional-edge routing function.

    Returns ``"tools"`` when the last assistant message has ``tool_calls``
    and the iteration cap hasn't been hit, otherwise ``"end"``.
    """

    def should_continue(state: AgentState) -> str:
        messages = state["messages"]
        if not messages:
            return "end"
        last = messages[-1]
        if (
            last.get("role") == "assistant"
            and last.get("tool_calls")
            and state["iterations"] < max_iterations
        ):
            return "tools"
        return "end"

    return should_continue
