"""LangGraph agent graph — reusable LLM ↔ tools loop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Awaitable, Callable

from loguru import logger
from langgraph.graph import END, StateGraph

from nanobot.agent.state import AgentState, make_should_continue
from nanobot.agent.tool_results import ToolResultCompressor
from nanobot.providers.base import LLMProvider
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.utils.helpers import build_assistant_message


def _strip_think(text: str | None) -> str | None:
    """Remove <think>…</think> blocks that some models embed in content."""
    import re

    if not text:
        return None
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or None


def _tool_hint(tool_calls: list) -> str:
    """Format tool calls as concise hint, e.g. 'web_search("query")'."""

    def _fmt(tc):
        args = (
            tc.arguments[0]
            if isinstance(tc.arguments, list)
            else tc.arguments
        ) or {}
        val = next(iter(args.values()), None) if isinstance(args, dict) else None
        if not isinstance(val, str):
            return tc.name
        return (
            f'{tc.name}("{val[:40]}…")'
            if len(val) > 40
            else f'{tc.name}("{val}")'
        )

    return ", ".join(_fmt(tc) for tc in tool_calls)


def _tool_call_hint(tc: dict) -> str:
    """Format an OpenAI-style tool call dict for progress updates."""
    name = tc["function"]["name"]
    try:
        args = json.loads(tc["function"]["arguments"])
    except Exception:
        return name
    if not isinstance(args, dict) or not args:
        return name
    val = next(iter(args.values()), None)
    if not isinstance(val, str):
        return name
    return f'{name}("{val[:40]}…")' if len(val) > 40 else f'{name}("{val}")'


async def llm_node(
    state: AgentState,
    *,
    provider: LLMProvider,
    tools: ToolRegistry,
    model: str,
    on_progress: Callable[..., Awaitable[None]] | None = None,
) -> dict:
    """Call the LLM and return the assistant response message."""

    response = await provider.chat_with_retry(
        messages=state["messages"],
        tools=tools.get_definitions(),
        model=model,
    )

    iteration = state.get("iterations", 0) + 1

    if response.has_tool_calls:
        if on_progress:
            thought = _strip_think(response.content)
            if thought:
                await on_progress(thought)
            hint = _tool_hint(response.tool_calls)
            hint = _strip_think(hint)
            await on_progress(hint, tool_hint=True)

        tool_call_dicts = [tc.to_openai_tool_call() for tc in response.tool_calls]
        assistant_msg = build_assistant_message(
            response.content,
            tool_calls=tool_call_dicts,
            reasoning_content=response.reasoning_content,
            thinking_blocks=response.thinking_blocks,
        )
        return {"messages": [assistant_msg], "iterations": iteration}
    else:
        clean = _strip_think(response.content)

        if response.finish_reason == "error":
            logger.error("LLM returned error: {}", (clean or "")[:200])
            clean = clean or "Sorry, I encountered an error calling the AI model."
            # Don't add to messages — avoids persisting error responses that
            # can poison session context and cause permanent 400 loops (#1303).
            return {"iterations": iteration, "error": clean}

        assistant_msg = build_assistant_message(
            clean,
            reasoning_content=response.reasoning_content,
            thinking_blocks=response.thinking_blocks,
        )
        return {"messages": [assistant_msg], "iterations": iteration}


async def tool_node(
    state: AgentState,
    *,
    tools: ToolRegistry,
    provider: LLMProvider | None = None,
    model: str | None = None,
    workspace: Path | None = None,
    on_progress: Callable[..., Awaitable[None]] | None = None,
) -> dict:
    """Execute tool calls from the last assistant message."""

    last_msg = state["messages"][-1]
    tool_calls = last_msg.get("tool_calls", [])
    compressor = ToolResultCompressor(workspace=workspace, provider=provider, model=model)

    tool_msgs = []
    tools_used = list(state.get("tools_used", []))

    for tc in tool_calls:
        name = tc["function"]["name"]
        args = json.loads(tc["function"]["arguments"])
        tools_used.append(name)
        args_str = json.dumps(args, ensure_ascii=False)
        logger.info("Tool call: {}({})", name, args_str[:200])
        if on_progress:
            await on_progress(f"Starting {_tool_call_hint(tc)}", tool_hint=True)
        result = await tools.execute(name, args)
        if on_progress:
            if isinstance(result, str) and "timed out" in result.lower():
                await on_progress(f"{name} timed out; trying another path", tool_hint=True)
            elif isinstance(result, str) and result.startswith("Error"):
                await on_progress(f"{name} returned an error; adjusting", tool_hint=True)
            else:
                await on_progress(f"{name} completed", tool_hint=True)
        if isinstance(result, str):
            result = await compressor.compress(name, tc["id"], result)
        tool_msgs.append(
            {
                "role": "tool",
                "tool_call_id": tc["id"],
                "name": name,
                "content": result,
            }
        )

    return {"messages": tool_msgs, "tools_used": tools_used}


def create_agent_graph(
    *,
    provider: LLMProvider,
    tools: ToolRegistry,
    model: str,
    max_iterations: int,
    workspace: Path | None = None,
    on_progress: Callable[..., Awaitable[None]] | None = None,
):
    """Create a compiled LangGraph StateGraph for the agent loop.

    Graph structure::

        START → llm_node → [conditional]
                  ↑           ├── tool_calls & iter < max → tool_node → llm (loop)
                  └───────────┘   (via tool_node → llm edge)
                               └── otherwise → END
    """

    graph = StateGraph(AgentState)

    async def _llm_node(state: AgentState) -> dict:
        return await llm_node(
            state,
            provider=provider,
            tools=tools,
            model=model,
            on_progress=on_progress,
        )

    async def _tool_node(state: AgentState) -> dict:
        return await tool_node(
            state,
            tools=tools,
            provider=provider,
            model=model,
            workspace=workspace,
            on_progress=on_progress,
        )

    graph.add_node("llm", _llm_node)
    graph.add_node("tools", _tool_node)

    graph.set_entry_point("llm")

    graph.add_conditional_edges(
        "llm",
        make_should_continue(max_iterations),
        {"tools": "tools", "end": END},
    )
    graph.add_edge("tools", "llm")

    return graph.compile()
