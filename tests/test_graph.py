"""Unit tests for the LangGraph agent graph."""

from __future__ import annotations

import pytest

from nanobot.agent.graph import create_agent_graph, llm_node, tool_node
from nanobot.agent.state import AgentState, make_should_continue
from nanobot.agent.tool_results import ToolResultCompressor
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest


# ── helpers ──────────────────────────────────────────────────────────────────


class FakeTool(Tool):
    """A tool that echoes its input."""

    @property
    def name(self) -> str:
        return "fake_tool"

    @property
    def description(self) -> str:
        return "A fake tool for testing."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"input": {"type": "string"}},
            "required": ["input"],
        }

    async def execute(self, **kwargs) -> str:
        return f"fake_tool result: {kwargs.get('input', '')}"


class ScriptedProvider(LLMProvider):
    """Returns pre-scripted LLM responses in order."""

    def __init__(self, responses):
        super().__init__()
        self._responses = list(responses)
        self.calls: int = 0

    async def chat(self, *args, **kwargs) -> LLMResponse:
        self.calls += 1
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def get_default_model(self) -> str:
        return "test-model"


def _make_tools() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(FakeTool())
    return reg


class LargeTool(Tool):
    @property
    def name(self) -> str:
        return "large_tool"

    @property
    def description(self) -> str:
        return "Returns a large payload."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }

    async def execute(self, **kwargs) -> str:
        return kwargs["text"]


def _make_large_tools() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(LargeTool())
    return reg


def _make_state(messages=None, iterations=0, tools_used=None):
    return {
        "messages": list(messages or []),
        "iterations": iterations,
        "tools_used": list(tools_used or []),
    }


# ── make_should_continue ─────────────────────────────────────────────────────


class TestShouldContinue:
    def test_returns_tools_when_last_msg_has_tool_calls_and_under_limit(self):
        fn = make_should_continue(max_iterations=5)
        state = _make_state(
            messages=[{"role": "assistant", "content": None, "tool_calls": [{}]}],
            iterations=3,
        )
        assert fn(state) == "tools"

    def test_returns_end_when_no_tool_calls(self):
        fn = make_should_continue(max_iterations=5)
        state = _make_state(
            messages=[{"role": "assistant", "content": "done"}],
            iterations=3,
        )
        assert fn(state) == "end"

    def test_returns_end_when_at_max_iterations(self):
        fn = make_should_continue(max_iterations=5)
        state = _make_state(
            messages=[{"role": "assistant", "content": None, "tool_calls": [{}]}],
            iterations=5,
        )
        assert fn(state) == "end"

    def test_returns_end_when_no_messages(self):
        fn = make_should_continue(max_iterations=5)
        state = _make_state(messages=[], iterations=0)
        assert fn(state) == "end"


# ── llm_node ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_llm_node_returns_assistant_message_without_tools():
    provider = ScriptedProvider([LLMResponse(content="Hello!")])
    tools = _make_tools()
    state = _make_state(messages=[{"role": "user", "content": "hi"}])

    result = await llm_node(state, provider=provider, tools=tools, model="test")

    assert len(result["messages"]) == 1
    assert result["messages"][0]["role"] == "assistant"
    assert result["messages"][0]["content"] == "Hello!"
    assert result["iterations"] == 1


@pytest.mark.asyncio
async def test_llm_node_returns_assistant_message_with_tool_calls():
    provider = ScriptedProvider([
        LLMResponse(
            content="Let me search...",
            tool_calls=[ToolCallRequest(id="1", name="fake_tool", arguments={"input": "test"})],
        )
    ])
    tools = _make_tools()
    state = _make_state(messages=[{"role": "user", "content": "search"}])

    result = await llm_node(state, provider=provider, tools=tools, model="test")

    assert result["messages"][0]["role"] == "assistant"
    assert result["messages"][0]["tool_calls"] is not None
    assert result["iterations"] == 1


@pytest.mark.asyncio
async def test_llm_node_stores_error_when_finish_reason_is_error():
    """Error responses are NOT put in messages (#1303)."""
    provider = ScriptedProvider([
        LLMResponse(content="401 unauthorized", finish_reason="error"),
    ])
    tools = _make_tools()
    state = _make_state(messages=[{"role": "user", "content": "hi"}])

    result = await llm_node(state, provider=provider, tools=tools, model="test")

    assert "messages" not in result
    assert result["error"] == "401 unauthorized"
    assert result["iterations"] == 1


# ── tool_node ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tool_node_executes_tool_calls_and_returns_results():
    tools = _make_tools()
    state = _make_state(
        messages=[
            {"role": "user", "content": "search"},
            {
                "role": "assistant",
                "content": "ok",
                "tool_calls": [
                    {"id": "1", "type": "function", "function": {"name": "fake_tool", "arguments": '{"input": "hello"}'}}
                ],
            },
        ],
    )

    result = await tool_node(state, tools=tools)

    assert len(result["messages"]) == 1
    assert result["messages"][0]["role"] == "tool"
    assert result["messages"][0]["name"] == "fake_tool"
    assert "fake_tool result" in result["messages"][0]["content"]
    assert result["tools_used"] == ["fake_tool"]


@pytest.mark.asyncio
async def test_tool_node_streams_tool_progress():
    tools = _make_tools()
    state = _make_state(
        messages=[
            {"role": "user", "content": "search"},
            {
                "role": "assistant",
                "content": "ok",
                "tool_calls": [
                    {"id": "1", "type": "function", "function": {"name": "fake_tool", "arguments": '{"input": "hello"}'}}
                ],
            },
        ],
    )
    progress: list[tuple[str, bool]] = []

    async def on_progress(content: str, *, tool_hint: bool = False) -> None:
        progress.append((content, tool_hint))

    await tool_node(state, tools=tools, on_progress=on_progress)

    assert progress == [
        ('Starting fake_tool("hello")', True),
        ("fake_tool completed", True),
    ]


@pytest.mark.asyncio
async def test_tool_node_compresses_large_results_and_spills_to_disk(tmp_path):
    tools = _make_large_tools()
    large_text = "A" * 13_500
    state = _make_state(
        messages=[
            {"role": "user", "content": "analyze"},
            {
                "role": "assistant",
                "content": "ok",
                "tool_calls": [
                    {"id": "1", "type": "function", "function": {"name": "large_tool", "arguments": f'{{"text": "{large_text}"}}'}}
                ],
            },
        ],
    )

    result = await tool_node(state, tools=tools, workspace=tmp_path)

    content = result["messages"][0]["content"]
    assert "[Tool result compressed: large_tool]" in content
    assert "Stored full output:" in content
    assert "Use read_file with offset/limit" in content
    artifacts = list((tmp_path / "memory" / "artifacts" / "tool-results").glob("*.txt"))
    assert len(artifacts) == 1
    assert artifacts[0].read_text(encoding="utf-8") == large_text


@pytest.mark.asyncio
async def test_tool_result_compressor_summarizes_large_results(tmp_path):
    provider = ScriptedProvider([
        LLMResponse(content="Summary:\nfirst chunk"),
        LLMResponse(content="Summary:\nsecond chunk"),
        LLMResponse(content="Global summary:\ncombined"),
    ])
    compressor = ToolResultCompressor(workspace=tmp_path, provider=provider, model="test")
    compressor.INLINE_CHAR_BUDGET = 30
    compressor.CHUNK_CHAR_BUDGET = 20
    compressor.CHUNK_OVERLAP = 0

    result = await compressor.compress("large_tool", "abc", "A" * 20 + "B" * 20)

    assert "Whole-document digest:" in result
    assert "Global summary:\ncombined" in result
    assert provider.calls == 3


# ── create_agent_graph (end-to-end) ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_graph_single_turn_no_tools():
    """LLM responds without tool calls → should end immediately."""
    provider = ScriptedProvider([LLMResponse(content="Hi there!")])
    tools = _make_tools()
    graph = create_agent_graph(
        provider=provider, tools=tools, model="test", max_iterations=5,
    )
    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "hello"}], "iterations": 0, "tools_used": []},
        config={"recursion_limit": 50},
    )
    assert provider.calls == 1
    # Last assistant message should be the final response
    assert any(
        m["role"] == "assistant" and m["content"] == "Hi there!"
        for m in result["messages"]
    )


@pytest.mark.asyncio
async def test_graph_multi_turn_tool_then_end():
    """LLM calls tool, tool executes, LLM responds without tools → end."""
    provider = ScriptedProvider([
        LLMResponse(
            content="calling tool...",
            tool_calls=[ToolCallRequest(id="1", name="fake_tool", arguments={"input": "query"})],
        ),
        LLMResponse(content="Final answer based on tool result."),
    ])
    tools = _make_tools()
    graph = create_agent_graph(
        provider=provider, tools=tools, model="test", max_iterations=5,
    )
    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "answer"}], "iterations": 0, "tools_used": []},
        config={"recursion_limit": 50},
    )
    assert provider.calls == 2
    assert result["tools_used"] == ["fake_tool"]
    assert any(
        m["role"] == "assistant" and m["content"] == "Final answer based on tool result."
        for m in result["messages"]
    )


@pytest.mark.asyncio
async def test_graph_max_iterations_enforced():
    """Graph stops after max_iterations even with repeated tool calls."""
    responses = []
    for _ in range(5):
        responses.append(LLMResponse(
            content="still searching...",
            tool_calls=[ToolCallRequest(id="1", name="fake_tool", arguments={"input": "q"})],
        ))
    responses.append(LLMResponse(content="This should never be returned."))
    provider = ScriptedProvider(responses)
    tools = _make_tools()
    graph = create_agent_graph(
        provider=provider, tools=tools, model="test", max_iterations=3,
    )
    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "go"}], "iterations": 0, "tools_used": []},
        config={"recursion_limit": 50},
    )
    # Should stop at iteration 3, not reach the 6th response
    assert provider.calls == 3
    assert result["iterations"] == 3
    # No final assistant message without tool_calls
    last_msgs = result["messages"][-3:]
    assert not any(
        m.get("role") == "assistant" and m.get("content") and not m.get("tool_calls")
        for m in last_msgs
    )


@pytest.mark.asyncio
async def test_graph_error_handling():
    """LLM returns error → should end and store error."""
    provider = ScriptedProvider([
        LLMResponse(content="401 unauthorized", finish_reason="error"),
    ])
    tools = _make_tools()
    graph = create_agent_graph(
        provider=provider, tools=tools, model="test", max_iterations=5,
    )
    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "hi"}], "iterations": 0, "tools_used": []},
        config={"recursion_limit": 50},
    )
    assert result["error"] == "401 unauthorized"
    # Error message should NOT be in messages
    assert not any(
        m.get("role") == "assistant" and "unauthorized" in str(m.get("content", ""))
        for m in result["messages"]
    )
