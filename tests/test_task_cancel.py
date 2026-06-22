"""Tests for /stop task cancellation."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_loop():
    """Create a minimal AgentLoop with mocked dependencies."""
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    workspace = MagicMock()
    workspace.__truediv__ = MagicMock(return_value=MagicMock())

    with patch("nanobot.agent.loop.ContextBuilder"), \
         patch("nanobot.agent.loop.SessionManager"), \
         patch("nanobot.agent.loop.SubagentManager") as MockSubMgr:
        MockSubMgr.return_value.cancel_by_session = AsyncMock(return_value=0)
        loop = AgentLoop(bus=bus, provider=provider, workspace=workspace)
    return loop, bus


class TestHandleStop:
    @pytest.mark.asyncio
    async def test_stop_no_active_task(self):
        from nanobot.bus.events import InboundMessage

        loop, bus = _make_loop()
        msg = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="/stop")
        await loop._handle_stop(msg)
        out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        assert "No active task" in out.content

    @pytest.mark.asyncio
    async def test_stop_cancels_active_task(self):
        from nanobot.bus.events import InboundMessage

        loop, bus = _make_loop()
        cancelled = asyncio.Event()

        async def slow_task():
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        task = asyncio.create_task(slow_task())
        await asyncio.sleep(0)
        loop._active_tasks["test:c1"] = [task]

        msg = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="/stop")
        await loop._handle_stop(msg)

        assert cancelled.is_set()
        out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        assert "stopped" in out.content.lower()

    @pytest.mark.asyncio
    async def test_stop_cancels_multiple_tasks(self):
        from nanobot.bus.events import InboundMessage

        loop, bus = _make_loop()
        events = [asyncio.Event(), asyncio.Event()]

        async def slow(idx):
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                events[idx].set()
                raise

        tasks = [asyncio.create_task(slow(i)) for i in range(2)]
        await asyncio.sleep(0)
        loop._active_tasks["test:c1"] = tasks

        msg = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="/stop")
        await loop._handle_stop(msg)

        assert all(e.is_set() for e in events)
        out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        assert "2 task" in out.content


class TestDispatch:
    @pytest.mark.asyncio
    async def test_dispatch_processes_and_publishes(self):
        from nanobot.bus.events import InboundMessage, OutboundMessage

        loop, bus = _make_loop()
        msg = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="hello")
        loop._process_message = AsyncMock(
            return_value=OutboundMessage(channel="test", chat_id="c1", content="hi")
        )
        await loop._dispatch(msg)
        out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        assert out.content == "hi"

    @pytest.mark.asyncio
    async def test_session_lock_serializes_same_session(self):
        from nanobot.bus.events import InboundMessage, OutboundMessage

        loop, bus = _make_loop()
        order = []

        async def mock_process(m, **kwargs):
            order.append(f"start-{m.content}")
            await asyncio.sleep(0.05)
            order.append(f"end-{m.content}")
            return OutboundMessage(channel="test", chat_id="c1", content=m.content)

        loop._process_message = mock_process
        msg1 = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="a")
        msg2 = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="b")

        t1 = asyncio.create_task(loop._dispatch(msg1))
        t2 = asyncio.create_task(loop._dispatch(msg2))
        await asyncio.gather(t1, t2)
        assert order == ["start-a", "end-a", "start-b", "end-b"]

    @pytest.mark.asyncio
    async def test_different_sessions_can_process_concurrently(self):
        from nanobot.bus.events import InboundMessage, OutboundMessage

        loop, bus = _make_loop()
        started = asyncio.Event()
        release = asyncio.Event()
        order = []

        async def mock_process(m, **kwargs):
            order.append(f"start-{m.chat_id}")
            if m.chat_id == "c1":
                started.set()
                await release.wait()
            return OutboundMessage(channel="test", chat_id=m.chat_id, content=m.content)

        loop._process_message = mock_process
        msg1 = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="a")
        msg2 = InboundMessage(channel="test", sender_id="u2", chat_id="c2", content="b")

        t1 = asyncio.create_task(loop._dispatch(msg1))
        await started.wait()
        t2 = asyncio.create_task(loop._dispatch(msg2))
        await asyncio.sleep(0)
        release.set()
        await asyncio.gather(t1, t2)

        assert order == ["start-c1", "start-c2"]

    def test_agent_loop_passes_tool_timeout_to_registry(self, tmp_path):
        from nanobot.agent.loop import AgentLoop
        from nanobot.bus.queue import MessageBus

        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"

        loop = AgentLoop(
            bus=MessageBus(),
            provider=provider,
            workspace=tmp_path,
            tool_timeout=7,
        )

        assert loop.tools.default_timeout == 7

    @pytest.mark.asyncio
    async def test_dispatch_times_out_stuck_turn(self, tmp_path):
        from nanobot.agent.loop import AgentLoop
        from nanobot.bus.events import InboundMessage
        from nanobot.bus.queue import MessageBus

        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        loop = AgentLoop(
            bus=bus,
            provider=provider,
            workspace=tmp_path,
            turn_timeout_s=0.01,
        )

        async def stuck(_msg):
            await asyncio.sleep(10)

        loop._process_message = stuck
        msg = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="hello")

        await loop._dispatch(msg)
        out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)

        assert out.kind == "error"
        assert out.correlation_id == msg.event_id
        assert "timed out" in out.content


class TestRunInterruptPolicy:
    @pytest.mark.asyncio
    async def test_same_session_followup_queues_without_cancelling(self):
        from nanobot.bus.events import InboundMessage, OutboundMessage

        loop, bus = _make_loop()
        started = asyncio.Event()
        release = asyncio.Event()
        cancelled = asyncio.Event()
        order: list[str] = []

        async def mock_process(msg):
            order.append(f"start-{msg.content}")
            if msg.content == "a":
                started.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
            order.append(f"end-{msg.content}")
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=msg.content)

        loop._process_message = mock_process
        runner = asyncio.create_task(loop.run())
        await bus.publish_inbound(InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="a"))
        await started.wait()
        await bus.publish_inbound(InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="补充一下"))
        await asyncio.sleep(0.05)

        assert not cancelled.is_set()
        assert order == ["start-a"]

        release.set()
        assert (await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)).content == "a"
        assert (await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)).content == "补充一下"
        loop.stop()
        await asyncio.wait_for(runner, timeout=2.0)

    @pytest.mark.asyncio
    async def test_interrupt_prefix_cancels_and_processes_new_message(self):
        from nanobot.bus.events import InboundMessage, OutboundMessage

        loop, bus = _make_loop()
        started = asyncio.Event()
        cancelled = asyncio.Event()
        order: list[str] = []

        async def mock_process(msg):
            order.append(f"start-{msg.content}")
            if msg.content == "a":
                started.set()
                try:
                    await asyncio.sleep(60)
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
            order.append(f"end-{msg.content}")
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=msg.content)

        loop._process_message = mock_process
        runner = asyncio.create_task(loop.run())
        await bus.publish_inbound(InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="a"))
        await started.wait()
        await bus.publish_inbound(InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="停，改成 b"))

        # Interrupt is acknowledged but NOT dispatched to LLM as a message
        assert (await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)).content == "Stopped. Send a new message to continue."
        assert cancelled.is_set()
        # The interrupt content is not forwarded to LLM — only cancellation happens
        assert order == ["start-a"]
        loop.stop()
        await asyncio.wait_for(runner, timeout=2.0)

    @pytest.mark.asyncio
    async def test_recall_control_cancels_without_dispatching_message(self):
        from nanobot.bus.events import InboundMessage, OutboundMessage

        loop, bus = _make_loop()
        started = asyncio.Event()
        cancelled = asyncio.Event()
        order: list[str] = []

        async def mock_process(msg):
            order.append(f"start-{msg.content}")
            if msg.content == "a":
                started.set()
                try:
                    await asyncio.sleep(60)
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=msg.content)

        loop._process_message = mock_process
        runner = asyncio.create_task(loop.run())
        await bus.publish_inbound(InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="a"))
        await started.wait()
        await bus.publish_inbound(InboundMessage(
            channel="test",
            sender_id="u1",
            chat_id="c1",
            content="",
            metadata={"_control": "recall", "message_id": "m1"},
        ))
        await asyncio.sleep(0.05)

        assert cancelled.is_set()
        assert order == ["start-a"]
        assert bus.outbound_size == 0
        loop.stop()
        await asyncio.wait_for(runner, timeout=2.0)


class TestSubagentCancellation:
    @pytest.mark.asyncio
    async def test_cancel_by_session(self):
        from nanobot.agent.subagent import SubagentManager
        from nanobot.bus.queue import MessageBus

        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        mgr = SubagentManager(provider=provider, workspace=MagicMock(), bus=bus)

        cancelled = asyncio.Event()

        async def slow():
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        task = asyncio.create_task(slow())
        await asyncio.sleep(0)
        mgr._running_tasks["sub-1"] = task
        mgr._session_tasks["test:c1"] = {"sub-1"}

        count = await mgr.cancel_by_session("test:c1")
        assert count == 1
        assert cancelled.is_set()

    @pytest.mark.asyncio
    async def test_cancel_by_session_no_tasks(self):
        from nanobot.agent.subagent import SubagentManager
        from nanobot.bus.queue import MessageBus

        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        mgr = SubagentManager(provider=provider, workspace=MagicMock(), bus=bus)
        assert await mgr.cancel_by_session("nonexistent") == 0

    @pytest.mark.asyncio
    async def test_subagent_preserves_reasoning_fields_in_tool_turn(self, monkeypatch, tmp_path):
        from nanobot.agent.subagent import SubagentManager
        from nanobot.bus.queue import MessageBus
        from nanobot.providers.base import LLMResponse, ToolCallRequest

        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"

        captured_second_call: list[dict] = []

        call_count = {"n": 0}

        async def scripted_chat_with_retry(*, messages, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return LLMResponse(
                    content="thinking",
                    tool_calls=[ToolCallRequest(id="call_1", name="list_dir", arguments={})],
                    reasoning_content="hidden reasoning",
                    thinking_blocks=[{"type": "thinking", "thinking": "step"}],
                )
            captured_second_call[:] = messages
            return LLMResponse(content="done", tool_calls=[])
        provider.chat_with_retry = scripted_chat_with_retry
        mgr = SubagentManager(provider=provider, workspace=tmp_path, bus=bus)

        async def fake_execute(self, name, arguments):
            return "tool result"

        monkeypatch.setattr("nanobot.agent.tools.registry.ToolRegistry.execute", fake_execute)

        await mgr._run_subagent("sub-1", "do task", "label", {"channel": "test", "chat_id": "c1"})

        assistant_messages = [
            msg for msg in captured_second_call
            if msg.get("role") == "assistant" and msg.get("tool_calls")
        ]
        assert len(assistant_messages) == 1
        assert assistant_messages[0]["reasoning_content"] == "hidden reasoning"
        assert assistant_messages[0]["thinking_blocks"] == [{"type": "thinking", "thinking": "step"}]
