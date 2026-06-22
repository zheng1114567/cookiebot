import asyncio

import pytest

from nanobot.agent.tools.message import MessageTool
from nanobot.bus.events import OutboundMessage


@pytest.mark.asyncio
async def test_message_tool_returns_error_when_no_target_context() -> None:
    tool = MessageTool()
    result = await tool.execute(content="test")
    assert result == "Error: No target channel/chat specified"


@pytest.mark.asyncio
async def test_message_tool_context_is_task_local() -> None:
    sent: list[OutboundMessage] = []

    async def send(msg: OutboundMessage) -> None:
        sent.append(msg)

    tool = MessageTool(send_callback=send)

    async def run_turn(chat_id: str) -> bool:
        tool.set_context("test", chat_id)
        tool.start_turn()
        await tool.execute(content=f"hello {chat_id}")
        return tool.sent_in_current_turn()

    results = await asyncio.gather(run_turn("c1"), run_turn("c2"))

    assert results == [True, True]
    assert {msg.chat_id for msg in sent} == {"c1", "c2"}
