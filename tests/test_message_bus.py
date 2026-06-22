import asyncio

import pytest

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus


def test_messages_have_lifecycle_ids() -> None:
    inbound = InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="hi")
    outbound = OutboundMessage(
        channel="cli",
        chat_id="direct",
        content="ok",
        correlation_id=inbound.event_id,
        kind="progress",
    )

    assert inbound.event_id
    assert outbound.event_id
    assert outbound.correlation_id == inbound.event_id
    assert outbound.kind == "progress"
    assert inbound.attempt == 0
    assert outbound.attempt == 0


@pytest.mark.asyncio
async def test_message_bus_supports_backpressure() -> None:
    bus = MessageBus(maxsize=1)
    await bus.publish_inbound(InboundMessage(channel="cli", sender_id="u", chat_id="c", content="one"))

    blocked = asyncio.create_task(
        bus.publish_inbound(InboundMessage(channel="cli", sender_id="u", chat_id="c", content="two"))
    )
    await asyncio.sleep(0)
    assert not blocked.done()

    consumed = await bus.consume_inbound()
    assert consumed.content == "one"
    await blocked
    assert (await bus.consume_inbound()).content == "two"


def test_message_bus_default_is_unbounded_for_library_usage() -> None:
    bus = MessageBus()

    assert bus.inbound.maxsize == 0
    assert bus.outbound.maxsize == 0
