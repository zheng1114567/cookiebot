from nanobot.bus.queue import MessageBus
from nanobot.channels.mochat import MochatChannel, MochatConfig


async def test_mochat_recall_event_publishes_control_message() -> None:
    bus = MessageBus()
    channel = MochatChannel(
        MochatConfig(enabled=True, claw_token="token", allow_from=["user1"]),
        bus,
    )

    await channel._handle_notify_chat_message(
        {
            "groupId": "group1",
            "converseId": "panel1",
            "author": "user1",
            "messageId": "msg1",
        },
        "notify:chat.message.recall",
    )

    msg = await bus.consume_inbound()
    assert msg.channel == "mochat"
    assert msg.chat_id == "panel1"
    assert msg.sender_id == "user1"
    assert msg.content == ""
    assert msg.metadata["_control"] == "recall"
    assert msg.metadata["message_id"] == "msg1"


async def test_mochat_delete_event_publishes_control_message() -> None:
    bus = MessageBus()
    channel = MochatChannel(
        MochatConfig(enabled=True, claw_token="token", allow_from=["user1"]),
        bus,
    )

    await channel._handle_notify_chat_message(
        {
            "groupId": "group1",
            "panelId": "panel1",
            "author": "user1",
            "_id": "msg2",
        },
        "notify:chat.message.delete",
    )

    msg = await bus.consume_inbound()
    assert msg.metadata["_control"] == "delete"
    assert msg.metadata["message_id"] == "msg2"
