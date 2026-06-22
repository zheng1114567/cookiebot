"""WeCom self-built-app channel — HTTP callback mode for 企业微信自建应用."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib.util
import json
import random
import socket
import string
import struct
import time
from collections import OrderedDict
from typing import Any
from xml.etree import ElementTree

from Crypto.Cipher import AES
from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import Base
from pydantic import Field

AIOHTTP_AVAILABLE = importlib.util.find_spec("aiohttp") is not None


class WecomSelfbuiltConfig(Base):
    """WeCom self-built app channel configuration (callback/HTTP mode)."""

    enabled: bool = False
    corp_id: str = ""
    agent_id: str = ""
    secret: str = ""
    token: str = ""
    encoding_aes_key: str = ""
    callback_port: int = 9898
    callback_path: str = "/wxcomapp"
    callback_host: str = "0.0.0.0"
    allow_from: list[str] = Field(default_factory=list)
    welcome_message: str = ""


# ── WeCom message crypto (WXBizMsgCrypt) ────────────────────────────────────


class WXBizMsgCrypt:
    """WeCom message encryption/decryption (AES-256-CBC)."""

    def __init__(self, token: str, encoding_aes_key: str, corp_id: str):
        self.token = token
        self.corp_id = corp_id
        self.aes_key = base64.b64decode(encoding_aes_key + "=")

    def _pad(self, text: bytes) -> bytes:
        block_size = 32
        pad = block_size - (len(text) % block_size)
        return text + bytes([pad] * pad)

    def _unpad(self, text: bytes) -> bytes:
        pad = text[-1]
        return text[:-pad]

    def encrypt(self, msg: str) -> str:
        """Encrypt a reply message."""
        random_bytes = bytes(random.getrandbits(8) for _ in range(16))
        raw = random_bytes + struct.pack("!I", len(msg.encode())) + msg.encode() + self.corp_id.encode()
        padded = self._pad(raw)
        cipher = AES.new(self.aes_key, AES.MODE_CBC, self.aes_key[:16])
        encrypted = cipher.encrypt(padded)
        encrypted_b64 = base64.b64encode(encrypted).decode()

        timestamp = str(int(time.time()))
        nonce = "".join(random.choices(string.digits, k=10))
        signature = self._sign(timestamp, nonce, encrypted_b64)

        return f"""<xml>
<Encrypt><![CDATA[{encrypted_b64}]]></Encrypt>
<MsgSignature><![CDATA[{signature}]]></MsgSignature>
<TimeStamp>{timestamp}</TimeStamp>
<Nonce><![CDATA[{nonce}]]></Nonce>
</xml>"""

    def decrypt(self, timestamp: str, nonce: str, signature: str, encrypted: str) -> str | None:
        """Decrypt an incoming message. Returns XML string or None on failure."""
        if signature != self._sign(timestamp, nonce, encrypted):
            return None

        cipher = AES.new(self.aes_key, AES.MODE_CBC, self.aes_key[:16])
        try:
            decrypted = cipher.decrypt(base64.b64decode(encrypted))
            decrypted = self._unpad(decrypted)
            # Skip 16 random bytes + 4 bytes msg length
            content = decrypted[20:]
            # Strip corp_id suffix
            content = content[: -len(self.corp_id.encode())]
            return content.decode()
        except Exception:
            return None

    def verify_url(self, timestamp: str, nonce: str, echostr: str, signature: str) -> str | None:
        """Verify callback URL. Returns decrypted echostr or None."""
        if signature != self._sign(timestamp, nonce, echostr):
            return None
        cipher = AES.new(self.aes_key, AES.MODE_CBC, self.aes_key[:16])
        try:
            decrypted = cipher.decrypt(base64.b64decode(echostr))
            decrypted = self._unpad(decrypted)
            content = decrypted[16:]  # skip 16 random bytes
            content = content[: -len(self.corp_id.encode())]
            return content.decode()
        except Exception:
            return None

    def _sign(self, timestamp: str, nonce: str, encrypted: str) -> str:
        params = sorted([self.token, timestamp, nonce, encrypted])
        return hashlib.sha1("".join(params).encode()).hexdigest()


# ── Channel ──────────────────────────────────────────────────────────────────


class WecomSelfbuiltChannel(BaseChannel):
    """WeCom self-built app channel using HTTP callback mode.

    Starts a local aiohttp HTTP server to receive callbacks from WeCom.
    Uses the WeCom API to send reply messages.
    """

    name = "wecom_selfbuilt"
    display_name = "WeCom Self-built"

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return WecomSelfbuiltConfig().model_dump(by_alias=True)

    def __init__(self, config: Any, bus: MessageBus):
        if isinstance(config, dict):
            config = WecomSelfbuiltConfig.model_validate(config)
        super().__init__(config, bus)
        self.config: WecomSelfbuiltConfig = config
        self._crypto = WXBizMsgCrypt(
            config.token, config.encoding_aes_key, config.corp_id
        )
        self._access_token: str | None = None
        self._token_expires: float = 0
        self._processed_ids: OrderedDict[str, None] = OrderedDict()
        self._server: Any = None
        self._runner: Any = None

    # -- start / stop ---------------------------------------------------------

    async def start(self) -> None:
        if not AIOHTTP_AVAILABLE:
            logger.error("aiohttp not installed. Run: pip install aiohttp")
            return

        if not self.config.corp_id or not self.config.secret:
            logger.error("WeCom self-built: corp_id and secret required")
            return

        import aiohttp.web as web

        self._running = True
        app = web.Application()
        app.router.add_get(self.config.callback_path, self._handle_verify)
        app.router.add_post(self.config.callback_path, self._handle_message)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(
            runner,
            host=self.config.callback_host,
            port=self.config.callback_port,
        )
        await site.start()
        self._runner = runner

        logger.info(
            "WeCom self-built channel started on http://{}:{}{}",
            self.config.callback_host,
            self.config.callback_port,
            self.config.callback_path,
        )

        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        self._running = False
        if self._runner:
            await self._runner.cleanup()

    # -- HTTP handlers --------------------------------------------------------

    async def _handle_verify(self, request: Any) -> Any:
        """GET: URL verification callback."""
        import aiohttp.web as web

        params = request.query
        signature = params.get("msg_signature", "")
        timestamp = params.get("timestamp", "")
        nonce = params.get("nonce", "")
        echostr = params.get("echostr", "")

        result = self._crypto.verify_url(timestamp, nonce, echostr, signature)
        if result is None:
            logger.warning("WeCom URL verification failed")
            return web.Response(text="fail")
        logger.info("WeCom URL verified OK")
        return web.Response(text=result)

    async def _handle_message(self, request: Any) -> Any:
        """POST: message callback."""
        import aiohttp.web as web

        try:
            body = await request.text()
        except Exception:
            return web.Response(text="")

        signature = request.query.get("msg_signature", "")
        timestamp = request.query.get("timestamp", "")
        nonce = request.query.get("nonce", "")

        root = ElementTree.fromstring(body)
        encrypted = root.findtext("Encrypt", "")
        if not encrypted:
            return web.Response(text="")

        decrypted = self._crypto.decrypt(timestamp, nonce, signature, encrypted)
        if decrypted is None:
            logger.warning("WeCom message decrypt failed")
            return web.Response(text="")

        msg_root = ElementTree.fromstring(decrypted)
        msg_type = msg_root.findtext("MsgType", "")
        content = msg_root.findtext("Content", "") or ""
        from_user = msg_root.findtext("FromUserName", "")
        msg_id = msg_root.findtext("MsgId", "")

        # Dedup
        if msg_id:
            if msg_id in self._processed_ids:
                return web.Response(text="")
            self._processed_ids[msg_id] = None
            if len(self._processed_ids) > 500:
                self._processed_ids.popitem(last=False)

        if msg_type == "event":
            event_type = msg_root.findtext("Event", "")
            logger.info("WeCom event: {} from {}", event_type, from_user)
            if event_type == "enter_agent" and self.config.welcome_message:
                await self.bus.publish_inbound(
                    _make_inbound("system", from_user, "system:ping", {})
                )
                await self.send(
                    OutboundMessage(
                        channel=self.name,
                        chat_id=from_user,
                        content=self.config.welcome_message,
                    )
                )
            return web.Response(text="")

        if msg_type != "text" or not content:
            return web.Response(text="")

        logger.info("WeCom msg from {}: {}", from_user, content[:80])

        from nanobot.bus.events import InboundMessage

        await self.bus.publish_inbound(
            InboundMessage(
                channel=self.name,
                sender_id=from_user,
                chat_id=from_user,
                content=content,
                metadata={"message_id": msg_id} if msg_id else {},
            )
        )
        return web.Response(text="")

    # -- send reply via WeCom API --------------------------------------------

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message via WeCom API."""
        if not msg.content or not msg.content.strip():
            return

        try:
            token = await self._get_access_token()
            if not token:
                return

            import httpx

            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token={token}",
                    json={
                        "touser": msg.chat_id,
                        "msgtype": "text",
                        "agentid": int(self.config.agent_id),
                        "text": {"content": msg.content},
                    },
                    timeout=30,
                )
                data = resp.json()
                if data.get("errcode") != 0:
                    logger.warning(
                        "WeCom send failed: {} — {}", data.get("errcode"), data.get("errmsg")
                    )
        except Exception:
            logger.exception("WeCom send error")

    async def _get_access_token(self) -> str | None:
        if self._access_token and time.time() < self._token_expires - 60:
            return self._access_token

        try:
            import httpx

            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "https://qyapi.weixin.qq.com/cgi-bin/gettoken",
                    params={
                        "corpid": self.config.corp_id,
                        "corpsecret": self.config.secret,
                    },
                    timeout=15,
                )
                data = resp.json()
                if data.get("errcode") != 0:
                    logger.error(
                        "WeCom gettoken failed: {} — {}", data.get("errcode"), data.get("errmsg")
                    )
                    return None
                self._access_token = data["access_token"]
                self._token_expires = time.time() + data.get("expires_in", 7200)
                logger.info("WeCom access_token obtained")
                return self._access_token
        except Exception:
            logger.exception("WeCom gettoken error")
            return None


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_inbound(channel: str, sender_id: str, chat_id: str, metadata: dict):
    """Build a lightweight inbound message without importing events."""
    from nanobot.bus.events import InboundMessage

    return InboundMessage(
        channel=channel,
        sender_id=sender_id,
        chat_id=chat_id,
        content="",  # placeholder — replaced in handler
        metadata=metadata,
    )
