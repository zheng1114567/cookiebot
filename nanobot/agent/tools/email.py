"""Email tools backed by the real IMAP/SMTP email channel config."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from typing import Any

from nanobot.agent.tools.base import Tool
from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.email import EmailChannel, EmailConfig


class EmailTool(Tool):
    """Search and send email using configured IMAP/SMTP credentials."""

    def __init__(self, config: Any):
        self.accounts: dict[str, EmailConfig] = {}
        if isinstance(config, dict):
            raw_accounts = config.get("accounts") or {}
            for name, account in raw_accounts.items():
                merged = {
                    "enabled": config.get("enabled", True),
                    "consentGranted": config.get(
                        "consentGranted",
                        config.get("consent_granted", False),
                    ),
                    **account,
                }
                parsed = EmailConfig.model_validate(merged)
                if parsed.enabled:
                    self.accounts[str(name)] = parsed
            config = EmailConfig.model_validate(config)
        self.config: EmailConfig = config

    @property
    def name(self) -> str:
        return "email"

    @property
    def description(self) -> str:
        accounts = ", ".join(sorted(self.accounts))
        account_note = (
            f" Configured accounts: {accounts}. You must pass account when more than one "
            "account is configured."
            if accounts else ""
        )
        return (
            "Search real email via IMAP and send email via SMTP. "
            "Use this for inbox monitoring, important-email checks, and email replies."
            f"{account_note}"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        schema = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["search", "unread", "send"],
                    "description": "Email action to perform",
                },
                "account": {
                    "type": "string",
                    "description": (
                        "Optional account name when multiple email accounts are configured"
                    ),
                },
                "query": {
                    "type": "string",
                    "description": "Keyword filter applied to sender, subject, and body",
                },
                "from_address": {
                    "type": "string",
                    "description": "Filter search results by sender email address",
                },
                "subject": {
                    "type": "string",
                    "description": "Subject filter for search, or subject for send",
                },
                "days": {
                    "type": "integer",
                    "description": "For search: look back this many days, default 1",
                    "minimum": 1,
                    "maximum": 30,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum messages to return, default 10",
                    "minimum": 1,
                    "maximum": 50,
                },
                "to": {
                    "type": "string",
                    "description": "Recipient email address for send",
                },
                "content": {
                    "type": "string",
                    "description": "Plain-text email body for send",
                },
            },
            "required": ["action"],
        }
        if self.accounts:
            schema["properties"]["account"]["enum"] = sorted(self.accounts)
            if len(self.accounts) > 1:
                schema["required"] = ["action", "account"]
        return schema

    async def execute(
        self,
        action: str,
        account: str = "",
        query: str = "",
        from_address: str = "",
        subject: str = "",
        days: int = 1,
        limit: int = 10,
        to: str = "",
        content: str = "",
        **kwargs: Any,
    ) -> str:
        config = self._resolve_config(account)
        if isinstance(config, str):
            return config

        if not config.consent_granted:
            return (
                "Error: email tool requires explicit consent. Set "
                "channels.email.consentGranted=true after user approval."
            )

        if action in {"search", "unread"}:
            return await self._search(config, action, query, from_address, subject, days, limit)
        if action == "send":
            return await self._send(config, to, subject, content)
        return f"Error: unknown email action '{action}'"

    def _resolve_config(self, account: str) -> EmailConfig | str:
        account = account.strip()
        if account:
            if account not in self.accounts:
                available = ", ".join(sorted(self.accounts)) or "(none)"
                return f"Error: unknown email account '{account}'. Available accounts: {available}"
            return self.accounts[account]
        if self.accounts:
            if len(self.accounts) == 1:
                return next(iter(self.accounts.values()))
            available = ", ".join(sorted(self.accounts))
            return f"Error: multiple email accounts configured. Specify account: {available}"
        return self.config

    @staticmethod
    def _channel(config: EmailConfig) -> EmailChannel:
        return EmailChannel(config, MessageBus())

    async def _search(
        self,
        config: EmailConfig,
        action: str,
        query: str,
        from_address: str,
        subject: str,
        days: int,
        limit: int,
    ) -> str:
        missing = []
        if not config.imap_host:
            missing.append("imap_host")
        if not config.imap_username:
            missing.append("imap_username")
        if not config.imap_password:
            missing.append("imap_password")
        if missing:
            return f"Error: email IMAP is not configured, missing: {', '.join(missing)}"

        channel = self._channel(config)
        if action == "unread":
            messages = await asyncio.to_thread(channel._fetch_new_messages)
        else:
            end = date.today() + timedelta(days=1)
            start = end - timedelta(days=max(1, min(days, 30)))
            messages = await asyncio.to_thread(
                channel.fetch_messages_between_dates,
                start,
                end,
                max(1, min(limit, 50)),
            )

        filtered = self._filter_messages(messages, query, from_address, subject)
        filtered = filtered[: max(1, min(limit, 50))]
        if not filtered:
            return "No matching emails found."

        lines = [f"Found {len(filtered)} email(s):"]
        for index, item in enumerate(filtered, 1):
            meta = item.get("metadata", {})
            body = str(item.get("content", "")).strip()
            body = body[:1000] + "\n... (truncated)" if len(body) > 1000 else body
            lines.extend(
                [
                    f"\n[{index}] From: {item.get('sender', '')}",
                    f"Subject: {item.get('subject', meta.get('subject', ''))}",
                    f"Date: {meta.get('date', '')}",
                    f"UID: {meta.get('uid', '')}",
                    body,
                ]
            )
        return "\n".join(lines)

    @staticmethod
    def _filter_messages(
        messages: list[dict[str, Any]],
        query: str,
        from_address: str,
        subject: str,
    ) -> list[dict[str, Any]]:
        query = query.lower().strip()
        from_address = from_address.lower().strip()
        subject = subject.lower().strip()
        result = []
        for item in messages:
            sender = str(item.get("sender", "")).lower()
            item_subject = str(item.get("subject", "")).lower()
            content = str(item.get("content", "")).lower()
            if from_address and from_address not in sender:
                continue
            if subject and subject not in item_subject:
                continue
            if query and query not in f"{sender}\n{item_subject}\n{content}":
                continue
            result.append(item)
        return result

    async def _send(self, config: EmailConfig, to: str, subject: str, content: str) -> str:
        missing = []
        if not config.smtp_host:
            missing.append("smtp_host")
        if not config.smtp_username:
            missing.append("smtp_username")
        if not config.smtp_password:
            missing.append("smtp_password")
        if missing:
            return f"Error: email SMTP is not configured, missing: {', '.join(missing)}"
        if not to:
            return "Error: to is required for send"
        if not content:
            return "Error: content is required for send"

        msg = OutboundMessage(
            channel="email",
            chat_id=to,
            content=content,
            metadata={"subject": subject or f"nanobot update {datetime.now().date()}"},
        )
        await self._channel(config).send(msg)
        return f"Email sent to {to}"
