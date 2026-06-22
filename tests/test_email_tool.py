import pytest

from nanobot.agent.tools.email import EmailTool


def _multi_config() -> dict:
    return {
        "enabled": True,
        "consentGranted": True,
        "accounts": {
            "a": {
                "imapHost": "imap.a.example",
                "imapUsername": "a@example.com",
                "imapPassword": "secret-a",
                "smtpHost": "smtp.a.example",
                "smtpUsername": "a@example.com",
                "smtpPassword": "secret-a",
            },
            "b": {
                "imapHost": "imap.b.example",
                "imapUsername": "b@example.com",
                "imapPassword": "secret-b",
                "smtpHost": "smtp.b.example",
                "smtpUsername": "b@example.com",
                "smtpPassword": "secret-b",
            },
        },
    }


@pytest.mark.asyncio
async def test_email_tool_requires_account_when_multiple_configured() -> None:
    tool = EmailTool(_multi_config())

    result = await tool.execute(action="search")

    assert "multiple email accounts configured" in result
    assert "a, b" in result


def test_email_tool_resolves_named_account() -> None:
    tool = EmailTool(_multi_config())

    resolved = tool._resolve_config("b")

    assert not isinstance(resolved, str)
    assert resolved.imap_host == "imap.b.example"


def test_email_tool_schema_requires_account_for_multiple_accounts() -> None:
    tool = EmailTool(_multi_config())

    schema = tool.parameters

    assert schema["properties"]["account"]["enum"] == ["a", "b"]
    assert "account" in schema["required"]


def test_email_tool_ignores_disabled_accounts() -> None:
    config = _multi_config()
    config["accounts"]["a"]["enabled"] = False
    tool = EmailTool(config)

    schema = tool.parameters

    assert sorted(tool.accounts) == ["b"]
    assert schema["properties"]["account"]["enum"] == ["b"]
    assert "account" not in schema["required"]


@pytest.mark.asyncio
async def test_email_tool_unknown_account_returns_error() -> None:
    tool = EmailTool(_multi_config())

    result = await tool.execute(action="search", account="missing")

    assert "unknown email account" in result
