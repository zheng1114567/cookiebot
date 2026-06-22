"""Unit tests for sandbox policy engine and approval manager."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from nanobot.config.schema import SandboxConfig
from nanobot.security.sandbox import (
    BUILTIN_DENY_PATTERNS,
    ApprovalManager,
    PolicyDecision,
    SandboxPolicy,
)


# ── SandboxPolicy ───────────────────────────────────────────────────────────


class TestSandboxPolicy:
    def test_off_mode_always_safe(self):
        config = SandboxConfig(mode="off")
        policy = SandboxPolicy(config)
        assert policy.classify_exec("rm -rf /") == PolicyDecision.SAFE
        assert policy.classify_exec("sudo shutdown now") == PolicyDecision.SAFE
        assert policy.classify_exec("ls -la") == PolicyDecision.SAFE

    def test_builtin_deny_blocks_dangerous_commands(self):
        config = SandboxConfig(mode="advisory", use_builtin_deny=True)
        policy = SandboxPolicy(config)
        assert policy.classify_exec("rm -rf /tmp/build") == PolicyDecision.DANGEROUS
        assert policy.classify_exec("shutdown now") == PolicyDecision.DANGEROUS
        assert policy.classify_exec("dd if=/dev/zero of=/dev/sda") == PolicyDecision.DANGEROUS
        assert policy.classify_exec(":(){ :|:& };:") == PolicyDecision.DANGEROUS

    def test_safe_commands_pass(self):
        config = SandboxConfig(mode="advisory", use_builtin_deny=True)
        policy = SandboxPolicy(config)
        assert policy.classify_exec("ls -la") == PolicyDecision.SAFE
        assert policy.classify_exec("echo hello") == PolicyDecision.SAFE
        assert policy.classify_exec("python -c 'print(1)'") == PolicyDecision.SAFE

    def test_custom_deny_patterns(self):
        config = SandboxConfig(
            mode="advisory",
            use_builtin_deny=False,
            deny_patterns=[r"\bsudo\b", r"\bchmod\s+777\b"],
        )
        policy = SandboxPolicy(config)
        # Custom patterns catch these
        assert policy.classify_exec("sudo apt install") == PolicyDecision.DANGEROUS
        assert policy.classify_exec("chmod 777 /etc/passwd") == PolicyDecision.DANGEROUS
        # Builtin deny is disabled, so this passes
        assert policy.classify_exec("rm -rf /tmp") == PolicyDecision.SAFE

    def test_approve_patterns_trigger_approval(self):
        config = SandboxConfig(
            mode="advisory",
            require_approval=True,
            approve_patterns=[r"\bpip\s+install\b", r"\bdocker\s+rm\b"],
        )
        policy = SandboxPolicy(config)
        assert policy.classify_exec("pip install numpy") == PolicyDecision.NEEDS_APPROVAL
        assert policy.classify_exec("docker rm container") == PolicyDecision.NEEDS_APPROVAL
        assert policy.classify_exec("pip install --upgrade pip") == PolicyDecision.NEEDS_APPROVAL
        assert policy.classify_exec("ls -la") == PolicyDecision.SAFE

    def test_strict_mode_requires_approval(self):
        config = SandboxConfig(mode="strict")
        policy = SandboxPolicy(config)
        assert policy.classify_exec("ls -la") == PolicyDecision.NEEDS_APPROVAL
        assert policy.classify_exec("echo hello") == PolicyDecision.NEEDS_APPROVAL
        # Deny patterns still take priority
        assert policy.classify_exec("rm -rf /") == PolicyDecision.DANGEROUS

    def test_deny_takes_priority_over_approve(self):
        config = SandboxConfig(
            mode="advisory",
            approve_patterns=[r"\brm\b"],  # would trigger approval
        )
        policy = SandboxPolicy(config)
        # rm -rf matches builtin deny BEFORE approve check
        assert policy.classify_exec("rm -rf /tmp") == PolicyDecision.DANGEROUS

    def test_classify_tool_dispatches_by_name(self):
        config = SandboxConfig(mode="off")
        policy = SandboxPolicy(config)
        # All non-exec tools are SAFE in off mode
        assert policy.classify_tool("read_file", {"path": "/etc/passwd"}) == PolicyDecision.SAFE
        assert policy.classify_tool("web_search", {"query": "test"}) == PolicyDecision.SAFE

    def test_classify_tool_strict_mode_file_writes(self):
        config = SandboxConfig(mode="strict")
        policy = SandboxPolicy(config)
        assert policy.classify_tool("write_file", {"path": "test.txt"}) == PolicyDecision.NEEDS_APPROVAL
        assert policy.classify_tool("edit_file", {"path": "test.txt"}) == PolicyDecision.NEEDS_APPROVAL
        assert policy.classify_tool("read_file", {"path": "test.txt"}) == PolicyDecision.SAFE

    def test_invalid_regex_pattern_does_not_crash(self):
        config = SandboxConfig(
            mode="advisory",
            deny_patterns=[r"[invalid"],
            approve_patterns=[r"[also_invalid"],
        )
        policy = SandboxPolicy(config)
        # Invalid regexes are silently skipped
        assert policy.classify_exec("ls -la") == PolicyDecision.SAFE


# ── ApprovalManager ─────────────────────────────────────────────────────────


class TestApprovalManager:
    def _make_manager(self, tmp_path: Path, **kw):
        config = SandboxConfig(mode="advisory", require_approval=True, **kw)
        policy = SandboxPolicy(config)
        bus = AsyncMock()
        bus.publish_outbound = AsyncMock()
        audit = tmp_path / "audit.jsonl"
        mgr = ApprovalManager(policy=policy, audit_log_path=audit, bus=bus)
        mgr.bind(session_key="test:1", channel="cli", chat_id="direct")
        return mgr, bus, audit

    @pytest.mark.asyncio
    async def test_safe_command_runs_immediately(self, tmp_path: Path):
        mgr, bus, audit = self._make_manager(tmp_path)
        result = await mgr.check_and_request("exec", {"command": "ls -la"})
        assert result is None  # OK to proceed
        bus.publish_outbound.assert_not_called()
        assert audit.exists()
        entries = [json.loads(l) for l in audit.read_text().splitlines() if l]
        assert entries[0]["decision"] == "auto_ran"

    @pytest.mark.asyncio
    async def test_dangerous_command_blocked(self, tmp_path: Path):
        mgr, bus, audit = self._make_manager(tmp_path)
        result = await mgr.check_and_request("exec", {"command": "rm -rf /"})
        assert result is not None
        assert "blocked" in result.lower()
        bus.publish_outbound.assert_not_called()
        entries = [json.loads(l) for l in audit.read_text().splitlines() if l]
        assert entries[0]["decision"] == "blocked"

    @pytest.mark.asyncio
    async def test_approval_request_sent_and_accepted(self, tmp_path: Path):
        mgr, bus, audit = self._make_manager(
            tmp_path, approve_patterns=[r"\bpip\s+install\b"]
        )

        # Simulate user responding "yes" in background after 50ms
        async def _approve_later():
            await asyncio.sleep(0.05)
            mgr.respond("test:1", approved=True)

        asyncio.create_task(_approve_later())

        result = await mgr.check_and_request("exec", {"command": "pip install numpy"})

        assert result is None  # approved → OK to proceed
        bus.publish_outbound.assert_called_once()
        entries = [json.loads(l) for l in audit.read_text().splitlines() if l]
        assert entries[0]["decision"] == "approved"
        assert entries[0]["user"] == "yes"

    @pytest.mark.asyncio
    async def test_approval_request_rejected(self, tmp_path: Path):
        mgr, bus, audit = self._make_manager(
            tmp_path, approve_patterns=[r"\bpip\s+install\b"]
        )

        async def _deny_later():
            await asyncio.sleep(0.05)
            mgr.respond("test:1", approved=False)

        asyncio.create_task(_deny_later())

        result = await mgr.check_and_request("exec", {"command": "pip install numpy"})

        assert result is not None
        assert "rejected" in result.lower()
        entries = [json.loads(l) for l in audit.read_text().splitlines() if l]
        assert entries[0]["decision"] == "denied"
        assert entries[0]["user"] == "no"

    @pytest.mark.asyncio
    async def test_approval_timeout(self, tmp_path: Path):
        mgr, bus, audit = self._make_manager(
            tmp_path,
            approve_patterns=[r"\bpip\s+install\b"],
            approval_timeout_s=1,  # fast timeout for test
        )

        result = await mgr.check_and_request("exec", {"command": "pip install numpy"})

        assert result is not None
        assert "timed out" in result.lower()
        assert not mgr.has_pending("test:1")

    def test_is_approval_response(self):
        for yes in ("yes", "y", "YES", "Y", "approve", "ok", "true", "1"):
            assert ApprovalManager.is_approval_response(yes)
        for no in ("no", "n", "NO", "deny", "reject", "false", "0", "cancel"):
            assert ApprovalManager.is_approval_response(no)
        assert not ApprovalManager.is_approval_response("hello")
        assert not ApprovalManager.is_approval_response("maybe")
        assert not ApprovalManager.is_approval_response("")

    def test_has_pending(self, tmp_path: Path):
        mgr, _, _ = self._make_manager(tmp_path)
        assert not mgr.has_pending("test:1")
        # Simulate pending
        mgr._pending_events["test:1"] = asyncio.Event()
        assert mgr.has_pending("test:1")
        assert not mgr.has_pending("test:2")

    def test_audit_writes_summary(self, tmp_path: Path):
        mgr, _, audit = self._make_manager(tmp_path)
        mgr._write_audit("exec", {"command": "ls -la /tmp"}, "auto_ran")
        entries = [json.loads(l) for l in audit.read_text().splitlines() if l]
        assert len(entries) == 1
        assert entries[0]["tool"] == "exec"
        assert entries[0]["summary"] == "exec: ls -la /tmp"
        assert entries[0]["session"] == "test:1"

    def test_bind_changes_context(self, tmp_path: Path):
        mgr, _, _ = self._make_manager(tmp_path)
        mgr.bind(session_key="other:2", channel="telegram", chat_id="456")
        assert mgr._session_key == "other:2"
        assert mgr._channel == "telegram"
        assert mgr._chat_id == "456"

    def test_effective_deny_patterns_includes_builtin(self):
        config = SandboxConfig(mode="advisory", use_builtin_deny=True)
        policy = SandboxPolicy(config)
        assert len(policy.effective_deny_patterns) == len(BUILTIN_DENY_PATTERNS)

        config2 = SandboxConfig(mode="advisory", use_builtin_deny=False)
        policy2 = SandboxPolicy(config2)
        assert len(policy2.effective_deny_patterns) == 0
