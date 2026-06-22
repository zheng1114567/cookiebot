import asyncio
from typing import Any

from nanobot.agent.middleware import OutboundDefaultsMiddleware
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.sandbox_runner import DockerSandboxRunner, ExecRunResult
from nanobot.agent.tools.shell import ExecTool
from nanobot.sandbox.client import SandboxClientRunner
from nanobot.bus.events import OutboundMessage


class SampleTool(Tool):
    @property
    def name(self) -> str:
        return "sample"

    @property
    def description(self) -> str:
        return "sample tool"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 2},
                "count": {"type": "integer", "minimum": 1, "maximum": 10},
                "mode": {"type": "string", "enum": ["fast", "full"]},
                "meta": {
                    "type": "object",
                    "properties": {
                        "tag": {"type": "string"},
                        "flags": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["tag"],
                },
            },
            "required": ["query", "count"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return "ok"


class SlowTool(Tool):
    @property
    def name(self) -> str:
        return "slow"

    @property
    def description(self) -> str:
        return "slow tool"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        await asyncio.sleep(10)
        return "done"


class MiddlewareTool(Tool):
    @property
    def name(self) -> str:
        return "middleware_tool"

    @property
    def description(self) -> str:
        return "tool for middleware tests"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

    async def execute(self, **kwargs: Any) -> str:
        return kwargs["text"]


class PrefixMiddleware:
    async def before_execute(self, tool_name: str, params: dict[str, Any]) -> dict[str, Any] | str | None:
        updated = dict(params)
        updated["text"] = "pre-" + updated["text"]
        return updated

    async def after_execute(self, tool_name: str, params: dict[str, Any], result: str) -> str:
        return result + "-post"


def test_validate_params_missing_required() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "hi"})
    assert "missing required count" in "; ".join(errors)


def test_validate_params_type_and_range() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "hi", "count": 0})
    assert any("count must be >= 1" in e for e in errors)

    errors = tool.validate_params({"query": "hi", "count": "2"})
    assert any("count should be integer" in e for e in errors)


def test_validate_params_enum_and_min_length() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "h", "count": 2, "mode": "slow"})
    assert any("query must be at least 2 chars" in e for e in errors)
    assert any("mode must be one of" in e for e in errors)


def test_validate_params_nested_object_and_array() -> None:
    tool = SampleTool()
    errors = tool.validate_params(
        {
            "query": "hi",
            "count": 2,
            "meta": {"flags": [1, "ok"]},
        }
    )
    assert any("missing required meta.tag" in e for e in errors)
    assert any("meta.flags[0] should be string" in e for e in errors)


def test_validate_params_ignores_unknown_fields() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "hi", "count": 2, "extra": "x"})
    assert errors == []


async def test_registry_returns_validation_error() -> None:
    reg = ToolRegistry()
    reg.register(SampleTool())
    result = await reg.execute("sample", {"query": "hi"})
    assert "Invalid parameters" in result


async def test_registry_times_out_hung_tool() -> None:
    reg = ToolRegistry(default_timeout=0.01)
    reg.register(SlowTool())

    result = await reg.execute("slow", {})

    assert "timed out after 0.01 seconds" in result
    assert "try a different approach" in result


async def test_registry_runs_tool_middlewares() -> None:
    reg = ToolRegistry()
    reg.register(MiddlewareTool())
    reg.register_middleware(PrefixMiddleware())

    result = await reg.execute("middleware_tool", {"text": "value"})

    assert result == "pre-value-post"


async def test_outbound_defaults_middleware_normalizes_metadata() -> None:
    middleware = OutboundDefaultsMiddleware()

    msg = await middleware.before_send(OutboundMessage(channel="cli", chat_id="direct", content="ok", metadata=None))

    assert msg is not None
    assert msg.metadata == {}


def test_exec_extract_absolute_paths_keeps_full_windows_path() -> None:
    cmd = r"type C:\user\workspace\txt"
    paths = ExecTool._extract_absolute_paths(cmd)
    assert paths == [r"C:\user\workspace\txt"]


def test_exec_extract_absolute_paths_ignores_relative_posix_segments() -> None:
    cmd = ".venv/bin/python script.py"
    paths = ExecTool._extract_absolute_paths(cmd)
    assert "/bin/python" not in paths


def test_exec_extract_absolute_paths_captures_posix_absolute_paths() -> None:
    cmd = "cat /tmp/data.txt > /tmp/out.txt"
    paths = ExecTool._extract_absolute_paths(cmd)
    assert "/tmp/data.txt" in paths
    assert "/tmp/out.txt" in paths


def test_exec_extract_absolute_paths_captures_home_paths() -> None:
    cmd = "cat ~/.nanobot/config.json > ~/out.txt"
    paths = ExecTool._extract_absolute_paths(cmd)
    assert "~/.nanobot/config.json" in paths
    assert "~/out.txt" in paths


def test_exec_extract_absolute_paths_captures_quoted_paths() -> None:
    cmd = 'cat "/tmp/data.txt" "~/.nanobot/config.json"'
    paths = ExecTool._extract_absolute_paths(cmd)
    assert "/tmp/data.txt" in paths
    assert "~/.nanobot/config.json" in paths


def test_exec_guard_blocks_home_path_outside_workspace(tmp_path) -> None:
    tool = ExecTool(restrict_to_workspace=True)
    error = tool._guard_command("cat ~/.nanobot/config.json", str(tmp_path))
    assert error == "Error: Command blocked by safety guard (path outside working dir)"


def test_exec_guard_blocks_quoted_home_path_outside_workspace(tmp_path) -> None:
    tool = ExecTool(restrict_to_workspace=True)
    error = tool._guard_command('cat "~/.nanobot/config.json"', str(tmp_path))
    assert error == "Error: Command blocked by safety guard (path outside working dir)"


# --- cast_params tests ---


class CastTestTool(Tool):
    """Minimal tool for testing cast_params."""

    def __init__(self, schema: dict[str, Any]) -> None:
        self._schema = schema

    @property
    def name(self) -> str:
        return "cast_test"

    @property
    def description(self) -> str:
        return "test tool for casting"

    @property
    def parameters(self) -> dict[str, Any]:
        return self._schema

    async def execute(self, **kwargs: Any) -> str:
        return "ok"


def test_cast_params_string_to_int() -> None:
    tool = CastTestTool(
        {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
        }
    )
    result = tool.cast_params({"count": "42"})
    assert result["count"] == 42
    assert isinstance(result["count"], int)


def test_cast_params_string_to_number() -> None:
    tool = CastTestTool(
        {
            "type": "object",
            "properties": {"rate": {"type": "number"}},
        }
    )
    result = tool.cast_params({"rate": "3.14"})
    assert result["rate"] == 3.14
    assert isinstance(result["rate"], float)


def test_cast_params_string_to_bool() -> None:
    tool = CastTestTool(
        {
            "type": "object",
            "properties": {"enabled": {"type": "boolean"}},
        }
    )
    assert tool.cast_params({"enabled": "true"})["enabled"] is True
    assert tool.cast_params({"enabled": "false"})["enabled"] is False
    assert tool.cast_params({"enabled": "1"})["enabled"] is True


def test_cast_params_array_items() -> None:
    tool = CastTestTool(
        {
            "type": "object",
            "properties": {
                "nums": {"type": "array", "items": {"type": "integer"}},
            },
        }
    )
    result = tool.cast_params({"nums": ["1", "2", "3"]})
    assert result["nums"] == [1, 2, 3]


def test_cast_params_nested_object() -> None:
    tool = CastTestTool(
        {
            "type": "object",
            "properties": {
                "config": {
                    "type": "object",
                    "properties": {
                        "port": {"type": "integer"},
                        "debug": {"type": "boolean"},
                    },
                },
            },
        }
    )
    result = tool.cast_params({"config": {"port": "8080", "debug": "true"}})
    assert result["config"]["port"] == 8080
    assert result["config"]["debug"] is True


def test_cast_params_bool_not_cast_to_int() -> None:
    """Booleans should not be silently cast to integers."""
    tool = CastTestTool(
        {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
        }
    )
    result = tool.cast_params({"count": True})
    assert result["count"] is True
    errors = tool.validate_params(result)
    assert any("count should be integer" in e for e in errors)


def test_cast_params_preserves_empty_string() -> None:
    """Empty strings should be preserved for string type."""
    tool = CastTestTool(
        {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
    )
    result = tool.cast_params({"name": ""})
    assert result["name"] == ""


def test_cast_params_bool_string_false() -> None:
    """Test that 'false', '0', 'no' strings convert to False."""
    tool = CastTestTool(
        {
            "type": "object",
            "properties": {"flag": {"type": "boolean"}},
        }
    )
    assert tool.cast_params({"flag": "false"})["flag"] is False
    assert tool.cast_params({"flag": "False"})["flag"] is False
    assert tool.cast_params({"flag": "0"})["flag"] is False
    assert tool.cast_params({"flag": "no"})["flag"] is False
    assert tool.cast_params({"flag": "NO"})["flag"] is False


def test_cast_params_bool_string_invalid() -> None:
    """Invalid boolean strings should not be cast."""
    tool = CastTestTool(
        {
            "type": "object",
            "properties": {"flag": {"type": "boolean"}},
        }
    )
    # Invalid strings should be preserved (validation will catch them)
    result = tool.cast_params({"flag": "random"})
    assert result["flag"] == "random"
    result = tool.cast_params({"flag": "maybe"})
    assert result["flag"] == "maybe"


def test_cast_params_invalid_string_to_int() -> None:
    """Invalid strings should not be cast to integer."""
    tool = CastTestTool(
        {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
        }
    )
    result = tool.cast_params({"count": "abc"})
    assert result["count"] == "abc"  # Original value preserved
    result = tool.cast_params({"count": "12.5.7"})
    assert result["count"] == "12.5.7"


def test_cast_params_invalid_string_to_number() -> None:
    """Invalid strings should not be cast to number."""
    tool = CastTestTool(
        {
            "type": "object",
            "properties": {"rate": {"type": "number"}},
        }
    )
    result = tool.cast_params({"rate": "not_a_number"})
    assert result["rate"] == "not_a_number"


def test_validate_params_bool_not_accepted_as_number() -> None:
    """Booleans should not pass number validation."""
    tool = CastTestTool(
        {
            "type": "object",
            "properties": {"rate": {"type": "number"}},
        }
    )
    errors = tool.validate_params({"rate": False})
    assert any("rate should be number" in e for e in errors)


def test_cast_params_none_values() -> None:
    """Test None handling for different types."""
    tool = CastTestTool(
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer"},
                "items": {"type": "array"},
                "config": {"type": "object"},
            },
        }
    )
    result = tool.cast_params(
        {
            "name": None,
            "count": None,
            "items": None,
            "config": None,
        }
    )
    # None should be preserved for all types
    assert result["name"] is None
    assert result["count"] is None
    assert result["items"] is None
    assert result["config"] is None


def test_cast_params_single_value_not_auto_wrapped_to_array() -> None:
    """Single values should NOT be automatically wrapped into arrays."""
    tool = CastTestTool(
        {
            "type": "object",
            "properties": {"items": {"type": "array"}},
        }
    )
    # Non-array values should be preserved (validation will catch them)
    result = tool.cast_params({"items": 5})
    assert result["items"] == 5  # Not wrapped to [5]
    result = tool.cast_params({"items": "text"})
    assert result["items"] == "text"  # Not wrapped to ["text"]


# --- ExecTool enhancement tests ---


async def test_exec_always_returns_exit_code() -> None:
    """Exit code should appear in output even on success (exit 0)."""
    tool = ExecTool()
    result = await tool.execute(command="echo hello")
    assert "Exit code: 0" in result
    assert "hello" in result


async def test_exec_head_tail_truncation() -> None:
    """Long output should preserve both head and tail."""
    tool = ExecTool()
    # Generate output that exceeds _MAX_OUTPUT (10_000 chars)
    # Use python to generate output to avoid command line length limits
    result = await tool.execute(
        command="python -c \"print('A' * 6000 + '\\n' + 'B' * 6000)\""
    )
    assert "chars truncated" in result
    # Head portion should start with As
    assert result.startswith("A")
    # Tail portion should end with the exit code which comes after Bs
    assert "Exit code:" in result


async def test_exec_timeout_parameter() -> None:
    """LLM-supplied timeout should override the constructor default."""
    tool = ExecTool(timeout=60)
    # A very short timeout should cause the command to be killed
    result = await tool.execute(command="python -c \"import time; time.sleep(10)\"", timeout=1)
    assert "timed out" in result
    assert "1 seconds" in result


async def test_exec_timeout_capped_at_max() -> None:
    """Timeout values above _MAX_TIMEOUT should be clamped."""
    tool = ExecTool()
    # Should not raise — just clamp to 600
    result = await tool.execute(command="echo ok", timeout=9999)
    assert "Exit code: 0" in result


class FakeRunner:
    def __init__(self, result: ExecRunResult) -> None:
        self.result = result
        self.requests = []

    async def run(self, request):
        self.requests.append(request)
        return self.result

    async def aclose(self):
        self.closed = True


async def test_exec_uses_custom_runner_and_formats_result(tmp_path) -> None:
    runner = FakeRunner(ExecRunResult(stdout="ok", stderr="", exit_code=0, backend="fake"))
    tool = ExecTool(
        runner=runner,
        working_dir=str(tmp_path),
        workspace_root=str(tmp_path),
    )

    result = await tool.execute(command="echo ok")

    assert "ok" in result
    assert "Exit code: 0" in result
    assert runner.requests[0].workspace_root == str(tmp_path)


async def test_exec_aclose_closes_runner(tmp_path) -> None:
    runner = FakeRunner(ExecRunResult(stdout="ok", stderr="", exit_code=0, backend="fake"))
    runner.closed = False
    tool = ExecTool(
        runner=runner,
        working_dir=str(tmp_path),
        workspace_root=str(tmp_path),
    )

    await tool.aclose()

    assert runner.closed is True


def test_exec_chooses_docker_runner() -> None:
    tool = ExecTool(sandbox_backend="docker")

    assert isinstance(tool.runner, DockerSandboxRunner)


def test_exec_chooses_sandboxd_runner() -> None:
    tool = ExecTool(sandbox_backend="sandboxd_local", sandbox_profile="workspace_write_no_net")

    assert isinstance(tool.runner, SandboxClientRunner)
    assert tool.runner.backend_name == "local"
    assert tool.runner.profile == "workspace_write_no_net"


def test_docker_runner_builds_isolated_command(tmp_path) -> None:
    workspace = tmp_path / "ws"
    nested = workspace / "sub"
    nested.mkdir(parents=True)
    runner = DockerSandboxRunner(
        image="python:3.11-slim",
        network_enabled=False,
        memory_mb=256,
        pids_limit=64,
        user="1000:1000",
        cpus=0.5,
        seccomp_profile="/tmp/seccomp.json",
    )

    argv = runner.build_argv(
        request=type("Req", (), {
            "command": "echo hi",
            "cwd": str(nested),
            "timeout": 5,
            "env": {"PATH": "/usr/bin"},
            "workspace_root": str(workspace),
            "profile": "workspace_write_no_net",
        })()
    )

    assert argv[:3] == ["docker", "run", "--rm"]
    assert "--read-only" in argv
    assert "--cap-drop" in argv
    assert "ALL" in argv
    assert "--security-opt" in argv
    assert "no-new-privileges" in argv
    assert "--memory" in argv
    assert "256m" in argv
    assert "--cpus" in argv
    assert "0.5" in argv
    assert "--pids-limit" in argv
    assert "64" in argv
    assert "--user" in argv
    assert "1000:1000" in argv
    assert "--tmpfs" in argv
    assert "seccomp=/tmp/seccomp.json" in argv
    assert "--network" in argv
    assert "none" in argv
    assert str(workspace) + ":/workspace" in argv
    assert "/workspace/sub" in argv


def test_docker_runner_mounts_read_only_workspace_for_read_only_profile(tmp_path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True)
    runner = DockerSandboxRunner()

    argv = runner.build_argv(
        request=type("Req", (), {
            "command": "ls",
            "cwd": str(workspace),
            "timeout": 5,
            "env": {},
            "workspace_root": str(workspace),
            "profile": "read_only",
        })()
    )

    assert str(workspace) + ":/workspace:ro" in argv


def test_sandbox_client_argv_uses_module() -> None:
    runner = SandboxClientRunner(python_executable="python")

    assert runner.build_argv() == ["python", "-m", "nanobot.sandbox.server"]
