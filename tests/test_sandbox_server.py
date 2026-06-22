import json
import sys

import pytest

from nanobot.agent.tools.sandbox_runner import ExecRunRequest
from nanobot.sandbox.client import SandboxClientRunner
from nanobot.sandbox.protocol import SandboxRequest
from nanobot.sandbox.server import handle_message, handle_once


@pytest.mark.asyncio
async def test_sandbox_server_handles_local_command(tmp_path) -> None:
    request = SandboxRequest(
        command=f'"{sys.executable}" -c "print(\'hello\')"',
        cwd=str(tmp_path),
        timeout=5,
        env={},
        workspace_root=str(tmp_path),
        profile="workspace_write",
        backend="local",
    )

    response = await handle_once(json.dumps(request.to_dict()))

    assert response.exit_code == 0
    assert "hello" in response.stdout
    assert response.profile == "workspace_write"


@pytest.mark.asyncio
async def test_sandbox_server_shutdown_message_returns_none() -> None:
    response = await handle_message('{"_control":"shutdown"}')

    assert response is None


@pytest.mark.asyncio
async def test_sandbox_client_reuses_persistent_server(tmp_path) -> None:
    runner = SandboxClientRunner(backend_name="local", python_executable=sys.executable)

    first = await runner.run(
        ExecRunRequest(
            command=f'"{sys.executable}" -c "print(\'one\')"',
            cwd=str(tmp_path),
            timeout=5,
            env={},
            workspace_root=str(tmp_path),
            profile="workspace_write",
        )
    )
    process_id = id(runner._process)
    second = await runner.run(
        ExecRunRequest(
            command=f'"{sys.executable}" -c "print(\'two\')"',
            cwd=str(tmp_path),
            timeout=5,
            env={},
            workspace_root=str(tmp_path),
            profile="workspace_write",
        )
    )

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert "one" in first.stdout
    assert "two" in second.stdout
    assert runner._process is not None
    assert id(runner._process) == process_id

    await runner.aclose()
    assert runner._process is None


@pytest.mark.asyncio
async def test_tool_registry_aclose_closes_registered_tools() -> None:
    from nanobot.agent.tools.registry import ToolRegistry

    class _ClosableTool:
        name = "closable"

        def to_schema(self):
            return {}

        async def aclose(self):
            self.closed = True

    tool = _ClosableTool()
    tool.closed = False
    registry = ToolRegistry()
    registry._tools["closable"] = tool  # focused lifecycle test

    await registry.aclose()

    assert tool.closed is True
