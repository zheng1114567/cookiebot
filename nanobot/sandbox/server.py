"""sandboxd: isolated command execution over JSON-over-stdio."""

from __future__ import annotations

import asyncio
import json
import sys

from nanobot.agent.tools.sandbox_runner import DockerSandboxRunner, ExecRunRequest, LocalSandboxRunner
from nanobot.sandbox.protocol import SandboxRequest, SandboxResponse

_PROFILE_NETWORK = {
    "read_only": False,
    "workspace_write": False,
    "workspace_write_no_net": False,
    "workspace_write_net": True,
    "full_access": True,
}


def _make_runner(request: SandboxRequest):
    if request.backend == "docker":
        return DockerSandboxRunner(network_enabled=_PROFILE_NETWORK.get(request.profile, False))
    return LocalSandboxRunner()


async def handle_once(raw: str) -> SandboxResponse:
    try:
        request = SandboxRequest.from_dict(json.loads(raw))
    except Exception as exc:
        return SandboxResponse(exit_code=1, error=f"invalid request: {exc}")

    runner = _make_runner(request)
    result = await runner.run(
        ExecRunRequest(
            command=request.command,
            cwd=request.cwd,
            timeout=request.timeout,
            env=request.env,
            workspace_root=request.workspace_root,
            profile=request.profile,
        )
    )
    return SandboxResponse(
        stdout=result.stdout,
        stderr=result.stderr,
        exit_code=result.exit_code,
        timed_out=result.timed_out,
        backend=result.backend,
        profile=request.profile,
    )


async def handle_message(raw: str) -> SandboxResponse | None:
    """Handle one line-delimited protocol message."""
    try:
        data = json.loads(raw)
    except Exception as exc:
        return SandboxResponse(exit_code=1, error=f"invalid request: {exc}")
    if isinstance(data, dict) and data.get("_control") == "shutdown":
        return None
    return await handle_once(raw)


async def _main() -> int:
    while True:
        raw = await asyncio.to_thread(sys.stdin.readline)
        if not raw:
            return 0
        raw = raw.strip()
        if not raw:
            continue
        response = await handle_message(raw)
        if response is None:
            return 0
        sys.stdout.write(json.dumps(response.to_dict(), ensure_ascii=False) + "\n")
        sys.stdout.flush()
    return 0


def main() -> int:
    return asyncio.run(_main())


if __name__ == "__main__":
    raise SystemExit(main())
