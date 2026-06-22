"""Client-side sandbox runner that delegates execution to sandboxd."""

from __future__ import annotations

import asyncio
import atexit
import json
import sys

from nanobot.agent.tools.sandbox_runner import ExecRunRequest, ExecRunResult
from nanobot.sandbox.protocol import SandboxRequest, SandboxResponse


class SandboxClientRunner:
    """Spawn sandboxd and execute one request over stdio."""

    backend = "sandboxd"

    def __init__(
        self,
        *,
        backend_name: str = "local",
        profile: str = "workspace_write",
        python_executable: str | None = None,
    ) -> None:
        self.backend_name = backend_name
        self.profile = profile
        self.python_executable = python_executable or sys.executable
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._atexit_registered = False

    def build_argv(self) -> list[str]:
        return [self.python_executable, "-m", "nanobot.sandbox.server"]

    async def _ensure_process(self) -> asyncio.subprocess.Process:
        process = self._process
        if process and process.returncode is None:
            return process
        self._process = await asyncio.create_subprocess_exec(
            *self.build_argv(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        if not self._atexit_registered:
            atexit.register(self._atexit_cleanup)
            self._atexit_registered = True
        return self._process

    def _atexit_cleanup(self) -> None:
        process = self._process
        if process and process.returncode is None:
            process.kill()

    async def run(self, request: ExecRunRequest) -> ExecRunResult:
        payload = json.dumps(
            SandboxRequest(
                command=request.command,
                cwd=request.cwd,
                timeout=request.timeout,
                env=request.env,
                workspace_root=request.workspace_root,
                profile=request.profile or self.profile,
                backend=self.backend_name,
            ).to_dict(),
            ensure_ascii=False,
        ) + "\n"
        async with self._lock:
            try:
                process = await self._ensure_process()
                assert process.stdin is not None
                assert process.stdout is not None
                process.stdin.write(payload.encode("utf-8"))
                await process.stdin.drain()
                stdout = await process.stdout.readline()
            except Exception as exc:
                return ExecRunResult(stderr=str(exc), exit_code=1, backend=self.backend)

            if not stdout:
                stderr_text = ""
                if process.stderr is not None:
                    try:
                        stderr_text = await asyncio.wait_for(process.stderr.read(), timeout=0.2)
                    except Exception:
                        stderr_text = b""
                return ExecRunResult(
                    stderr=stderr_text.decode("utf-8", errors="replace") if stderr_text else "sandboxd closed unexpectedly",
                    exit_code=1,
                    backend=self.backend,
                )

            try:
                response = SandboxResponse.from_dict(json.loads(stdout.decode("utf-8", errors="replace")))
            except Exception as exc:
                return ExecRunResult(
                    stderr=f"invalid sandbox response: {exc}",
                    exit_code=1,
                    backend=self.backend,
                )
            if response.error:
                return ExecRunResult(stderr=response.error, exit_code=1, backend=self.backend)
            return ExecRunResult(
                stdout=response.stdout,
                stderr=response.stderr,
                exit_code=response.exit_code,
                timed_out=response.timed_out,
                backend=f"{self.backend}:{response.backend}",
            )

    async def aclose(self) -> None:
        """Shut down the persistent sandboxd process."""
        async with self._lock:
            process = self._process
            if not process or process.returncode is not None:
                self._process = None
                return
            try:
                if process.stdin is not None:
                    process.stdin.write(b'{"_control":"shutdown"}\n')
                    await process.stdin.drain()
                    process.stdin.close()
                await asyncio.wait_for(process.wait(), timeout=1.0)
            except Exception:
                process.kill()
                try:
                    await asyncio.wait_for(process.wait(), timeout=1.0)
                except Exception:
                    pass
            finally:
                self._process = None
