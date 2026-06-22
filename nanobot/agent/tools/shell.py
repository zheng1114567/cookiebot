"""Shell execution tool."""

import asyncio
import os
import re
from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.sandbox_runner import (
    DockerSandboxRunner,
    ExecRunRequest,
    ExecRunResult,
    LocalSandboxRunner,
    SandboxRunner,
)
from nanobot.sandbox.client import SandboxClientRunner
from nanobot.security.sandbox import BUILTIN_DENY_PATTERNS


class ExecTool(Tool):
    """Tool to execute shell commands."""

    def __init__(
        self,
        timeout: int = 60,
        working_dir: str | None = None,
        deny_patterns: list[str] | None = None,
        allow_patterns: list[str] | None = None,
        restrict_to_workspace: bool = False,
        path_append: str = "",
        sandbox_backend: str = "local",
        sandbox_profile: str = "workspace_write",
        docker_image: str = "python:3.11-slim",
        docker_network_enabled: bool = False,
        docker_memory_mb: int = 512,
        docker_pids_limit: int = 128,
        docker_user: str = "65534:65534",
        docker_cpus: float = 1.0,
        docker_seccomp_profile: str | None = None,
        workspace_root: str | None = None,
        runner: SandboxRunner | None = None,
    ):
        self.timeout = timeout
        self.working_dir = working_dir
        self.deny_patterns = deny_patterns or BUILTIN_DENY_PATTERNS
        self.allow_patterns = allow_patterns or []
        self.restrict_to_workspace = restrict_to_workspace
        self.path_append = path_append
        self.sandbox_backend = sandbox_backend
        self.sandbox_profile = sandbox_profile
        self.workspace_root = workspace_root
        self.runner = runner or self._make_runner(
            sandbox_backend=sandbox_backend,
            sandbox_profile=sandbox_profile,
            docker_image=docker_image,
            docker_network_enabled=docker_network_enabled,
            docker_memory_mb=docker_memory_mb,
            docker_pids_limit=docker_pids_limit,
            docker_user=docker_user,
            docker_cpus=docker_cpus,
            docker_seccomp_profile=docker_seccomp_profile,
        )

    @property
    def name(self) -> str:
        return "exec"

    _MAX_TIMEOUT = 600
    _MAX_OUTPUT = 10_000

    @property
    def description(self) -> str:
        return "Execute a shell command and return its output. Use with caution."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute",
                },
                "working_dir": {
                    "type": "string",
                    "description": "Optional working directory for the command",
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        "Timeout in seconds. Increase for long-running commands "
                        "like compilation or installation (default 60, max 600)."
                    ),
                    "minimum": 1,
                    "maximum": 600,
                },
            },
            "required": ["command"],
        }

    async def execute(
        self, command: str, working_dir: str | None = None,
        timeout: int | None = None, **kwargs: Any,
    ) -> str:
        cwd = working_dir or self.working_dir or os.getcwd()
        guard_error = self._guard_command(command, cwd)
        if guard_error:
            return guard_error

        effective_timeout = min(timeout or self.timeout, self._MAX_TIMEOUT)

        env = os.environ.copy()
        if self.path_append:
            env["PATH"] = env.get("PATH", "") + os.pathsep + self.path_append

        try:
            result = await self.runner.run(
                ExecRunRequest(
                    command=command,
                    cwd=cwd,
                    timeout=effective_timeout,
                    env=env,
                    workspace_root=self.workspace_root,
                    profile=self.sandbox_profile,
                )
            )
            return self._format_result(result, effective_timeout)
        except Exception as e:
            return f"Error executing command: {str(e)}"

    @classmethod
    def _make_runner(
        cls,
        *,
        sandbox_backend: str,
        sandbox_profile: str,
        docker_image: str,
        docker_network_enabled: bool,
        docker_memory_mb: int,
        docker_pids_limit: int,
        docker_user: str,
        docker_cpus: float,
        docker_seccomp_profile: str | None,
    ) -> SandboxRunner:
        if sandbox_backend == "sandboxd_local":
            return SandboxClientRunner(backend_name="local", profile=sandbox_profile)
        if sandbox_backend == "sandboxd_docker":
            return SandboxClientRunner(backend_name="docker", profile=sandbox_profile)
        if sandbox_backend == "docker":
            return DockerSandboxRunner(
                image=docker_image,
                network_enabled=docker_network_enabled,
                memory_mb=docker_memory_mb,
                pids_limit=docker_pids_limit,
                user=docker_user,
                cpus=docker_cpus,
                seccomp_profile=docker_seccomp_profile,
            )
        return LocalSandboxRunner()

    def _format_result(self, result: ExecRunResult, effective_timeout: int) -> str:
        if result.timed_out:
            return f"Error: Command timed out after {effective_timeout} seconds"

        output_parts = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr.strip():
            output_parts.append(f"STDERR:\n{result.stderr}")
        output_parts.append(f"\nExit code: {result.exit_code}")

        text = "\n".join(output_parts) if output_parts else "(no output)"
        max_len = self._MAX_OUTPUT
        if len(text) > max_len:
            half = max_len // 2
            text = (
                text[:half]
                + f"\n\n... ({len(text) - max_len:,} chars truncated) ...\n\n"
                + text[-half:]
            )
        return text

    async def aclose(self) -> None:
        """Close a persistent sandbox runner when one exists."""
        close = getattr(self.runner, "aclose", None)
        if callable(close):
            await close()

    def _guard_command(self, command: str, cwd: str) -> str | None:
        """Best-effort safety guard for potentially destructive commands."""
        cmd = command.strip()
        lower = cmd.lower()

        for pattern in self.deny_patterns:
            if re.search(pattern, lower):
                return "Error: Command blocked by safety guard (dangerous pattern detected)"

        if self.allow_patterns:
            if not any(re.search(p, lower) for p in self.allow_patterns):
                return "Error: Command blocked by safety guard (not in allowlist)"

        from nanobot.security.network import contains_internal_url
        if contains_internal_url(cmd):
            return "Error: Command blocked by safety guard (internal/private URL detected)"

        if self.restrict_to_workspace:
            if "..\\" in cmd or "../" in cmd:
                return "Error: Command blocked by safety guard (path traversal detected)"

            cwd_path = Path(cwd).resolve()

            for raw in self._extract_absolute_paths(cmd):
                try:
                    expanded = os.path.expandvars(raw.strip())
                    p = Path(expanded).expanduser().resolve()
                except Exception:
                    continue
                if p.is_absolute() and cwd_path not in p.parents and p != cwd_path:
                    return "Error: Command blocked by safety guard (path outside working dir)"

        return None

    @staticmethod
    def _extract_absolute_paths(command: str) -> list[str]:
        win_paths = re.findall(r"[A-Za-z]:\\[^\s\"'|><;]+", command)   # Windows: C:\...
        posix_paths = re.findall(r"(?:^|[\s|>'\"])(/[^\s\"'>;|<]+)", command) # POSIX: /absolute only
        home_paths = re.findall(r"(?:^|[\s|>'\"])(~[^\s\"'>;|<]*)", command) # POSIX/Windows home shortcut: ~
        return win_paths + posix_paths + home_paths
