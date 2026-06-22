"""Execution backends for shell commands."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class ExecRunRequest:
    """Normalized command execution request."""

    command: str
    cwd: str
    timeout: int
    env: dict[str, str]
    workspace_root: str | None = None
    profile: str = "workspace_write"


@dataclass
class ExecRunResult:
    """Command execution result independent of the concrete backend."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    timed_out: bool = False
    backend: str = "local"


class SandboxRunner(Protocol):
    """Backend contract for isolated command execution."""

    async def run(self, request: ExecRunRequest) -> ExecRunResult:
        """Execute a request and return a normalized result."""


class LocalSandboxRunner:
    """Direct subprocess execution on the host machine."""

    backend = "local"

    async def run(self, request: ExecRunRequest) -> ExecRunResult:
        try:
            process = await asyncio.create_subprocess_shell(
                request.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=request.cwd,
                env=request.env,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=request.timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    pass
                return ExecRunResult(
                    timed_out=True,
                    backend=self.backend,
                )

            return ExecRunResult(
                stdout=stdout.decode("utf-8", errors="replace") if stdout else "",
                stderr=stderr.decode("utf-8", errors="replace") if stderr else "",
                exit_code=process.returncode,
                backend=self.backend,
            )
        except Exception as exc:
            return ExecRunResult(stderr=str(exc), exit_code=1, backend=self.backend)


class DockerSandboxRunner:
    """Run commands inside a short-lived Docker container."""

    backend = "docker"

    def __init__(
        self,
        *,
        image: str = "python:3.11-slim",
        network_enabled: bool = False,
        shell: str = "/bin/sh",
        memory_mb: int = 512,
        pids_limit: int = 128,
        user: str = "65534:65534",
        read_only_root: bool = True,
        cpus: float = 1.0,
        seccomp_profile: str | None = None,
    ) -> None:
        self.image = image
        self.network_enabled = network_enabled
        self.shell = shell
        self.memory_mb = memory_mb
        self.pids_limit = pids_limit
        self.user = user
        self.read_only_root = read_only_root
        self.cpus = cpus
        self.seccomp_profile = seccomp_profile

    async def run(self, request: ExecRunRequest) -> ExecRunResult:
        argv = self.build_argv(request)
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=request.timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    pass
                return ExecRunResult(
                    timed_out=True,
                    backend=self.backend,
                )

            return ExecRunResult(
                stdout=stdout.decode("utf-8", errors="replace") if stdout else "",
                stderr=stderr.decode("utf-8", errors="replace") if stderr else "",
                exit_code=process.returncode,
                backend=self.backend,
            )
        except FileNotFoundError:
            return ExecRunResult(
                stderr="docker executable not found",
                exit_code=1,
                backend=self.backend,
            )
        except Exception as exc:
            return ExecRunResult(stderr=str(exc), exit_code=1, backend=self.backend)

    def build_argv(self, request: ExecRunRequest) -> list[str]:
        mount_src, workdir, workspace_read_only = self._resolve_mount(request)
        argv = [
            "docker",
            "run",
            "--rm",
            "--workdir",
            workdir,
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(self.pids_limit),
            "--memory",
            f"{self.memory_mb}m",
            "--cpus",
            str(self.cpus),
            "--user",
            self.user,
            "-v",
            f"{mount_src}:/workspace" + (":ro" if workspace_read_only else ""),
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev,size=64m",
            "--tmpfs",
            "/run:rw,noexec,nosuid,nodev,size=16m",
        ]
        if self.read_only_root:
            argv.append("--read-only")
        if self.seccomp_profile:
            argv.extend(["--security-opt", f"seccomp={self.seccomp_profile}"])
        if not self.network_enabled:
            argv.extend(["--network", "none"])
        path_value = request.env.get("PATH")
        if path_value:
            argv.extend(["-e", f"PATH={path_value}"])
        argv.extend([self.image, self.shell, "-lc", request.command])
        return argv

    @staticmethod
    def _resolve_mount(request: ExecRunRequest) -> tuple[str, str, bool]:
        cwd = Path(request.cwd).resolve()
        root = Path(request.workspace_root).resolve() if request.workspace_root else cwd
        workspace_read_only = request.profile == "read_only"
        try:
            relative = cwd.relative_to(root)
            workdir = "/workspace"
            if relative.parts:
                workdir += "/" + "/".join(relative.parts)
            return str(root), workdir, workspace_read_only
        except ValueError:
            return str(cwd), "/workspace", workspace_read_only
