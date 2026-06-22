"""JSON protocol types for sandbox command execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class SandboxRequest:
    """One command execution request for sandboxd."""

    command: str
    cwd: str
    timeout: int
    env: dict[str, str]
    workspace_root: str | None = None
    profile: str = "workspace_write"
    backend: str = "local"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SandboxRequest":
        return cls(
            command=str(data.get("command", "")),
            cwd=str(data.get("cwd", "")),
            timeout=int(data.get("timeout", 60)),
            env=dict(data.get("env", {})),
            workspace_root=data.get("workspace_root"),
            profile=str(data.get("profile", "workspace_write")),
            backend=str(data.get("backend", "local")),
        )


@dataclass
class SandboxResponse:
    """Normalized sandboxd response."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    timed_out: bool = False
    backend: str = "local"
    profile: str = "workspace_write"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SandboxResponse":
        return cls(
            stdout=str(data.get("stdout", "")),
            stderr=str(data.get("stderr", "")),
            exit_code=int(data.get("exit_code", 0)),
            timed_out=bool(data.get("timed_out", False)),
            backend=str(data.get("backend", "local")),
            profile=str(data.get("profile", "workspace_write")),
            error=data.get("error"),
        )
