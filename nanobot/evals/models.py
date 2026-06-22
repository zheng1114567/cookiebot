"""Data models for agent evals."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class EvalScenario:
    """A scenario seed plus expectations used to create eval cases."""

    id: str
    suite: str
    task_type: str
    user_goal: str
    variants: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    expected: dict[str, Any] = field(default_factory=dict)
    metrics: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalScenario":
        required = ["id", "suite", "task_type", "user_goal"]
        missing = [key for key in required if not data.get(key)]
        if missing:
            raise ValueError(f"Scenario missing required fields: {', '.join(missing)}")
        return cls(
            id=str(data["id"]),
            suite=str(data["suite"]),
            task_type=str(data["task_type"]),
            user_goal=str(data["user_goal"]),
            variants=dict(data.get("variants") or {}),
            environment=dict(data.get("environment") or {}),
            expected=dict(data.get("expected") or {}),
            metrics=list(data.get("metrics") or []),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "suite": self.suite,
            "task_type": self.task_type,
            "user_goal": self.user_goal,
            "variants": self.variants,
            "environment": self.environment,
            "expected": self.expected,
            "metrics": self.metrics,
        }


@dataclass(slots=True)
class EvalCase:
    """A concrete variant/fixture to run."""

    scenario_id: str
    suite: str
    task_type: str
    variant_id: str
    user_goal: str
    expected: dict[str, Any]
    environment: dict[str, Any] = field(default_factory=dict)
    fixture: str | None = None


@dataclass(slots=True)
class EvalObservation:
    """Observed agent behavior for one eval case."""

    final_response: str
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    final_state: dict[str, Any] = field(default_factory=dict)
    notified_user: bool | None = None
    latency_ms: int = 0
    cost_estimate: float = 0.0
    constraints_satisfied: bool = True
    recovered: bool | None = None


@dataclass(slots=True)
class EvalRunRecord:
    """Judged result for one eval case."""

    run_id: str
    scenario_id: str
    suite: str
    task_type: str
    variant_id: str
    status: str
    scores: dict[str, float | None]
    metrics: dict[str, float | None]
    failure_modes: list[str]
    judge_reason: str
    expected: dict[str, Any]
    observation: dict[str, Any]
    model: str = "spec"
    agent_version: str = "unknown"
    fixture: str | None = None
    created_at: str = field(default_factory=_utc_now_iso)

    @classmethod
    def create(
        cls,
        case: EvalCase,
        status: str,
        scores: dict[str, float | None],
        metrics: dict[str, float | None],
        failure_modes: list[str],
        judge_reason: str,
        observation: EvalObservation,
        model: str = "spec",
        agent_version: str = "unknown",
    ) -> "EvalRunRecord":
        return cls(
            run_id=str(uuid4()),
            scenario_id=case.scenario_id,
            suite=case.suite,
            task_type=case.task_type,
            variant_id=case.variant_id,
            status=status,
            scores=scores,
            metrics=metrics,
            failure_modes=failure_modes,
            judge_reason=judge_reason,
            expected=case.expected,
            observation={
                "final_response": observation.final_response,
                "tool_trace": observation.tool_trace,
                "final_state": observation.final_state,
                "notified_user": observation.notified_user,
                "latency_ms": observation.latency_ms,
                "cost_estimate": observation.cost_estimate,
                "constraints_satisfied": observation.constraints_satisfied,
                "recovered": observation.recovered,
            },
            model=model,
            agent_version=agent_version,
            fixture=case.fixture,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "scenario_id": self.scenario_id,
            "suite": self.suite,
            "task_type": self.task_type,
            "variant_id": self.variant_id,
            "fixture": self.fixture,
            "model": self.model,
            "agent_version": self.agent_version,
            "status": self.status,
            "scores": self.scores,
            "metrics": self.metrics,
            "failure_modes": self.failure_modes,
            "judge_reason": self.judge_reason,
            "expected": self.expected,
            "observation": self.observation,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalRunRecord":
        return cls(
            run_id=str(data["run_id"]),
            scenario_id=str(data["scenario_id"]),
            suite=str(data["suite"]),
            task_type=str(data["task_type"]),
            variant_id=str(data["variant_id"]),
            fixture=data.get("fixture"),
            model=str(data.get("model") or "unknown"),
            agent_version=str(data.get("agent_version") or "unknown"),
            status=str(data["status"]),
            scores=dict(data.get("scores") or {}),
            metrics=dict(data.get("metrics") or {}),
            failure_modes=list(data.get("failure_modes") or []),
            judge_reason=str(data.get("judge_reason") or ""),
            expected=dict(data.get("expected") or {}),
            observation=dict(data.get("observation") or {}),
            created_at=str(data.get("created_at") or _utc_now_iso()),
        )


@dataclass(slots=True)
class EvalReport:
    """Aggregated eval results."""

    suite: str
    run_id: str
    records: list[EvalRunRecord]
    summary: dict[str, float | int | None]
    created_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "summary": self.summary,
            "records": [record.to_dict() for record in self.records],
        }
