"""Eval executors."""

from __future__ import annotations

from dataclasses import dataclass

from nanobot.evals.agent_executor import AgentLoopExecutor
from nanobot.evals.models import EvalCase, EvalObservation


@dataclass(slots=True)
class SpecExecutor:
    """Deterministic executor used to validate the eval pipeline."""

    latency_ms: int = 100

    def run(self, case: EvalCase) -> EvalObservation:
        tools = [
            {"name": tool_name, "success": True}
            for tool_name in case.expected.get("required_tools", [])
        ]
        for requirement in case.expected.get("required_any_tools", []):
            if isinstance(requirement, str):
                tools.append({"name": requirement, "success": True})
            else:
                options = [str(item) for item in requirement]
                if options:
                    tools.append({"name": options[0], "success": True})

        state = dict(case.expected.get("state") or {})
        if case.expected.get("required_email_accounts"):
            state["email_accounts_checked"] = list(case.expected["required_email_accounts"])
        if case.expected.get("required_paths_read"):
            state["paths_read"] = list(case.expected["required_paths_read"])
        should_notify = case.expected.get("should_notify")
        recovered = True if case.expected.get("recovery_required") else None
        if recovered:
            state.setdefault("recovery_plan_provided", True)
            state.setdefault("safe_failure", True)

        fixture_note = f" fixture={case.fixture}" if case.fixture else ""
        required_markers = [str(item) for item in case.expected.get("response_must_include") or []]
        for group in case.expected.get("response_must_include_any", []):
            if isinstance(group, list) and group:
                required_markers.append(str(group[0]))
        response_suffix = ""
        if required_markers:
            response_suffix = " " + " ".join(required_markers)
        return EvalObservation(
            final_response=f"Spec executor completed {case.scenario_id}{fixture_note}.{response_suffix}",
            tool_trace=tools,
            final_state=state,
            notified_user=should_notify,
            latency_ms=self.latency_ms,
            cost_estimate=0.0,
            constraints_satisfied=True,
            recovered=recovered,
        )


__all__ = ["AgentLoopExecutor", "SpecExecutor"]
