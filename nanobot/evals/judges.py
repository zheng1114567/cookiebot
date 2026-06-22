"""Judging, aggregation, and regression helpers for evals."""

from __future__ import annotations

from collections import Counter
from typing import Any, TYPE_CHECKING

from nanobot.evals.models import EvalCase, EvalObservation, EvalRunRecord

if TYPE_CHECKING:
    from nanobot.evals.judge_llm import LLMJudge

RATE_METRICS = {
    "task_completion_rate",
    "useful_completion_rate",
    "tool_success_rate",
    "constraint_adherence_rate",
    "recovery_rate",
    "notification_precision",
    "notification_recall",
}


def _observed_tool_names(trace: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for item in trace:
        raw_name = str(item.get("name") or "")
        if not raw_name:
            continue
        names.add(raw_name)
        if "." in raw_name:
            names.add(raw_name.split(".", 1)[0])
    return names


def _matches_tool_requirement(
    observed_tools: set[str],
    requirement: str | list[str] | tuple[str, ...] | set[str],
) -> bool:
    if isinstance(requirement, str):
        return requirement in observed_tools
    options = {str(item) for item in requirement}
    return bool(observed_tools & options)


def _normalize_path(value: str) -> str:
    return value.replace("\\", "/").strip()


def _contains_all_markers(text: str, markers: list[str]) -> bool:
    lowered = text.lower()
    return all(marker.lower() in lowered for marker in markers)


def _contains_any_marker_group(text: str, marker_groups: list[list[str]]) -> bool:
    lowered = text.lower()
    return all(any(marker.lower() in lowered for marker in group) for group in marker_groups)


def _contains_no_markers(text: str, markers: list[str]) -> bool:
    lowered = text.lower()
    return all(marker.lower() not in lowered for marker in markers)


def judge_case(
    case: EvalCase,
    observation: EvalObservation,
    model: str = "spec",
    agent_version: str = "unknown",
) -> EvalRunRecord:
    """Judge one case with rule/state checks."""
    failures: list[str] = []

    required_tools = set(case.expected.get("required_tools") or [])
    required_any_tools = list(case.expected.get("required_any_tools") or [])
    forbidden_tools = set(case.expected.get("forbidden_tools") or [])
    called_tools = _observed_tool_names(observation.tool_trace)
    missing_tools = sorted(required_tools - called_tools)
    missing_tool_groups = [
        requirement
        for requirement in required_any_tools
        if not _matches_tool_requirement(called_tools, requirement)
    ]
    used_forbidden_tools = sorted(forbidden_tools & called_tools)
    if missing_tools:
        failures.append("missing_tool_use")
    if missing_tool_groups:
        failures.append("missing_tool_option")
    if used_forbidden_tools:
        failures.append("forbidden_tool_use")

    tool_successes = []
    for item in observation.tool_trace:
        if not required_tools and not required_any_tools and not forbidden_tools:
            tool_successes.append(bool(item.get("success", False)))
            continue
        observed_names = _observed_tool_names([item])
        if required_tools & observed_names:
            tool_successes.append(bool(item.get("success", False)))
            continue
        if any(_matches_tool_requirement(observed_names, requirement) for requirement in required_any_tools):
            tool_successes.append(bool(item.get("success", False)))
    tool_score = (
        sum(1 for item in tool_successes if item) / len(tool_successes)
        if tool_successes else 1.0
    )
    if tool_score < 1:
        failures.append("tool_failure")

    expected_state = dict(case.expected.get("state") or {})
    state_misses = [
        key for key, value in expected_state.items()
        if observation.final_state.get(key) != value
    ]
    if state_misses:
        failures.append("state_mismatch")

    required_paths = {
        _normalize_path(str(item))
        for item in case.expected.get("required_paths_read") or []
    }
    if required_paths:
        observed_paths = {
            _normalize_path(str(item))
            for item in observation.final_state.get("paths_read") or []
        }
        missing_paths = sorted(required_paths - observed_paths)
        if missing_paths:
            failures.append("missing_required_path")

    required_markers = [str(item) for item in case.expected.get("response_must_include") or []]
    if required_markers and not _contains_all_markers(observation.final_response, required_markers):
        failures.append("missing_required_content")

    required_any_marker_groups = [
        [str(marker) for marker in group]
        for group in case.expected.get("response_must_include_any") or []
        if isinstance(group, list) and group
    ]
    if required_any_marker_groups and not _contains_any_marker_group(
        observation.final_response,
        required_any_marker_groups,
    ):
        failures.append("missing_required_content")

    forbidden_markers = [str(item) for item in case.expected.get("response_must_not_include") or []]
    if forbidden_markers and not _contains_no_markers(observation.final_response, forbidden_markers):
        failures.append("included_forbidden_content")

    required_email_accounts = set(case.expected.get("required_email_accounts") or [])
    if required_email_accounts:
        checked_accounts = set(observation.final_state.get("email_accounts_checked") or [])
        missing_accounts = sorted(required_email_accounts - checked_accounts)
        if missing_accounts:
            failures.append("missing_email_account")

    expected_notify = case.expected.get("should_notify")
    notification_score: float | None = None
    notification_skipped = bool(observation.final_state.get("notification_judgement_skipped"))
    if expected_notify is not None and not notification_skipped:
        notification_score = 1.0 if observation.notified_user is expected_notify else 0.0
        if notification_score == 0 and expected_notify:
            failures.append("under_notified")
        elif notification_score == 0:
            failures.append("over_notified")

    constraint_score = 1.0 if observation.constraints_satisfied else 0.0
    if constraint_score == 0:
        failures.append("ignored_constraint")

    recovery_score: float | None = None
    if case.expected.get("recovery_required"):
        recovery_score = 1.0 if observation.recovered else 0.0
        if recovery_score == 0:
            failures.append("recovery_failed")

    completion_score = 1.0
    if (
        missing_tools
        or missing_tool_groups
        or state_misses
        or used_forbidden_tools
        or ("missing_required_path" in failures)
        or ("missing_required_content" in failures)
        or ("included_forbidden_content" in failures)
    ):
        completion_score = 0.0
    elif failures:
        completion_score = 0.5

    status = "success"
    if completion_score == 0:
        status = "failed"
    elif completion_score < 1:
        status = "partial"

    metrics = {
        "task_completion": completion_score,
        "constraint_adherence": constraint_score,
        "tool_success": tool_score,
        "notification_correct": notification_score,
        "recovery": recovery_score,
        "latency_ms": float(observation.latency_ms),
        "cost_estimate": observation.cost_estimate,
    }
    scores = {
        "completion_score": completion_score,
        "tool_score": tool_score,
        "constraint_score": constraint_score,
        "notification_score": notification_score,
        "recovery_score": recovery_score,
    }
    if not failures:
        reason = "All rule and state checks passed."
    else:
        parts = list(failures)
        if missing_tool_groups:
            parts.append(f"missing_any_of={missing_tool_groups}")
        if used_forbidden_tools:
            parts.append(f"forbidden={used_forbidden_tools}")
        reason = ", ".join(parts)
    return EvalRunRecord.create(
        case=case,
        status=status,
        scores=scores,
        metrics=metrics,
        failure_modes=failures,
        judge_reason=reason,
        observation=observation,
        model=model,
        agent_version=agent_version,
    )


def _avg(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def aggregate(records: list[EvalRunRecord]) -> dict[str, float | int | None]:
    total = len(records)
    success = sum(1 for record in records if record.status == "success")
    partial = sum(1 for record in records if record.status == "partial")
    failed = sum(1 for record in records if record.status == "failed")

    notified = [
        record for record in records
        if record.expected.get("should_notify") is not None
        and not record.observation.get("final_state", {}).get("notification_judgement_skipped")
    ]
    true_positive = sum(
        1 for record in notified
        if record.expected.get("should_notify") is True
        and record.observation.get("notified_user") is True
    )
    false_positive = sum(
        1 for record in notified
        if record.expected.get("should_notify") is False
        and record.observation.get("notified_user") is True
    )
    false_negative = sum(
        1 for record in notified
        if record.expected.get("should_notify") is True
        and record.observation.get("notified_user") is not True
    )

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive else None
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative else None
    )

    return {
        "total": total,
        "success": success,
        "partial": partial,
        "failed": failed,
        "task_completion_rate": success / total if total else None,
        "useful_completion_rate": (success + partial * 0.5) / total if total else None,
        "tool_success_rate": _avg([
            float(record.metrics["tool_success"])
            for record in records
            if record.metrics.get("tool_success") is not None
        ]),
        "constraint_adherence_rate": _avg([
            float(record.metrics["constraint_adherence"])
            for record in records
            if record.metrics.get("constraint_adherence") is not None
        ]),
        "recovery_rate": _avg([
            float(record.metrics["recovery"])
            for record in records
            if record.metrics.get("recovery") is not None
        ]),
        "notification_precision": precision,
        "notification_recall": recall,
        "avg_latency_ms": _avg([
            float(record.metrics["latency_ms"])
            for record in records
            if record.metrics.get("latency_ms") is not None
        ]),
        "cost_per_success": (
            sum(float(record.metrics.get("cost_estimate") or 0.0) for record in records) / success
            if success else None
        ),
    }


def aggregate_by_task_type(
    records: list[EvalRunRecord],
) -> dict[str, dict[str, float | int | None]]:
    groups: dict[str, list[EvalRunRecord]] = {}
    for record in records:
        groups.setdefault(record.task_type, []).append(record)
    return {task_type: aggregate(group) for task_type, group in sorted(groups.items())}


def failure_mode_counts(records: list[EvalRunRecord]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(record.failure_modes)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def compare_summaries(
    baseline: dict[str, float | int | None],
    current: dict[str, float | int | None],
) -> list[dict[str, float | str | None]]:
    metrics = [
        "task_completion_rate",
        "useful_completion_rate",
        "tool_success_rate",
        "constraint_adherence_rate",
        "recovery_rate",
        "notification_precision",
        "notification_recall",
        "avg_latency_ms",
        "cost_per_success",
    ]
    rows: list[dict[str, float | str | None]] = []
    for metric in metrics:
        base = baseline.get(metric)
        cur = current.get(metric)
        delta = None if base is None or cur is None else float(cur) - float(base)
        rows.append({"metric": metric, "baseline": base, "current": cur, "delta": delta})
    return rows


def check_thresholds(
    summary: dict[str, float | int | None],
    thresholds: dict[str, float | None],
) -> list[dict[str, float | str | None]]:
    failures: list[dict[str, float | str | None]] = []
    for metric, threshold in thresholds.items():
        if threshold is None:
            continue
        value = summary.get(metric)
        if value is None:
            failures.append({"metric": metric, "actual": None, "threshold": threshold})
        elif float(value) < threshold:
            failures.append({"metric": metric, "actual": float(value), "threshold": threshold})
    return failures


def detect_regressions(
    baseline_records: list[EvalRunRecord],
    current_records: list[EvalRunRecord],
    max_drop: float,
) -> list[dict[str, float | str | None]]:
    baseline = aggregate(baseline_records)
    current = aggregate(current_records)
    regressions: list[dict[str, float | str | None]] = []
    for row in compare_summaries(baseline, current):
        metric = str(row["metric"])
        delta = row["delta"]
        if metric not in RATE_METRICS or delta is None:
            continue
        if float(delta) < -max_drop:
            regressions.append(row)
    return regressions


async def llm_judge_case(
    judge: LLMJudge,
    record: EvalRunRecord,
    case: EvalCase,
    observation: EvalObservation,
) -> EvalRunRecord:
    """Augment an existing rule-judged record with LLM-as-Judge scores.

    This wraps the result of ``judge_case()`` with additional LLM-evaluated
    dimensions: relevance, helpfulness, faithfulness.
    """
    judgment = await judge.evaluate(
        query=case.user_goal,
        response=observation.final_response,
        context=observation.final_state.get("paths_read", ""),
        expected=str(case.expected),
    )
    return LLMJudge.merge_into_run_record(record, judgment)

