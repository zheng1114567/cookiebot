"""Scenario loading and case materialization helpers."""

from __future__ import annotations

import json
from pathlib import Path

from nanobot.evals.default_scenarios import DEFAULT_SCENARIOS
from nanobot.evals.models import EvalCase, EvalRunRecord, EvalScenario


def load_scenarios(path: Path | None = None) -> list[EvalScenario]:
    """Load scenarios from a JSON file/directory or return built-ins."""
    if path is None:
        return [EvalScenario.from_dict(item) for item in DEFAULT_SCENARIOS]

    path = path.expanduser().resolve()
    if path.is_dir():
        scenarios: list[EvalScenario] = []
        for item in sorted(path.glob("*.json")):
            scenarios.extend(load_scenarios(item))
        return scenarios

    data = json.loads(path.read_text(encoding="utf-8"))
    raw_items = data if isinstance(data, list) else data.get("scenarios", [data])
    return [EvalScenario.from_dict(item) for item in raw_items]


def load_run_records(path: Path) -> list[EvalRunRecord]:
    """Load run records from a JSONL report."""
    records: list[EvalRunRecord] = []
    with path.expanduser().resolve().open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(EvalRunRecord.from_dict(json.loads(line)))
    return records


def _dynamic_variants(scenario: EvalScenario) -> list[str]:
    config = scenario.variants.get("generation") or {}
    count = int(config.get("count") or 0)
    dimensions = list(config.get("dimensions") or [])
    templates = {
        "time_change": "把时间表达换一种说法，但保持原任务目标：{goal}",
        "constraint": "增加一个合理限制并保持原任务目标：{goal}",
        "paraphrase": "用更口语化的方式表达同一个目标：{goal}",
        "noise": "加入无关上下文，但核心目标不变：{goal}",
        "timeout": "在工具可能超时的情况下完成：{goal}",
        "permission": "在可能缺少权限时完成：{goal}",
        "malformed_response": "在工具返回格式异常时完成：{goal}",
    }
    if not dimensions:
        dimensions = ["paraphrase"]
    generated: list[str] = []
    for index in range(count):
        dimension = dimensions[index % len(dimensions)]
        template = templates.get(dimension, "生成一个等价变体：{goal}")
        generated.append(template.format(goal=scenario.user_goal))
    return generated


def materialize_cases(scenario: EvalScenario) -> list[EvalCase]:
    """Expand a scenario into concrete variant/fixture cases."""
    variant_goals = [scenario.user_goal]
    variant_goals.extend(str(item) for item in scenario.variants.get("static") or [])
    variant_goals.extend(_dynamic_variants(scenario))

    notify_on = list(scenario.expected.get("notify_on") or [])
    suppress_on = list(scenario.expected.get("suppress_on") or [])
    fixtures: list[tuple[str | None, bool | None]] = [(None, None)]
    if notify_on or suppress_on:
        fixtures = [(fixture, True) for fixture in notify_on]
        fixtures.extend((fixture, False) for fixture in suppress_on)

    cases: list[EvalCase] = []
    for variant_index, goal in enumerate(variant_goals):
        for fixture, should_notify in fixtures:
            expected = dict(scenario.expected)
            if should_notify is not None:
                expected["should_notify"] = should_notify
            cases.append(
                EvalCase(
                    scenario_id=scenario.id,
                    suite=scenario.suite,
                    task_type=scenario.task_type,
                    variant_id=f"v{variant_index:02d}",
                    user_goal=goal,
                    expected=expected,
                    environment=scenario.environment,
                    fixture=fixture,
                )
            )
    return cases
