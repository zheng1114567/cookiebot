"""Scenario runner plus backward-compatible eval helpers."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from uuid import uuid4

from nanobot.evals.executors import SpecExecutor
from nanobot.evals.judges import (
    RATE_METRICS,
    aggregate,
    aggregate_by_task_type,
    check_thresholds,
    compare_summaries,
    detect_regressions,
    failure_mode_counts,
    judge_case,
    llm_judge_case,
)
from nanobot.evals.models import EvalReport, EvalScenario
from nanobot.evals.reports import (
    check_rag_capability_thresholds,
    compare_rag_scorecards,
    detect_rag_regressions,
    rag_scorecard,
    render_compare_markdown,
    render_markdown,
)
from nanobot.evals.scenarios import load_run_records, load_scenarios, materialize_cases


class EvalRunner:
    """Run eval scenarios and write reports."""

    def __init__(self, executor: SpecExecutor | None = None) -> None:
        self.executor = executor or SpecExecutor()

    def run(
        self,
        scenarios: list[EvalScenario],
        suite: str = "core",
        model: str = "spec",
        agent_version: str = "unknown",
        max_cases: int | None = None,
    ) -> EvalReport:
        selected = [scenario for scenario in scenarios if suite == "all" or scenario.suite == suite]
        if not selected:
            raise ValueError(f"No eval scenarios found for suite: {suite}")
        records = []
        for scenario in selected:
            for case in materialize_cases(scenario):
                if max_cases is not None and len(records) >= max_cases:
                    break
                observation = self.executor.run(case)
                records.append(
                    judge_case(
                        case,
                        observation,
                        model=model,
                        agent_version=agent_version,
                    )
                )
            if max_cases is not None and len(records) >= max_cases:
                break
        return EvalReport(
            suite=suite,
            run_id=str(uuid4()),
            records=records,
            summary=aggregate(records),
        )

    async def arun(
        self,
        scenarios: list[EvalScenario],
        suite: str = "core",
        model: str = "spec",
        agent_version: str = "unknown",
        max_cases: int | None = None,
    ) -> EvalReport:
        selected = [scenario for scenario in scenarios if suite == "all" or scenario.suite == suite]
        if not selected:
            raise ValueError(f"No eval scenarios found for suite: {suite}")
        records = []
        for scenario in selected:
            for case in materialize_cases(scenario):
                if max_cases is not None and len(records) >= max_cases:
                    break
                maybe_observation = self.executor.run(case)
                observation = (
                    await maybe_observation
                    if inspect.isawaitable(maybe_observation)
                    else maybe_observation
                )
                records.append(
                    judge_case(
                        case,
                        observation,
                        model=model,
                        agent_version=agent_version,
                    )
                )
            if max_cases is not None and len(records) >= max_cases:
                break
        return EvalReport(
            suite=suite,
            run_id=str(uuid4()),
            records=records,
            summary=aggregate(records),
        )

    def write_report(self, report: EvalReport, output_dir: Path) -> tuple[Path, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_dir / f"{report.run_id}.jsonl"
        md_path = output_dir / f"{report.run_id}.md"
        summary_path = output_dir / f"{report.run_id}.summary.json"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for record in report.records:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        md_path.write_text(render_markdown(report), encoding="utf-8")
        summary_payload = {
            "suite": report.suite,
            "run_id": report.run_id,
            "created_at": report.created_at,
            "summary": report.summary,
            "by_task_type": aggregate_by_task_type(report.records),
            "rag_scorecard": rag_scorecard(report.records),
            "failure_modes": failure_mode_counts(report.records),
        }
        summary_path.write_text(
            json.dumps(summary_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / "latest.md").write_text(render_markdown(report), encoding="utf-8")
        (output_dir / "latest.jsonl").write_text(
            jsonl_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        (output_dir / "latest.summary.json").write_text(
            summary_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        return jsonl_path, md_path


__all__ = [
    "RATE_METRICS",
    "EvalRunner",
    "aggregate",
    "aggregate_by_task_type",
    "check_thresholds",
    "compare_summaries",
    "detect_regressions",
    "failure_mode_counts",
    "judge_case",
    "llm_judge_case",
    "load_run_records",
    "load_scenarios",
    "materialize_cases",
    "check_rag_capability_thresholds",
    "compare_rag_scorecards",
    "detect_rag_regressions",
    "render_compare_markdown",
    "render_markdown",
]
