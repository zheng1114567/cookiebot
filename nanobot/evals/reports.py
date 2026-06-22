"""Markdown report rendering helpers."""

from __future__ import annotations

from nanobot.evals.judges import aggregate, aggregate_by_task_type, compare_summaries, failure_mode_counts
from nanobot.evals.models import EvalReport, EvalRunRecord


def _format_pct(value: float | int | None) -> str:
    return "n/a" if value is None else f"{float(value) * 100:.1f}%"


def _rag_capability(task_type: str) -> str | None:
    mapping = {
        "rag_single_source_extract": "Grounded Extraction",
        "rag_multi_source_compare": "Cross-Source Comparison",
        "rag_insufficient_context_refusal": "Insufficient-Context Refusal",
        "rag_conflict_resolution": "Conflict Handling",
        "rag_irrelevant_context_filter": "Irrelevant-Context Filtering",
        "rag_multi_hop_join": "Multi-Hop Synthesis",
    }
    return mapping.get(task_type)


def rag_scorecard(records: list[EvalRunRecord]) -> dict[str, dict[str, float | int | None]]:
    groups: dict[str, list[EvalRunRecord]] = {}
    for record in records:
        capability = _rag_capability(record.task_type)
        if capability:
            groups.setdefault(capability, []).append(record)
    return {name: aggregate(group) for name, group in sorted(groups.items())}


def check_rag_capability_thresholds(
    records: list[EvalRunRecord],
    thresholds: dict[str, float | None],
) -> list[dict[str, float | str | None]]:
    """Return RAG capability thresholds that failed."""
    scorecard = rag_scorecard(records)
    failures: list[dict[str, float | str | None]] = []
    for capability, threshold in thresholds.items():
        if threshold is None:
            continue
        group = scorecard.get(capability)
        value = None if group is None else group.get("task_completion_rate")
        if value is None:
            failures.append({"metric": capability, "actual": None, "threshold": threshold})
        elif float(value) < threshold:
            failures.append({"metric": capability, "actual": float(value), "threshold": threshold})
    return failures


def compare_rag_scorecards(
    baseline_records: list[EvalRunRecord],
    current_records: list[EvalRunRecord],
) -> list[dict[str, float | str | None]]:
    baseline = rag_scorecard(baseline_records)
    current = rag_scorecard(current_records)
    capabilities = sorted(set(baseline) | set(current))
    rows: list[dict[str, float | str | None]] = []
    for capability in capabilities:
        base = None if capability not in baseline else baseline[capability].get("task_completion_rate")
        cur = None if capability not in current else current[capability].get("task_completion_rate")
        delta = None if base is None or cur is None else float(cur) - float(base)
        rows.append({"metric": capability, "baseline": base, "current": cur, "delta": delta})
    return rows


def detect_rag_regressions(
    baseline_records: list[EvalRunRecord],
    current_records: list[EvalRunRecord],
    max_drop: float,
) -> list[dict[str, float | str | None]]:
    regressions: list[dict[str, float | str | None]] = []
    for row in compare_rag_scorecards(baseline_records, current_records):
        delta = row["delta"]
        if delta is None:
            continue
        if float(delta) < -max_drop:
            regressions.append(row)
    return regressions


def render_markdown(report: EvalReport) -> str:
    summary = report.summary

    def pct(key: str) -> str:
        value = summary.get(key)
        return "n/a" if value is None else f"{float(value) * 100:.1f}%"

    lines = [
        f"# Agent Eval Report: {report.suite}",
        "",
        f"- Run ID: `{report.run_id}`",
        f"- Created: `{report.created_at}`",
        f"- Cases: {summary.get('total', 0)}",
        "- Pass / Partial / Fail: "
        f"{summary.get('success', 0)} / "
        f"{summary.get('partial', 0)} / "
        f"{summary.get('failed', 0)}",
        "",
        "## Metrics",
        "",
        f"- Task Completion Rate: {pct('task_completion_rate')}",
        f"- Useful Completion Rate: {pct('useful_completion_rate')}",
        f"- Tool Success Rate: {pct('tool_success_rate')}",
        f"- Constraint Adherence Rate: {pct('constraint_adherence_rate')}",
        f"- Recovery Rate: {pct('recovery_rate')}",
        f"- Notification Precision: {pct('notification_precision')}",
        f"- Notification Recall: {pct('notification_recall')}",
        f"- Average Latency: {summary.get('avg_latency_ms') or 0:.0f} ms",
        f"- Cost per Success: {summary.get('cost_per_success') or 0:.4f}",
        "",
        "## Task Types",
        "",
    ]
    by_task_type = aggregate_by_task_type(report.records)
    if not by_task_type:
        lines.append("No task type data.")
    else:
        lines.extend([
            "| Task Type | Cases | Completion | Tool Success | Constraints |",
            "| --- | ---: | ---: | ---: | ---: |",
        ])
        for task_type, item in by_task_type.items():
            lines.append(
                f"| {task_type} | {item.get('total', 0)} | "
                f"{_format_pct(item.get('task_completion_rate'))} | "
                f"{_format_pct(item.get('tool_success_rate'))} | "
                f"{_format_pct(item.get('constraint_adherence_rate'))} |"
            )

    lines.extend([
        "",
        "## RAG Scorecard",
        "",
    ])
    rag_groups = rag_scorecard(report.records)
    if not rag_groups:
        lines.append("No RAG scenarios.")
    else:
        lines.extend([
            "| Capability | Cases | Completion | Tool Success | Constraints |",
            "| --- | ---: | ---: | ---: | ---: |",
        ])
        for capability, item in rag_groups.items():
            lines.append(
                f"| {capability} | {item.get('total', 0)} | "
                f"{_format_pct(item.get('task_completion_rate'))} | "
                f"{_format_pct(item.get('tool_success_rate'))} | "
                f"{_format_pct(item.get('constraint_adherence_rate'))} |"
            )

    lines.extend([
        "",
        "## Failure Modes",
        "",
    ])
    failures_by_mode = failure_mode_counts(report.records)
    if not failures_by_mode:
        lines.append("No failure modes.")
    else:
        for mode, count in failures_by_mode.items():
            lines.append(f"- `{mode}`: {count}")

    lines.extend([
        "",
        "## Failures",
        "",
    ])
    failing = [record for record in report.records if record.status != "success"]
    if not failing:
        lines.append("No failures.")
    else:
        for record in failing[:20]:
            fixture = f" fixture={record.fixture}" if record.fixture else ""
            lines.append(
                f"- `{record.scenario_id}` `{record.variant_id}`: "
                f"{record.status}{fixture} ({record.judge_reason})"
            )
    lines.append("")
    return "\n".join(lines)


def render_compare_markdown(
    baseline_records: list[EvalRunRecord],
    current_records: list[EvalRunRecord],
) -> str:
    baseline = aggregate(baseline_records)
    current = aggregate(current_records)
    rows = compare_summaries(baseline, current)

    lines = [
        "# Agent Eval Compare",
        "",
        f"- Baseline cases: {baseline.get('total', 0)}",
        f"- Current cases: {current.get('total', 0)}",
        "",
        "| Metric | Baseline | Current | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        metric = str(row["metric"])
        base = row["baseline"]
        cur = row["current"]
        delta = row["delta"]
        if metric in {"avg_latency_ms", "cost_per_success"}:
            base_text = "n/a" if base is None else f"{float(base):.4f}"
            cur_text = "n/a" if cur is None else f"{float(cur):.4f}"
            delta_text = "n/a" if delta is None else f"{float(delta):+.4f}"
        else:
            base_text = _format_pct(base if isinstance(base, (int, float)) else None)
            cur_text = _format_pct(cur if isinstance(cur, (int, float)) else None)
            delta_text = "n/a" if delta is None else f"{float(delta) * 100:+.1f} pp"
        lines.append(f"| {metric} | {base_text} | {cur_text} | {delta_text} |")

    lines.extend([
        "",
        "## RAG Capability Compare",
        "",
    ])
    rag_rows = compare_rag_scorecards(baseline_records, current_records)
    if not rag_rows:
        lines.append("No RAG capability data.")
    else:
        lines.extend([
            "| Capability | Baseline | Current | Delta |",
            "| --- | ---: | ---: | ---: |",
        ])
        for row in rag_rows:
            base_text = _format_pct(row["baseline"] if isinstance(row["baseline"], (int, float)) else None)
            cur_text = _format_pct(row["current"] if isinstance(row["current"], (int, float)) else None)
            delta_text = "n/a" if row["delta"] is None else f"{float(row['delta']) * 100:+.1f} pp"
            lines.append(f"| {row['metric']} | {base_text} | {cur_text} | {delta_text} |")
    lines.append("")
    return "\n".join(lines)
