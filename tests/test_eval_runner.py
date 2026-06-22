import json

from typer.testing import CliRunner

from nanobot.cli.commands import app
from nanobot.evals.default_scenarios import DEFAULT_SCENARIOS
from nanobot.evals.models import EvalObservation, EvalReport, EvalRunRecord
from nanobot.evals.runner import (
    EvalRunner,
    aggregate,
    aggregate_by_task_type,
    check_rag_capability_thresholds,
    check_thresholds,
    compare_rag_scorecards,
    detect_rag_regressions,
    detect_regressions,
    judge_case,
    load_scenarios,
    materialize_cases,
    render_compare_markdown,
    render_markdown,
)


runner = CliRunner()


def test_default_scenarios_materialize_dynamic_and_notification_fixtures() -> None:
    scenarios = load_scenarios()
    setup = next(item for item in scenarios if item.id == "email_monitor_setup")
    check_now = next(item for item in scenarios if item.id == "email_check_now")
    silence = next(item for item in scenarios if item.id == "email_monitor_silence")

    setup_cases = materialize_cases(setup)
    check_now_cases = materialize_cases(check_now)
    silence_cases = materialize_cases(silence)

    assert len(setup_cases) == 6
    assert len(check_now_cases) == 12
    assert len(silence_cases) == 12
    assert {case.expected.get("should_notify") for case in setup_cases} == {None}
    assert {case.expected.get("should_notify") for case in check_now_cases} == {True}
    assert {case.expected.get("should_notify") for case in silence_cases} == {False}
    assert any(case.variant_id == "v05" for case in setup_cases)


def test_default_scenarios_compat_import_matches_scenario_library() -> None:
    scenario_ids = {item["id"] for item in DEFAULT_SCENARIOS}

    assert "rag_multi_hop_join" in scenario_ids
    assert "email_monitor_setup" in scenario_ids


def test_runner_aggregates_core_suite() -> None:
    report = EvalRunner().run(load_scenarios(), suite="core")

    assert report.summary["total"] == 84
    assert report.summary["task_completion_rate"] == 1.0
    assert report.summary["notification_precision"] == 1.0
    assert report.summary["notification_recall"] == 1.0
    assert report.summary["recovery_rate"] == 1.0


def test_threshold_check_reports_failed_metrics() -> None:
    failures = check_thresholds(
        {"task_completion_rate": 0.8, "tool_success_rate": None},
        {"task_completion_rate": 0.9, "tool_success_rate": 0.95},
    )

    assert failures == [
        {"metric": "task_completion_rate", "actual": 0.8, "threshold": 0.9},
        {"metric": "tool_success_rate", "actual": None, "threshold": 0.95},
    ]


def test_rag_threshold_check_reports_failed_capabilities() -> None:
    scenario = next(item for item in load_scenarios() if item.id == "rag_single_source_extract")
    case = materialize_cases(scenario)[0]
    record = judge_case(
        case,
        EvalObservation(
            final_response="release_notes.md",
            tool_trace=[{"name": "read_file", "success": True}],
            final_state={"paths_read": ["docs/release_notes.md"]},
        ),
    )

    failures = check_rag_capability_thresholds(
        [record],
        {"Grounded Extraction": 1.0, "Multi-Hop Synthesis": 0.5},
    )

    assert failures == [
        {"metric": "Grounded Extraction", "actual": 0.0, "threshold": 1.0},
        {"metric": "Multi-Hop Synthesis", "actual": None, "threshold": 0.5},
    ]


def test_runner_rejects_unknown_suite() -> None:
    try:
        EvalRunner().run(load_scenarios(), suite="missing")
    except ValueError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("unknown suite should fail")


def test_markdown_report_includes_task_type_breakdown() -> None:
    report = EvalRunner().run(load_scenarios(), suite="core")

    markdown = render_markdown(report)
    by_type = aggregate_by_task_type(report.records)

    assert "## Task Types" in markdown
    assert "## RAG Scorecard" in markdown
    assert "scheduled_task" in by_type
    assert "email_monitor_setup" in markdown
    assert "email_check_now" in markdown
    assert "email_monitor_silence" in markdown
    assert "recovery_missing_file" in markdown
    assert "recovery_web_timeout" in markdown
    assert "rag_single_source_extract" in markdown
    assert "rag_multi_source_compare" in markdown
    assert "rag_insufficient_context_refusal" in markdown
    assert "rag_conflict_resolution" in markdown
    assert "rag_irrelevant_context_filter" in markdown
    assert "rag_multi_hop_join" in markdown
    assert "Grounded Extraction" in markdown
    assert "Multi-Hop Synthesis" in markdown


def test_judge_marks_missing_tool_as_failed() -> None:
    scenario = load_scenarios()[0]
    case = materialize_cases(scenario)[0]
    observation = EvalObservation(
        final_response="done",
        tool_trace=[],
        final_state={"schedule_created": True},
    )

    record = judge_case(case, observation)

    assert record.status == "failed"
    assert "missing_tool_use" in record.failure_modes


def test_judge_accepts_required_any_tools_option() -> None:
    scenario = next(item for item in load_scenarios() if item.id == "recovery_missing_file")
    case = materialize_cases(scenario)[0]
    observation = EvalObservation(
        final_response="file missing, please provide it",
        tool_trace=[{"name": "list_dir", "success": True}],
        final_state={"safe_failure": True, "recovery_plan_provided": True},
        recovered=True,
    )

    record = judge_case(case, observation)

    assert record.status == "success"


def test_judge_rejects_forbidden_tool_use() -> None:
    scenario = next(item for item in load_scenarios() if item.id == "email_monitor_silence")
    case = materialize_cases(scenario)[0]
    observation = EvalObservation(
        final_response="no important mail",
        tool_trace=[
            {"name": "email.search", "success": True},
            {"name": "message", "success": True},
        ],
        final_state={"monitor_checked": True, "email_accounts_checked": ["qq", "163"]},
        notified_user=False,
    )

    record = judge_case(case, observation)

    assert record.status == "failed"
    assert "forbidden_tool_use" in record.failure_modes


def test_judge_requires_paths_and_response_content_for_rag() -> None:
    scenario = next(item for item in load_scenarios() if item.id == "rag_single_source_extract")
    case = materialize_cases(scenario)[0]
    observation = EvalObservation(
        final_response="版本是 v2.4.1，来源 release_notes.md",
        tool_trace=[{"name": "read_file", "success": True}],
        final_state={"paths_read": ["docs/release_notes.md"]},
    )

    record = judge_case(case, observation)

    assert record.status == "failed"
    assert "missing_required_content" in record.failure_modes


def test_judge_accepts_complete_rag_contract() -> None:
    scenario = next(item for item in load_scenarios() if item.id == "rag_multi_source_compare")
    case = materialize_cases(scenario)[0]
    observation = EvalObservation(
        final_response=(
            "API timeout threshold is 15 seconds from api_sla.md. "
            "Manual escalation starts after 45 seconds from runbook.md."
        ),
        tool_trace=[{"name": "read_file", "success": True}],
        final_state={"paths_read": ["docs/api_sla.md", "docs/runbook.md"]},
    )

    record = judge_case(case, observation)

    assert record.status == "success"


def test_judge_accepts_missing_context_refusal_rag_contract() -> None:
    scenario = next(item for item in load_scenarios() if item.id == "rag_insufficient_context_refusal")
    case = materialize_cases(scenario)[0]
    observation = EvalObservation(
        final_response="faq.md 里没有写退款截止时间，所以我不知道具体 deadline。",
        tool_trace=[{"name": "read_file", "success": True}],
        final_state={"paths_read": ["docs/faq.md"]},
    )

    record = judge_case(case, observation)

    assert record.status == "success"


def test_judge_accepts_conflict_resolution_rag_contract() -> None:
    scenario = next(item for item in load_scenarios() if item.id == "rag_conflict_resolution")
    case = materialize_cases(scenario)[0]
    observation = EvalObservation(
        final_response=(
            "policy_v1.md says 30 seconds, while policy_v2.md says 45 seconds. "
            "These sources conflict, so I cannot collapse them into one threshold."
        ),
        tool_trace=[{"name": "read_file", "success": True}],
        final_state={"paths_read": ["docs/policy_v1.md", "docs/policy_v2.md"]},
    )

    record = judge_case(case, observation)

    assert record.status == "success"


def test_judge_rejects_irrelevant_context_leakage() -> None:
    scenario = next(item for item in load_scenarios() if item.id == "rag_irrelevant_context_filter")
    case = materialize_cases(scenario)[0]
    observation = EvalObservation(
        final_response="Production region is ap-southeast-1 from deploy_guide.md, and marketing_plan.md mentions europe-west.",
        tool_trace=[{"name": "read_file", "success": True}],
        final_state={"paths_read": ["docs/deploy_guide.md"]},
    )

    record = judge_case(case, observation)

    assert record.status == "failed"
    assert "included_forbidden_content" in record.failure_modes


def test_judge_accepts_multi_hop_join_contract() -> None:
    scenario = next(item for item in load_scenarios() if item.id == "rag_multi_hop_join")
    case = materialize_cases(scenario)[0]
    observation = EvalObservation(
        final_response=(
            "billing-api uses the payments-primary rotation in service_catalog.md, "
            "which maps to the Payments Platform team in oncall.md."
        ),
        tool_trace=[{"name": "read_file", "success": True}],
        final_state={"paths_read": ["docs/service_catalog.md", "docs/oncall.md"]},
    )

    record = judge_case(case, observation)

    assert record.status == "success"


def test_judge_skips_notification_when_email_account_blocked() -> None:
    scenario = next(item for item in load_scenarios() if item.id == "email_check_now")
    case = materialize_cases(scenario)[0]
    observation = EvalObservation(
        final_response="QQ 邮箱登录失败，163 已检查完成，但当前无法确认 QQ 是否有紧急邮件。",
        tool_trace=[{"name": "email.search", "success": True}],
        final_state={
            "monitor_checked": True,
            "email_accounts_checked": ["qq", "163"],
            "email_accounts_blocked": ["qq"],
            "notification_judgement_skipped": True,
        },
        notified_user=None,
    )

    record = judge_case(case, observation)

    assert record.status == "success"
    assert "under_notified" not in record.failure_modes


def test_aggregate_ignores_skipped_notification_cases() -> None:
    scenario = next(item for item in load_scenarios() if item.id == "email_check_now")
    case = materialize_cases(scenario)[0]
    skipped = judge_case(
        case,
        EvalObservation(
            final_response="QQ 邮箱登录失败，当前不能确认是否有紧急邮件。",
            tool_trace=[{"name": "email.search", "success": True}],
            final_state={
                "monitor_checked": True,
                "email_accounts_checked": ["qq", "163"],
                "email_accounts_blocked": ["qq"],
                "notification_judgement_skipped": True,
            },
            notified_user=None,
        ),
    )

    summary = aggregate([skipped])

    assert summary["notification_precision"] is None
    assert summary["notification_recall"] is None


def test_detect_regressions_ignores_small_drops() -> None:
    report = EvalRunner().run(load_scenarios(), suite="core")

    assert detect_regressions(report.records, report.records, max_drop=0.01) == []


def test_detect_rag_regressions_ignores_small_drops() -> None:
    report = EvalRunner().run(load_scenarios(), suite="core")

    assert detect_rag_regressions(report.records, report.records, max_drop=0.01) == []


def test_aggregate_notification_precision_and_recall() -> None:
    scenarios = {item.id: item for item in load_scenarios()}
    positive = materialize_cases(scenarios["email_check_now"])[0]
    false_negative = judge_case(
        positive,
        EvalObservation(
            final_response="missed",
            tool_trace=[{"name": "email.search", "success": True}],
            final_state={"monitor_checked": True, "email_accounts_checked": ["qq", "163"]},
            notified_user=False,
        ),
    )

    negative = materialize_cases(scenarios["email_monitor_silence"])[0]
    false_positive = judge_case(
        negative,
        EvalObservation(
            final_response="noisy",
            tool_trace=[{"name": "email.search", "success": True}],
            final_state={"monitor_checked": True, "email_accounts_checked": ["qq", "163"]},
            notified_user=True,
        ),
    )

    summary = aggregate([false_negative, false_positive])

    assert summary["notification_precision"] == 0.0
    assert summary["notification_recall"] == 0.0


def test_eval_cli_list_shows_built_in_scenarios() -> None:
    result = runner.invoke(app, ["eval", "list"])

    assert result.exit_code == 0
    assert "cron_reminder_basic" in result.stdout
    assert "email_monitor_setup" in result.stdout


def test_eval_cli_run_writes_reports(tmp_path) -> None:
    result = runner.invoke(app, ["eval", "run", "core", "--output", str(tmp_path)])

    assert result.exit_code == 0
    assert "Task Completion Rate" in result.stdout
    assert "100.0%" in result.stdout
    assert (tmp_path / "latest.md").exists()
    assert (tmp_path / "latest.jsonl").exists()
    assert (tmp_path / "latest.summary.json").exists()
    assert list(tmp_path.glob("*.jsonl"))
    summary = json.loads((tmp_path / "latest.summary.json").read_text(encoding="utf-8"))
    assert "rag_scorecard" in summary
    assert "Grounded Extraction" in summary["rag_scorecard"]


def test_eval_cli_run_threshold_failure_exits_two(tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "eval",
            "run",
            "core",
            "--output",
            str(tmp_path),
            "--min-completion",
            "1.01",
        ],
    )

    assert result.exit_code == 2
    assert "Gate failed" in result.stdout


def test_eval_cli_run_rag_threshold_failure_exits_two(tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "eval",
            "run",
            "core",
            "--output",
            str(tmp_path),
            "--min-rag-grounding",
            "1.01",
        ],
    )

    assert result.exit_code == 2
    assert "RAG gate failed" in result.stdout


def test_eval_cli_run_rag_threshold_passes(tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "eval",
            "run",
            "core",
            "--output",
            str(tmp_path),
            "--min-rag-grounding",
            "1.0",
            "--min-rag-refusal",
            "1.0",
            "--min-rag-conflict",
            "1.0",
            "--min-rag-multihop",
            "1.0",
        ],
    )

    assert result.exit_code == 0


def test_eval_cli_run_unknown_suite_fails(tmp_path) -> None:
    result = runner.invoke(app, ["eval", "run", "missing", "--output", str(tmp_path)])

    assert result.exit_code == 1
    assert "No eval scenarios found" in result.stdout


def test_eval_cli_loads_custom_scenario_file(tmp_path) -> None:
    scenario_file = tmp_path / "scenario.json"
    scenario = dict(DEFAULT_SCENARIOS[0])
    scenario["id"] = "custom_cron"
    scenario["suite"] = "custom"
    scenario_file.write_text(json.dumps(scenario), encoding="utf-8")

    result = runner.invoke(
        app,
        ["eval", "run", "custom", "--path", str(scenario_file), "--output", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert "Agent Eval Report: custom" in result.stdout


def test_eval_compare_renders_delta(tmp_path) -> None:
    runner_impl = EvalRunner()
    report = runner_impl.run(load_scenarios(), suite="core")
    baseline_path, _ = runner_impl.write_report(report, tmp_path)
    current_path = tmp_path / "latest.jsonl"

    result = runner.invoke(app, ["eval", "compare", str(baseline_path), str(current_path)])

    assert result.exit_code == 0
    assert "Agent Eval Compare" in result.stdout
    assert "task_completion_rate" in result.stdout


def test_eval_compare_regression_gate_passes_for_same_run(tmp_path) -> None:
    runner_impl = EvalRunner()
    report = runner_impl.run(load_scenarios(), suite="core")
    baseline_path, _ = runner_impl.write_report(report, tmp_path)
    current_path = tmp_path / "latest.jsonl"

    result = runner.invoke(
        app,
        [
            "eval",
            "compare",
            str(baseline_path),
            str(current_path),
            "--max-regression",
            "0",
        ],
    )

    assert result.exit_code == 0


def test_eval_compare_rag_regression_gate_fails(tmp_path) -> None:
    runner_impl = EvalRunner()
    report = runner_impl.run(load_scenarios(), suite="core")
    baseline_path, _ = runner_impl.write_report(report, tmp_path)
    degraded = [EvalRunRecord.from_dict(record.to_dict()) for record in report.records]
    for record in degraded:
        if record.task_type == "rag_multi_hop_join":
            record.status = "failed"
            record.metrics["task_completion"] = 0.0
            record.scores["completion_score"] = 0.0
    current_path = tmp_path / "degraded.jsonl"
    current_path.write_text(
        "\n".join(json.dumps(record.to_dict(), ensure_ascii=False) for record in degraded) + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "eval",
            "compare",
            str(baseline_path),
            str(current_path),
            "--max-rag-regression",
            "0",
        ],
    )

    assert result.exit_code == 2
    assert "RAG regression failed" in result.stdout


def test_detect_regressions_flags_rate_drop() -> None:
    report = EvalRunner().run(load_scenarios(), suite="core")
    degraded = [EvalRunRecord.from_dict(record.to_dict()) for record in report.records]
    degraded[0].status = "failed"
    degraded[0].metrics["task_completion"] = 0.0

    regressions = detect_regressions(report.records, degraded, max_drop=0.0)

    assert any(row["metric"] == "task_completion_rate" for row in regressions)


def test_detect_rag_regressions_flags_capability_drop() -> None:
    report = EvalRunner().run(load_scenarios(), suite="core")
    degraded = [EvalRunRecord.from_dict(record.to_dict()) for record in report.records]
    for record in degraded:
        if record.task_type == "rag_single_source_extract":
            record.status = "failed"
            record.metrics["task_completion"] = 0.0
            record.scores["completion_score"] = 0.0

    regressions = detect_rag_regressions(report.records, degraded, max_drop=0.0)

    assert any(row["metric"] == "Grounded Extraction" for row in regressions)


def test_render_compare_markdown_handles_same_run() -> None:
    report = EvalRunner().run(load_scenarios(), suite="core")

    markdown = render_compare_markdown(report.records, report.records)

    assert "Agent Eval Compare" in markdown
    assert "RAG Capability Compare" in markdown
    assert "+0.0 pp" in markdown


def test_compare_rag_scorecards_returns_capability_rows() -> None:
    report = EvalRunner().run(load_scenarios(), suite="core")

    rows = compare_rag_scorecards(report.records, report.records)

    assert any(row["metric"] == "Grounded Extraction" for row in rows)
    assert any(row["metric"] == "Multi-Hop Synthesis" for row in rows)


def test_markdown_report_includes_judge_reason_for_failures() -> None:
    scenario = next(item for item in load_scenarios() if item.id == "email_check_now")
    case = materialize_cases(scenario)[0]
    record = judge_case(
        case,
        EvalObservation(
            final_response="scheduled only",
            tool_trace=[{"name": "cron.create", "success": True}],
            final_state={"schedule_created": True, "email_accounts_checked": ["qq", "163"]},
            notified_user=False,
        ),
    )

    report = EvalReport(
        suite="core",
        run_id="r1",
        records=[record],
        summary=aggregate([record]),
    )
    markdown = render_markdown(report)

    assert "missing_tool_use" in markdown
    assert "state_mismatch" in markdown


def test_judge_accepts_email_monitor_setup_without_immediate_check() -> None:
    scenario = next(item for item in load_scenarios() if item.id == "email_monitor_setup")
    case = materialize_cases(scenario)[0]
    observation = EvalObservation(
        final_response="scheduled",
        tool_trace=[{"name": "cron.create", "success": True}],
        final_state={
            "schedule_created": True,
            "email_accounts_checked": ["qq", "163"],
            "monitor_job_mentions_email": True,
            "monitor_job_mentions_importance": True,
            "monitor_job_mentions_silence": True,
        },
    )

    record = judge_case(case, observation)

    assert record.status == "success"


def test_judge_rejects_email_check_now_when_only_schedule_created() -> None:
    scenario = next(item for item in load_scenarios() if item.id == "email_check_now")
    case = materialize_cases(scenario)[0]
    observation = EvalObservation(
        final_response="scheduled only",
        tool_trace=[{"name": "cron.create", "success": True}],
        final_state={
            "schedule_created": True,
            "email_accounts_checked": ["qq", "163"],
        },
        notified_user=False,
    )

    record = judge_case(case, observation)

    assert record.status == "failed"
    assert "missing_tool_use" in record.failure_modes
    assert "state_mismatch" in record.failure_modes
