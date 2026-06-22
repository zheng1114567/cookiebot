from nanobot.evals.agent_executor import AgentLoopExecutor
from nanobot.evals.models import EvalCase


def _case(task_type: str) -> EvalCase:
    return EvalCase(
        scenario_id=task_type,
        suite="core",
        task_type=task_type,
        variant_id="v00",
        user_goal="test",
        expected={},
    )


def test_build_prompt_for_email_monitor_setup_mentions_cron_and_both_accounts() -> None:
    prompt = AgentLoopExecutor._build_prompt(_case("email_monitor_setup"))

    assert "Create a recurring cron-based monitor" in prompt
    assert "qq and 163" in prompt
    assert "stay silent" in prompt


def test_build_prompt_for_email_check_now_requires_immediate_check() -> None:
    prompt = AgentLoopExecutor._build_prompt(_case("email_check_now"))

    assert "Use the real email tool right now" in prompt
    assert "instead of only scheduling a future monitor" in prompt
    assert "not logged in" in prompt


def test_email_accounts_from_trace_and_schedule_collects_qq_and_163() -> None:
    accounts = AgentLoopExecutor._email_accounts_from_trace_and_schedule(
        [{"name": "email", "params": {"account": "qq"}}],
        "每天检查 163 邮箱，有重要邮件再通知我",
    )

    assert accounts == ["163", "qq"]


def test_build_prompt_for_recovery_missing_file_mentions_no_invention() -> None:
    prompt = AgentLoopExecutor._build_prompt(_case("recovery_missing_file"))

    assert "file read or adjacent file inspection" in prompt
    assert "do not invent its contents" in prompt


def test_build_prompt_for_recovery_web_timeout_mentions_retry() -> None:
    prompt = AgentLoopExecutor._build_prompt(_case("recovery_web_timeout"))

    assert "web fetch" in prompt
    assert "retry or access step" in prompt


def test_build_prompt_for_rag_mentions_local_documents_and_sources() -> None:
    prompt = AgentLoopExecutor._build_prompt(_case("rag_single_source_extract"))

    assert "local workspace file tools" in prompt
    assert "Include the source file names" in prompt


def test_build_prompt_for_missing_context_rag_forbids_guessing() -> None:
    prompt = AgentLoopExecutor._build_prompt(_case("rag_insufficient_context_refusal"))

    assert "does not provide it" in prompt
    assert "Do not guess" in prompt


def test_build_prompt_for_conflicting_rag_requires_calling_out_conflict() -> None:
    prompt = AgentLoopExecutor._build_prompt(_case("rag_conflict_resolution"))

    assert "documents disagree" in prompt
    assert "call out the conflict" in prompt


def test_build_prompt_for_irrelevant_context_rag_mentions_distractors() -> None:
    prompt = AgentLoopExecutor._build_prompt(_case("rag_irrelevant_context_filter"))

    assert "Ignore distractor content" in prompt
    assert "deployment document" in prompt


def test_build_prompt_for_multi_hop_rag_mentions_joining_sources() -> None:
    prompt = AgentLoopExecutor._build_prompt(_case("rag_multi_hop_join"))

    assert "Join the answer across the two sources" in prompt
    assert "both source file names" in prompt


def test_paths_read_from_trace_normalizes_paths() -> None:
    paths = AgentLoopExecutor._paths_read_from_trace([
        {"name": "read_file", "params": {"path": "docs\\release_notes.md"}},
        {"name": "list_dir", "params": {"path": "docs"}},
    ])

    assert paths == ["docs", "docs/release_notes.md"]


def test_blocked_email_accounts_from_trace_detects_login_failure() -> None:
    blocked = AgentLoopExecutor._blocked_email_accounts_from_trace([
        {
            "name": "email",
            "params": {"account": "qq"},
            "result_preview": "Error: QQ login failed, IMAP unavailable",
        },
        {
            "name": "email",
            "params": {"account": "163"},
            "result_preview": "Found 1 email(s)",
        },
    ])

    assert blocked == ["qq"]
