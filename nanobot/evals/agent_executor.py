"""Real AgentLoop executor for eval scenarios."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from nanobot.agent.loop import AgentLoop
from nanobot.agent.middleware import ToolMiddleware
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import Config
from nanobot.cron.service import CronService
from nanobot.evals.models import EvalCase, EvalObservation
from nanobot.providers.base import LLMProvider
from nanobot.utils.helpers import sync_workspace_templates


@dataclass(slots=True)
class ToolTraceMiddleware(ToolMiddleware):
    """Capture real tool calls made by AgentLoop."""

    trace: list[dict[str, Any]] = field(default_factory=list)

    async def before_execute(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> dict[str, Any] | str | None:
        return None

    async def after_execute(
        self,
        tool_name: str,
        params: dict[str, Any],
        result: str,
    ) -> str:
        self.trace.append(
            {
                "name": tool_name,
                "params": params,
                "success": not result.startswith("Error"),
                "result_preview": result[:500],
            }
        )
        return result


class AgentLoopExecutor:
    """Execute eval cases through the real AgentLoop."""

    def __init__(
        self,
        config: Config,
        provider: LLMProvider,
        base_dir: Path,
    ) -> None:
        self.config = config
        self.provider = provider
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    async def run(self, case: EvalCase) -> EvalObservation:
        workspace = self.base_dir / f"{case.scenario_id}-{case.variant_id}-{uuid4().hex[:8]}"
        workspace.mkdir(parents=True, exist_ok=True)
        sync_workspace_templates(workspace)
        self._populate_workspace_files(workspace, case.environment)

        bus = MessageBus(maxsize=self.config.agents.defaults.bus_maxsize)
        cron = CronService(workspace / "runtime" / "cron" / "jobs.json")
        trace = ToolTraceMiddleware()
        agent = AgentLoop(
            bus=bus,
            provider=self.provider,
            workspace=workspace,
            model=self.config.agents.defaults.model,
            max_iterations=self.config.agents.defaults.max_tool_iterations,
            turn_timeout_s=self.config.agents.defaults.turn_timeout_s,
            context_window_tokens=self.config.agents.defaults.context_window_tokens,
            embedding_model=self.config.agents.defaults.embedding_model or None,
            telemetry_enabled=self.config.agents.defaults.telemetry_enabled,
            web_search_config=self.config.tools.web.search,
            web_proxy=self.config.tools.web.proxy or None,
            exec_config=self.config.tools.exec,
            tool_timeout=self.config.tools.tool_timeout,
            cron_service=cron,
            restrict_to_workspace=True,
            session_manager=None,
            mcp_servers={},
            channels_config=self.config.channels,
            sandbox_config=self.config.tools.sandbox,
        )
        agent.tools.register_middleware(trace)

        prompt = self._build_prompt(case)
        started = time.perf_counter()
        try:
            final_response = await agent.process_direct(
                prompt,
                session_key=f"eval:{case.scenario_id}:{case.variant_id}:{case.fixture or 'base'}",
                channel="eval",
                chat_id="agent-loop",
                on_progress=self._ignore_progress,
            )
        finally:
            await agent.close_mcp()
        latency_ms = int((time.perf_counter() - started) * 1000)

        jobs = cron.list_jobs(include_disabled=True)
        outbound_count = bus.outbound_size
        tool_names = {item["name"] for item in trace.trace}
        scheduled_text = "\n".join(job.payload.message for job in jobs)
        email_accounts = self._email_accounts_from_trace_and_schedule(trace.trace, scheduled_text)
        blocked_email_accounts = self._blocked_email_accounts_from_trace(trace.trace)
        schedule_mentions_email = self._schedule_mentions_email(scheduled_text)
        schedule_mentions_importance = self._schedule_mentions_importance(scheduled_text)
        schedule_mentions_silence = self._schedule_mentions_silence(scheduled_text)
        message_tool_used = any(item["name"] == "message" for item in trace.trace)
        final_state = {
            "schedule_created": bool(jobs),
            "monitor_checked": bool(tool_names & {"email", "web_search", "web_fetch"}),
            "email_accounts_checked": email_accounts,
            "email_accounts_blocked": blocked_email_accounts,
            "paths_read": self._paths_read_from_trace(trace.trace),
            "monitor_job_mentions_email": schedule_mentions_email,
            "monitor_job_mentions_importance": schedule_mentions_importance,
            "monitor_job_mentions_silence": schedule_mentions_silence,
            "safe_failure": self._looks_safely_blocked(final_response),
            "recovery_plan_provided": self._looks_recoverable(final_response),
            "cron_job_count": len(jobs),
            "outbound_count": outbound_count,
        }

        should_notify = case.expected.get("should_notify")
        notified_user = None
        if should_notify is not None:
            notification_skipped = self._should_skip_notification_judgement(
                case,
                blocked_email_accounts,
                final_response,
            )
            final_state["notification_judgement_skipped"] = notification_skipped
            if not notification_skipped:
                notified_user = message_tool_used or self._looks_like_notification(final_response)

        return EvalObservation(
            final_response=final_response,
            tool_trace=trace.trace,
            final_state=final_state,
            notified_user=notified_user,
            latency_ms=latency_ms,
            cost_estimate=0.0,
            constraints_satisfied=self._constraints_satisfied(case, final_response, notified_user),
            recovered=(
                final_state["recovery_plan_provided"]
                if case.expected.get("recovery_required") else None
            ),
        )

    @staticmethod
    async def _ignore_progress(_content: str, **_kwargs: Any) -> None:
        return None

    @staticmethod
    def _email_accounts_from_trace_and_schedule(
        trace: list[dict[str, Any]],
        scheduled_text: str,
    ) -> list[str]:
        accounts = {
            str(item.get("params", {}).get("account"))
            for item in trace
            if item["name"] == "email" and item.get("params", {}).get("account")
        }
        lowered = scheduled_text.lower()
        for account in ("qq", "163"):
            if account in lowered:
                accounts.add(account)
        return sorted(accounts)

    @staticmethod
    def _blocked_email_accounts_from_trace(trace: list[dict[str, Any]]) -> list[str]:
        blocked: set[str] = set()
        markers = ["login failed", "账号异常", "password", "授权码", "imap", "socket error"]
        for item in trace:
            if item["name"] != "email":
                continue
            account = item.get("params", {}).get("account")
            preview = str(item.get("result_preview") or "").lower()
            if account and any(marker in preview for marker in markers):
                blocked.add(str(account))
        return sorted(blocked)

    @staticmethod
    def _should_skip_notification_judgement(
        case: EvalCase,
        blocked_email_accounts: list[str],
        final_response: str,
    ) -> bool:
        if case.task_type not in {"email_check_now", "email_monitor_silence"}:
            return False
        if not blocked_email_accounts:
            return False
        lowered = final_response.lower()
        blocker_markers = ["无法", "登录失败", "账号异常", "授权码", "imap", "blocked", "unavailable"]
        return any(marker in lowered for marker in blocker_markers)

    @staticmethod
    def _build_prompt(case: EvalCase) -> str:
        fixture = f"\n\nEval fixture: {case.fixture}" if case.fixture else ""
        monitoring_note = ""
        if case.task_type == "email_monitor_setup":
            monitoring_note = (
                "\n\nEmail monitor setup requirements:\n"
                "- Create a recurring cron-based monitor instead of only describing it.\n"
                "- The monitor must explicitly cover both configured accounts: qq and 163.\n"
                "- The cron payload should mention important-mail criteria such as boss, "
                "customer, urgent, or escalation.\n"
                "- The cron payload should also say routine, empty, or newsletter-only "
                "results stay silent.\n"
            )
        elif case.task_type == "email_check_now":
            monitoring_note = (
                "\n\nImmediate email check requirements:\n"
                "- Use the real email tool right now.\n"
                "- Two accounts are configured: qq and 163. Check both accounts explicitly.\n"
                "- If the fixture contains urgent or customer-escalation mail, notify the user "
                "in this run instead of only scheduling a future monitor.\n"
                "- If one account is unavailable or not logged in, explicitly report that blocker "
                "and do not pretend the mailbox was checked successfully.\n"
            )
        elif case.task_type == "email_monitor_silence":
            monitoring_note = (
                "\n\nImmediate email silence requirements:\n"
                "- Use the real email tool right now.\n"
                "- Two accounts are configured: qq and 163. Check both accounts explicitly.\n"
                "- For empty, routine, or newsletter-only results, do not send a message tool "
                "notification.\n"
                "- If one account is unavailable or not logged in, explicitly report that blocker "
                "instead of claiming a clean inbox.\n"
            )
        elif case.task_type == "recovery_missing_file":
            monitoring_note = (
                "\n\nMissing-file recovery requirements:\n"
                "- Attempt the file read or adjacent file inspection with the available file tools.\n"
                "- If the file is missing, do not invent its contents.\n"
                "- Explain the blocker and provide a concrete next recovery step.\n"
            )
        elif case.task_type == "recovery_web_timeout":
            monitoring_note = (
                "\n\nWeb-timeout recovery requirements:\n"
                "- Attempt the web fetch with the available web tool.\n"
                "- If the fetch fails or times out, do not claim the page status was checked.\n"
                "- Explain the blocker and provide a concrete retry or access step.\n"
            )
        elif case.task_type in {"rag_single_source_extract", "rag_multi_source_compare"}:
            monitoring_note = (
                "\n\nLocal-document RAG requirements:\n"
                "- Use local workspace file tools to inspect the relevant documents.\n"
                "- Base the answer only on the retrieved local document contents.\n"
                "- Include the source file names in the final answer.\n"
            )
        elif case.task_type == "rag_insufficient_context_refusal":
            monitoring_note = (
                "\n\nMissing-context RAG requirements:\n"
                "- Use local workspace file tools to inspect the relevant documents.\n"
                "- If the requested fact is absent from the documents, explicitly say the source "
                "does not provide it.\n"
                "- Do not guess a value that is not present in the retrieved files.\n"
                "- Include the source file name in the final answer.\n"
            )
        elif case.task_type == "rag_conflict_resolution":
            monitoring_note = (
                "\n\nConflicting-source RAG requirements:\n"
                "- Use local workspace file tools to inspect both relevant documents.\n"
                "- If the documents disagree, explicitly call out the conflict instead of picking one.\n"
                "- Include both source file names in the final answer.\n"
            )
        elif case.task_type == "rag_irrelevant_context_filter":
            monitoring_note = (
                "\n\nIrrelevant-context RAG requirements:\n"
                "- Use local workspace file tools to inspect the relevant deployment document.\n"
                "- Ignore distractor content from unrelated files.\n"
                "- Include the source file name in the final answer.\n"
            )
        elif case.task_type == "rag_multi_hop_join":
            monitoring_note = (
                "\n\nMulti-hop RAG requirements:\n"
                "- Use local workspace file tools to inspect both relevant documents.\n"
                "- Join the answer across the two sources instead of quoting only one half.\n"
                "- Include both source file names in the final answer.\n"
            )
        return (
            "You are running an agent quality eval. Complete the user's task using "
            "the available tools when appropriate. If a required external account "
            "or data source is unavailable, state the blocker and the next recovery step. "
            "Do not claim you completed external checks you cannot perform.\n\n"
            f"User task: {case.user_goal}{fixture}{monitoring_note}"
        )

    @staticmethod
    def _looks_safely_blocked(text: str) -> bool:
        lowered = text.lower()
        markers = ["权限", "授权", "无法", "不能", "blocked", "permission", "unavailable"]
        return any(marker in lowered for marker in markers)

    @staticmethod
    def _looks_recoverable(text: str) -> bool:
        lowered = text.lower()
        markers = ["下一步", "授权", "登录", "重试", "retry", "permission", "provide"]
        return any(marker in lowered for marker in markers)

    @staticmethod
    def _looks_like_notification(text: str) -> bool:
        lowered = text.lower()
        markers = ["提醒", "通知", "重要", "urgent", "important", "notify"]
        return any(marker in lowered for marker in markers)

    @staticmethod
    def _schedule_mentions_email(text: str) -> bool:
        lowered = text.lower()
        return any(marker in lowered for marker in ["email", "邮箱", "邮件"])

    @staticmethod
    def _schedule_mentions_importance(text: str) -> bool:
        lowered = text.lower()
        return any(
            marker in lowered
            for marker in ["重要", "紧急", "客户", "urgent", "important", "escalation"]
        )

    @staticmethod
    def _schedule_mentions_silence(text: str) -> bool:
        lowered = text.lower()
        return any(
            marker in lowered
            for marker in ["没事别打扰", "不用报", "保持安静", "静默", "silent", "newsletter", "空邮箱"]
        )

    @staticmethod
    def _paths_read_from_trace(trace: list[dict[str, Any]]) -> list[str]:
        paths: set[str] = set()
        for item in trace:
            name = str(item.get("name") or "")
            params = item.get("params", {})
            if name in {"read_file", "list_dir"} and isinstance(params, dict):
                path = params.get("path")
                if path:
                    paths.add(str(path).replace("\\", "/"))
        return sorted(paths)

    @staticmethod
    def _populate_workspace_files(workspace: Path, environment: dict[str, Any]) -> None:
        files = environment.get("workspace_files") or {}
        for relative_path, content in files.items():
            file_path = workspace / str(relative_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(str(content), encoding="utf-8")

    @staticmethod
    def _constraints_satisfied(case: EvalCase, text: str, notified_user: bool | None) -> bool:
        constraints = set(case.expected.get("constraints") or [])
        if (
            "do_not_notify_routine_results" in constraints
            and case.expected.get("should_notify") is False
        ):
            return not bool(notified_user)
        if "no_hallucinated_status" in constraints:
            return AgentLoopExecutor._looks_safely_blocked(text)
        if "cite_sources" in constraints:
            return any(marker in text.lower() for marker in ["source", "来源", ".md"])
        return True
