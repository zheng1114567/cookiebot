"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.attachments import preprocess_inbound_attachments
from nanobot.agent.context import ContextBuilder
from nanobot.agent.context_budget import ContextBudget
from nanobot.agent.graph import create_agent_graph
from nanobot.agent.memory import MemoryConsolidator
from nanobot.agent.middleware import OutboundDefaultsMiddleware, OutboundMiddleware
from nanobot.agent.skills import BUILTIN_SKILLS_DIR
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.email import EmailTool
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.memory_feedback import FeedbackMemoryTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.session.manager import Session, SessionManager
from nanobot.utils.helpers import ensure_dir, estimate_prompt_tokens_chain

if TYPE_CHECKING:
    from nanobot.config.schema import ChannelsConfig, ExecToolConfig, SandboxConfig, WebSearchConfig
    from nanobot.cron.service import CronService


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    _TOOL_RESULT_MAX_CHARS = 16_000
    _STOP_ALIASES = frozenset({"停", "停止", "取消", "别做了", "stop", "cancel"})
    _INTERRUPT_PREFIXES = (
        "停，",
        "停,",
        "停止，",
        "停止,",
        "取消，",
        "取消,",
        "别做了，",
        "别做了,",
        "先停",
        "先别",
        "stop ",
        "cancel ",
        "/interrupt ",
    )

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 40,
        turn_timeout_s: int | None = 600,
        context_window_tokens: int = 65_536,
        embedding_model: str | None = None,
        telemetry_enabled: bool = True,
        web_search_config: WebSearchConfig | None = None,
        web_proxy: str | None = None,
        exec_config: ExecToolConfig | None = None,
        tool_timeout: int | None = 120,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
        sandbox_config: "SandboxConfig | None" = None,
    ):
        from nanobot.config.schema import ExecToolConfig, SandboxConfig, WebSearchConfig
        from nanobot.security.sandbox import ApprovalManager, SandboxPolicy

        self.bus = bus
        self.channels_config = channels_config
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.turn_timeout_s = turn_timeout_s
        self.context_window_tokens = context_window_tokens
        self.embedding_model = embedding_model or None
        self.telemetry_enabled = telemetry_enabled
        self.web_search_config = web_search_config or WebSearchConfig()
        self.web_proxy = web_proxy
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace

        self.context = ContextBuilder(workspace)

        # Context budget management — dynamically trim system prompt to fit model context
        self._context_budget = ContextBudget(
            provider=provider,
            model=self.model,
            context_window_tokens=self.context_window_tokens,
            tool_definitions=lambda: self.tools.get_definitions(),
        )
        self.context.set_context_budget(self._context_budget)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry(default_timeout=tool_timeout)
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            web_search_config=self.web_search_config,
            web_proxy=web_proxy,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )

        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # session_key -> tasks
        self._background_tasks: list[asyncio.Task] = []
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._outbound_middlewares: list[OutboundMiddleware] = [OutboundDefaultsMiddleware()]

        # Sandbox (safety mode)
        sandbox_cfg = sandbox_config or SandboxConfig()
        self._sandbox_policy = SandboxPolicy(sandbox_cfg)
        self._approval_manager = ApprovalManager(
            policy=self._sandbox_policy,
            audit_log_path=Path(sandbox_cfg.audit_log).expanduser(),
            bus=self.bus,
        )

        self.memory_consolidator = MemoryConsolidator(
            workspace=workspace,
            provider=provider,
            model=self.model,
            sessions=self.sessions,
            context_window_tokens=context_window_tokens,
            build_messages=self.context.build_messages,
            get_tool_definitions=self.tools.get_definitions,
        )
        self._register_default_tools()
        self.telemetry_log_path = ensure_dir(self.workspace / "runtime") / "turns.jsonl"

    def register_outbound_middleware(self, middleware: OutboundMiddleware) -> None:
        """Register an outbound middleware."""
        self._outbound_middlewares.append(middleware)

    async def _publish_outbound(self, message: OutboundMessage) -> None:
        """Run outbound middleware stack, then emit to the bus."""
        current: OutboundMessage | None = message
        for middleware in self._outbound_middlewares:
            if current is None:
                return
            current = await middleware.before_send(current)
        if current is not None:
            await self.bus.publish_outbound(current)

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        extra_read = [BUILTIN_SKILLS_DIR] if allowed_dir else None
        self.tools.register(ReadFileTool(workspace=self.workspace, allowed_dir=allowed_dir, extra_allowed_dirs=extra_read))
        for cls in (WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
            path_append=self.exec_config.path_append,
            sandbox_backend=self.exec_config.sandbox_backend,
            sandbox_profile=self.exec_config.sandbox_profile,
            docker_image=self.exec_config.docker_image,
            docker_network_enabled=self.exec_config.docker_network_enabled,
            docker_memory_mb=self.exec_config.docker_memory_mb,
            docker_pids_limit=self.exec_config.docker_pids_limit,
            docker_user=self.exec_config.docker_user,
            docker_cpus=self.exec_config.docker_cpus,
            docker_seccomp_profile=self.exec_config.docker_seccomp_profile,
            workspace_root=str(self.workspace),
            deny_patterns=(
                self._sandbox_policy.effective_deny_patterns
                if self._sandbox_policy.config.mode != "off"
                else None
            ),
        ))
        self.tools.register(WebSearchTool(config=self.web_search_config, proxy=self.web_proxy))
        self.tools.register(WebFetchTool(proxy=self.web_proxy))
        if self.channels_config and (email_config := getattr(self.channels_config, "email", None)):
            self.tools.register(EmailTool(email_config))
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(SpawnTool(manager=self.subagents))
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))
        self.tools.register(FeedbackMemoryTool(self.context.memory))

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        self._mcp_connecting = True
        from nanobot.agent.tools.mcp import connect_mcp_servers
        try:
            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)
            self._mcp_connected = True
        except BaseException as e:
            logger.error("Failed to connect MCP servers (will retry next message): {}", e)
            if self._mcp_stack:
                try:
                    await self._mcp_stack.aclose()
                except Exception:
                    pass
                self._mcp_stack = None
        finally:
            self._mcp_connecting = False

    def _set_tool_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Update context for all tools that need routing info."""
        for name in ("message", "spawn", "cron"):
            if tool := self.tools.get(name):
                if hasattr(tool, "set_context"):
                    tool.set_context(channel, chat_id, *([message_id] if name == "message" else []))


    async def _run_graph(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> tuple[str | None, list[str], list[dict]]:
        """Run the agent using a compiled LangGraph StateGraph."""
        graph = create_agent_graph(
            provider=self.provider,
            tools=self.tools,
            model=self.model,
            max_iterations=self.max_iterations,
            workspace=self.workspace,
            on_progress=on_progress,
        )
        result = await graph.ainvoke(
            {"messages": initial_messages, "iterations": 0, "tools_used": []},
            config={"recursion_limit": (self.max_iterations + 5) * 3},
        )

        messages: list[dict] = result["messages"]

        # Extract final content from the last assistant message without tool calls
        final_content: str | None = None
        for m in reversed(messages):
            if (
                m.get("role") == "assistant"
                and m.get("content")
                and not m.get("tool_calls")
            ):
                final_content = m["content"]
                break

        # Check for error stored by llm_node (not persisted to messages, #1303)
        if final_content is None and result.get("error"):
            final_content = result["error"]

        if final_content is None and result["iterations"] >= self.max_iterations:
            logger.warning("Max iterations ({}) reached", self.max_iterations)
            final_content = (
                f"I reached the maximum number of tool call iterations ({self.max_iterations}) "
                "without completing the task. You can try breaking the task into smaller steps."
            )

        return final_content, result.get("tools_used", []), messages

    async def _embed_query(self, text: str) -> list[float] | None:
        """Best-effort query embedding for memory retrieval."""
        if not text.strip():
            return None
        embed = getattr(self.provider, "embed", None)
        if not callable(embed):
            return None
        try:
            result = await embed([text], model=self.embedding_model)
        except TypeError:
            try:
                result = await embed([text])
            except Exception:
                return None
        except Exception:
            return None
        if result and isinstance(result, list) and result[0]:
            return result[0]
        return None

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.warning("Error consuming inbound message: {}, continuing...", e)
                continue

            cmd = msg.content.strip().lower()
            if self._is_recall_control(msg):
                await self._handle_recall_control(msg)
            elif cmd == "/stop" or cmd in self._STOP_ALIASES:
                await self._handle_stop(msg)
            elif cmd == "/restart":
                await self._handle_restart(msg)
            elif self._approval_manager.has_pending(msg.session_key):
                # Sandbox approval response — resolve directly without dispatch
                if self._approval_manager.is_approval_response(msg.content):
                    approved = cmd in ("yes", "y", "approve", "ok", "1")
                    self._approval_manager.respond(msg.session_key, approved)
                else:
                    await self._publish_outbound(OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content="An approval request is pending. Please reply 'yes' or 'no'.",
                    ))
            else:
                if self._is_interrupt_request(msg.content):
                    await self._cancel_active_session(msg.session_key)
                    # Acknowledge interrupt without dispatching to LLM
                    await self._publish_outbound(OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content="Stopped. Send a new message to continue.",
                    ))
                    continue
                # If the session already has queued messages, skip this one
                # and let the latest message be processed (avoid backlog buildup).
                if self._has_queued_for_session(msg.session_key):
                    logger.debug("Skipping queued message for {} — newer message pending", msg.session_key)
                    continue
                task = asyncio.create_task(self._dispatch(msg))
                self._active_tasks.setdefault(msg.session_key, []).append(task)
                task.add_done_callback(lambda t, k=msg.session_key: self._active_tasks.get(k, []) and self._active_tasks[k].remove(t) if t in self._active_tasks.get(k, []) else None)

    def _is_interrupt_request(self, content: str) -> bool:
        """Return True when a message explicitly redirects the current task."""
        text = content.strip().lower()
        return any(text.startswith(prefix) for prefix in self._INTERRUPT_PREFIXES)

    @staticmethod
    def _is_recall_control(msg: InboundMessage) -> bool:
        """Return True for channel-level recall/delete control events."""
        control = (msg.metadata or {}).get("_control")
        return control in {"recall", "delete", "message_recall", "message_delete"}

    async def _handle_recall_control(self, msg: InboundMessage) -> None:
        """Treat message recall/delete as a silent stop for the current session,
        and remove the recalled message from session history."""
        cancelled = await self._cancel_active_session(msg.session_key)
        sub_cancelled = await self.subagents.cancel_by_session(msg.session_key)
        total = cancelled + sub_cancelled
        if total:
            logger.info(
                "Stopped {} task(s) for {} due to {} control",
                total,
                msg.session_key,
                (msg.metadata or {}).get("_control"),
            )

        # Remove the recalled message from session history
        recall_msg_id = (msg.metadata or {}).get("message_id") or msg.content
        if recall_msg_id:
            session = self.sessions.get_or_create(msg.session_key)
            removed = session.remove_message_by_id(recall_msg_id)
            if removed:
                logger.info("Removed recalled message '{}' from session history", recall_msg_id)
                self.sessions.save(session)

    def _has_queued_for_session(self, session_key: str) -> bool:
        """Check if there are more inbound messages for the same session.

        Used to skip duplicate queued messages when the user sends multiple
        messages in quick succession — only the latest one will be processed.
        """
        try:
            # Peek at the next queued message without removing it
            remaining = self.bus.inbound_size
            if remaining <= 0:
                return False
            # We can't peek into asyncio.Queue, so we check if there are
            # queued items and the current dispatch hasn't started yet
            # (active_tasks for this session has pending entries).
            active = self._active_tasks.get(session_key, [])
            if len(active) > 0:
                return True
        except Exception:
            pass
        return False

    async def _cancel_active_session(self, session_key: str) -> int:
        """Cancel active and queued dispatch tasks for one session."""
        tasks = self._active_tasks.pop(session_key, [])
        cancelled = 0
        for task in list(tasks):
            if not task.done():
                task.cancel()
                cancelled += 1
        if cancelled:
            logger.info("Interrupted {} task(s) for {}", cancelled, session_key)
            await asyncio.sleep(0)
        return cancelled

    async def _handle_stop(self, msg: InboundMessage) -> None:
        """Cancel all active tasks and subagents for the session."""
        tasks = self._active_tasks.pop(msg.session_key, [])
        cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        sub_cancelled = await self.subagents.cancel_by_session(msg.session_key)
        total = cancelled + sub_cancelled
        content = f"Stopped {total} task(s)." if total else "No active task to stop."
        await self._publish_outbound(OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=content,
        ))

    async def _handle_restart(self, msg: InboundMessage) -> None:
        """Restart the process in-place via os.execv."""
        await self._publish_outbound(OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content="Restarting...",
        ))

        async def _do_restart():
            await asyncio.sleep(1)
            # Use -m nanobot instead of sys.argv[0] for Windows compatibility
            # (sys.argv[0] may be just "nanobot" without full path on Windows)
            os.execv(sys.executable, [sys.executable, "-m", "nanobot"] + sys.argv[1:])

        asyncio.create_task(_do_restart())

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Process a message under its session lock."""
        async with self._get_session_lock(msg.session_key):
            try:
                if self.turn_timeout_s and self.turn_timeout_s > 0:
                    response = await asyncio.wait_for(
                        self._process_message(msg),
                        timeout=self.turn_timeout_s,
                    )
                else:
                    response = await self._process_message(msg)
                if response is not None:
                    await self._publish_outbound(response)
                elif msg.channel == "cli":
                    await self._publish_outbound(OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content="", metadata=msg.metadata or {},
                    ))
            except asyncio.CancelledError:
                logger.info("Task cancelled for session {}", msg.session_key)
                raise
            except asyncio.TimeoutError:
                logger.warning(
                    "Task timed out for session {} after {}s",
                    msg.session_key,
                    self.turn_timeout_s,
                )
                await self._publish_outbound(OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=(
                        f"Task timed out after {self.turn_timeout_s} seconds. "
                        "The current turn was cancelled; send a new message to continue."
                    ),
                    metadata=msg.metadata or {},
                    correlation_id=msg.event_id,
                    kind="error",
                ))
            except Exception:
                logger.exception("Error processing message for session {}", msg.session_key)
                await self._publish_outbound(OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Sorry, I encountered an error.",
                ))

    async def close_mcp(self) -> None:
        """Drain pending work, close tools/subagents, then close MCP connections."""
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()
        try:
            await self.subagents.aclose()
        except Exception:
            pass
        try:
            await self.tools.aclose()
        except Exception:
            pass
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None

    def _schedule_background(self, coro) -> None:
        """Schedule a coroutine as a tracked background task (drained on shutdown)."""
        task = asyncio.create_task(coro)
        self._background_tasks.append(task)
        task.add_done_callback(self._discard_background_task)

    def _discard_background_task(self, task: asyncio.Task) -> None:
        """Remove a completed background task if it is still tracked."""
        try:
            self._background_tasks.remove(task)
        except ValueError:
            pass

    def _get_session_lock(self, session_key: str) -> asyncio.Lock:
        """Return a lock that serializes work for one session only."""
        lock = self._session_locks.get(session_key)
        if lock is None:
            lock = asyncio.Lock()
            self._session_locks[session_key] = lock
        return lock

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response."""
        turn_started = time.perf_counter()
        # System messages: parse origin from chat_id ("channel:chat_id")
        if msg.channel == "system":
            channel, chat_id = (msg.chat_id.split(":", 1) if ":" in msg.chat_id
                                else ("cli", msg.chat_id))
            logger.info("Processing system message from {}", msg.sender_id)
            key = f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(key)
            await self.memory_consolidator.maybe_consolidate_by_tokens(session)
            self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"))
            history = session.get_history(max_messages=0)
            # Subagent results should be assistant role, other system messages use user role
            current_role = "assistant" if msg.sender_id == "subagent" else "user"
            messages = self.context.build_messages(
                history=history,
                current_message=msg.content, channel=channel, chat_id=chat_id,
                current_role=current_role,
                available_budget=self._context_budget.available,
            )
            self._approval_manager.bind(
                session_key=key, channel=channel, chat_id=chat_id,
            )
            self.tools.set_approval_handler(self._approval_manager.check_and_request)

            try:
                final_content, _, all_msgs = await self._run_graph(messages)
            finally:
                self.tools.set_approval_handler(None)
            self._save_turn(session, all_msgs, 1 + len(history))
            self.sessions.save(session)
            self._schedule_background(self.memory_consolidator.maybe_consolidate_by_tokens(session))
            self._record_turn_telemetry(
                session_key=key,
                channel=channel,
                chat_id=chat_id,
                prompt_messages=messages,
                tools_used=[],
                final_content=final_content,
                started_at=turn_started,
                kind="system",
            )
            return OutboundMessage(channel=channel, chat_id=chat_id,
                                  content=final_content or "Background task completed.")

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)

        # Slash commands
        cmd = msg.content.strip().lower()
        if cmd == "/new":
            snapshot = session.messages[session.last_consolidated:]
            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)

            if snapshot:
                self._schedule_background(self.memory_consolidator.archive_messages(snapshot))

            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="New session started.")
        if cmd == "/help":
            lines = [
                "🐈 cookiebot commands:",
                "/new — Start a new conversation",
                "/stop — Stop the current task",
                "/restart — Restart the bot",
                "/help — Show available commands",
            ]
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content="\n".join(lines),
            )
        await self.memory_consolidator.maybe_consolidate_by_tokens(session)

        current_content = msg.content
        current_media = msg.media if msg.media else None
        if current_media:
            attachment_result = await preprocess_inbound_attachments(
                current_media,
                supports_multimodal=self.provider.supports_multimodal(self.model),
            )
            if attachment_result.unsupported_message:
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=attachment_result.unsupported_message,
                    metadata=msg.metadata or {},
                    correlation_id=msg.event_id,
                    kind="error",
                )
            if attachment_result.content_suffix:
                current_content = f"{current_content}{attachment_result.content_suffix}"
            current_media = attachment_result.media or None

        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        history = session.get_history(max_messages=0)

        # Short-term running summary: when conversation is long, summarize early
        # turns so they don't consume the whole context window.
        running_summary = session.metadata.get("running_summary") or None
        short_history = history
        if running_summary and len(history) > 12:
            short_history = history[-12:]
        elif len(history) > 20:
            # Schedule a background summary update for next turn
            self._schedule_background(
                self.memory_consolidator.update_running_summary(session, history)
            )

        query_vector = await self._embed_query(current_content)

        initial_messages = self.context.build_messages(
            history=short_history,
            current_message=current_content,
            media=current_media,
            channel=msg.channel, chat_id=msg.chat_id,
            running_summary=running_summary,
            query_vector=query_vector,
            available_budget=self._context_budget.available,
        )

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self._publish_outbound(OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=content,
                metadata=meta,
                correlation_id=msg.event_id,
                kind="tool_hint" if tool_hint else "progress",
            ))

        # Bind sandbox approval context for this turn
        self._approval_manager.bind(
            session_key=key, channel=msg.channel, chat_id=msg.chat_id,
        )
        self.tools.set_approval_handler(self._approval_manager.check_and_request)

        try:
            final_content, _, all_msgs = await self._run_graph(
                initial_messages, on_progress=on_progress or _bus_progress,
            )
        finally:
            self.tools.set_approval_handler(None)

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        self._save_turn(session, all_msgs, 1 + len(history))
        self.sessions.save(session)
        self._schedule_background(self.memory_consolidator.maybe_consolidate_by_tokens(session))
        tools_used = [
            m.get("name")
            for m in all_msgs
            if m.get("role") == "tool" and isinstance(m.get("name"), str)
        ]
        self._record_turn_telemetry(
            session_key=key,
            channel=msg.channel,
            chat_id=msg.chat_id,
            prompt_messages=initial_messages,
            tools_used=tools_used,
            final_content=final_content,
            started_at=turn_started,
            kind="user",
        )

        if (
            (mt := self.tools.get("message"))
            and isinstance(mt, MessageTool)
            and mt.sent_in_current_turn()
        ):
            return None

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata=msg.metadata or {},
            correlation_id=msg.event_id,
        )

    def _record_turn_telemetry(
        self,
        *,
        session_key: str,
        channel: str,
        chat_id: str,
        prompt_messages: list[dict[str, Any]],
        tools_used: list[str],
        final_content: str | None,
        started_at: float,
        kind: str,
    ) -> None:
        """Append a small JSONL record for local debugging and performance inspection."""
        if not self.telemetry_enabled:
            return
        try:
            prompt_tokens, counter = estimate_prompt_tokens_chain(
                self.provider,
                self.model,
                prompt_messages,
                self.tools.get_definitions(),
            )
            entry = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "session_key": session_key,
                "channel": channel,
                "chat_id": chat_id,
                "kind": kind,
                "duration_ms": int((time.perf_counter() - started_at) * 1000),
                "prompt_tokens_estimate": prompt_tokens,
                "prompt_token_counter": counter,
                "prompt_message_count": len(prompt_messages),
                "tools_used": tools_used,
                "response_chars": len(final_content or ""),
            }
            with open(self.telemetry_log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _save_turn(self, session: Session, messages: list[dict], skip: int) -> None:
        """Save new-turn messages into session, truncating large tool results."""
        from datetime import datetime
        for m in messages[skip:]:
            entry = dict(m)
            role, content = entry.get("role"), entry.get("content")
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue  # skip empty assistant messages — they poison session context
            if role == "tool" and isinstance(content, str) and len(content) > self._TOOL_RESULT_MAX_CHARS:
                entry["content"] = content[:self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
            elif role == "user":
                if isinstance(content, str) and content.startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                    # Strip the runtime-context prefix, keep only the user text.
                    parts = content.split("\n\n", 1)
                    if len(parts) > 1 and parts[1].strip():
                        entry["content"] = parts[1]
                    else:
                        continue
                if isinstance(content, list):
                    filtered = []
                    for c in content:
                        if c.get("type") == "text" and isinstance(c.get("text"), str) and c["text"].startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                            continue  # Strip runtime context from multimodal messages
                        if (c.get("type") == "image_url"
                                and c.get("image_url", {}).get("url", "").startswith("data:image/")):
                            path = (c.get("_meta") or {}).get("path", "")
                            placeholder = f"[image: {path}]" if path else "[image]"
                            filtered.append({"type": "text", "text": placeholder})
                        else:
                            filtered.append(c)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
        session.updated_at = datetime.now()

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """Process a message directly (for CLI or cron usage)."""
        await self._connect_mcp()
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)
        if self.turn_timeout_s and self.turn_timeout_s > 0:
            response = await asyncio.wait_for(
                self._process_message(msg, session_key=session_key, on_progress=on_progress),
                timeout=self.turn_timeout_s,
            )
        else:
            response = await self._process_message(msg, session_key=session_key, on_progress=on_progress)
        return response.content if response else ""
