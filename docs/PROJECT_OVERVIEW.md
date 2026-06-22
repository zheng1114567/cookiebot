# cookiebot Project Overview

cookiebot is a lightweight personal AI assistant runtime built around a small Python agent core,
multi-channel messaging, persistent memory, and tool/MCP execution.

The internal Python package is still named `nanobot` for compatibility with the existing source
layout, but the project name and CLI entrypoint are `cookiebot`.

## What It Does

cookiebot can:

- Run as a local CLI assistant.
- Run as a gateway connected to chat platforms.
- Use LLM providers through LiteLLM and provider adapters.
- Execute native tools such as file operations, shell commands, web search/fetch, messaging, cron, and subagents.
- Connect MCP servers and expose MCP tools as agent tools.
- Maintain session history and layered memory.
- Stream observable work progress while the agent is running.
- Run shell execution through a persistent sandbox worker, with Docker as the hardened local default.

## Core Runtime

```text
user/channel
  -> InboundMessage
  -> MessageBus
  -> AgentLoop
  -> ContextBuilder + SessionManager + MemoryStore
  -> LangGraph LLM/tool loop
  -> ToolRegistry / MCP / native tools
  -> OutboundMessage
  -> channel.send()
```

## Important Modules

| Area | Path | Purpose |
| --- | --- | --- |
| Agent loop | `nanobot/agent/loop.py` | Turn orchestration, session locks, progress, timeout handling |
| Graph execution | `nanobot/agent/graph.py` | LangGraph LLM/tool loop |
| Memory | `nanobot/agent/memory.py` | Long-term memory, history archive, GraphRAG, consolidation |
| Sessions | `nanobot/session/manager.py` | JSONL session persistence |
| Tools | `nanobot/agent/tools/` | Native tool implementations and registry |
| MCP | `nanobot/agent/tools/mcp.py` | MCP server connection and tool wrapping |
| Channels | `nanobot/channels/` | Chat platform adapters |
| Bus | `nanobot/bus/` | Inbound/outbound event structures and queues |
| Config | `nanobot/config/schema.py` | Pydantic runtime configuration |
| CLI | `nanobot/cli/commands.py` | `cookiebot` commands |

## Memory

Memory is split across:

- `MEMORY.md`: long-term user profile and stable facts.
- `HISTORY.md`: append-only conversation summaries.
- `graph.json`: GraphRAG medium-term memory.
- `projects/*.md`: project-level profiles.
- `sessions/*.jsonl`: raw session turns.

Recent memory improvements:

- Query embeddings are used during retrieval.
- Irrelevant graph fallback results were removed.
- Graph nodes are merged instead of replaced.
- Medium-term graph nodes are grouped hierarchically under project containers or daily conversation categories.
- Memory and session writes use atomic replace.

## State and Timeout Design

State is layered:

- `AgentState` is graph-local state.
- `AgentLoop` owns outer task state and session locks.
- Tool routing state is task-local through `ContextVar`.

Timeout boundaries:

- Whole turn: `agents.defaults.turnTimeoutS`
- Native tools: `tools.toolTimeout`
- Shell: `tools.exec.timeout`
- MCP tools: `tools.mcpServers.<name>.toolTimeout`
- Channel sends: `channels.sendTimeoutS`

Timeouts return recoverable errors where possible so the agent can continue or the user can retry.

## Communication

Messages carry lifecycle fields:

- `event_id`
- `correlation_id`
- `attempt`
- outbound `kind`

Progress updates are streamed as outbound events. The runtime reports observable work steps such as
tool starts, completions, errors, and timeouts without exposing hidden model reasoning.

## Development Entry Points

```bash
pip install -e .
cookiebot --help
cookiebot onboard
cookiebot agent
cookiebot gateway
```

Focused tests:

```bash
python -m pytest tests/test_graph.py tests/test_tool_validation.py tests/test_task_cancel.py
```
