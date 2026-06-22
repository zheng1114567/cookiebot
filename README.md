# cookiebot

cookiebot is a lightweight personal AI assistant framework. It keeps the original `nanobot`
Python package layout internally, but the product and CLI entrypoint are now `cookiebot`.

The project focuses on a practical agent runtime:

- Multi-channel chat gateway for CLI, Telegram, Feishu, Slack, Discord, WhatsApp, Email, QQ, Matrix, WeCom, DingTalk, and Mochat.
- Provider-agnostic LLM access through LiteLLM and custom provider adapters.
- Built-in tools for files, shell execution, web search/fetch, messaging, cron jobs, MCP tools, and subagents.
- Persistent sessions and memory with long-term memory, history archive, project profiles, and GraphRAG retrieval.
- Observable agent execution with progress updates for tool starts, completions, errors, and timeouts.
- Timeout boundaries for native tools, MCP tools, channel sends, and whole agent turns.
- Workspace-local skills and templates for customization.

## Current Architecture

```text
nanobot/
├── agent/          # Agent loop, LangGraph execution, memory, skills, tools
├── bus/            # Inbound/outbound message events and queues
├── channels/       # Chat platform adapters and channel plugin discovery
├── cli/            # Typer CLI commands exposed as `cookiebot`
├── config/         # Pydantic config schema and loading
├── cron/           # Scheduled task storage and execution
├── heartbeat/      # Periodic workspace task checks
├── providers/      # LLM provider adapters and registry
├── security/       # Sandbox policy, network guards, approvals
├── session/        # JSONL conversation persistence
└── skills/         # Built-in skills
```

## Install for Development

```bash
pip install -e .
```

The console script is:

```bash
cookiebot --help
```

## Quick Start

Initialize config and workspace:

```bash
cookiebot onboard
```

Edit the generated config, usually at `~/.nanobot/config.json`, and set a provider key/model. Example:

```json
{
  "providers": {
    "openrouter": {
      "apiKey": "sk-or-v1-..."
    }
  },
  "agents": {
    "defaults": {
      "provider": "openrouter",
      "model": "anthropic/claude-opus-4-5"
    }
  }
}
```

Run an interactive local agent:

```bash
cookiebot agent
```

Run a gateway for enabled chat channels:

```bash
cookiebot gateway
```

## Runtime Configuration Highlights

Important default controls:

```json
{
  "agents": {
    "defaults": {
      "contextWindowTokens": 65536,
      "turnTimeoutS": 600,
      "embeddingModel": ""
    }
  },
  "channels": {
    "sendProgress": true,
    "sendToolHints": false,
    "sendTimeoutS": 30
  },
  "tools": {
    "toolTimeout": 120,
    "restrictToWorkspace": false,
    "exec": {
      "timeout": 60,
      "sandboxBackend": "sandboxd_docker",
      "sandboxProfile": "workspace_write_no_net"
    }
  }
}
```

Recommended local hardened shell execution:

```json
{
  "tools": {
    "exec": {
      "sandboxBackend": "sandboxd_docker",
      "sandboxProfile": "workspace_write_no_net",
      "dockerImage": "python:3.11-slim",
      "dockerMemoryMb": 512,
      "dockerPidsLimit": 128,
      "dockerCpus": 1.0,
      "dockerUser": "65534:65534"
    }
  }
}
```

This keeps `exec` off the host process, disables container networking by default, and limits memory, CPU, and process count.

Timeout behavior:

- `agents.defaults.turnTimeoutS`: upper bound for one agent turn.
- `tools.toolTimeout`: registry-level timeout for native tools.
- `tools.exec.timeout`: default shell command timeout.
- `tools.mcpServers.<name>.toolTimeout`: per-MCP-tool timeout.
- `channels.sendTimeoutS`: outbound channel send timeout.

## Memory Model

cookiebot uses a layered memory system:

- `memory/MEMORY.md`: long-term user facts.
- `memory/HISTORY.md`: append-only conversation archive.
- `memory/graph.json`: medium-term GraphRAG nodes and edges.
- `memory/projects/*.md`: project-specific profiles.
- `sessions/*.jsonl`: raw conversation turns with `last_consolidated` offsets.

Memory retrieval now uses query embeddings when available, validates embedding dimensions,
uses similarity thresholds, and avoids returning unrelated fallback nodes.

## Agent State Model

The runtime separates state into three layers:

- `AgentState`: LangGraph internal state for messages, iterations, used tools, and errors.
- `AgentLoop`: outer turn state for sessions, channel routing, memory retrieval, approvals, progress, active tasks, and timeout boundaries.
- Tool context: task-local `ContextVar` values for message/spawn/cron/approval routing.

Same-session work is serialized. Different sessions can run concurrently.

## Communication Model

Inbound and outbound messages include lifecycle metadata:

- `event_id`
- `correlation_id`
- `attempt`
- outbound `kind`: `message`, `progress`, `tool_hint`, `approval_request`, or `error`

This provides traceability and a foundation for future retry/ack/dead-letter behavior.

## Useful Commands

```bash
cookiebot status
cookiebot agent -m "hello"
cookiebot gateway
cookiebot channels status
cookiebot channels login
cookiebot provider login openai-codex
```

## Development Checks

Run focused tests:

```bash
python -m pytest tests/test_graph.py tests/test_tool_validation.py tests/test_task_cancel.py
```

Run the broader suite:

```bash
python -m pytest
```

Some optional channel tests require optional dependencies, such as Matrix dependencies.
