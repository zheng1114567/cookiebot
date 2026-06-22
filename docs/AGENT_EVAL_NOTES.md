# Agent Eval And Tooling Notes

Date: 2026-06-17

## What Changed

### Eval System

Added an agent eval MVP under `nanobot/evals/`:

- Built-in scenario definitions for scheduled tasks, email monitoring, and grounded recovery.
- Dynamic variant expansion from scenario seeds.
- Rule/state judge with structured run records and explicit scenario contracts.
- Markdown, JSONL, and summary JSON report output.
- CLI commands:
  - `cookiebot eval list`
  - `cookiebot eval run core`
  - `cookiebot eval run core --executor agent`
  - `cookiebot eval compare baseline.jsonl current.jsonl`
  - `cookiebot eval report`
- CI-style gates:
  - `--min-completion`
  - `--min-tool-success`
  - `--min-notification-recall`
  - `--max-regression`

The default executor remains `spec` for deterministic framework checks. The real loop executor is available with `--executor agent`.

### Real AgentLoop Eval

Added `AgentLoopExecutor` so eval cases can run through the real `AgentLoop.process_direct()` path.

It records:

- Final response.
- Tool calls and params.
- Tool success/failure.
- Cron jobs created.
- Email accounts checked.
- Message tool notifications.
- Latency.

The executor runs cases in isolated temporary eval workspaces under the selected report output directory, so normal workspace files and cron jobs are not polluted by eval runs.

### Email Tool

Added a real `email` agent tool backed by the existing `EmailChannel` IMAP/SMTP implementation.

Supported actions:

- `search`
- `unread`
- `send`

The tool supports multiple configured accounts. Current configured accounts:

- `qq`
- `163`

For multiple accounts, `account` is required and exposed as an enum in the tool schema. This helps the model reliably call:

```json
{"action": "search", "account": "qq", "days": 1, "limit": 5}
```

and:

```json
{"action": "search", "account": "163", "days": 1, "limit": 5}
```

No secrets are documented here.

### Config Updates

Current relevant runtime choices:

- LLM provider: `deepseek`
- Web search provider: `duckduckgo`
- Exec backend: `local`
- Email channel: enabled, with `qq` and `163` accounts
- Feishu channel: enabled, with app credentials, verification token, and encrypt key configured

## Recent Eval Results

### Before Email Tool

Real `AgentLoop` core eval had strong scheduled-task behavior but weak monitoring/recovery:

- Scheduled tasks: 100%
- Monitoring: 0% success
- Recovery: 0% success

The main issue was that the eval asked for email-style monitoring but the agent did not have a real email tool.

### After Email Tool

After adding the real email tool and configuring QQ/163, monitoring improved. A full core run showed:

- Task Completion Rate: 16.7%
- Useful Completion Rate: 30.6%
- Tool Success Rate: 89.3%
- Constraint Adherence Rate: 72.2%
- Notification Precision: 50.0%
- Notification Recall: 100.0%

After tightening the tool schema and monitoring judge, a 10-case smoke run showed:

- Cases: 10
- Pass / Partial / Fail: 8 / 0 / 2
- Task Completion Rate: 80.0%
- Tool Success Rate: 100.0%
- Constraint Adherence Rate: 100.0%
- Notification Precision: 100.0%
- Notification Recall: 100.0%

The remaining smoke failures were due to judge interpretation around scheduled email monitoring, not core tool connectivity. The judge was then refined to use explicit contracts instead of loose monitoring heuristics.

## Product Thinking

The eval system should measure agent reliability across real workflows, not just static prompt/answer quality.

Useful scorecard dimensions:

- Task completion.
- Useful completion.
- Tool success.
- Constraint adherence.
- Notification precision.
- Notification recall.
- Recovery behavior.
- Latency.
- Cost per success.

For background agents, notification quality is especially important:

- Notify when there is actionable or important information.
- Stay silent for empty, routine, or newsletter-only results.
- Avoid claiming external checks were completed if the required tool/account is unavailable.

## Current Known Issues

### Monitoring Eval

The monitoring eval is now split into three explicit behaviors:

- `email_monitor_setup`
- `email_check_now`
- `email_monitor_silence`

This makes the pass criteria clearer:

- Setup tasks require cron creation and schedule content checks.
- Immediate-check tasks require actual email tool use in the current run.
- Silence tasks separately measure "do not notify for routine results".

### Recovery Eval

Recovery scenarios are now grounded in currently available tools:

- `recovery_missing_file`
- `recovery_web_timeout`

The key design change is that recovery tasks no longer depend on vague "check project status" prompts.

### Eval Judge

The judge should continue moving from keyword/state heuristics toward explicit scenario contracts:

- Required immediate tools.
- Required scheduled behavior.
- Allowed alternative tools.
- Forbidden shortcut tools.
- Required account coverage.
- Whether final response counts as a notification.
- Whether message tool use counts as notification.

Current contract fields used in `expected` include:

- `required_tools`
- `required_any_tools`
- `forbidden_tools`
- `required_email_accounts`
- `state`
- `constraints`
- `recovery_required`

## Next Steps

1. Run a full real-loop core eval after the contract refinements.

2. Add more grounded recovery cases:
   - `exec` timeout.
   - `email` missing/unknown account.
   - MCP unavailable.

3. Add scenario-local fixture contracts:
   - exact missing path.
   - exact timeout URL.
   - expected blocker class.

4. Add a stable CI command such as:

```powershell
python -m nanobot eval run core --executor agent --min-completion 0.7 --min-tool-success 0.9
```

5. Compare against previous JSONL reports with:

```powershell
python -m nanobot eval compare baseline.jsonl current.jsonl --max-regression 0.05
```
