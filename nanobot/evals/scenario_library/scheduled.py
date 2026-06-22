"""Scheduled-task eval scenarios."""

SCHEDULED_SCENARIOS: list[dict] = [
    {
        "id": "cron_reminder_basic",
        "suite": "core",
        "task_type": "scheduled_task",
        "user_goal": "每天晚上 9 点提醒我复盘今天的工作",
        "variants": {
            "static": [
                "工作日晚上 9 点叫我写当天复盘",
                "从明天开始，每天 21:00 提醒我做工作复盘",
            ],
            "generation": {
                "count": 3,
                "dimensions": ["time_change", "constraint", "paraphrase"],
            },
        },
        "environment": {
            "clock": "2026-06-16T09:00:00+08:00",
            "tools": {"cron": {"mode": "fake"}, "message": {"mode": "capture"}},
        },
        "expected": {
            "required_tools": ["cron"],
            "state": {"schedule_created": True},
            "constraints": ["recurring_schedule", "time_preserved"],
        },
        "metrics": [
            "task_completion",
            "constraint_adherence",
            "tool_success",
            "latency_to_completion",
        ],
    },
]
