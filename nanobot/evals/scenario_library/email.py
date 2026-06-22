"""Email-monitoring eval scenarios."""

EMAIL_SCENARIOS: list[dict] = [
    {
        "id": "email_monitor_setup",
        "suite": "core",
        "task_type": "email_monitor_setup",
        "user_goal": "帮我设置一个每天早上检查 qq 和 163 两个邮箱的监控，有重要事项再通知我，没事别打扰我",
        "variants": {
            "static": [
                "设置工作日早上检查 qq 和 163 收件箱的定时任务，只提醒紧急邮件",
                "给 qq、163 邮箱配个每天巡检规则，老板或客户的重要邮件再告诉我，普通内容不用报",
            ],
            "generation": {
                "count": 3,
                "dimensions": ["noise", "constraint", "paraphrase"],
            },
        },
        "environment": {
            "clock": "2026-06-16T09:00:00+08:00",
            "tools": {
                "email": {"mode": "fake"},
                "message": {"mode": "capture"},
                "cron": {"mode": "fake"},
            },
        },
        "expected": {
            "required_tools": ["cron"],
            "forbidden_tools": ["message"],
            "required_email_accounts": ["qq", "163"],
            "state": {
                "schedule_created": True,
                "monitor_job_mentions_email": True,
                "monitor_job_mentions_importance": True,
                "monitor_job_mentions_silence": True,
            },
        },
        "metrics": [
            "task_completion",
            "constraint_adherence",
            "tool_success",
            "latency_to_completion",
        ],
    },
    {
        "id": "email_check_now",
        "suite": "core",
        "task_type": "email_check_now",
        "user_goal": "现在检查 qq 和 163 两个邮箱，有老板催办或客户升级邮件就马上告诉我",
        "variants": {
            "static": [
                "立刻看看 qq 和 163 收件箱，发现紧急邮件就通知我",
                "马上查 qq、163 邮件，只要老板或客户的重要邮件结果",
            ],
            "generation": {
                "count": 3,
                "dimensions": ["noise", "constraint", "paraphrase"],
            },
        },
        "environment": {
            "clock": "2026-06-16T09:00:00+08:00",
            "tools": {
                "email": {"mode": "fake"},
                "message": {"mode": "capture"},
                "cron": {"mode": "fake"},
            },
        },
        "expected": {
            "required_tools": ["email"],
            "forbidden_tools": ["cron"],
            "required_email_accounts": ["qq", "163"],
            "state": {"monitor_checked": True},
            "notify_on": ["urgent_boss_email", "customer_escalation"],
        },
        "metrics": [
            "task_completion",
            "notification_recall",
            "constraint_adherence",
            "tool_success",
        ],
    },
    {
        "id": "email_monitor_silence",
        "suite": "core",
        "task_type": "email_monitor_silence",
        "user_goal": "现在检查 qq 和 163 两个邮箱，没有重要事项就别打扰我",
        "variants": {
            "static": [
                "立刻查 qq 和 163 邮箱，空邮箱或只有订阅邮件时不要通知我",
                "马上看 qq、163 收件箱，只有重要邮件才报，普通和 newsletter 都保持安静",
            ],
            "generation": {
                "count": 3,
                "dimensions": ["noise", "constraint", "paraphrase"],
            },
        },
        "environment": {
            "clock": "2026-06-16T09:00:00+08:00",
            "tools": {
                "email": {"mode": "fake"},
                "message": {"mode": "capture"},
                "cron": {"mode": "fake"},
            },
        },
        "expected": {
            "required_tools": ["email"],
            "forbidden_tools": ["message", "cron"],
            "required_email_accounts": ["qq", "163"],
            "state": {"monitor_checked": True},
            "suppress_on": ["empty_inbox", "newsletter_only"],
            "constraints": ["do_not_notify_routine_results"],
        },
        "metrics": [
            "task_completion",
            "notification_precision",
            "constraint_adherence",
            "tool_success",
        ],
    },
]
