"""RAG eval scenarios."""

RAG_SCENARIOS: list[dict] = [
    {
        "id": "rag_single_source_extract",
        "suite": "core",
        "task_type": "rag_single_source_extract",
        "user_goal": "根据 workspace 里的 docs/release_notes.md，告诉我当前版本和已知阻塞问题，并标明来源文件",
        "variants": {
            "static": [
                "去 docs/release_notes.md 找当前版本号和 blocker，回答时带上来源",
                "只依据 release notes 说出版本和已知阻塞项，不要编造别的内容",
            ],
            "generation": {"count": 3, "dimensions": ["paraphrase", "constraint", "noise"]},
        },
        "environment": {
            "workspace_files": {
                "docs/release_notes.md": (
                    "# Release Notes\n\n"
                    "Version: v2.4.1\n"
                    "Known blocker: OAuth refresh bug affects the mobile callback flow.\n"
                    "Status: release candidate.\n"
                ),
            },
            "tools": {"read_file": {"mode": "fake"}, "list_dir": {"mode": "fake"}, "message": {"mode": "capture"}},
        },
        "expected": {
            "required_tools": ["read_file"],
            "required_paths_read": ["docs/release_notes.md"],
            "forbidden_tools": ["web_fetch", "web_search", "message"],
            "response_must_include": ["v2.4.1", "OAuth refresh bug", "release_notes.md"],
            "constraints": ["cite_sources"],
        },
        "metrics": ["task_completion", "constraint_adherence", "tool_success"],
    },
    {
        "id": "rag_multi_source_compare",
        "suite": "core",
        "task_type": "rag_multi_source_compare",
        "user_goal": "对比 workspace 里的 docs/api_sla.md 和 docs/runbook.md，总结 API 超时阈值和人工升级阈值，并标明两个来源文件",
        "variants": {
            "static": [
                "比较 api_sla 和 runbook，两边关于超时和升级的要求分别是什么，回答时带来源",
                "只根据这两个文档，汇总 timeout 阈值和 escalation 条件",
            ],
            "generation": {"count": 3, "dimensions": ["paraphrase", "constraint", "noise"]},
        },
        "environment": {
            "workspace_files": {
                "docs/api_sla.md": (
                    "# API SLA\n\n"
                    "Request timeout threshold: 15 seconds.\n"
                    "Clients should retry once before surfacing an error.\n"
                ),
                "docs/runbook.md": (
                    "# Incident Runbook\n\n"
                    "Escalate to the on-call engineer after 45 seconds of sustained timeout symptoms.\n"
                    "Page the incident lead if three retries fail.\n"
                ),
            },
            "tools": {"read_file": {"mode": "fake"}, "list_dir": {"mode": "fake"}, "message": {"mode": "capture"}},
        },
        "expected": {
            "required_any_tools": [["read_file", "list_dir"]],
            "required_paths_read": ["docs/api_sla.md", "docs/runbook.md"],
            "forbidden_tools": ["web_fetch", "web_search", "message"],
            "response_must_include": ["15 seconds", "45 seconds", "api_sla.md", "runbook.md"],
            "constraints": ["cite_sources"],
        },
        "metrics": ["task_completion", "constraint_adherence", "tool_success"],
    },
    {
        "id": "rag_insufficient_context_refusal",
        "suite": "core",
        "task_type": "rag_insufficient_context_refusal",
        "user_goal": "只根据 workspace 里的 docs/faq.md，告诉我退款截止时间；如果文档没有写，就明确说不知道并标明来源文件",
        "variants": {
            "static": [
                "去 faq.md 找退款 deadline，没有就直接说明文档没提供",
                "仅依据 FAQ 回答退款截止时间；如果缺失，不要猜",
            ],
            "generation": {"count": 3, "dimensions": ["paraphrase", "constraint", "noise"]},
        },
        "environment": {
            "workspace_files": {
                "docs/faq.md": (
                    "# FAQ\n\n"
                    "Refunds may be requested by contacting support.\n"
                    "Processing usually completes within 5 business days.\n"
                ),
            },
            "tools": {"read_file": {"mode": "fake"}, "list_dir": {"mode": "fake"}, "message": {"mode": "capture"}},
        },
        "expected": {
            "required_tools": ["read_file"],
            "required_paths_read": ["docs/faq.md"],
            "forbidden_tools": ["web_fetch", "web_search", "message"],
            "response_must_include": ["faq.md"],
            "response_must_include_any": [["没有写", "not specified", "not provided", "unknown"]],
            "constraints": ["cite_sources", "refuse_when_missing_context"],
        },
        "metrics": ["task_completion", "constraint_adherence", "tool_success"],
    },
    {
        "id": "rag_conflict_resolution",
        "suite": "core",
        "task_type": "rag_conflict_resolution",
        "user_goal": "根据 workspace 里的 docs/policy_v1.md 和 docs/policy_v2.md，告诉我超时升级阈值；如果两份文档冲突，要明确指出冲突并标明来源文件",
        "variants": {
            "static": [
                "比较 policy_v1 和 policy_v2 的 escalation timeout，冲突时不要替我选一个",
                "只根据两份 policy 文档回答升级阈值，并说明是否一致",
            ],
            "generation": {"count": 3, "dimensions": ["paraphrase", "constraint", "noise"]},
        },
        "environment": {
            "workspace_files": {
                "docs/policy_v1.md": "# Policy v1\n\nEscalate after 30 seconds of repeated timeout symptoms.\n",
                "docs/policy_v2.md": "# Policy v2\n\nEscalate after 45 seconds of repeated timeout symptoms.\n",
            },
            "tools": {"read_file": {"mode": "fake"}, "list_dir": {"mode": "fake"}, "message": {"mode": "capture"}},
        },
        "expected": {
            "required_tools": ["read_file"],
            "required_paths_read": ["docs/policy_v1.md", "docs/policy_v2.md"],
            "forbidden_tools": ["web_fetch", "web_search", "message"],
            "response_must_include": ["30 seconds", "45 seconds", "policy_v1.md", "policy_v2.md"],
            "response_must_include_any": [["冲突", "conflict", "inconsistent", "disagree"]],
            "constraints": ["cite_sources", "flag_conflicting_sources"],
        },
        "metrics": ["task_completion", "constraint_adherence", "tool_success"],
    },
    {
        "id": "rag_irrelevant_context_filter",
        "suite": "core",
        "task_type": "rag_irrelevant_context_filter",
        "user_goal": "根据 workspace 里的 docs/deploy_guide.md，告诉我生产环境部署区域，并标明来源；不要被其他无关文档干扰",
        "variants": {
            "static": [
                "只根据 deploy_guide 找生产部署区域，回答时带来源",
                "看部署指南说 production region 是什么，别把别的文档内容混进来",
            ],
            "generation": {"count": 3, "dimensions": ["paraphrase", "constraint", "noise"]},
        },
        "environment": {
            "workspace_files": {
                "docs/deploy_guide.md": "# Deploy Guide\n\nProduction region: ap-southeast-1.\nStaging region: us-west-2.\n",
                "docs/marketing_plan.md": "# Marketing Plan\n\nCampaign region focus: europe-west.\nTagline refresh for Q4 launch.\n",
            },
            "tools": {"read_file": {"mode": "fake"}, "list_dir": {"mode": "fake"}, "message": {"mode": "capture"}},
        },
        "expected": {
            "required_tools": ["read_file"],
            "required_paths_read": ["docs/deploy_guide.md"],
            "forbidden_tools": ["web_fetch", "web_search", "message"],
            "response_must_include": ["ap-southeast-1", "deploy_guide.md"],
            "response_must_not_include": ["europe-west", "marketing_plan.md"],
            "constraints": ["cite_sources", "ignore_irrelevant_context"],
        },
        "metrics": ["task_completion", "constraint_adherence", "tool_success"],
    },
    {
        "id": "rag_multi_hop_join",
        "suite": "core",
        "task_type": "rag_multi_hop_join",
        "user_goal": "根据 workspace 里的 docs/service_catalog.md 和 docs/oncall.md，告诉我哪个服务归属哪个 on-call 团队，并标明来源文件",
        "variants": {
            "static": [
                "结合 service_catalog 和 oncall 文档，说出 billing-api 对应的值班团队",
                "从两份文档拼出 billing-api 的 owner team，回答时带来源",
            ],
            "generation": {"count": 3, "dimensions": ["paraphrase", "constraint", "noise"]},
        },
        "environment": {
            "workspace_files": {
                "docs/service_catalog.md": "# Service Catalog\n\nService: billing-api\nOwner rotation: payments-primary\n",
                "docs/oncall.md": "# On-call Directory\n\nRotation: payments-primary\nTeam: Payments Platform\n",
            },
            "tools": {"read_file": {"mode": "fake"}, "list_dir": {"mode": "fake"}, "message": {"mode": "capture"}},
        },
        "expected": {
            "required_tools": ["read_file"],
            "required_paths_read": ["docs/service_catalog.md", "docs/oncall.md"],
            "forbidden_tools": ["web_fetch", "web_search", "message"],
            "response_must_include": ["billing-api", "payments-primary", "Payments Platform", "service_catalog.md", "oncall.md"],
            "constraints": ["cite_sources", "multi_hop_synthesis"],
        },
        "metrics": ["task_completion", "constraint_adherence", "tool_success"],
    },
]
