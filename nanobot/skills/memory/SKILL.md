---
name: memory
description: Three-tier memory system with automatic GraphRAG retrieval.
always: true
---

# Memory

## Three-Tier Structure

Memory is automatically managed across three tiers. You don't need to grep or manually search.

| Tier | Storage | What | How to access |
|---|---|---|---|
| Short-term | Session JSONL | Current conversation, raw messages | Always in context. Long conversations get a running summary + recent messages. |
| Medium-term | `memory/graph.json` | Hierarchical knowledge graph: projects, daily categories, topics, decisions, facts, conversations. Keyword + graph retrieval. | Automatically injected in grouped `## Related Context`. Project memories stay under their project; everyday conversations are grouped by daily category. Auto-expires non-structural nodes after 30 days. |
| Long-term | `memory/MEMORY.md` | Persistent user profile: preferences, traits, habits, personal info. | Always loaded in `## Long-term Memory`. |

## How Retrieval Works

Each time you are called, the system:
1. Extracts keywords from the user's message
2. Matches against the knowledge graph (medium-term)
3. Expands 1-hop subgraph for related nodes
4. Detects project context or daily category and boosts relevant nodes
5. Groups retrieved memories by project or daily category
6. Injects top results into your system prompt

**You don't need to grep HISTORY.md.** Retrieval is automatic.

## Project Profiles

When the user works on a project, relevant graph nodes are attached under `project:{project_name}` and relevant context is grouped together in `memory/projects/{project_name}.md`. This profile is auto-injected when the project is detected in the current conversation.

## Daily Categories

When the conversation is not tied to a specific project, medium-term graph nodes are attached under a `daily:{category}` container such as planning, learning, health, finance, family, travel, errands, work, or general.

## When to Update MEMORY.md

Only write to `memory/MEMORY.md` when you discover NEW persistent facts about the user:
- User preferences ("I prefer dark mode")
- Personal info ("I live in Beijing")
- Key decisions ("We decided to use LangGraph")

Do NOT write project-related information to MEMORY.md — that belongs in the knowledge graph and project profiles.

## Auto-consolidation

After each conversation turn, the system automatically:
- Extracts entities, topics, decisions → writes to knowledge graph
- Detects project affiliation or daily category → updates hierarchical graph grouping
- Updates project profile when project-specific information changes
- Identifies new user traits → updates MEMORY.md (rare)

No manual action needed.
