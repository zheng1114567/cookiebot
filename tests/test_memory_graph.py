"""Unit tests for MemoryGraph, ProjectProfile, and retrieval."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from nanobot.agent.memory import (
    MemoryGraph,
    MemoryStore,
    ProjectProfile,
    _cosine_similarity,
    _tokenize,
)

# ── _tokenize ───────────────────────────────────────────────────────────────


def test_tokenize_extracts_lowercase_words():
    assert _tokenize("Hello World 123") == {"hello", "world", "123"}


def test_tokenize_filters_single_char_words():
    assert _tokenize("a bb c d eee") == {"bb", "eee"}


# ── _cosine_similarity ──────────────────────────────────────────────────────


def test_cosine_identical_vectors():
    v = [1.0, 2.0, 3.0]
    assert _cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors():
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_zero_vector():
    assert _cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


# ── MemoryGraph ─────────────────────────────────────────────────────────────


class TestMemoryGraph:
    def test_add_and_retrieve_nodes(self, tmp_path: Path):
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        g = MemoryGraph(memory_dir)

        g.add_nodes([
            {
                "id": "topic:test",
                "type": "topic",
                "name": "Test Topic",
                "summary": "A topic about testing",
                "tags": ["test", "unit"],
            }
        ])

        # Should be retrievable
        results = g.retrieve("testing", max_entries=5)
        assert len(results) >= 1
        assert results[0]["name"] == "Test Topic"

    def test_add_nodes_merges_existing_metadata(self, tmp_path: Path):
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        g = MemoryGraph(memory_dir)

        g.add_nodes([
            {
                "id": "topic:test",
                "type": "topic",
                "name": "Test",
                "summary": "Original summary",
                "tags": ["old"],
                "first_seen": "2026-01-01T00:00:00",
                "last_seen": "2026-01-01T00:00:00",
                "n_mentions": 2,
            }
        ])
        g.add_nodes([
            {
                "id": "topic:test",
                "type": "topic",
                "name": "Test Updated",
                "summary": "Updated summary",
                "tags": ["new"],
            }
        ])

        data = json.loads(g.graph_file.read_text(encoding="utf-8"))
        node = data["nodes"][0]
        assert node["first_seen"] == "2026-01-01T00:00:00"
        assert node["n_mentions"] == 3
        assert node["summary"] == "Updated summary"
        assert node["tags"] == ["new", "old"]

    def test_retrieve_no_match_returns_empty(self, tmp_path: Path):
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        g = MemoryGraph(memory_dir)

        g.add_nodes([
            {
                "id": "topic:python",
                "type": "topic",
                "name": "Python",
                "summary": "Python programming",
                "tags": ["python"],
            }
        ])

        results = g.retrieve("zzz_xyzzy_nonexistent", max_entries=5)
        assert results == []

    def test_vector_retrieval_uses_similarity_threshold_and_dimensions(self, tmp_path: Path):
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        g = MemoryGraph(memory_dir)

        g.add_nodes([
            {
                "id": "topic:match",
                "type": "topic",
                "name": "Match",
                "summary": "Vector match",
                "tags": ["vector"],
                "embedding": [1.0, 0.0],
            },
            {
                "id": "topic:wrong-dim",
                "type": "topic",
                "name": "Wrong Dimension",
                "summary": "Should be skipped",
                "tags": ["vector"],
                "embedding": [1.0, 0.0, 0.0],
            },
            {
                "id": "topic:orthogonal",
                "type": "topic",
                "name": "Orthogonal",
                "summary": "Below threshold",
                "tags": ["vector"],
                "embedding": [0.0, 1.0],
            },
        ])

        results = g.retrieve("unrelated words", query_vector=[1.0, 0.0], max_entries=10)
        ids = {r["id"] for r in results}
        assert "topic:match" in ids
        assert "topic:wrong-dim" not in ids
        assert "topic:orthogonal" not in ids

    def test_hybrid_retrieval_merges_dense_and_sparse_candidates(self, tmp_path: Path):
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        g = MemoryGraph(memory_dir)

        g.add_nodes([
            {
                "id": "topic:dense",
                "type": "topic",
                "name": "Preference",
                "summary": "User prefers concise implementation notes for code reviews.",
                "tags": ["preference", "review"],
                "embedding": [1.0, 0.0],
            },
            {
                "id": "topic:sparse",
                "type": "topic",
                "name": "Storage quota billing",
                "summary": "Storage quota billing policy has exact-match retrieval terms.",
                "tags": ["storage", "quota", "billing"],
                "embedding": [0.0, 1.0],
            },
        ])

        results = g.retrieve("storage quota billing", query_vector=[1.0, 0.0], max_entries=10)
        ids = {result["id"] for result in results}

        assert "topic:dense" in ids
        assert "topic:sparse" in ids
        assert g.last_retrieval_telemetry["dense_candidates"] >= 1
        assert g.last_retrieval_telemetry["sparse_candidates"] >= 1
        assert g.last_retrieval_telemetry["merged_candidates"] >= 2

    def test_semantic_chunking_adds_neighbor_window_context(self, tmp_path: Path):
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        g = MemoryGraph(memory_dir)
        intro = " ".join(f"intro{i}" for i in range(360))
        middle = " ".join(f"middle{i}" for i in range(360))
        ending = " ".join(f"ending{i}" for i in range(360))

        g.add_nodes([
            {
                "id": "doc:long",
                "type": "fact",
                "name": "Long retrieval document",
                "summary": f"{intro}.\n\n{middle}.\n\n{ending}.",
                "tags": ["long", "retrieval", "document"],
            }
        ])

        data = json.loads(g.graph_file.read_text(encoding="utf-8"))
        chunks = [node for node in data["nodes"] if node.get("doc_id") == "doc:long"]

        assert len(chunks) >= 2
        assert all(chunk.get("body_text") for chunk in chunks)
        assert all(chunk.get("expanded_context_text") for chunk in chunks)
        assert chunks[1]["expanded_context_text"] != chunks[1]["body_text"]

    def test_add_edges_and_subgraph_expansion(self, tmp_path: Path):
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        g = MemoryGraph(memory_dir)

        g.add_nodes([
            {"id": "topic:a", "type": "topic", "name": "A", "summary": "Topic A", "tags": ["a"]},
            {"id": "topic:b", "type": "topic", "name": "B", "summary": "Topic B", "tags": ["b"]},
            {"id": "topic:c", "type": "topic", "name": "C", "summary": "Topic C", "tags": ["c"]},
        ])
        g.add_edges([
            {"source": "topic:a", "target": "topic:b", "type": "related"},
        ])

        # Query for A → should get A and B (connected) but not C
        results = g.retrieve("Topic A", max_entries=10)
        ids = {r["id"] for r in results}
        assert "topic:a" in ids
        assert "topic:b" in ids
        # C may or may not be present (seed match fallback)

    def test_project_detection(self, tmp_path: Path):
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        g = MemoryGraph(memory_dir)

        g.add_nodes([
            {"id": "project:nanobot", "type": "project", "name": "nanobot", "summary": "AI assistant", "tags": ["ai"]},
            {"id": "topic:memory", "type": "topic", "name": "Memory", "summary": "Memory system", "tags": ["memory"]},
        ])
        g.add_edges([
            {"source": "project:nanobot", "target": "topic:memory", "type": "contains"},
        ])

        project_id = g._detect_project("我在改 nanobot 的记忆系统")
        assert project_id == "project:nanobot"

    def test_project_boost_in_retrieval(self, tmp_path: Path):
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        g = MemoryGraph(memory_dir)

        g.add_nodes([
            {"id": "project:nanobot", "type": "project", "name": "nanobot", "summary": "AI assistant", "tags": ["ai"]},
            {"id": "topic:mem", "type": "topic", "name": "Memory", "summary": "Memory system redesign", "tags": ["memory"]},
            {"id": "topic:dep", "type": "topic", "name": "Dependencies", "summary": "Project dependencies", "tags": ["deps"]},
        ])
        g.add_edges([
            {"source": "project:nanobot", "target": "topic:mem", "type": "contains"},
            {"source": "project:nanobot", "target": "topic:dep", "type": "contains"},
        ])

        results = g.retrieve("nanobot 项目", max_entries=5)
        # Both project-connected nodes should appear with boosted scores
        names = {r["name"] for r in results}
        assert "Memory" in names or "Dependencies" in names

    def test_prepare_project_hierarchy_groups_child_nodes(self, tmp_path: Path):
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        g = MemoryGraph(memory_dir)

        nodes, edges = g.prepare_hierarchy(
            [
                {
                    "id": "decision:memory-hierarchy",
                    "type": "decision",
                    "name": "Memory hierarchy",
                    "summary": "Add project-scoped hierarchy to GraphRAG memory.",
                    "tags": ["memory", "graphrag"],
                }
            ],
            project_name="nanobot",
        )
        g.add_nodes(nodes)
        g.add_edges(edges)

        data = json.loads(g.graph_file.read_text(encoding="utf-8"))
        node_map = {node["id"]: node for node in data["nodes"]}
        child = node_map["decision:memory-hierarchy"]
        assert "scope:projects" in node_map
        assert node_map["project:nanobot"]["parent_id"] == "scope:projects"
        assert child["scope"] == "project"
        assert child["project"] == "nanobot"
        assert child["parent_id"] == "project:nanobot"
        assert child["path"] == ["projects", "nanobot", "general"]
        assert len([node for node in data["nodes"] if node["id"] == "project:nanobot"]) == 1
        assert ("project:nanobot", "decision:memory-hierarchy", "contains") in {
            (edge["source"], edge["target"], edge["type"]) for edge in data["edges"]
        }

        results = g.retrieve("nanobot 项目", max_entries=10)
        assert "decision:memory-hierarchy" in {result["id"] for result in results}

    def test_prepare_daily_hierarchy_classifies_conversation_nodes(self, tmp_path: Path):
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        g = MemoryGraph(memory_dir)

        nodes, edges = g.prepare_hierarchy(
            [
                {
                    "id": "conversation:dinner",
                    "type": "conversation",
                    "name": "Dinner planning",
                    "summary": "User discussed where to eat dinner.",
                    "tags": ["dinner"],
                }
            ],
            daily_category="food",
        )
        g.add_nodes(nodes)
        g.add_edges(edges)

        data = json.loads(g.graph_file.read_text(encoding="utf-8"))
        node_map = {node["id"]: node for node in data["nodes"]}
        child = node_map["conversation:dinner"]
        assert "scope:daily" in node_map
        assert "daily:food" in node_map
        assert child["scope"] == "daily"
        assert child["category"] == "food"
        assert child["parent_id"] == "daily:food"
        assert child["path"] == ["daily", "food"]

        results = g.retrieve("food 日常", max_entries=10)
        assert "conversation:dinner" in {result["id"] for result in results}

    def test_memory_context_groups_related_entries_by_hierarchy(self, tmp_path: Path):
        store = MemoryStore(tmp_path)
        project_nodes, project_edges = store.graph.prepare_hierarchy(
            [
                {
                    "id": "topic:project-memory",
                    "type": "topic",
                    "name": "Project memory",
                    "summary": "Nanobot project memory changes.",
                    "tags": ["memory"],
                }
            ],
            project_name="nanobot",
        )
        daily_nodes, daily_edges = store.graph.prepare_hierarchy(
            [
                {
                    "id": "conversation:meal",
                    "type": "conversation",
                    "name": "Meal discussion",
                    "summary": "Everyday food preference discussion.",
                    "tags": ["food"],
                }
            ],
            daily_category="food",
        )
        store.graph.add_nodes(project_nodes + daily_nodes)
        store.graph.add_edges(project_edges + daily_edges)

        context = store.get_memory_context("nanobot food")

        assert "### Project: nanobot" in context
        assert "Project memory" in context
        assert "### Daily: food" in context
        assert "Meal discussion" in context

    def test_memory_context_marks_confirmation_needed(self, tmp_path: Path):
        store = MemoryStore(tmp_path)
        store.graph.add_nodes([
            {
                "id": "fact:conflict",
                "type": "fact",
                "name": "Runtime choice",
                "summary": "Use uvicorn workers=2 for this service in production.",
                "tags": ["runtime", "deploy", "production"],
                "needs_confirmation": True,
            }
        ])

        context = store.get_memory_context("runtime choice")
        assert "[needs confirmation]" in context

    def test_store_delete_memory_and_audit(self, tmp_path: Path):
        store = MemoryStore(tmp_path)
        store.graph.add_nodes([
            {
                "id": "fact:wrong",
                "type": "fact",
                "name": "Wrong memory",
                "summary": "This memory is incorrect and should be removable.",
                "tags": ["wrong", "memory", "delete"],
            }
        ])

        assert store.delete_memory("fact:wrong", reason="test") is True
        report = store.audit_memories()
        assert report["total_nodes"] == 0

    def test_expire_removes_old_nodes(self, tmp_path: Path):
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        g = MemoryGraph(memory_dir)
        stale_time = (datetime.now() - timedelta(days=45)).isoformat()

        g.add_nodes([
            {
                "id": "topic:old",
                "type": "topic",
                "name": "Old Topic",
                "summary": "Very old",
                "tags": ["old"],
                "last_seen": stale_time,
            },
            {
                "id": "topic:recent",
                "type": "topic",
                "name": "Recent Topic",
                "summary": "Just now",
                "tags": ["recent"],
            },
        ])

        removed = g.expire(max_age_days=30, hard_delete_days=90)
        assert removed == 0
        data = json.loads(g.graph_file.read_text(encoding="utf-8"))
        old_node = next(node for node in data["nodes"] if node["id"] == "topic:old")
        assert old_node["stale"] is True

    def test_expire_hard_deletes_very_old_nodes(self, tmp_path: Path):
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        g = MemoryGraph(memory_dir)

        g.add_nodes([
            {
                "id": "topic:very-old",
                "type": "topic",
                "name": "Very Old Topic",
                "summary": "Ancient memory",
                "tags": ["old"],
                "last_seen": "2020-01-01T00:00:00",
            }
        ])

        removed = g.expire(max_age_days=30, hard_delete_days=90)
        assert removed == 1

    def test_duplicate_edges_skipped(self, tmp_path: Path):
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        g = MemoryGraph(memory_dir)

        g.add_nodes([
            {"id": "topic:x", "type": "topic", "name": "X", "summary": "X", "tags": ["x"]},
            {"id": "topic:y", "type": "topic", "name": "Y", "summary": "Y", "tags": ["y"]},
        ])

        g.add_edges([{"source": "topic:x", "target": "topic:y", "type": "related"}])
        g.add_edges([{"source": "topic:x", "target": "topic:y", "type": "related"}])

        assert len(g._graph["edges"]) == 1  # duplicate skipped

    def test_low_value_nodes_are_filtered(self, tmp_path: Path):
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        g = MemoryGraph(memory_dir)

        g.add_nodes([
            {"id": "conversation:noise", "type": "conversation", "name": "ok", "summary": "yes", "tags": []}
        ])

        data = json.loads(g.graph_file.read_text(encoding="utf-8"))
        assert data["nodes"] == []

    def test_conflict_keeps_recent_node_and_preserves_versions(self, tmp_path: Path):
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        g = MemoryGraph(memory_dir)

        g.add_nodes([
            {
                "id": "fact:pref",
                "type": "fact",
                "name": "Editor preference",
                "summary": "User prefers Vim for editing Python files.",
                "tags": ["editor", "vim", "python"],
            }
        ])
        g.add_nodes([
            {
                "id": "fact:pref",
                "type": "fact",
                "name": "Editor preference",
                "summary": "User now prefers VS Code for editing Python files.",
                "tags": ["editor", "vscode", "python"],
            }
        ])

        data = json.loads(g.graph_file.read_text(encoding="utf-8"))
        node = next(node for node in data["nodes"] if node["id"] == "fact:pref")
        assert node["summary"] == "User now prefers VS Code for editing Python files."
        assert node["conflict"]["needs_confirmation"] is True
        assert len(node["versions"]) == 1

    def test_sensitive_nodes_are_encrypted(self, monkeypatch, tmp_path: Path):
        monkeypatch.setenv("NANOBOT_MEMORY_SECRET", "secret-key")
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        g = MemoryGraph(memory_dir)

        g.add_nodes([
            {
                "id": "fact:key",
                "type": "fact",
                "name": "API Key",
                "summary": "The api_key is sk-1234567890abcdef and should stay private.",
                "tags": ["api", "secret", "credential"],
                "importance": "high",
            }
        ])

        data = json.loads(g.graph_file.read_text(encoding="utf-8"))
        node = next(node for node in data["nodes"] if node["id"] == "fact:key")
        assert node["is_sensitive"] is True
        assert node["needs_confirmation"] is True
        assert node["summary"] == "[encrypted sensitive memory]"
        assert node["body_text"] == "[encrypted sensitive memory]"
        assert node["expanded_context_text"] == "[encrypted sensitive memory]"
        results = g.retrieve("API Key", max_entries=1)
        assert "sk-1234567890abcdef" in results[0]["summary"]
        assert "sk-1234567890abcdef" in results[0]["expanded_context_text"]

    def test_delete_node_removes_memory_and_edges(self, tmp_path: Path):
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        g = MemoryGraph(memory_dir)

        g.add_nodes([
            {"id": "topic:a", "type": "topic", "name": "A topic", "summary": "A useful memory topic", "tags": ["topic", "memory"]},
            {"id": "topic:b", "type": "topic", "name": "B topic", "summary": "Another useful memory topic", "tags": ["topic", "memory"]},
        ])
        g.add_edges([{"source": "topic:a", "target": "topic:b", "type": "related"}])

        assert g.delete_node("topic:a", reason="incorrect_memory") is True
        data = json.loads(g.graph_file.read_text(encoding="utf-8"))
        assert {node["id"] for node in data["nodes"]} == {"topic:b"}
        assert data["edges"] == []

    def test_audit_reports_sensitive_conflicted_and_stale_nodes(self, monkeypatch, tmp_path: Path):
        monkeypatch.setenv("NANOBOT_MEMORY_SECRET", "secret-key")
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        g = MemoryGraph(memory_dir)
        stale_time = (datetime.now() - timedelta(days=45)).isoformat()

        g.add_nodes([
            {
                "id": "fact:conflict",
                "type": "fact",
                "name": "Preference",
                "summary": "User prefers espresso after lunch every workday.",
                "tags": ["preference", "coffee", "workday"],
                "last_seen": "2026-01-01T00:00:00",
            }
        ])
        g.add_nodes([
            {
                "id": "fact:conflict",
                "type": "fact",
                "name": "Preference",
                "summary": "User now avoids espresso after lunch every workday.",
                "tags": ["preference", "coffee", "workday"],
            }
        ])
        g.add_nodes([
            {
                "id": "fact:secret",
                "type": "fact",
                "name": "Secret token",
                "summary": "The access_token is sk-abcdef1234567890 for staging.",
                "tags": ["token", "secret", "staging"],
                "importance": "high",
            }
        ])
        g.add_nodes([
            {
                "id": "fact:stale",
                "type": "fact",
                "name": "Old decision",
                "summary": "Use Redis caching for background job fan-out in this service.",
                "tags": ["redis", "cache", "background"],
                "last_seen": stale_time,
            }
        ])
        g.expire(max_age_days=30, hard_delete_days=400)

        report = g.audit()
        assert report["sensitive_nodes"] >= 1
        assert report["conflicted_nodes"] >= 1
        assert report["stale_nodes"] >= 1
        assert report["needs_confirmation"]


# ── ProjectProfile ──────────────────────────────────────────────────────────


class TestProjectProfile:
    def test_get_nonexistent_returns_none(self, tmp_path: Path):
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        pp = ProjectProfile(memory_dir)
        assert pp.get("nonexistent") is None

    def test_update_and_read(self, tmp_path: Path):
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        pp = ProjectProfile(memory_dir)

        pp.update("nanobot", "# nanobot\n\nTest content")
        assert pp.get("nanobot") == "# nanobot\n\nTest content"

    def test_merge_appends(self, tmp_path: Path):
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        pp = ProjectProfile(memory_dir)

        pp.update("test", "# Test\n\nInitial content")
        pp.merge("test", "New info")
        result = pp.get("test")
        assert "Initial content" in result
        assert "New info" in result

    def test_list_projects(self, tmp_path: Path):
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        pp = ProjectProfile(memory_dir)

        pp.update("proj_a", "content a")
        pp.update("proj_b", "content b")

        projects = pp.list_projects()
        assert "proj_a" in projects
        assert "proj_b" in projects
