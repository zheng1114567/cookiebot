"""Memory system for persistent agent memory."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import re
import time
import weakref
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from nanobot.utils.helpers import ensure_dir, estimate_message_tokens, estimate_prompt_tokens_chain

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import Session, SessionManager


_SAVE_MEMORY_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": (
                "Save consolidated memory to persistent storage. "
                "Always produce graph_nodes and history_entry. "
                "Only produce memory_update when you discover NEW persistent facts about the user. "
                "Only produce project_update when the conversation adds new information to a project."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "graph_nodes": {
                        "type": "array",
                        "description": (
                            "1-5 knowledge graph nodes extracted from the conversation. "
                            "Each node has: id (unique, prefixed by type:), type (topic/decision/fact/project/conversation), "
                            "name (short title), summary (one paragraph), tags (3-8 keywords for retrieval). "
                            "Optional node fields: category, scope, parent_id. "
                            "For project nodes, the id should be 'project:<name>'."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "type": {"type": "string"},
                                "name": {"type": "string"},
                                "summary": {"type": "string"},
                                "tags": {"type": "array", "items": {"type": "string"}},
                                "category": {"type": "string"},
                                "scope": {"type": "string"},
                                "parent_id": {"type": "string"},
                            },
                            "required": ["id", "type", "name", "summary", "tags"],
                        },
                    },
                    "graph_edges": {
                        "type": "array",
                        "description": (
                            "Relationships between graph nodes. Each edge has source (node id), "
                            "target (node id), and type (contains/follows_up/related/depends_on/initiated)."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "source": {"type": "string"},
                                "target": {"type": "string"},
                                "type": {"type": "string"},
                            },
                            "required": ["source", "target", "type"],
                        },
                    },
                    "project_name": {
                        "type": "string",
                        "description": "If this conversation is about a specific project, its name.",
                    },
                    "daily_category": {
                        "type": "string",
                        "description": (
                            "If this is not about a specific project, classify the everyday "
                            "conversation with a short category such as planning, learning, "
                            "health, finance, family, travel, errands, work, or general."
                        ),
                    },
                    "project_update": {
                        "type": "string",
                        "description": "New information to merge into the project profile. Only when relevant.",
                    },
                    "history_entry": {
                        "type": "string",
                        "description": "A paragraph summarizing key events/decisions/topics. Start with [YYYY-MM-DD HH:MM].",
                    },
                    "memory_update": {
                        "type": ["string", "null"],
                        "description": "Full updated MEMORY.md content, or null if no new persistent user facts discovered.",
                    },
                },
                "required": ["graph_nodes", "graph_edges", "history_entry"],
            },
        },
    }
]


def _ensure_text(value: Any) -> str:
    """Normalize tool-call payload values to text for file storage."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _normalize_save_memory_args(args: Any) -> dict[str, Any] | None:
    """Normalize provider tool-call arguments to the expected dict shape."""
    if isinstance(args, str):
        args = json.loads(args)
    if isinstance(args, list):
        return args[0] if args and isinstance(args[0], dict) else None
    return args if isinstance(args, dict) else None

_TOOL_CHOICE_ERROR_MARKERS = (
    "tool_choice",
    "toolchoice",
    "does not support",
    'should be ["none", "auto"]',
)


def _atomic_write_text(path: Path, content: str) -> None:
    """Write text via a same-directory temp file, then atomically replace."""
    ensure_dir(path.parent)
    tmp = path.with_name(f".{path.name}.{id(content)}.tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def _is_tool_choice_unsupported(content: str | None) -> bool:
    """Detect provider errors caused by forced tool_choice being unsupported."""
    text = (content or "").lower()
    return any(m in text for m in _TOOL_CHOICE_ERROR_MARKERS)


def _utcnow() -> datetime:
    return datetime.now()


def _safe_iso(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value
    return _utcnow().isoformat()


def _clip_text(value: Any, limit: int = 200) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def _signal_score(text: str) -> float:
    normalized = _normalize_whitespace(text)
    if not normalized:
        return 0.0
    unique_ratio = len(set(normalized.lower().split())) / max(1, len(normalized.split()))
    length_score = min(len(normalized) / 120, 1.0)
    return round((unique_ratio * 0.45) + (length_score * 0.55), 3)


def _keystream_xor(data: bytes, key: bytes, nonce: bytes) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < len(data):
        block = hmac.new(key, nonce + counter.to_bytes(4, "big"), hashlib.sha256).digest()
        out.extend(block)
        counter += 1
    return bytes(a ^ b for a, b in zip(data, out[: len(data)]))


class MemoryCipher:
    """Small dependency-free encryption helper for sensitive memory payloads."""

    def __init__(self, key_material: str | None):
        self._key = hashlib.sha256((key_material or "").encode("utf-8")).digest() if key_material else None

    @property
    def enabled(self) -> bool:
        return self._key is not None

    def encrypt(self, text: str) -> dict[str, str] | None:
        if not self._key or not text:
            return None
        nonce = os.urandom(16)
        cipher = _keystream_xor(text.encode("utf-8"), self._key, nonce)
        return {
            "alg": "hmac-sha256-xor-v1",
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "ciphertext": base64.b64encode(cipher).decode("ascii"),
        }

    def decrypt(self, payload: dict[str, Any] | None) -> str | None:
        if not self._key or not isinstance(payload, dict):
            return None
        nonce = payload.get("nonce")
        ciphertext = payload.get("ciphertext")
        if not isinstance(nonce, str) or not isinstance(ciphertext, str):
            return None
        try:
            raw_nonce = base64.b64decode(nonce)
            raw_cipher = base64.b64decode(ciphertext)
        except Exception:
            return None
        try:
            plain = _keystream_xor(raw_cipher, self._key, raw_nonce)
            return plain.decode("utf-8")
        except Exception:
            return None


# ── keyword tokenizer ──────────────────────────────────────────────────────


def _tokenize(text: str) -> set[str]:
    """Lowercase word extraction for keyword matching."""
    return {w.lower() for w in re.findall(r"\w+", text) if len(w) > 1}


def _token_list(text: str) -> list[str]:
    """Lowercase token stream for sparse scoring and light chunk sizing."""
    return [w.lower() for w in re.findall(r"\w+", text) if len(w) > 1]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity (no numpy dependency)."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── MemoryGraph ─────────────────────────────────────────────────────────────


class MemoryGraph:
    """Medium-term memory as a knowledge graph with embedding vectors.

    Hybrid GraphRAG retrieval:
    1. Embed query → ANN / cosine similarity → seed nodes
    2. 1-hop subgraph expansion
    3. Project boost + time decay → ranked results

    Storage: ``memory/graph.json`` — a single JSON file with nodes and edges.
    Nodes store their embedding inline as a float list.

    ANN indexing (hnswlib) is used when available (>50 nodes), falling back
    to brute-force cosine similarity otherwise.
    """

    _ANN_MIN_NODES = 50   # Build ANN index only above this threshold
    _ANN_EF_CONSTRUCTION = 200
    _ANN_M = 32

    def __init__(self, memory_dir: Path):
        self.memory_dir = ensure_dir(memory_dir)
        self.graph_file = self.memory_dir / "graph.json"
        self._graph: dict = {"nodes": [], "edges": []}
        self.audit_file = self.memory_dir / "audit.jsonl"
        self.cipher = MemoryCipher(os.getenv("NANOBOT_MEMORY_SECRET"))
        self.last_retrieval_telemetry: dict[str, Any] = {}

        # ANN index (lazy-built, hnswlib)
        self._index = None
        self._index_ids: list[str] = []
        self._index_dirty = True
        self._hnswlib_available = self._check_hnswlib()

    @staticmethod
    def _check_hnswlib() -> bool:
        """Check if hnswlib is available. Returns False if not installed."""
        try:
            import hnswlib  # noqa: F401
            return True
        except ImportError:
            return False

    def _build_index(self) -> None:
        """Build or rebuild the HNSW index from graph nodes that have embeddings."""
        try:
            import numpy as np
        except ImportError:
            self._index_dirty = False
            return

        self._index = None
        self._index_ids = []

        embed_nodes = [
            (i, n) for i, n in enumerate(self._graph["nodes"])
            if isinstance(n.get("embedding"), list) and len(n["embedding"]) >= 2
        ]
        if len(embed_nodes) < self._ANN_MIN_NODES:
            self._index_dirty = False
            return

        dim = len(embed_nodes[0][1]["embedding"])
        ids = np.array([n[0] for n in embed_nodes], dtype=np.int64)
        data = np.array([n[1]["embedding"] for n in embed_nodes], dtype=np.float32)

        try:
            import hnswlib
            index = hnswlib.Index(space="cosine", dim=dim)
            index.init_index(
                max_elements=max(len(embed_nodes) * 2, 200),
                ef_construction=self._ANN_EF_CONSTRUCTION,
                M=self._ANN_M,
            )
            index.add_items(data, ids)
            self._index = index
            self._index_ids = [n[1]["id"] for n in embed_nodes]
            self._index_dirty = False
            logger.debug(
                "Built ANN index with {} nodes (dim={})",
                len(embed_nodes), dim,
            )
        except Exception as exc:
            logger.warning("Failed to build ANN index: {}", exc)

    def _ann_search(
        self,
        query_vec: list[float],
        top_k: int = 30,
    ) -> list[tuple[float, dict]]:
        """ANN search using hnswlib index. Falls back to brute-force if unavailable."""
        try:
            import numpy as np
        except ImportError:
            return self._brute_force_search(query_vec, self._graph["nodes"], top_k)

        if self._index_dirty or self._index is None:
            self._build_index()

        if self._index is not None:
            try:
                q = np.array([query_vec], dtype=np.float32)
                labels, distances = self._index.knn_query(q, k=min(top_k, self._index.element_count))
                node_map = {i: n for i, n in enumerate(self._graph["nodes"])}
                results = []
                for label, dist in zip(labels[0], distances[0]):
                    if label < len(self._graph["nodes"]):
                        node = node_map.get(label)
                        if node is not None:
                            similarity = 1.0 - float(dist)
                            results.append((similarity, node))
                return results
            except Exception as exc:
                logger.debug("ANN search failed, falling back to brute-force: {}", exc)

        return self._brute_force_search(query_vec, self._graph["nodes"], top_k)

    @staticmethod
    def _brute_force_search(
        query_vec: list[float],
        candidates: list[dict],
        top_k: int = 5,
        min_similarity: float = 0.25,
    ) -> list[tuple[float, dict]]:
        """Brute-force cosine similarity over candidate nodes that have embeddings."""
        scored = []
        for node in candidates:
            emb = node.get("embedding")
            if not emb or len(emb) < 2 or len(emb) != len(query_vec):
                continue
            sim = _cosine_similarity(query_vec, emb)
            if sim >= min_similarity:
                scored.append((sim, node))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

        # ANN index (lazy-built, hnswlib)
        self._index = None
        self._index_ids: list[str] = []
        self._index_dirty = True
        self._hnswlib_available = self._check_hnswlib()

    @staticmethod
    def _is_sensitive_text(text: str) -> bool:
        import re

        patterns = (
            r"\bapi[_-]?key\b",
            r"\baccess[_-]?token\b",
            r"\brefresh[_-]?token\b",
            r"\bsecret\b",
            r"\bpassword\b",
            r"\bsk-[A-Za-z0-9_\-]{8,}\b",
            r"\bAKIA[0-9A-Z]{16}\b",
        )
        lowered = text.lower()
        return any(re.search(pattern, lowered, re.IGNORECASE) for pattern in patterns)

    @classmethod
    def _node_text(cls, node: dict) -> str:
        tags = " ".join(str(t) for t in node.get("tags", []) if t)
        return " ".join([
            str(node.get("name", "")),
            str(node.get("body_text") or node.get("summary", "")),
            tags,
        ]).strip()

    @staticmethod
    def _semantic_segments(text: str) -> list[str]:
        """Split text on stable semantic-ish boundaries without heavy NLP deps."""
        normalized = _normalize_whitespace(text)
        if not normalized:
            return []
        blocks = [
            block.strip()
            for block in re.split(r"(?:\n\s*){2,}", text)
            if block.strip()
        ]
        if not blocks:
            blocks = [normalized]

        segments: list[str] = []
        for block in blocks:
            if re.match(r"^\s*(#{1,6}\s+|\d+[.)]\s+|[-*]\s+)", block):
                segments.append(_normalize_whitespace(block))
                continue
            parts = re.split(r"(?<=[.!?。！？])\s+", block.strip())
            segments.extend(_normalize_whitespace(part) for part in parts if part.strip())
        return [segment for segment in segments if segment]

    @classmethod
    def _semantic_chunks(
        cls,
        text: str,
        *,
        min_tokens: int = 300,
        max_tokens: int = 800,
    ) -> list[str]:
        """Build chunks from paragraph/list/sentence boundaries."""
        segments = cls._semantic_segments(text)
        if not segments:
            return []

        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        def flush() -> None:
            nonlocal current, current_tokens
            if current:
                chunks.append("\n".join(current).strip())
                current = []
                current_tokens = 0

        for segment in segments:
            tokens = _token_list(segment)
            token_count = len(tokens)
            if token_count > max_tokens:
                flush()
                words = segment.split()
                if len(words) <= max_tokens:
                    chunks.append(segment)
                    continue
                for idx in range(0, len(words), max_tokens):
                    chunks.append(" ".join(words[idx: idx + max_tokens]).strip())
                continue
            if current and current_tokens + token_count > max_tokens:
                flush()
            current.append(segment)
            current_tokens += token_count
            if current_tokens >= min_tokens:
                flush()
        flush()
        return chunks

    @staticmethod
    def _window_text(text: str, max_tokens: int, *, tail: bool = False) -> str:
        words = text.split()
        if len(words) <= max_tokens:
            return text.strip()
        selected = words[-max_tokens:] if tail else words[:max_tokens]
        return " ".join(selected).strip()

    @classmethod
    def semantic_chunk_nodes(cls, nodes: list[dict]) -> list[dict]:
        """Expand long nodes into semantic chunks with neighbor context windows."""
        prepared: list[dict] = []
        for node in nodes:
            summary = str(node.get("summary", "")).strip()
            chunks = cls._semantic_chunks(summary)
            if len(chunks) <= 1:
                enriched = dict(node)
                enriched.setdefault("doc_id", node.get("doc_id") or node.get("id"))
                enriched.setdefault("chunk_id", node.get("chunk_id") or node.get("id"))
                enriched.setdefault("chunk_position", int(node.get("chunk_position", 0) or 0))
                enriched.setdefault("title", node.get("title") or node.get("name"))
                enriched.setdefault("body_text", summary)
                enriched.setdefault("expanded_context_text", summary)
                prepared.append(enriched)
                continue

            base_id = str(node.get("id"))
            for idx, body in enumerate(chunks):
                chunk = dict(node)
                chunk["id"] = f"{base_id}:chunk:{idx + 1}"
                chunk["doc_id"] = node.get("doc_id") or base_id
                chunk["canonical_source_id"] = base_id
                chunk["chunk_id"] = chunk["id"]
                chunk["chunk_position"] = idx
                chunk["chunk_count"] = len(chunks)
                chunk["title"] = node.get("title") or node.get("name")
                chunk["name"] = f"{node.get('name', base_id)} (part {idx + 1}/{len(chunks)})"
                left = cls._window_text(chunks[idx - 1], 160, tail=True) if idx > 0 else ""
                right = cls._window_text(chunks[idx + 1], 160) if idx + 1 < len(chunks) else ""
                chunk["body_text"] = body
                chunk["expanded_context_text"] = "\n".join(
                    part for part in [left, body, right] if part
                )
                chunk["summary"] = body
                prepared.append(chunk)
        return prepared

    @classmethod
    def _generalization_value(cls, node: dict) -> float:
        text = cls._node_text(node)
        base = _signal_score(text)
        node_type = str(node.get("type", "")).lower()
        if node_type in {"decision", "fact", "project"}:
            base += 0.2
        if node.get("importance") in {"high", "critical"}:
            base += 0.2
        if len(node.get("tags", [])) >= 2:
            base += 0.1
        return round(min(base, 1.0), 3)

    @classmethod
    def _has_conflict(cls, old: dict, new: dict) -> bool:
        old_summary = _normalize_whitespace(str(old.get("summary", "")))
        new_summary = _normalize_whitespace(str(new.get("summary", "")))
        if old_summary and new_summary and old_summary != new_summary:
            return True
        old_name = _normalize_whitespace(str(old.get("name", "")))
        new_name = _normalize_whitespace(str(new.get("name", "")))
        return bool(old_name and new_name and old_name != new_name)

    @staticmethod
    def _importance_score(node: dict) -> float:
        raw = node.get("importance_score")
        if isinstance(raw, (int, float)):
            return float(raw)
        label = str(node.get("importance", "")).lower()
        return {
            "critical": 1.0,
            "high": 0.85,
            "medium": 0.55,
            "low": 0.25,
        }.get(label, 0.35)

    def _write_audit(self, event: str, **payload: Any) -> None:
        record = {
            "timestamp": _utcnow().isoformat(),
            "event": event,
            **payload,
        }
        existing = self.audit_file.read_text(encoding="utf-8") if self.audit_file.exists() else ""
        _atomic_write_text(self.audit_file, existing + json.dumps(record, ensure_ascii=False) + "\n")

    def _protect_sensitive_node(self, node: dict) -> dict:
        protected = dict(node)
        text = self._node_text(protected)
        if not text or not self._is_sensitive_text(text):
            return protected
        protected["is_sensitive"] = True
        protected["needs_confirmation"] = True
        protected["summary_preview"] = _clip_text(protected.get("summary", ""), 80)
        encrypted = self.cipher.encrypt(str(protected.get("summary", "")))
        if encrypted:
            protected["summary_encrypted"] = encrypted
            protected["summary"] = "[encrypted sensitive memory]"
            for field in ("body_text", "expanded_context_text"):
                field_value = str(protected.get(field, ""))
                if field_value:
                    field_encrypted = self.cipher.encrypt(field_value)
                    if field_encrypted:
                        protected[f"{field}_encrypted"] = field_encrypted
                protected[field] = "[encrypted sensitive memory]"
        else:
            protected["summary"] = "[sensitive memory omitted until encryption key is configured]"
            protected["body_text"] = "[sensitive memory omitted until encryption key is configured]"
            protected["expanded_context_text"] = "[sensitive memory omitted until encryption key is configured]"
        self._write_audit(
            "sensitive_memory_detected",
            node_id=str(protected.get("id", "")),
            node_type=str(protected.get("type", "")),
        )
        return protected

    def _restore_summary(self, node: dict) -> dict:
        restored = dict(node)
        encrypted = restored.get("summary_encrypted")
        if encrypted:
            plain = self.cipher.decrypt(encrypted)
            if plain:
                restored["summary"] = plain
        for field in ("body_text", "expanded_context_text"):
            encrypted_field = restored.get(f"{field}_encrypted")
            if encrypted_field:
                plain = self.cipher.decrypt(encrypted_field)
                if plain:
                    restored[field] = plain
        return restored

    def _filter_noise(self, nodes: list[dict]) -> list[dict]:
        filtered: list[dict] = []
        for node in nodes:
            if node.get("persistent") or node.get("type") in {"scope", "project", "category"}:
                stored = dict(node)
                stored.setdefault("importance_score", self._importance_score(stored))
                stored["generalization_score"] = 1.0
                filtered.append(self._protect_sensitive_node(stored))
                continue
            score = self._generalization_value(node)
            text = _normalize_whitespace(self._node_text(node))
            if score < 0.2 or (len(text) < 8 and len(node.get("tags", [])) < 2):
                self._write_audit(
                    "memory_filtered_low_value",
                    node_id=str(node.get("id", "")),
                    score=score,
                )
                continue
            stored = dict(node)
            stored.setdefault("importance_score", self._importance_score(stored))
            stored["generalization_score"] = score
            filtered.append(self._protect_sensitive_node(stored))
        return filtered

    @staticmethod
    def _label(value: Any, default: str | None = None) -> str | None:
        """Coerce optional hierarchy labels to clean text."""
        if value is None:
            return default
        text = str(value).strip()
        return text or default

    @staticmethod
    def _slug(value: str) -> str:
        """Stable JSON id fragment for hierarchy container nodes."""
        import re

        slug = re.sub(r"[^\w]+", "-", value.strip().lower()).strip("-")
        return slug or "general"

    @classmethod
    def _project_id(cls, project_name: str) -> str:
        return f"project:{project_name.strip()}"

    @classmethod
    def _daily_category_id(cls, category: str) -> str:
        return f"daily:{cls._slug(category)}"

    @staticmethod
    def _projects_root_node() -> dict:
        return {
            "id": "scope:projects",
            "type": "scope",
            "name": "Projects",
            "summary": "Top-level container for project-scoped medium-term memories.",
            "tags": ["project", "projects"],
            "scope": "projects",
            "persistent": True,
        }

    @staticmethod
    def _daily_root_node() -> dict:
        return {
            "id": "scope:daily",
            "type": "scope",
            "name": "Daily Conversations",
            "summary": "Top-level container for everyday conversation memories.",
            "tags": ["daily", "conversation"],
            "scope": "daily",
            "persistent": True,
        }

    @classmethod
    def _project_node(cls, project_name: str) -> dict:
        project_id = cls._project_id(project_name)
        return {
            "id": project_id,
            "type": "project",
            "name": project_name,
            "summary": f"Project-scoped memories for {project_name}.",
            "tags": ["project", project_name],
            "scope": "project",
            "scope_id": project_id,
            "parent_id": "scope:projects",
            "category": "project",
            "path": ["projects", project_name],
            "persistent": True,
        }

    @classmethod
    def _daily_category_node(cls, category: str) -> dict:
        category_id = cls._daily_category_id(category)
        return {
            "id": category_id,
            "type": "category",
            "name": f"Daily: {category}",
            "summary": f"Everyday conversation memories about {category}.",
            "tags": ["daily", "conversation", category],
            "scope": "daily",
            "scope_id": category_id,
            "parent_id": "scope:daily",
            "category": category,
            "path": ["daily", category],
            "persistent": True,
        }

    @staticmethod
    def _merge_node_metadata(base: dict, update: dict) -> dict:
        merged = dict(base)
        merged.update({k: v for k, v in update.items() if v not in (None, "", [])})
        if base.get("tags") or update.get("tags"):
            merged["tags"] = sorted({
                *(str(t) for t in base.get("tags", []) if t),
                *(str(t) for t in update.get("tags", []) if t),
            })
        return merged

    @classmethod
    def _project_name_from_nodes(cls, nodes: list[dict]) -> str | None:
        """Infer a project name when the model emitted a project node but omitted project_name."""
        for node in nodes:
            if node.get("type") == "project":
                name = cls._label(node.get("name"))
                if name:
                    return name
                node_id = str(node.get("id", ""))
                if node_id.startswith("project:"):
                    return cls._label(node_id.split(":", 1)[1])
            project = cls._label(node.get("project"))
            if node.get("scope") == "project" and project:
                return project
            parent_id = str(node.get("parent_id") or node.get("scope_id") or "")
            if parent_id.startswith("project:"):
                return cls._label(parent_id.split(":", 1)[1])
        return None

    @classmethod
    def _infer_daily_category(cls, nodes: list[dict]) -> str:
        """Best-effort category for non-project conversations."""
        text_parts: list[str] = []
        for node in nodes:
            text_parts.extend([
                str(node.get("name", "")),
                str(node.get("summary", "")),
                " ".join(str(t) for t in node.get("tags", []) if t),
            ])
        text = " ".join(text_parts).lower()
        buckets = (
            ("planning", ("plan", "schedule", "calendar", "todo", "task", "安排", "计划", "待办", "日程")),
            ("learning", ("learn", "study", "course", "paper", "reading", "学习", "课程", "论文", "读书")),
            ("health", ("health", "doctor", "sleep", "workout", "exercise", "健康", "医生", "睡眠", "运动")),
            ("finance", ("money", "budget", "finance", "invoice", "tax", "账单", "预算", "财务", "发票", "税")),
            ("family", ("family", "friend", "relationship", "孩子", "家人", "朋友", "关系")),
            ("travel", ("travel", "trip", "flight", "hotel", "旅行", "出差", "航班", "酒店")),
            ("errands", ("shopping", "buy", "errand", "delivery", "购买", "购物", "快递", "办事")),
            ("work", ("work", "meeting", "office", "career", "工作", "会议", "同事", "客户")),
        )
        for category, keywords in buckets:
            if any(keyword in text for keyword in keywords):
                return category
        return "general"

    @classmethod
    def prepare_hierarchy(
        cls,
        nodes: list[dict],
        *,
        project_name: Any = None,
        daily_category: Any = None,
    ) -> tuple[list[dict], list[dict]]:
        """Attach extracted memory nodes to project or daily hierarchy containers.

        The method is intentionally schema-compatible: old nodes still work, and new
        hierarchy metadata is added only as extra fields/edges in ``graph.json``.
        """
        valid_nodes = [dict(n) for n in nodes if isinstance(n, dict) and n.get("id")]
        if not valid_nodes:
            return [], []

        organized: list[dict] = []
        by_id: dict[str, dict] = {}
        edges: list[dict] = []

        def add_node(node: dict) -> None:
            node_id = str(node.get("id", ""))
            if not node_id:
                return
            if node_id in by_id:
                by_id[node_id].update(cls._merge_node_metadata(by_id[node_id], node))
                return
            stored = dict(node)
            by_id[node_id] = stored
            organized.append(stored)

        def add_edge(source: str, target: str, edge_type: str = "contains") -> None:
            if source and target and source != target:
                edges.append({"source": source, "target": target, "type": edge_type})

        project = cls._label(project_name) or cls._project_name_from_nodes(valid_nodes)
        if project:
            project_id = cls._project_id(project)
            add_node(cls._projects_root_node())
            add_node(cls._project_node(project))
            add_edge("scope:projects", project_id)
            for node in valid_nodes:
                node_id = str(node["id"])
                enriched = dict(node)
                if node_id == project_id or enriched.get("type") == "project":
                    enriched.setdefault("id", project_id)
                    enriched.setdefault("type", "project")
                    enriched.setdefault("name", project)
                    enriched.setdefault("scope", "project")
                    enriched.setdefault("scope_id", project_id)
                    enriched.setdefault("parent_id", "scope:projects")
                    enriched.setdefault("category", "project")
                    enriched.setdefault("path", ["projects", project])
                    enriched.setdefault("persistent", True)
                    add_node(enriched)
                    add_edge("scope:projects", node_id)
                    continue
                category = cls._label(enriched.get("category"), "general") or "general"
                enriched.setdefault("scope", "project")
                enriched.setdefault("project", project)
                enriched.setdefault("scope_id", project_id)
                enriched.setdefault("parent_id", project_id)
                enriched.setdefault("category", category)
                enriched.setdefault("path", ["projects", project, category])
                add_node(enriched)
                add_edge(project_id, node_id)
            return organized, edges

        fallback_category = cls._label(daily_category) or cls._infer_daily_category(valid_nodes)
        add_node(cls._daily_root_node())
        for node in valid_nodes:
            node_id = str(node["id"])
            enriched = dict(node)
            category = cls._label(enriched.get("category"), fallback_category) or "general"
            category_id = cls._daily_category_id(category)
            add_node(cls._daily_category_node(category))
            add_edge("scope:daily", category_id)
            if node_id == category_id or enriched.get("type") == "category":
                enriched.setdefault("scope", "daily")
                enriched.setdefault("scope_id", category_id)
                enriched.setdefault("parent_id", "scope:daily")
                enriched.setdefault("category", category)
                enriched.setdefault("path", ["daily", category])
                enriched.setdefault("persistent", True)
                add_node(enriched)
                continue
            enriched.setdefault("scope", "daily")
            enriched.setdefault("scope_id", category_id)
            enriched.setdefault("parent_id", category_id)
            enriched.setdefault("category", category)
            enriched.setdefault("path", ["daily", category])
            add_node(enriched)
            add_edge(category_id, node_id)
        return organized, edges

    # -- load / save ---------------------------------------------------------

    def _load(self) -> None:
        if self.graph_file.exists():
            try:
                self._graph = json.loads(self.graph_file.read_text(encoding="utf-8"))
            except Exception:
                self._graph = {"nodes": [], "edges": []}

    def _save(self) -> None:
        _atomic_write_text(
            self.graph_file,
            json.dumps(self._graph, ensure_ascii=False, indent=2),
        )

    # -- add / expire --------------------------------------------------------

    def add_nodes(self, nodes: list[dict]) -> None:
        """Insert or merge nodes. Matches on ``id`` and preserves existing metadata."""
        self._load()
        self._index_dirty = True  # force ANN index rebuild on next search
        nodes = self.semantic_chunk_nodes(nodes)
        nodes = self._filter_noise(nodes)
        existing = {n["id"]: i for i, n in enumerate(self._graph["nodes"])}
        now = _utcnow().isoformat()
        for node in nodes:
            if node["id"] in existing:
                old = self._graph["nodes"][existing[node["id"]]]
                merged = dict(old)
                merged.update({k: v for k, v in node.items() if v not in (None, "", [])})
                merged["first_seen"] = old.get("first_seen") or node.get("first_seen") or now
                merged["last_seen"] = node.get("last_seen") or now
                merged["n_mentions"] = int(old.get("n_mentions", 0)) + 1
                merged["access_count"] = int(old.get("access_count", 0))
                merged["importance_score"] = max(
                    self._importance_score(old),
                    self._importance_score(node),
                )
                merged["tags"] = sorted({
                    *(str(t) for t in old.get("tags", []) if t),
                    *(str(t) for t in node.get("tags", []) if t),
                })
                if self._has_conflict(old, node):
                    versions = list(old.get("versions", []))
                    versions.append({
                        "saved_at": old.get("last_seen") or old.get("first_seen") or now,
                        "name": old.get("name"),
                        "summary": old.get("summary"),
                        "summary_encrypted": old.get("summary_encrypted"),
                    })
                    merged["versions"] = versions[-5:]
                    merged["conflict"] = {
                        "needs_confirmation": True,
                        "replaced_at": now,
                        "previous_summary_preview": _clip_text(old.get("summary_preview") or old.get("summary", ""), 80),
                    }
                    self._write_audit(
                        "memory_conflict_detected",
                        node_id=node["id"],
                        kept="newer",
                        preserved_versions=len(merged["versions"]),
                    )
                self._graph["nodes"][existing[node["id"]]] = merged
            else:
                new_node = dict(node)
                new_node.setdefault("first_seen", now)
                new_node.setdefault("last_seen", now)
                new_node["n_mentions"] = int(new_node.get("n_mentions", 1))
                new_node.setdefault("access_count", 0)
                new_node.setdefault("importance_score", self._importance_score(new_node))
                self._graph["nodes"].append(new_node)
                existing[new_node["id"]] = len(self._graph["nodes"]) - 1
        self._save()

    def add_edges(self, edges: list[dict]) -> None:
        """Insert edges. Skips duplicates (same source + target + type)."""
        self._load()
        seen = {
            (e["source"], e["target"], e.get("type", "related"))
            for e in self._graph["edges"]
        }
        for edge in edges:
            key = (edge["source"], edge["target"], edge.get("type", "related"))
            if key not in seen:
                seen.add(key)
                edge.setdefault("weight", 1)
                self._graph["edges"].append(edge)
        self._save()

    def expire(self, max_age_days: int = 30, hard_delete_days: int = 90) -> int:
        """Mark stale memories after *max_age_days* and hard-delete after *hard_delete_days*."""
        self._load()
        stale_cutoff = _utcnow() - timedelta(days=max_age_days)
        delete_cutoff = _utcnow() - timedelta(days=hard_delete_days)
        keep_nodes = []
        removed_ids: set[str] = set()
        for node in self._graph["nodes"]:
            try:
                last = datetime.fromisoformat(node["last_seen"])
            except (ValueError, KeyError):
                keep_nodes.append(node)
                continue
            updated = dict(node)
            updated["stale"] = last < stale_cutoff
            if not node.get("persistent") and last < delete_cutoff:
                removed_ids.add(node["id"])
                self._write_audit("memory_hard_deleted", node_id=node["id"], last_seen=node.get("last_seen"))
            else:
                keep_nodes.append(updated)
        self._graph["nodes"] = keep_nodes
        # Prune orphan edges
        self._graph["edges"] = [
            e
            for e in self._graph["edges"]
            if e["source"] not in removed_ids and e["target"] not in removed_ids
        ]
        self._save()
        return len(removed_ids)

    def delete_node(self, node_id: str, *, reason: str = "user_requested") -> bool:
        """Delete one memory node and related edges."""
        self._load()
        before = len(self._graph["nodes"])
        self._graph["nodes"] = [n for n in self._graph["nodes"] if n.get("id") != node_id]
        if len(self._graph["nodes"]) == before:
            return False
        self._graph["edges"] = [
            e
            for e in self._graph["edges"]
            if e.get("source") != node_id and e.get("target") != node_id
        ]
        self._save()
        self._write_audit("memory_deleted", node_id=node_id, reason=reason)
        return True

    # -- retrieval -----------------------------------------------------------

    def _match_seeds(self, query: str, top_k: int = 5) -> list[dict]:
        """Keyword overlap scoring. Fast pre-filter before vector comparison."""
        return [node for _, node in self._sparse_search(query, top_k=top_k)]

    def _sparse_search(self, query: str, top_k: int = 30) -> list[tuple[float, dict]]:
        """BM25-inspired sparse recall over titles, summaries, tags, and hierarchy."""
        q_words = _tokenize(query)
        if not q_words:
            return []
        q_lower = query.lower()
        scored: list[tuple[float, dict]] = []
        for node in self._graph["nodes"]:
            if node.get("type") == "scope":
                continue
            title = str(node.get("title") or node.get("name", ""))
            body = str(node.get("body_text") or node.get("summary", ""))
            tags = " ".join(str(t) for t in node.get("tags", []))
            hierarchy = " ".join(
                str(v)
                for v in [
                    node.get("scope", ""),
                    node.get("category", ""),
                    node.get("project", ""),
                    " ".join(str(p) for p in node.get("path", []) if p),
                ]
            )
            n_words = _tokenize(title + " " + body + " " + tags + " " + hierarchy)
            overlap = len(q_words & n_words)
            if overlap <= 0:
                continue

            title_words = _tokenize(title)
            tag_words = _tokenize(tags)
            title_hits = len(q_words & title_words)
            tag_hits = len(q_words & tag_words)
            field_boost = (title_hits * 0.25) + (tag_hits * 0.15)
            phrase_boost = 0.2 if q_lower and q_lower in f"{title} {body}".lower() else 0.0
            length_norm = max(1.0, len(n_words) ** 0.5)
            score = (overlap / max(1, len(q_words))) + field_boost + phrase_boost
            score += min(overlap / length_norm, 0.25)
            scored.append((score, node))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def _vector_search(
        self,
        query_vec: list[float],
        candidates: list[dict],
        top_k: int = 5,
        min_similarity: float = 0.25,
    ) -> list[tuple[float, dict]]:
        """Cosine similarity over candidate nodes that have embeddings.

        Uses ANN index when available (>50 nodes), falls back to brute-force.
        """
        if self._hnswlib_available and len(candidates) > self._ANN_MIN_NODES:
            ann_results = self._ann_search(query_vec, top_k=top_k * 4)
            # Filter to only candidates that are in the candidate set
            candidate_ids = {n.get("id") for n in candidates if n.get("id")}
            filtered = [(s, n) for s, n in ann_results if n.get("id") in candidate_ids]
            return filtered[:top_k]

        # Brute-force fallback
        scored = []
        for node in candidates:
            emb = node.get("embedding")
            if not emb or len(emb) < 2 or len(emb) != len(query_vec):
                continue
            sim = _cosine_similarity(query_vec, emb)
            if sim >= min_similarity:
                scored.append((sim, node))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def _expand_subgraph(self, seed_ids: set[str]) -> list[dict]:
        """1-hop expansion: find all nodes connected to seed_ids via edges."""
        connected_ids = set(seed_ids)
        for seed_id in seed_ids:
            connected_ids.update(self._hierarchy_node_ids(seed_id))
        for edge in self._graph["edges"]:
            if edge["source"] in seed_ids:
                connected_ids.add(edge["target"])
            if edge["target"] in seed_ids:
                connected_ids.add(edge["source"])

        node_map = {n["id"]: n for n in self._graph["nodes"]}
        subgraph = []
        for nid in connected_ids:
            if nid in node_map:
                subgraph.append(node_map[nid])
        return subgraph

    def _hierarchy_node_ids(self, root_id: str) -> set[str]:
        """Return a hierarchy container and all known descendants."""
        node_ids = {root_id}
        changed = True
        while changed:
            changed = False
            for node in self._graph["nodes"]:
                node_id = node.get("id")
                if not node_id or node_id in node_ids:
                    continue
                if node.get("parent_id") in node_ids or node.get("scope_id") == root_id:
                    node_ids.add(node_id)
                    changed = True
            for edge in self._graph["edges"]:
                if edge.get("type", "related") != "contains":
                    continue
                target = edge.get("target")
                if edge.get("source") in node_ids and target and target not in node_ids:
                    node_ids.add(target)
                    changed = True
        return node_ids

    def _detect_project(self, query: str) -> str | None:
        """Check if the query references a known project node."""
        q_lower = query.lower()
        for node in self._graph["nodes"]:
            if node.get("type") != "project":
                continue
            name = node.get("name", "").lower()
            tags = [
                t.lower()
                for t in node.get("tags", [])
                if str(t).lower() not in {"project", "projects"}
            ]
            node_id_name = node.get("id", "").replace("project:", "").lower()
            if (
                (name and name in q_lower)
                or (node_id_name and node_id_name in q_lower)
                or any(t and t in q_lower for t in tags)
            ):
                return node["id"]
        return None

    def _detect_daily_category(self, query: str) -> str | None:
        """Check if the query references a known daily conversation category."""
        q_lower = query.lower()
        for node in self._graph["nodes"]:
            if node.get("type") != "category" or node.get("scope") != "daily":
                continue
            name = str(node.get("name", "")).lower().replace("daily:", "").strip()
            labels = [
                name,
                str(node.get("category", "")),
                *[
                    str(t)
                    for t in node.get("tags", [])
                    if str(t).lower() not in {"daily", "conversation"}
                ],
            ]
            if any(label and label.lower() in q_lower for label in labels):
                return node["id"]

        inferred = self._infer_daily_category([{"name": query, "summary": query, "tags": []}])
        if inferred == "general":
            return None
        category_id = self._daily_category_id(inferred)
        if any(n.get("id") == category_id for n in self._graph["nodes"]):
            return category_id
        return None

    def retrieve(
        self,
        query: str,
        *,
        max_entries: int = 8,
        query_vector: list[float] | None = None,
    ) -> list[dict]:
        """Hybrid GraphRAG retrieval.

        1. Dense and sparse recall run independently, then candidates are deduplicated
        2. 1-hop subgraph expansion
        3. Manual weighted fusion + metadata boosts → final ranking
        """
        started = time.perf_counter()
        self._load()
        if not self._graph["nodes"]:
            return []

        # Step 1: dense + sparse recall, keeping source-specific scores.
        dense_results: list[tuple[float, dict]] = []
        if query_vector:
            dense_results = self._vector_search(query_vector, self._graph["nodes"], top_k=30)
        sparse_results = self._sparse_search(query, top_k=30)
        if not dense_results and not sparse_results:
            self.last_retrieval_telemetry = {
                "query_length": len(query),
                "dense_candidates": 0,
                "sparse_candidates": 0,
                "merged_candidates": 0,
                "expanded_candidates": 0,
                "returned": 0,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
            return []

        dense_scores = {node["id"]: score for score, node in dense_results}
        sparse_scores = {node["id"]: score for score, node in sparse_results}
        candidate_nodes: dict[str, dict] = {}
        for _, node in [*dense_results, *sparse_results]:
            candidate_nodes[node["id"]] = node
        seed_ids = set(candidate_nodes)

        # Step 2: expand subgraph
        subgraph = self._expand_subgraph(seed_ids)

        # Step 3: project detection and boost
        project_id = self._detect_project(query)
        project_nodes: set[str] = set()
        if project_id:
            project_nodes = self._hierarchy_node_ids(project_id) | {
                n["id"] for n in self._expand_subgraph({project_id})
            }

        daily_id = self._detect_daily_category(query)
        daily_nodes: set[str] = set()
        if daily_id:
            daily_nodes = self._hierarchy_node_ids(daily_id) | {
                n["id"] for n in self._expand_subgraph({daily_id})
            }

        # Ranking: LambdaMART-ready features with a hand-tuned weighted fallback.
        score_map: dict[str, float] = {}
        source_map: dict[str, set[str]] = {}
        for node_id, score in dense_scores.items():
            score_map[node_id] = max(score_map.get(node_id, 0.0), score * 0.48)
            source_map.setdefault(node_id, set()).add("dense")
        max_sparse = max(sparse_scores.values(), default=1.0)
        for node_id, score in sparse_scores.items():
            normalized = score / max(max_sparse, 1e-9)
            score_map[node_id] = score_map.get(node_id, 0.0) + (normalized * 0.36)
            source_map.setdefault(node_id, set()).add("sparse")

        for node in subgraph:
            sid = node["id"]
            if sid not in score_map:
                score_map[sid] = 0.16  # base score for graph-expanded nodes
                source_map.setdefault(sid, set()).add("graph")

            # Project boost
            if sid in project_nodes:
                score_map[sid] *= 1.5
            if sid in daily_nodes:
                score_map[sid] *= 1.25

            # Time decay (>7 days starts decaying, >30 days is 0)
            try:
                last = datetime.fromisoformat(node.get("last_seen", ""))
                days = (_utcnow() - last).days
            except (ValueError, KeyError):
                days = 0
            recency = 1.0 if days <= 3 else max(0.1, 1.0 - (days - 3) / 60)
            frequency = min((int(node.get("n_mentions", 1)) + int(node.get("access_count", 0))) / 8, 1.0)
            importance = self._importance_score(node)
            if node.get("stale"):
                recency *= 0.45
            title = str(node.get("title") or node.get("name", ""))
            title_hit = bool(_tokenize(query) & _tokenize(title))
            sparse_exact = 0.08 if title_hit else 0.0
            hybrid_bonus = 0.08 if {"dense", "sparse"} <= source_map.get(sid, set()) else 0.0
            position = int(node.get("chunk_position", 0) or 0)
            position_prior = max(0.0, 0.04 - (position * 0.004))
            score_map[sid] = (
                (score_map[sid] * 0.52)
                + (recency * 0.17)
                + (frequency * 0.11)
                + (importance * 0.16)
                + sparse_exact
                + hybrid_bonus
                + position_prior
            )

        # Sort and return top entries
        sorted_ids = sorted(score_map, key=lambda k: score_map[k], reverse=True)
        node_map = {n["id"]: n for n in subgraph}
        result = []
        for nid in sorted_ids:
            if nid in node_map and score_map[nid] > 0:
                node = self._restore_summary(node_map[nid])
                node["access_count"] = int(node.get("access_count", 0)) + 1
                node["_score"] = round(score_map[nid], 3)
                node["_retrieval_sources"] = sorted(source_map.get(nid, {"graph"}))
                result.append(node)
        if result:
            result_ids = {node["id"] for node in result[:max_entries]}
            for idx, stored in enumerate(self._graph["nodes"]):
                if stored.get("id") in result_ids:
                    self._graph["nodes"][idx]["access_count"] = int(stored.get("access_count", 0)) + 1
            self._save()
        self.last_retrieval_telemetry = {
            "query_length": len(query),
            "dense_candidates": len(dense_results),
            "sparse_candidates": len(sparse_results),
            "merged_candidates": len(candidate_nodes),
            "expanded_candidates": len(subgraph),
            "returned": len(result[:max_entries]),
            "source_contribution": {
                "dense": sum(1 for item in result[:max_entries] if "dense" in item.get("_retrieval_sources", [])),
                "sparse": sum(1 for item in result[:max_entries] if "sparse" in item.get("_retrieval_sources", [])),
                "graph": sum(1 for item in result[:max_entries] if "graph" in item.get("_retrieval_sources", [])),
            },
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }
        return result[:max_entries]

    def get_project_node_ids(self, project_id: str) -> set[str]:
        """Return all node IDs connected to a given project node."""
        self._load()
        return self._hierarchy_node_ids(project_id) | {
            n["id"] for n in self._expand_subgraph({project_id})
        }

    def audit(self, *, stale_after_days: int = 30) -> dict[str, Any]:
        """Return an audit snapshot for medium-term memory quality."""
        self._load()
        now = _utcnow()
        report = {
            "timestamp": now.isoformat(),
            "total_nodes": len(self._graph["nodes"]),
            "stale_nodes": 0,
            "sensitive_nodes": 0,
            "conflicted_nodes": 0,
            "needs_confirmation": [],
        }
        for node in self._graph["nodes"]:
            try:
                days = (now - datetime.fromisoformat(node.get("last_seen", ""))).days
            except (ValueError, TypeError):
                days = 0
            if days >= stale_after_days:
                report["stale_nodes"] += 1
            if node.get("is_sensitive"):
                report["sensitive_nodes"] += 1
            if node.get("conflict"):
                report["conflicted_nodes"] += 1
            if node.get("needs_confirmation") or node.get("conflict", {}).get("needs_confirmation"):
                report["needs_confirmation"].append({
                    "id": node.get("id"),
                    "name": node.get("name"),
                    "summary_preview": node.get("summary_preview") or _clip_text(node.get("summary", ""), 80),
                })
        self._write_audit("memory_audit", **report)
        return report


# -- embedding helper that runs inside async context -------------------------

async def _embed_with_provider(
    provider: "LLMProvider", texts: list[str]
) -> list[list[float] | None]:
    """Call provider.embed() with error handling. Returns None per-text on failure."""
    try:
        result = await provider.embed(texts)
        if result and len(result) == len(texts):
            return result
    except Exception:
        pass
    return [None] * len(texts)


# ── ProjectProfile ──────────────────────────────────────────────────────────


class ProjectProfile:
    """Per-project aggregated summaries stored as markdown files.

    Path: ``memory/projects/{project_name}.md``
    """

    def __init__(self, memory_dir: Path):
        self.projects_dir = ensure_dir(memory_dir / "projects")

    def get(self, project_name: str) -> str | None:
        """Read a project profile, or None if it doesn't exist."""
        file = self._path(project_name)
        if file.exists():
            return file.read_text(encoding="utf-8")
        return None

    def update(self, project_name: str, content: str) -> None:
        """Overwrite a project profile with new content."""
        _atomic_write_text(self._path(project_name), content)

    def merge(self, project_name: str, new_info: str) -> None:
        """Append new information to an existing project profile."""
        existing = self.get(project_name) or f"# {project_name}\n\n"
        merged = existing.rstrip() + "\n\n" + new_info
        self.update(project_name, merged)

    def list_projects(self) -> list[str]:
        """Return names of all known projects."""
        return [
            p.stem
            for p in self.projects_dir.glob("*.md")
            if p.is_file()
        ]

    def _path(self, name: str) -> Path:
        safe = name.replace("/", "_").replace("\\", "_").replace(":", "_")
        return self.projects_dir / f"{safe}.md"


class MemoryStore:
    """Three-tier memory: graph (medium-term) + MEMORY.md (long-term) + HISTORY.md (archive)."""

    _MAX_FAILURES_BEFORE_RAW_ARCHIVE = 3

    def __init__(self, workspace: Path):
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.secure_memory_file = self.memory_dir / "MEMORY.secure.json"
        self.history_file = self.memory_dir / "HISTORY.md"
        self.graph = MemoryGraph(self.memory_dir)
        self.projects = ProjectProfile(self.memory_dir)
        self._consecutive_failures = 0
        self.cipher = self.graph.cipher

    def read_long_term(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        _atomic_write_text(self.memory_file, content)

    def _contains_sensitive_content(self, content: str) -> bool:
        return self.graph._is_sensitive_text(content)

    def _write_secure_memory(self, content: str) -> bool:
        encrypted = self.cipher.encrypt(content)
        if not encrypted:
            return False
        _atomic_write_text(
            self.secure_memory_file,
            json.dumps(
                {
                    "needs_user_confirmation": True,
                    "updated_at": _utcnow().isoformat(),
                    "payload": encrypted,
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        self.graph._write_audit("sensitive_long_term_memory_queued")
        return True

    def append_history(self, entry: str) -> None:
        existing = self.history_file.read_text(encoding="utf-8") if self.history_file.exists() else ""
        _atomic_write_text(self.history_file, existing + entry.rstrip() + "\n\n")

    @staticmethod
    def _is_container_entry(entry: dict) -> bool:
        return entry.get("type") in {"scope", "category"}

    @staticmethod
    def _entry_group_label(entry: dict) -> str:
        scope = str(entry.get("scope") or "")
        parent_id = str(entry.get("parent_id") or "")
        scope_id = str(entry.get("scope_id") or "")
        if entry.get("type") == "project":
            return f"Project: {entry.get('name') or entry.get('id', '').replace('project:', '')}"
        if scope == "project" or parent_id.startswith("project:") or scope_id.startswith("project:"):
            project = str(entry.get("project") or "")
            if not project:
                project_ref = scope_id if scope_id.startswith("project:") else parent_id
                project = project_ref.replace("project:", "") or "unknown"
            return f"Project: {project}"
        if scope == "daily" or parent_id.startswith("daily:") or scope_id.startswith("daily:"):
            return f"Daily: {entry.get('category') or 'general'}"
        return "General"

    @classmethod
    def _format_related_context(cls, entries: list[dict]) -> str:
        visible_entries = [e for e in entries if not cls._is_container_entry(e)]
        if not visible_entries:
            visible_entries = [e for e in entries if e.get("type") != "scope"]
        if not visible_entries:
            return ""

        lines = ["## Related Context"]
        groups: dict[str, list[dict]] = {}
        for entry in visible_entries:
            groups.setdefault(cls._entry_group_label(entry), []).append(entry)

        for label, group_entries in groups.items():
            lines.append(f"### {label}")
            for entry in group_entries:
                name = entry.get("name", "?")
                summary = entry.get("expanded_context_text") or entry.get("summary", "")
                tags = entry.get("tags", [])
                tag_str = f" [{', '.join(tags)}]" if tags else ""
                confirm = " [needs confirmation]" if entry.get("needs_confirmation") or entry.get("conflict", {}).get("needs_confirmation") else ""
                lines.append(f"- **{name}**{tag_str}{confirm}: {summary}")
        return "\n".join(lines)

    def get_memory_context(
        self, query: str | None = None, query_vector: list[float] | None = None
    ) -> str:
        """Build memory context with medium-term retrieval + project profile + long-term."""
        parts: list[str] = []

        # Medium-term: vector + keyword + graph retrieval
        if query:
            try:
                entries = self.graph.retrieve(query, max_entries=8, query_vector=query_vector)
                if entries:
                    related_context = self._format_related_context(entries)
                    if related_context:
                        parts.append(related_context)
            except Exception:
                pass

            # Project profile (if a project is detected)
            try:
                project_id = self.graph._detect_project(query)
                if project_id:
                    proj_name = project_id.replace("project:", "")
                    profile = self.projects.get(proj_name)
                    if profile:
                        parts.insert(0, f"## Project Profile: {proj_name}\n\n{profile}")
            except Exception:
                pass

        # Long-term
        long_term = self.read_long_term()
        if long_term:
            parts.append(f"## Long-term Memory\n{long_term}")

        return "\n\n".join(parts) if parts else ""

    @staticmethod
    def _format_messages(messages: list[dict]) -> str:
        lines = []
        for message in messages:
            if not message.get("content"):
                continue
            tools = f" [tools: {', '.join(message['tools_used'])}]" if message.get("tools_used") else ""
            lines.append(
                f"[{message.get('timestamp', '?')[:16]}] {message['role'].upper()}{tools}: {message['content']}"
            )
        return "\n".join(lines)

    def delete_memory(self, memory_id: str, *, reason: str = "user_requested") -> bool:
        """Delete an incorrect graph memory by ID."""
        return self.graph.delete_node(memory_id, reason=reason)

    async def apply_feedback(
        self,
        action: str,
        *,
        node_id: str | None = None,
        correct_summary: str | None = None,
        reason: str = "user_feedback",
    ) -> str:
        """Apply user-driven feedback to the memory system.

        Supported actions:
        - ``delete``: remove a graph node by ID
        - ``correct``: update a node's summary with corrected information

        Returns a human-readable result string.
        """
        action = action.lower()
        if action == "delete":
            if not node_id:
                return "Error: node_id is required for delete"
            ok = self.graph.delete_node(node_id, reason=reason)
            if ok:
                self.graph._write_audit(
                    "feedback_delete",
                    node_id=node_id,
                    reason=reason,
                )
                return f"Deleted memory node '{node_id}'"
            return f"Error: memory node '{node_id}' not found"

        if action == "correct":
            if not node_id or not correct_summary:
                return "Error: node_id and correct_summary are required for correct"
            self._load_graph_unchecked()
            updated = False
            for idx, node in enumerate(self.graph._graph["nodes"]):
                if node.get("id") == node_id:
                    old_summary = node.get("summary", "")
                    old_name = node.get("name", "")
                    # Save version history
                    versions = list(node.get("versions", []))
                    versions.append({
                        "saved_at": _utcnow().isoformat(),
                        "name": old_name,
                        "summary": old_summary,
                    })
                    self.graph._graph["nodes"][idx]["versions"] = versions[-5:]
                    self.graph._graph["nodes"][idx]["summary"] = correct_summary
                    self.graph._graph["nodes"][idx]["last_seen"] = _utcnow().isoformat()
                    self.graph._graph["nodes"][idx]["needs_confirmation"] = False
                    self.graph._write_audit(
                        "feedback_correct",
                        node_id=node_id,
                        old_summary_preview=_clip_text(old_summary, 80),
                        new_summary_preview=_clip_text(correct_summary, 80),
                        reason=reason,
                    )
                    updated = True
                    break
            if updated:
                self.graph._save()
                return f"Corrected memory node '{node_id}'"
            return f"Error: memory node '{node_id}' not found"

        return f"Error: unknown feedback action '{action}' (supported: delete, correct)"

    def _load_graph_unchecked(self) -> None:
        """Load graph data without triggering save-on-load."""
        if self.graph.graph_file.exists():
            try:
                self.graph._graph = json.loads(self.graph.graph_file.read_text(encoding="utf-8"))
            except Exception:
                self.graph._graph = {"nodes": [], "edges": []}

    def audit_memories(self) -> dict[str, Any]:
        """Run a memory audit across medium and long-term stores."""
        report = self.graph.audit()
        report["has_long_term_memory"] = self.memory_file.exists()
        report["has_secure_pending_memory"] = self.secure_memory_file.exists()
        return report

    async def consolidate(
        self,
        messages: list[dict],
        provider: LLMProvider,
        model: str,
    ) -> bool:
        """Consolidate messages into medium-term graph + HISTORY.md + optional long-term."""
        if not messages:
            return True

        current_memory = self.read_long_term()
        prompt = f"""Process this conversation and call the save_memory tool with your consolidation.

## Current Long-term Memory (user profile)
{current_memory or "(empty)"}

## Conversation to Process
{self._format_messages(messages)}"""

        chat_messages = [
            {"role": "system", "content": (
                "You are a memory consolidation agent. Extract structured knowledge from the "
                "conversation and call the save_memory tool. Always produce graph_nodes and "
                "history_entry. Only produce memory_update when you discover NEW persistent "
                "facts about the user (preferences, traits, habits, personal info). "
                "Use project_name when the conversation is about a specific project, repo, "
                "product, or app so related graph nodes are grouped together. When it is not "
                "project-specific, use daily_category to classify the everyday conversation "
                "into a short reusable category."
            )},
            {"role": "user", "content": prompt},
        ]

        try:
            forced = {"type": "function", "function": {"name": "save_memory"}}
            response = await provider.chat_with_retry(
                messages=chat_messages,
                tools=_SAVE_MEMORY_TOOL,
                model=model,
                tool_choice=forced,
            )

            if response.finish_reason == "error" and _is_tool_choice_unsupported(
                response.content
            ):
                logger.warning("Forced tool_choice unsupported, retrying with auto")
                response = await provider.chat_with_retry(
                    messages=chat_messages,
                    tools=_SAVE_MEMORY_TOOL,
                    model=model,
                    tool_choice="auto",
                )

            if not response.has_tool_calls:
                logger.warning(
                    "Memory consolidation: LLM did not call save_memory "
                    "(finish_reason={}, content_len={}, content_preview={})",
                    response.finish_reason,
                    len(response.content or ""),
                    (response.content or "")[:200],
                )
                return self._fail_or_raw_archive(messages)

            args = _normalize_save_memory_args(response.tool_calls[0].arguments)
            if args is None:
                logger.warning("Memory consolidation: unexpected save_memory arguments")
                return self._fail_or_raw_archive(messages)

            # ── graph nodes ────────────────────────────────────────────
            graph_nodes = [
                dict(n) for n in (args.get("graph_nodes") or []) if isinstance(n, dict)
            ]
            graph_edges = [
                dict(e) for e in (args.get("graph_edges") or []) if isinstance(e, dict)
            ]
            if graph_nodes:
                graph_nodes, hierarchy_edges = self.graph.prepare_hierarchy(
                    graph_nodes,
                    project_name=args.get("project_name"),
                    daily_category=args.get("daily_category") or args.get("category"),
                )
                graph_nodes = self.graph.semantic_chunk_nodes(graph_nodes)
                # Generate embeddings from focused chunk bodies, not neighbor windows.
                summaries = [n.get("body_text") or n.get("summary", "") for n in graph_nodes]
                embeddings = await _embed_with_provider(provider, summaries)
                for i, node in enumerate(graph_nodes):
                    if embeddings[i]:
                        node["embedding"] = embeddings[i]
                graph_edges = [*hierarchy_edges, *graph_edges]
                self.graph.add_nodes(graph_nodes)

            # ── graph edges ────────────────────────────────────────────
            if graph_edges:
                self.graph.add_edges(graph_edges)

            # ── project profile ────────────────────────────────────────
            proj_name = args.get("project_name")
            proj_update = args.get("project_update")
            if proj_name and proj_update:
                self.projects.merge(str(proj_name), str(proj_update))

            # ── history archive ────────────────────────────────────────
            entry = args.get("history_entry")
            if entry:
                entry = _ensure_text(entry).strip()
                if entry:
                    self.append_history(entry)

            # ── long-term user profile ─────────────────────────────────
            update = args.get("memory_update")
            if update and update != current_memory:
                update = _ensure_text(update)
                if self._contains_sensitive_content(update):
                    self._write_secure_memory(update)
                else:
                    self.write_long_term(update)

            # ── expire old medium-term entries ─────────────────────────
            self.graph.expire(max_age_days=30, hard_delete_days=90)

            self._consecutive_failures = 0
            logger.info("Memory consolidation done for {} messages", len(messages))
            return True
        except Exception:
            logger.exception("Memory consolidation failed")
            return self._fail_or_raw_archive(messages)

    def _fail_or_raw_archive(self, messages: list[dict]) -> bool:
        """Increment failure count; after threshold, raw-archive messages and return True."""
        self._consecutive_failures += 1
        if self._consecutive_failures < self._MAX_FAILURES_BEFORE_RAW_ARCHIVE:
            return False
        self._raw_archive(messages)
        self._consecutive_failures = 0
        return True

    def _raw_archive(self, messages: list[dict]) -> None:
        """Fallback: dump raw messages to HISTORY.md without LLM summarization."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.append_history(
            f"[{ts}] [RAW] {len(messages)} messages\n"
            f"{self._format_messages(messages)}"
        )
        logger.warning(
            "Memory consolidation degraded: raw-archived {} messages", len(messages)
        )


class MemoryConsolidator:
    """Owns consolidation policy, locking, and session offset updates."""

    _MAX_CONSOLIDATION_ROUNDS = 2  # reduced from 5 — emergency valve only

    def __init__(
        self,
        workspace: Path,
        provider: LLMProvider,
        model: str,
        sessions: SessionManager,
        context_window_tokens: int,
        build_messages: Callable[..., list[dict[str, Any]]],
        get_tool_definitions: Callable[[], list[dict[str, Any]]],
    ):
        self.store = MemoryStore(workspace)
        self.provider = provider
        self.model = model
        self.sessions = sessions
        self.context_window_tokens = context_window_tokens
        self._build_messages = build_messages
        self._get_tool_definitions = get_tool_definitions
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()

    def get_lock(self, session_key: str) -> asyncio.Lock:
        """Return the shared consolidation lock for one session."""
        return self._locks.setdefault(session_key, asyncio.Lock())

    async def consolidate_messages(self, messages: list[dict[str, object]]) -> bool:
        """Archive a selected message chunk into persistent memory."""
        return await self.store.consolidate(messages, self.provider, self.model)

    def pick_consolidation_boundary(
        self,
        session: Session,
        tokens_to_remove: int,
    ) -> tuple[int, int] | None:
        """Pick a user-turn boundary that removes enough old prompt tokens."""
        start = session.last_consolidated
        if start >= len(session.messages) or tokens_to_remove <= 0:
            return None

        removed_tokens = 0
        last_boundary: tuple[int, int] | None = None
        for idx in range(start, len(session.messages)):
            message = session.messages[idx]
            if idx > start and message.get("role") == "user":
                last_boundary = (idx, removed_tokens)
                if removed_tokens >= tokens_to_remove:
                    return last_boundary
            removed_tokens += estimate_message_tokens(message)

        return last_boundary

    def estimate_session_prompt_tokens(self, session: Session) -> tuple[int, str]:
        """Estimate current prompt size for the normal session history view."""
        history = session.get_history(max_messages=0)
        channel, chat_id = (session.key.split(":", 1) if ":" in session.key else (None, None))
        probe_messages = self._build_messages(
            history=history,
            current_message="[token-probe]",
            channel=channel,
            chat_id=chat_id,
        )
        return estimate_prompt_tokens_chain(
            self.provider,
            self.model,
            probe_messages,
            self._get_tool_definitions(),
        )

    async def archive_messages(self, messages: list[dict[str, object]]) -> bool:
        """Archive messages with guaranteed persistence (retries until raw-dump fallback)."""
        if not messages:
            return True
        for _ in range(self.store._MAX_FAILURES_BEFORE_RAW_ARCHIVE):
            if await self.consolidate_messages(messages):
                return True
        return True

    async def maybe_consolidate_by_tokens(self, session: Session) -> None:
        """Loop: archive old messages until prompt fits within half the context window."""
        if not session.messages or self.context_window_tokens <= 0:
            return

        lock = self.get_lock(session.key)
        async with lock:
            target = self.context_window_tokens // 2
            estimated, source = self.estimate_session_prompt_tokens(session)
            if estimated <= 0:
                return
            if estimated < int(self.context_window_tokens * 0.9):
                logger.debug(
                    "Token consolidation idle {}: {}/{} via {}",
                    session.key,
                    estimated,
                    self.context_window_tokens,
                    source,
                )
                return

            for round_num in range(self._MAX_CONSOLIDATION_ROUNDS):
                if estimated <= target:
                    return

                boundary = self.pick_consolidation_boundary(session, max(1, estimated - target))
                if boundary is None:
                    logger.debug(
                        "Token consolidation: no safe boundary for {} (round {})",
                        session.key,
                        round_num,
                    )
                    return

                end_idx = boundary[0]
                chunk = session.messages[session.last_consolidated:end_idx]
                if not chunk:
                    return

                logger.info(
                    "Token consolidation round {} for {}: {}/{} via {}, chunk={} msgs",
                    round_num,
                    session.key,
                    estimated,
                    self.context_window_tokens,
                    source,
                    len(chunk),
                )
                if not await self.consolidate_messages(chunk):
                    return
                session.last_consolidated = end_idx
                self.sessions.save(session)

                estimated, source = self.estimate_session_prompt_tokens(session)
                if estimated <= 0:
                    return

    async def consolidate_recent_turn(self, session: "Session") -> None:
        """Background: consolidate into medium-term memory when enough messages
        have accumulated (>= 6 unconsolidated).

        Keeps the last 3 messages in short-term history so the agent doesn't
        lose conversational context right after consolidation.
        """
        lock = self.get_lock(session.key)
        async with lock:
            start = session.last_consolidated
            # Keep last 3 messages unconsolidated for conversational continuity
            keep = min(3, len(session.messages) - start)
            end = len(session.messages) - keep
            if end <= start:
                return
            chunk = session.messages[start:end]
            if len(chunk) < 6:
                return
            ok = await self.store.consolidate(chunk, self.provider, self.model)
            if ok:
                session.last_consolidated = end
                self.sessions.save(session)

    async def update_running_summary(
        self, session: "Session", messages: list[dict]
    ) -> None:
        """Generate a lightweight running summary of early conversation turns.

        Only processes messages before the last 6 (which stay raw). The summary
        is stored in ``session.metadata["running_summary"]`` for injection on
        the next turn.
        """
        lock = self.get_lock(session.key)
        async with lock:
            from nanobot.session.manager import Session

            early = messages[:-6] if len(messages) > 12 else messages[:-3]
            early = Session.filter_short_term_context(early, keep_recent_messages=0)
            if not early or len(early) < 3:
                return

            existing = session.metadata.get("running_summary", "")
            prefix = "Previous summary:\n" + existing + "\n\n" if existing else ""
            prompt = (
                f"{prefix}New conversation turns:\n"
                f"{self.store._format_messages(early)}\n\n"
                "Write a 2-3 sentence running summary covering key topics, decisions, "
                "and action items from the FULL conversation (previous summary + new turns). "
                "Keep it concise — it will be injected at the top of the next system prompt."
            )
            try:
                response = await self.provider.chat_with_retry(
                    messages=[{"role": "user", "content": prompt}],
                    tools=None,
                    model=self.model,
                )
                if response.content and response.finish_reason != "error":
                    session.metadata["running_summary"] = response.content.strip()
                    self.sessions.save(session)
            except Exception:
                pass  # best-effort, don't block message processing
