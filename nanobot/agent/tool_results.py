"""Helpers for constraining tool results before they re-enter the LLM context."""

from __future__ import annotations

import hashlib
import math
from datetime import datetime
from pathlib import Path

from nanobot.providers.base import LLMProvider
from nanobot.utils.helpers import ensure_dir


class ToolResultCompressor:
    """Store oversized tool output and return a compact, model-friendly digest."""

    INLINE_CHAR_BUDGET = 12_000
    CHUNK_CHAR_BUDGET = 12_000
    CHUNK_OVERLAP = 500
    MAX_CHUNKS = 24
    CHUNK_SUMMARY_MAX_CHARS = 900

    def __init__(
        self,
        *,
        workspace: Path | None,
        provider: LLMProvider | None,
        model: str | None,
    ) -> None:
        self.workspace = workspace
        self.provider = provider
        self.model = model

    async def compress(self, tool_name: str, tool_call_id: str, result: str) -> str:
        """Return a bounded tool message, spilling the full result to disk when needed."""
        if len(result) <= self.INLINE_CHAR_BUDGET:
            return result

        artifact_path = self._write_artifact(tool_name, tool_call_id, result)
        preview = self._head_tail_preview(result, budget=self.INLINE_CHAR_BUDGET // 2)

        summary = None
        if self.provider and self.model:
            summary = await self._summarize_large_result(tool_name, result)

        lines = [
            f"[Tool result compressed: {tool_name}]",
            f"Original length: {len(result):,} chars",
        ]
        if artifact_path:
            lines.append(f"Stored full output: {artifact_path}")
        if summary:
            lines.extend(["", "Whole-document digest:", summary])
        lines.extend(
            [
                "",
                "Preview:",
                preview,
            ]
        )
        if artifact_path:
            lines.append("")
            lines.append("Use read_file with offset/limit to inspect the stored full output.")
        return "\n".join(lines)

    def _write_artifact(self, tool_name: str, tool_call_id: str, result: str) -> str | None:
        if self.workspace is None:
            return None

        artifact_dir = ensure_dir(self.workspace / "memory" / "artifacts" / "tool-results")
        digest = hashlib.sha1(f"{tool_name}:{tool_call_id}:{len(result)}".encode("utf-8")).hexdigest()[:10]
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = artifact_dir / f"{stamp}-{tool_name}-{digest}.txt"
        path.write_text(result, encoding="utf-8")
        return str(path)

    @staticmethod
    def _head_tail_preview(text: str, *, budget: int) -> str:
        head = max(1, budget // 2)
        tail = max(1, budget - head)
        omitted = max(0, len(text) - head - tail)
        return (
            text[:head]
            + f"\n\n... ({omitted:,} chars omitted) ...\n\n"
            + text[-tail:]
        )

    def _chunk_text(self, text: str) -> list[str]:
        if len(text) <= self.CHUNK_CHAR_BUDGET:
            return [text]

        step = max(1, self.CHUNK_CHAR_BUDGET - self.CHUNK_OVERLAP)
        chunks = []
        for idx in range(0, len(text), step):
            chunk = text[idx:idx + self.CHUNK_CHAR_BUDGET]
            if not chunk:
                break
            chunks.append(chunk)
            if idx + self.CHUNK_CHAR_BUDGET >= len(text):
                break
        return chunks

    async def _summarize_large_result(self, tool_name: str, result: str) -> str | None:
        chunks = self._chunk_text(result)
        total_chunks = len(chunks)
        if not chunks:
            return None

        summaries: list[str] = []
        for idx, chunk in enumerate(chunks[: self.MAX_CHUNKS], start=1):
            prompt = (
                f"You are compressing a large tool result from `{tool_name}`.\n\n"
                f"Summarize chunk {idx}/{total_chunks} for downstream reasoning.\n"
                "Return plain text with these sections:\n"
                "Summary:\nKey facts:\nOpen questions:\nSignals:\n\n"
                "Keep names, numbers, errors, and decisions. Do not mention formatting.\n\n"
                f"Chunk:\n{chunk}"
            )
            response = await self.provider.chat_with_retry(
                messages=[{"role": "user", "content": prompt}],
                tools=None,
                model=self.model,
            )
            if response.finish_reason == "error" or not response.content:
                return None
            summaries.append(
                f"[Chunk {idx}/{total_chunks}]\n{response.content.strip()[: self.CHUNK_SUMMARY_MAX_CHARS]}"
            )

        omitted_chunks = max(0, total_chunks - len(summaries))
        combine_prompt = (
            f"The following are chunk summaries from a large `{tool_name}` tool result.\n"
            "Produce a whole-document digest for another agent.\n"
            "Return plain text with these sections:\n"
            "Global summary:\nCritical details:\nNotable evidence:\nUnresolved points:\n"
        )
        if omitted_chunks:
            combine_prompt += (
                f"\nOnly the first {len(summaries)} chunks were summarized directly out of "
                f"{total_chunks}; state that coverage is partial.\n"
            )
        combine_prompt += "\n\n" + "\n\n".join(summaries)

        response = await self.provider.chat_with_retry(
            messages=[{"role": "user", "content": combine_prompt}],
            tools=None,
            model=self.model,
        )
        if response.finish_reason == "error" or not response.content:
            return None

        digest = response.content.strip()
        if omitted_chunks:
            digest += (
                f"\nCoverage note: summarized {len(summaries)} of {total_chunks} chunks "
                "before synthesis."
            )
        return digest
