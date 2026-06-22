"""Context budget management — dynamically trim system prompt to fit model context window."""

from __future__ import annotations

from typing import Callable, TYPE_CHECKING

from nanobot.utils.helpers import estimate_prompt_tokens_chain

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider


class ContextBudget:
    """Calculate and allocate token budget for system prompt components.

    Priority (high → low when trimming):
      1. Identity + bootstrap files (AGENTS.md, SOUL.md, USER.md, TOOLS.md)
      2. Always-included skills
      3. Running summary (truncated)
      4. Medium-term memory (GraphRAG retrieval — lower-score entries trimmed first)
      5. Skill summaries (lowest priority, trimmed first)
    """

    # Minimum reserved for history + user message + tool definitions
    _HISTORY_RESERVE_RATIO = 0.35
    _MIN_SYSTEM_PROMPT = 512  # tokens — always keep at least this much system prompt

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        context_window_tokens: int,
        tool_definitions: list[dict] | Callable[[], list[dict]],
    ):
        self._provider = provider
        self._model = model
        self._context_window = context_window_tokens
        self._tool_definitions = tool_definitions

    @property
    def _tools(self) -> list[dict]:
        """Resolve tool definitions (lazy callable or direct list)."""
        if callable(self._tool_definitions):
            return self._tool_definitions()
        return self._tool_definitions

    @property
    def available(self) -> int:
        """Return max tokens available for system prompt content."""
        return max(
            self._MIN_SYSTEM_PROMPT,
            int(self._context_window * (1.0 - self._HISTORY_RESERVE_RATIO)),
        )

    def estimate(self, text: str) -> int:
        """Rough token estimate for a text string (fast, no API call)."""
        if not text:
            return 0
        estimated, _ = estimate_prompt_tokens_chain(
            self._provider,
            self._model,
            [{"role": "system", "content": text}],
            self._tools,
        )
        return estimated

    @staticmethod
    def trim_memory_context(memory_text: str, max_tokens: int) -> str:
        """Trim memory context by removing lowest-value entries (toward end).

        Memory format (from ``MemoryStore.get_memory_context``)::

            ## Related Context
            ### Project: foo
            - **entry** [tags]: summary
            ### Daily: general
            - **entry** [tags]: summary

            ## Long-term Memory
            ...

        Strategy: remove Related Context entries from the bottom up until
        the text fits within *max_tokens*.  Long-term Memory section is
        preserved if at all possible (it's compact).
        """
        if not memory_text:
            return ""

        # Rough token count: ~1 token per 4 chars for English text
        estimated = len(memory_text) // 4
        if estimated <= max_tokens:
            return memory_text

        # Split into sections
        sections = memory_text.split("\n## ")
        kept_sections: list[str] = []
        lt_memory: str | None = None

        for sec in sections:
            if sec.startswith("Long-term Memory"):
                lt_memory = sec
            else:
                kept_sections.append(sec)

        # Trim Related Context entries from the bottom
        trimmed = []
        for sec in kept_sections:
            if not sec.strip():
                continue
            lines = sec.split("\n")
            header = lines[:1]  # "## Related Context" or "### Group"
            entries = lines[1:]

            # Remove entries from the bottom until it fits
            while entries:
                test_text = "\n".join(trimmed + header + entries)
                if lt_memory:
                    test_text += "\n## " + lt_memory
                if len(test_text) // 4 <= max_tokens:
                    break
                entries.pop()

            trimmed.extend(header + entries)

        result = "\n".join(trimmed)
        if lt_memory:
            result += "\n## " + lt_memory

        return result

    @staticmethod
    def trim_summary(summary: str, max_tokens: int) -> str:
        """Truncate running summary to fit within token budget."""
        if not summary:
            return ""
        estimated = len(summary) // 4
        if estimated <= max_tokens:
            return summary
        max_chars = max_tokens * 4
        return summary[:max_chars] + "..."

    @staticmethod
    def trim_skills(skills_content: str, max_tokens: int) -> str:
        """Trim skills section to fit within token budget."""
        if not skills_content:
            return ""
        estimated = len(skills_content) // 4
        if estimated <= max_tokens:
            return skills_content
        max_chars = max_tokens * 4
        return skills_content[:max_chars] + "\n...(additional skills available via read_file)"
