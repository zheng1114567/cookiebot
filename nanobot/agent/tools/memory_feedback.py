"""Memory feedback tool — lets the LLM correct or delete memory based on user feedback."""

from typing import Any

from nanobot.agent.memory import MemoryStore
from nanobot.agent.tools.base import Tool


class FeedbackMemoryTool(Tool):
    """Tool that allows the agent to correct or delete memory entries."""

    def __init__(self, memory_store: MemoryStore):
        self._memory = memory_store

    @property
    def name(self) -> str:
        return "feedback_memory"

    @property
    def description(self) -> str:
        return (
            "Correct or delete a memory entry based on user feedback. "
            "Use 'delete' when the user says a memory is wrong and wants it removed. "
            "Use 'correct' when the user provides the correct information to replace a memory."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["delete", "correct"],
                    "description": "What to do: 'delete' removes the memory node; 'correct' updates its summary.",
                },
                "node_id": {
                    "type": "string",
                    "description": "The id of the memory graph node to modify (e.g. 'fact:xxx', 'topic:yyy'). Required for both actions.",
                },
                "correct_summary": {
                    "type": "string",
                    "description": "The corrected summary text. Required when action='correct'.",
                },
                "reason": {
                    "type": "string",
                    "description": "Optional reason for the change, e.g. 'user_says_wrong'.",
                },
            },
            "required": ["action", "node_id"],
        }

    async def execute(self, **kwargs: Any) -> str:
        action = str(kwargs.get("action", "")).lower()
        node_id = kwargs.get("node_id")
        correct_summary = kwargs.get("correct_summary")
        reason = str(kwargs.get("reason", "user_feedback"))

        return await self._memory.apply_feedback(
            action,
            node_id=node_id,
            correct_summary=correct_summary,
            reason=reason,
        )
