"""Built-in eval scenario library grouped by domain."""

from nanobot.evals.scenario_library.email import EMAIL_SCENARIOS
from nanobot.evals.scenario_library.rag import RAG_SCENARIOS
from nanobot.evals.scenario_library.recovery import RECOVERY_SCENARIOS
from nanobot.evals.scenario_library.scheduled import SCHEDULED_SCENARIOS

DEFAULT_SCENARIOS: list[dict] = [
    *SCHEDULED_SCENARIOS,
    *EMAIL_SCENARIOS,
    *RECOVERY_SCENARIOS,
    *RAG_SCENARIOS,
]

__all__ = [
    "DEFAULT_SCENARIOS",
    "EMAIL_SCENARIOS",
    "RAG_SCENARIOS",
    "RECOVERY_SCENARIOS",
    "SCHEDULED_SCENARIOS",
]
