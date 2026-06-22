"""LLM-as-Judge evaluator for agent responses and RAG quality."""

from __future__ import annotations

from typing import Any

from nanobot.evals.models import EvalObservation, EvalRunRecord


class LLMJudge:
    """Uses a configured LLM to evaluate response quality.

    Scores dimensions: relevance, helpfulness, faithfulness, overall.
    Each dimension is scored 1-5 with a reason.
    """

    _JUDGE_PROMPT = """You are an expert evaluator of AI assistant responses. Assess the response on these criteria:

## Relevance (1-5)
Does the response directly address the user's question? Is it on-topic?

## Helpfulness (1-5)
Does the response provide useful, actionable information? Does it solve the user's problem?

## Faithfulness (1-5)
Is the response accurate based on the provided context? Does it avoid hallucination or contradiction?

## Overall (1-5)
Your holistic judgment of response quality.

## Guidelines
- Be strict but fair. A score of 3 is "adequate" — not a failure.
- Mark hallucination harshly in faithfulness (1-2).
- Consider the task type: a code answer needs precision; a chat answer needs tone.

## Task
Evaluate the following:

### User Goal / Query
{query}

### Agent Response
{response}

### Context (provided to agent)
{context}

### Expected Criteria
{expected}

---

Return your evaluation in this exact JSON format:
```json
{{
  "relevance": {{"score": 1-5, "reason": "..."}},
  "helpfulness": {{"score": 1-5, "reason": "..."}},
  "faithfulness": {{"score": 1-5, "reason": "..."}},
  "overall": {{"score": 1-5, "reason": "..."}},
  "hallucinations": ["list specific hallucinated claims if any"],
  "strengths": ["key strengths"],
  "weaknesses": ["key weaknesses"]
}}
```"""

    def __init__(self, provider, model: str):
        self._provider = provider
        self._model = model

    async def evaluate(
        self,
        query: str,
        response: str,
        context: str = "",
        expected: str = "",
    ) -> dict[str, Any]:
        """Evaluate a single response using the judge LLM.

        Returns the parsed JSON evaluation dict.
        """
        prompt = self._JUDGE_PROMPT.format(
            query=query[:1000],
            response=response[:2000],
            context=context[:1500] if context else "(none)",
            expected=expected[:500] if expected else "(none)",
        )

        llm_response = await self._provider.chat(
            messages=[{"role": "user", "content": prompt}],
            model=self._model,
            temperature=0.1,
            max_tokens=1024,
        )

        result = self._parse_judgment(llm_response.content or "")
        return result

    @staticmethod
    def _parse_judgment(content: str) -> dict[str, Any]:
        """Extract JSON judgment from the LLM response."""
        import json
        import re

        # Try to extract JSON block
        json_match = re.search(r"```(?:json)?\s*\n?(\{.*?\})\s*\n?```", content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Fallback: try to find top-level JSON
        brace_start = content.find("{")
        brace_end = content.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            try:
                return json.loads(content[brace_start:brace_end + 1])
            except json.JSONDecodeError:
                pass

        # Last resort: return minimal structure
        return {
            "overall": {"score": 3, "reason": "Failed to parse LLM judge response"},
            "relevance": {"score": 3, "reason": "Parse failure"},
            "helpfulness": {"score": 3, "reason": "Parse failure"},
            "faithfulness": {"score": 3, "reason": "Parse failure"},
        }

    @staticmethod
    def merge_into_run_record(
        record: EvalRunRecord,
        judgment: dict[str, Any],
    ) -> EvalRunRecord:
        """Merge LLM judge scores into an existing run record's metrics."""
        for dim in ("relevance", "helpfulness", "faithfulness", "overall"):
            dim_data = judgment.get(dim, {})
            score = dim_data.get("score")
            if score is not None:
                record.metrics[f"llm_{dim}"] = float(score)

        h = judgment.get("hallucinations", [])
        if h:
            record.metrics["hallucination_count"] = float(len(h))
            record.failure_modes.append("llm_detected_hallucination")

        record.metrics["judge_quality"] = round(
            sum(
                float(judgment.get(d, {}).get("score", 3) or 3)
                for d in ("relevance", "helpfulness", "faithfulness", "overall")
            ) / 4,
            2,
        )
        return record
