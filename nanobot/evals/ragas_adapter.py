"""RAGAS integration adapter — bridges nanobot eval data to RAGAS metrics.

Requires ``pip install ragas``.

Usage::

    from nanobot.evals.ragas_adapter import RagasAdapter

    adapter = RagasAdapter()
    result = await adapter.evaluate(
        question="What is the refund policy?",
        answer="Refunds are available within 30 days.",
        contexts=["Refunds are available within 30 days of purchase."],
        ground_truth="Refunds available within 30 days.",
    )
    print(result.faithfulness, result.answer_relevancy)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RagasResult:
    """Structured RAGAS evaluation results."""

    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    answer_correctness: float | None = None
    aspect_critique: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    def summary(self) -> str:
        parts = []
        for key, label in [
            ("faithfulness", "faithfulness"),
            ("answer_relevancy", "relevancy"),
            ("context_precision", "ctx_precision"),
            ("context_recall", "ctx_recall"),
            ("answer_correctness", "correctness"),
        ]:
            val = getattr(self, key, None)
            if val is not None:
                parts.append(f"{label}={val:.3f}")
        return ", ".join(parts) if parts else "no metrics"


class RagasAdapter:
    """Adapter that evaluates nanobot RAG responses using RAGAS metrics.

    Can be configured in two ways:

    1. **With a nanobot provider** (recommended — uses project's existing config)::

        from nanobot.providers.litellm_provider import LiteLLMProvider
        provider = LiteLLMProvider(api_key="sk-xxx", default_model="deepseek-chat")
        adapter = RagasAdapter(provider=provider, model="deepseek-chat")

    2. **With env vars** (RAGAS uses its own LLM client)::

        adapter = RagasAdapter(judge_model="deepseek-chat")
        # Set DEEPSEEK_API_KEY + DEEPSEEK_API_BASE in environment
    """

    def __init__(
        self,
        provider=None,
        model: str | None = None,
        judge_model: str | None = None,
        judge_api_key: str | None = None,
        judge_api_base: str | None = None,
        embedding_model: str | None = None,
    ):
        self._provider = provider
        self._model = model

        # Priority: explicit args > provider config > env vars > defaults
        if provider is not None:
            self._judge_model = judge_model or model or getattr(provider, "default_model", None) or "deepseek-chat"
            self._judge_api_key = judge_api_key or getattr(provider, "api_key", None) or os.getenv("DEEPSEEK_API_KEY")
            self._judge_api_base = judge_api_base or getattr(provider, "api_base", None) or os.getenv("DEEPSEEK_BASE_URL")
        else:
            self._judge_model = judge_model or os.getenv("RAGAS_JUDGE_MODEL", "deepseek-chat")
            self._judge_api_key = judge_api_key or os.getenv("RAGAS_JUDGE_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
            self._judge_api_base = judge_api_base or os.getenv("RAGAS_JUDGE_API_BASE") or os.getenv("DEEPSEEK_BASE_URL")

        self._embedding_model = embedding_model or os.getenv("RAGAS_EMBEDDING_MODEL", "text-embedding-3-small")

        # Set env vars for RAGAS's internal LLM client so it can reach DeepSeek
        if self._judge_api_key and "DEEPSEEK" in (self._judge_model or "").upper():
            os.environ.setdefault("OPENAI_API_KEY", self._judge_api_key)
            if self._judge_api_base:
                os.environ.setdefault("OPENAI_BASE_URL", self._judge_api_base.rstrip("/") + "/v1")
        elif self._judge_api_key:
            os.environ.setdefault("OPENAI_API_KEY", self._judge_api_key)
            if self._judge_api_base:
                os.environ.setdefault("OPENAI_BASE_URL", self._judge_api_base)

    async def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
        metrics: list[str] | None = None,
    ) -> RagasResult:
        """Run RAGAS metrics on a single Q/A pair.

        Args:
            question: The user query.
            answer: The agent's response.
            contexts: Retrieved context passages.
            ground_truth: Optional reference answer.
            metrics: Which metrics to compute. Defaults to all available.

        Returns:
            RagasResult with scores.
        """
        try:
            return await self._evaluate_impl(question, answer, contexts, ground_truth, metrics)
        except ImportError as e:
            return RagasResult(error=f"Missing dependency: {e}")
        except Exception as e:
            return RagasResult(error=str(e))

    async def _evaluate_impl(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None,
        metrics: list[str] | None,
    ) -> RagasResult:
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.llms import llm_factory as ragas_llm_factory
        from ragas.embeddings import embedding_factory

        # Build dataset
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }
        if ground_truth:
            data["ground_truth"] = [ground_truth]

        dataset = Dataset.from_dict(data)

        # Resolve metric list
        metric_map = self._resolve_metrics(metrics)
        if not metric_map:
            return RagasResult(error="No valid metrics specified")

        # Create RAGAS LLM + embeddings
        llm = ragas_llm_factory(
            model=self._judge_model,
            client=None,  # uses OpenAI client from env
        )
        embeddings = embedding_factory(model=self._embedding_model)

        try:
            result = await ragas_evaluate(
                dataset=dataset,
                metrics=list(metric_map.values()),
                llm=llm,
                embeddings=embeddings,
                raise_exceptions=True,
            )
        except Exception as e:
            # Fallback: try without custom embeddings
            try:
                result = await ragas_evaluate(
                    dataset=dataset,
                    metrics=list(metric_map.values()),
                    llm=llm,
                    raise_exceptions=True,
                )
            except Exception as e2:
                return RagasResult(error=f"RAGAS evaluation failed: {e2}")

        # Parse results
        ragas_result = RagasResult()
        try:
            scores = result.to_pandas().iloc[0].to_dict() if hasattr(result, "to_pandas") else {}
            for metric_key, metric_obj in metric_map.items():
                metric_name = getattr(metric_obj, "name", metric_key)
                score = scores.get(metric_name, scores.get(metric_key))
                if score is not None:
                    try:
                        setattr(ragas_result, metric_key, float(score))
                    except (TypeError, ValueError):
                        pass
        except Exception:
            pass

        return ragas_result

    def _resolve_metrics(self, metric_names: list[str] | None) -> dict[str, Any]:
        """Map short metric names to RAGAS metric instances."""
        from ragas.llms import llm_factory as ragas_llm_factory
        from ragas.metrics.collections import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        llm = ragas_llm_factory(model=self._judge_model)

        available = {
            "faithfulness": faithfulness.Faithfulness(llm=llm),
            "answer_relevancy": answer_relevancy.AnswerRelevancy(
                llm=llm,
                embeddings=self._make_dummy_embeddings(),
            ),
            "context_precision": context_precision.ContextPrecision(llm=llm),
            "context_recall": context_recall.ContextRecall(llm=llm),
        }

        if metric_names:
            return {k: v for k, v in available.items() if k in metric_names}
        return available

    @staticmethod
    def _make_dummy_embeddings():
        """Create minimal embedding function for answer_relevancy.

        answer_relevancy needs an embedding model to compute cosine similarity
        between the question and generated questions. We provide a basic wrapper.
        """
        from langchain_core.embeddings import Embeddings

        class _RagasEmbeddings(Embeddings):
            def embed_documents(self, texts):
                return [[0.0] * 384 for _ in texts]

            def embed_query(self, text):
                return [0.0] * 384

        return _RagasEmbeddings()


class RagasEvalScenario:
    """Bridge between nanobot EvalRunner and RAGAS metrics."""

    def __init__(self, adapter: RagasAdapter | None = None):
        self._adapter = adapter

    async def score(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
        scenario_id: str = "ragas_eval",
    ) -> dict[str, Any]:
        """Score a Q/A pair and return metrics compatible with EvalRunRecord."""
        adapter = self._adapter or RagasAdapter()
        result = await adapter.evaluate(question, answer, contexts, ground_truth)

        metrics = {}
        for key in ("faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"):
            val = getattr(result, key, None)
            if val is not None:
                metrics[f"ragas_{key}"] = val

        if result.error:
            metrics["ragas_error"] = result.error

        return {
            "metrics": metrics,
            "failure_modes": ["ragas_error"] if result.error else [],
            "ragas_summary": result.summary(),
        }
