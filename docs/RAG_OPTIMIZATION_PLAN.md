# RAG Optimization Plan

## Background

The current project already has GraphRAG-style memory retrieval, query embedding support, and a
RAG evaluation scorecard covering grounded extraction, refusal, conflict handling, irrelevant
context filtering, and multi-hop synthesis.

The next step is to move from a basic retrieval pipeline to a production-oriented RAG stack with:

- semantic chunking instead of fixed-size splitting
- controlled query expansion instead of direct raw-query retrieval
- hybrid recall with learning-to-rank fusion
- strategy routing for rerank cost and latency control
- retrieval and answer quality monitoring with an online feedback loop

## Goals

- Improve retrieval recall without materially increasing hallucination risk.
- Improve first-hit precision so less irrelevant context reaches the generator.
- Reduce average rerank cost and p95 latency through query-aware routing.
- Make retrieval quality measurable with offline evals and online telemetry.

## Target Architecture

```text
source documents
  -> clean / normalize / deduplicate
  -> semantic chunking + neighbor window
  -> metadata tagging
  -> embedding + sparse index
  -> hybrid recall
  -> LambdaMART fusion
  -> strategy-based rerank routing
  -> context assembly
  -> grounded generation
  -> citation / monitoring / feedback
```

## Module Design

### 1. Semantic Chunking

Use NLP-aware segmentation to split by semantic boundaries such as headings, paragraphs, lists, and
stable sentence groups instead of only fixed token counts.

Recommended design:

- Keep a primary chunk body of roughly `300-800` tokens.
- Attach a `100-200` token left/right context window for retrieval display and generation context.
- Store `body_text` and `expanded_context_text` separately.
- Keep structural metadata such as `doc_id`, `section`, `title`, `updated_at`, `tenant`, and
  `permission_scope`.

Why this shape:

- The primary chunk stays focused enough for embedding quality.
- The extra window reduces context fracture at chunk boundaries.
- Decoupling body and display context avoids making the embedding input too noisy.

### 2. Query Preprocessing

Introduce a lightweight preprocessing stage before retrieval.

Pipeline:

1. Normalize the incoming query.
2. Use a small model to generate one or more expanded variants.
3. Compute embedding similarity between the original query and each expanded variant.
4. Discard any expanded query with similarity below `0.8`.
5. Keep the original query as the primary branch even when expansions are accepted.

Guardrails:

- Expanded queries must not replace the original query.
- Reject expansions that introduce new entities, dates, versions, or strong constraints that were
  not present in the original query.
- Log accepted and rejected expansions for later bad-case analysis.

Rationale:

- Query expansion improves recall on abbreviated or underspecified questions.
- Similarity gating reduces semantic drift from the small model.
- Keeping the original query preserves intent fidelity.

### 3. Hybrid Retrieval

Use multiple recall channels and merge their candidates:

- dense retrieval from embeddings
- sparse retrieval from BM25 or another inverted index
- metadata filtering for time, tenant, permissions, and document type
- optional graph or structured retrieval for relation-heavy queries

Recommended candidate flow:

- `top_k_dense = 30`
- `top_k_sparse = 30`
- deduplicate by chunk id or canonical source id
- feed the merged candidate set into ranking

This allows the system to recover both semantic matches and exact-match evidence.

### 4. LambdaMART Fusion Ranking

Use LambdaMART as the hybrid fusion layer instead of fixed manual weights.

Useful ranking features:

- dense similarity score
- sparse retrieval score
- title hit / heading hit
- exact entity match
- digit / date match
- chunk position in document
- document freshness
- source reliability tier
- permission match
- query length and query class
- expansion branch id

Training data sources:

- human relevance labels
- click-through or answer-adoption signals
- eval-derived positives and negatives
- curated bad cases from support or internal review

Fallback rule:

- If training data is too sparse, start with hand-tuned weighted fusion and keep LambdaMART behind a
  flag until labels are stable.

### 5. Strategy Routing for Rerank Optimization

Do not send every query through the same expensive rerank path.

Add a strategy router that classifies queries and chooses the cheapest path that is likely to
preserve quality.

Possible route classes:

- `simple_fact`: small candidate set, light rerank
- `exact_match`: favor sparse signals and metadata filters
- `multi_hop`: broader recall and stronger rerank
- `time_sensitive`: freshness-aware ranking
- `long_query`: query decomposition before recall

Routing signals:

- query length
- entity count
- time or version references
- numeric constraints
- classifier output
- prior failure patterns

Expected benefit:

- lower p50 and p95 latency
- lower rerank cost
- less unnecessary cross-encoder usage on easy queries

### 6. Context Assembly and Answer Grounding

After ranking, construct answer context deliberately instead of concatenating raw top-k chunks.

Recommended steps:

- merge adjacent chunks from the same source
- remove near-duplicate evidence
- prioritize source diversity when needed
- preserve source ids and chunk ids
- enforce token budget limits before generation

Generation constraints:

- answer from retrieved evidence only
- explicitly say when context is insufficient
- surface conflicting sources instead of resolving them silently
- attach citations to key claims

## Metrics and Monitoring

Monitoring should be split into retrieval, generation, and system layers.

### Retrieval Metrics

- `recall@k`
- `MRR`
- `NDCG@k`
- first-hit precision
- dense vs sparse contribution share
- accepted expansion rate
- rejected expansion rate
- recall lift from expansion branches

### Generation Metrics

- citation coverage rate
- grounded answer rate
- insufficient-context refusal accuracy
- conflict detection accuracy
- hallucination rate
- answer usefulness score

### System Metrics

- p50 / p95 end-to-end latency
- stage latency for preprocess, recall, rank, rerank, and generation
- token consumption
- route distribution by strategy class
- cache hit rate
- per-query cost

### Existing Eval Alignment

The repository already contains RAG eval coverage in `nanobot/evals/` for:

- grounded extraction
- cross-source comparison
- insufficient-context refusal
- conflict handling
- irrelevant-context filtering
- multi-hop synthesis

These scenarios should remain the minimum regression gate. New retrieval changes should also add:

- query expansion precision tests
- hybrid fusion ranking tests
- route-selection tests
- latency budget regression checks

## Rollout Plan

### Phase 1: Retrieval Baseline Hardening

- Add semantic chunking with neighbor windows.
- Add hybrid dense + sparse recall.
- Keep manual weighted fusion first.
- Add telemetry for candidate counts, latency, and source contribution.

### Phase 2: Query Expansion Controls

- Add small-model query expansion.
- Add embedding similarity gating at `0.8`.
- Keep original-query branch mandatory.
- Measure recall lift vs noise rate.

### Phase 3: Learning-to-Rank

- Introduce LambdaMART with offline training data.
- Compare against manual fusion on offline evals and online shadow traffic.
- Roll out behind a flag with per-route monitoring.

### Phase 4: Strategy Router

- Add query classification and route policies.
- Introduce light vs heavy rerank paths.
- Set latency SLOs and fail-safe fallback behavior.

### Phase 5: Continuous Optimization

- Build bad-case review loop.
- Add online A/B testing.
- Refresh ranking features and labels periodically.
- Extend eval coverage for domain-specific failure modes.

## Risks

- Semantic chunking can overfit one document style and hurt others.
- Query expansion can inject false constraints if guardrails are weak.
- LambdaMART adds operational complexity and depends on label quality.
- Routing can create fragmented behavior if route policies are not measurable.
- Metrics can drift if answer-quality annotation is too sparse or inconsistent.

## Implementation Notes for This Repository

Given the current codebase, the most natural integration points are:

- retrieval and scoring logic in [memory.py](/C:/Users/Administrator/Desktop/nanobot/nanobot/agent/memory.py)
- RAG eval scenarios in [rag.py](/C:/Users/Administrator/Desktop/nanobot/nanobot/evals/scenario_library/rag.py)
- scorecard and regression reporting in [reports.py](/C:/Users/Administrator/Desktop/nanobot/nanobot/evals/reports.py)
- CLI gates in [commands.py](/C:/Users/Administrator/Desktop/nanobot/nanobot/cli/commands.py)

If this plan is implemented incrementally, Phase 1 should land first because it improves retrieval
quality while keeping model behavior and operational complexity relatively stable.
