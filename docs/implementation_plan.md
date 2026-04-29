# DistillShield Implementation Plan

## Purpose
DistillShield is a local-first prototype that explores whether adaptive output protection against model distillation can be implemented, explained, evaluated, and demoed on a single developer laptop without relying on opaque classifiers.

## Core Design
Baseline components required for the prototype:
- synthetic scenario generation for four behaviour classes
- behavioural feature extraction
- grouped rule-based risk scoring
- explainability outputs with category scores and triggered rules
- adaptive policy selection
- mock teacher response generation
- output transformation and leakage proxy scoring
- FastAPI backend with SQLite persistence
- React dashboard
- policy-effectiveness evaluation and tests

## Architecture
The system follows this flow:

1. Session, query, response, and heuristic infrastructure events are ingested or simulated.
2. Feature extraction builds observed and proxy features from query text, timing, usage, and infrastructure context.
3. The grouped rule-based risk engine computes category-level scores for query pattern, reasoning extraction, automation, infrastructure, and legitimate use.
4. Escalation rules apply minimum class floors so severe signals are not diluted.
5. The explainability object captures confidence, category scores, top reasons, triggered rules, and risk reducers.
6. The policy engine maps the assessment into an adaptive output policy.
7. A mock teacher generates raw responses, then the transformation layer compresses, rewrites, restricts, or blocks output.
8. Risk assessments, policies, outputs, and evaluation artefacts are persisted for inspection.

## Repo Structure
- `apps/api`: FastAPI backend
- `apps/web`: React dashboard
- `packages/core`: shared enums, schemas, config, feature metadata
- `packages/storage`: SQLite models and repositories
- `packages/synthetic_data`: synthetic scenario generation
- `packages/feature_pipeline`: feature extraction and aggregation
- `packages/models`: rule-based risk engine and policy engine
- `packages/llm_adapter`: mock teacher, transformation engine, leakage proxy
- `packages/eval`: policy-effectiveness runner and metrics
- `data`: local database and generated artefacts
- `tests`: unit and integration coverage

## Assumptions
- No real production labels are available.
- Synthetic data is only for scenario simulation and evaluation.
- Infrastructure indicators are heuristic and demo-oriented unless backed by real telemetry.
- The prototype demonstrates transparent protective mechanics, not production-grade abuse detection.
