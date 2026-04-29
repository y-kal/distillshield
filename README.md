# DistillShield

DistillShield is a local-first, rule-based adaptive output protection prototype for reducing model-distillation leakage risk by changing the level of reasoning exposed to users based on behavioural risk signals.

## What the prototype includes
- Synthetic scenario generation for `normal`, `laboratory_legitimate`, `suspicious`, and `high_threat`
- Behavioural feature extraction across query, temporal, conversational, and heuristic infrastructure signals
- A grouped rule-based risk engine with category scores, escalation rules, and explainability
- Adaptive policy selection across `full_reasoning`, `compressed_reasoning`, `rewritten_reasoning`, `answer_only`, and `block`
- Mock teacher output transformation plus leakage and utility proxy evaluation
- FastAPI backend with SQLite persistence
- React dashboard for local inspection
- Policy-effectiveness evaluation and automated tests

## Architecture

`Session / Query Events -> Feature Extraction -> Grouped Rule-Based Risk Engine -> Explainability Object -> Policy Engine -> Mock Teacher -> Output Transformation -> Persistence / Dashboard -> Policy-Effectiveness Evaluation`

Repository layout:
- `apps/api`: FastAPI service
- `apps/web`: React + Vite dashboard
- `packages/core`: enums, schemas, config, feature metadata
- `packages/storage`: SQLite models and repositories
- `packages/synthetic_data`: synthetic scenario generation
- `packages/feature_pipeline`: feature extraction and aggregation
- `packages/models`: rules-only risk engine and policy engine
- `packages/llm_adapter`: mock teacher adapter, leakage proxy, transformations
- `packages/eval`: policy-effectiveness evaluation runner
- `docs`: implementation notes and limitations
- `tests`: unit and integration coverage

## Setup
1. Create the virtual environment.
2. Install Python dependencies in editable mode.
3. Install dashboard dependencies.
4. Generate scenario data if you want a seeded demo database.
5. Start the API and dashboard.
6. Run policy-effectiveness evaluation when needed.

Exact commands are in [RUN_INSTRUCTIONS.txt](/home/ykal/College/cf_ds/RUN_INSTRUCTIONS.txt).

## Scenario Generation And Evaluation
- `scripts/generate_scenarios.py` creates reproducible synthetic scenario sessions for demo, evaluation, and dashboard inspection.
- `scripts/evaluate.py` runs policy-effectiveness evaluation and writes JSON artefacts under `data/experiments`.

Evaluation focuses on:
- policy distribution by behaviour class
- leakage proxy reduction
- utility preservation proxy
- false block rates for normal and laboratory-legitimate scenarios
- adaptive degradation rates for suspicious and high-threat scenarios
- explainability reason coverage
- threshold sanity checks

## Backend Usage
Key endpoints:
- `POST /simulate/session`
- `POST /ingest/session`
- `POST /score/session`
- `POST /classify/session`
- `POST /policy/decide`
- `POST /transform/output`
- `POST /evaluate`
- `GET /sessions`
- `GET /sessions/{id}`
- `GET /users`
- `GET /features/{session_id}`
- `GET /experiments/latest`
- `GET /health`

Assessment responses expose:
- `predicted_class`
- `risk_score`
- `confidence`
- `category_scores`
- `top_reasons`
- `triggered_rules`
- `risk_reducers`
- `explainability`

OpenAPI docs are available at `http://localhost:8000/docs` when the API is running.

## Dashboard Usage
The dashboard is designed for local research demos. It can:
- list sessions and filter by class
- show extracted features and provenance
- display category scores, top reasons, triggered rules, and policy decisions
- compare mock teacher output with protected output
- display the latest policy-effectiveness evaluation summary

## What Is Real vs Mocked
Real in the prototype:
- local feature extraction
- grouped rule scoring
- policy selection
- persistence
- API orchestration
- dashboarding and evaluation flow

Mocked or heuristic:
- teacher responses
- synthetic scenario labels
- heuristic infrastructure abuse indicators
- leakage and utility proxies

The system does not train an LLM and does not train ML classifiers. Synthetic data is only used for scenario simulation, demo flows, and evaluation of the protection mechanics.

## Limitations
This is a research prototype. It demonstrates feasibility mechanics, not production-grade abuse detection or proven resistance against real model distillation. See [docs/limitations.md](/home/ykal/College/cf_ds/docs/limitations.md).
