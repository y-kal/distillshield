# DistillShield

DistillShield is a local-first research prototype for adaptive output protection against LLM distillation. It demonstrates a complete feasibility pipeline that observes user and session behavior, extracts heuristic and semantic features, computes risk, classifies sessions into behavioral classes, chooses an adaptive output policy, and transforms mock teacher outputs accordingly.

## What the prototype includes
- Synthetic dataset generation for `normal`, `laboratory_legitimate`, `suspicious`, and `high_threat`
- Modular feature extraction across six feature groups
- Transparent rule-based risk scoring with top reasons
- Tabular ML baselines with logistic regression, random forest, and LightGBM if available
- Optional sequence and graph model stubs that do not block the baseline path
- Ensemble risk scoring, policy decisions, output transformation, and leakage proxy scoring
- FastAPI backend with SQLite persistence
- React dashboard for research inspection
- Evaluation scripts, unit tests, integration tests, and local run instructions

## Architecture
The baseline pipeline is:

`User / Session / Query Events -> Feature Extraction -> Rule + ML Scoring -> Ensemble -> Policy Engine -> Mock Teacher -> Output Transformation -> Persisted Artifacts`

Repository layout:
- `apps/api`: FastAPI service
- `apps/web`: React + Vite dashboard
- `packages/core`: enums, schemas, config, feature metadata
- `packages/storage`: SQLite models and repositories
- `packages/synthetic_data`: seeded synthetic generators
- `packages/feature_pipeline`: feature extraction and aggregation
- `packages/models`: rule-based, ML, ensemble, optional stubs, policy engine
- `packages/llm_adapter`: mock teacher adapter, leakage proxy, transformations
- `packages/eval`: evaluation runner
- `docs`: implementation plan and limitations
- `tests`: unit and integration coverage

## Setup
The project is designed for a single laptop and uses a Python virtual environment plus a Node frontend toolchain.

1. Create the virtual environment.
2. Install Python dependencies in editable mode.
3. Install dashboard dependencies.
4. Generate synthetic data.
5. Train the baseline models.
6. Start the API and dashboard.

Exact commands are in [RUN_INSTRUCTIONS.txt](/home/ykal/College/cf_ds/RUN_INSTRUCTIONS.txt).

## Training And Evaluation
- `scripts/generate_synthetic.py` generates reproducible synthetic train/validation/test data and persists sessions in SQLite.
- `scripts/train_baseline.py` extracts tabular features and trains the baseline classifiers.
- `scripts/evaluate.py` compares the baseline pipeline on a synthetic held-out split and writes experiment metrics under `data/experiments`.

Metrics include:
- accuracy
- precision
- recall
- F1
- confusion matrix
- class report
- leakage proxy mean
- utility proxy mean

## Backend Usage
Key endpoints:
- `POST /simulate/session`
- `POST /ingest/session`
- `POST /score/session`
- `POST /classify/session`
- `POST /policy/decide`
- `POST /transform/output`
- `POST /train/baseline`
- `POST /evaluate`
- `GET /sessions`
- `GET /sessions/{id}`
- `GET /users`
- `GET /models`
- `GET /features/{session_id}`
- `GET /experiments/latest`
- `GET /health`

OpenAPI docs are available at `http://localhost:8000/docs` when the API is running.

## Dashboard Usage
The dashboard is built for research demos, not production operations. It can:
- list sessions and filter by class
- show stored session details and query history
- inspect extracted feature values and provenance
- display risk score, predicted class, confidence, policy, and reasons
- compare raw teacher output and transformed output
- display latest experiment summary and stored model metrics

Use the dashboard’s `Generate Demo Data` button to seed a local demo flow if the database is empty.

## What Is Real vs Mocked
Real in the prototype:
- modular feature extraction
- rule-based and tabular ML scoring
- policy selection
- persistence
- API orchestration
- local evaluation and dashboarding

Mocked or heuristic:
- teacher responses
- user behavior labels
- infrastructure abuse indicators
- leakage score
- optional sequence and graph scores

## Limitations
This is a research prototype. It does not claim production-grade security or validated resistance against real model distillation. See [docs/limitations.md](/home/ykal/College/cf_ds/docs/limitations.md) for the detailed caveats.

## Future Work
- replace synthetic labels with partially annotated real traffic or replay traces
- improve embedding and sequential modeling
- add calibration plots and richer experiment comparison surfaces
- validate output utility degradation with task-specific benchmarks
- test stronger transformation policies against actual distillation pipelines
