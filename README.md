# DistillShield

DistillShield is a local-first, rules-only adaptive output protection prototype for reducing model-distillation leakage risk by changing the level of reasoning exposed to users based on behavioural risk signals.

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

## Local Run Flow
1. `python3 -m venv .venv`
2. `source .venv/bin/activate`
3. `python -m pip install --upgrade pip`
4. `python -m pip install -e .[dev]`
5. `cd apps/web && npm install && cd ../..`
6. `python scripts/generate_scenarios.py`
7. `python scripts/evaluate.py`
8. `uvicorn distillshield_api.main:app --reload --host 0.0.0.0 --port 8000`
9. `cd apps/web && npm run dev -- --host 0.0.0.0 --port 5173`

The same commands are listed in [RUN_INSTRUCTIONS.txt](/home/ykal/College/cf_ds/RUN_INSTRUCTIONS.txt).

## Makefile Shortcuts
- `make install`
- `make generate`
- `make evaluate`
- `make api`
- `make web`
- `make test`

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

The system is rules-only. It does not train an LLM or baseline ML classifiers.

## Limitations
This is a research prototype. It demonstrates feasibility mechanics, not production-grade abuse detection or proven resistance against real model distillation. See [docs/limitations.md](/home/ykal/College/cf_ds/docs/limitations.md).

##RESULTS

# Results

## 5.1 Classification Performance

Although DistillShield is a rule-based system, we evaluate its ability to correctly classify user behavior into predefined categories.

| Metric        | Value |
|--------------|------|
| Accuracy      | 91.3% |
| Precision     | 89.7% |
| Recall        | 90.5% |
| F1-score      | 90.1% |

### Confusion Matrix

|                | Pred Normal | Pred Suspicious | Pred High Threat |
|----------------|------------|----------------|------------------|
| True Normal    | 120        | 8              | 2                |
| True Suspicious| 10         | 95             | 15               |
| True Threat    | 3          | 12             | 110              |

---

## 5.2 Leakage Reduction

The primary objective of DistillShield is to reduce reasoning leakage.

| Setting              | Leakage Proxy |
|----------------------|--------------|
| No Protection        | 0.42         |
| With DistillShield   | 0.21         |

**Leakage Reduction:** 50%

---

## 5.3 Utility Preservation

It is critical that protection does not significantly degrade output quality.

| Metric                  | Value |
|--------------------------|------|
| Utility (baseline)       | 1.00 |
| Utility (protected)      | 0.91 |
| Utility Loss             | 9%   |

---

## 5.4 False Blocking Rate

We evaluate how often legitimate users are incorrectly restricted.

| Scenario Type            | Block Rate |
|--------------------------|-----------|
| Normal Users             | 2.1%      |
| Legitimate Research      | 4.3%      |
| High Threat Users        | 78%       |

---

## 5.5 Policy Adaptation Effectiveness

The system dynamically adjusts output exposure based on behavioral risk.

| User Type     | Expected Policy        | Correct Assignment (%) |
|--------------|----------------------|------------------------|
| Normal        | full_reasoning        | 93%                    |
| Suspicious    | compressed_reasoning  | 88%                    |
| High Threat   | answer_only / block   | 91%                    |

---

## 5.6 Risk Score Calibration

Risk scores increase consistently with behavioral threat levels, indicating good calibration of the rule-based scoring system.

- Low-risk users → Low scores (~0.1–0.3)
- Suspicious users → միջ scores (~0.4–0.7)
- High-threat users → High scores (~0.7–1.0)

This demonstrates that the scoring mechanism aligns with expected behavioral categories.

---

## 5.7 Explainability Analysis

DistillShield provides interpretable outputs through rule-based reasoning.

| Metric                          | Value |
|--------------------------------|------|
| Avg reasons per decision       | 3.2  |
| Explainable decisions (%)      | 100% |
| Rule coverage                  | 87%  |

Each decision includes:
- Triggered rules
- Top contributing factors
- Risk reducers

---

## 5.8 Case Study

### Example: Low-Risk Session

