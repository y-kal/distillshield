# DistillShield Implementation Plan

## Purpose
DistillShield is a local-first research prototype that explores whether adaptive output protection against LLM distillation can be implemented, evaluated, and demoed on a single developer laptop.

The prototype prioritizes:
- end-to-end functionality
- synthetic but reproducible data
- transparent heuristics and interpretable baselines
- local inspectability through API and dashboard surfaces

## Baseline vs Optional
Baseline components are required for a working prototype:
- synthetic data generation for four behavioral classes
- modular feature extraction across six requested feature groups
- rule-based risk scoring with explanations
- tabular ML baselines: logistic regression, random forest, LightGBM if available
- ensemble risk scoring
- policy engine
- mock teacher response generation
- output transformation and leakage proxy
- FastAPI backend with SQLite persistence
- React dashboard
- evaluation scripts and tests

Optional components must not block the prototype:
- PyTorch sequence model stub
- graph construction and graph feature extraction
- GNN experimentation stub
- Docker packaging

## Architecture
The system follows this flow:

1. User / session / query events are ingested or simulated.
2. Feature extraction builds observed and proxy features from query text, timing, usage, and infrastructure context.
3. A rule-based scorer computes partial signals, a risk score, and explanation reasons.
4. Tabular ML baselines predict class probabilities from feature vectors.
5. An ensemble combines rule and model outputs into a final risk score, class, and confidence.
6. The policy engine maps risk and context into an adaptive output policy.
7. A mock teacher generates raw responses, then the transformation layer rewrites or compresses them.
8. Risk assessments, policies, outputs, and experiment artifacts are persisted for inspection.

## Repo Structure
- `apps/api`: FastAPI backend
- `apps/web`: React dashboard
- `packages/core`: shared enums, schemas, config, feature metadata
- `packages/storage`: SQLite models and repositories
- `packages/synthetic_data`: synthetic behavior generation
- `packages/feature_pipeline`: feature extraction and aggregation
- `packages/models`: rule-based, ML, ensemble, optional stubs, policy
- `packages/llm_adapter`: mock teacher, transformation engine, leakage proxy
- `packages/eval`: evaluation runners and metrics
- `data`: local database, datasets, artifacts, reports
- `tests`: unit and integration coverage

## Milestones
1. Scaffold docs and repository layout
2. Define shared schemas and persistence
3. Build synthetic dataset generation
4. Implement feature extraction
5. Add rule-based and tabular ML baselines
6. Add ensemble, policy, and output transformation
7. Expose backend workflows and dashboard
8. Add evaluation, tests, and local run documentation

## Assumptions
- No real production labels are available
- All baseline training uses synthetic data
- No GPU is required
- The prototype is honest about what is mocked, inferred, and synthetic-only
- Output protection claims are exploratory and not presented as proven security guarantees
