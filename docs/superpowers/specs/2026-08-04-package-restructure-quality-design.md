# Design: Package restructure + code quality (medium)

**Date:** 2026-08-04  
**Project:** ProjectAILearning  
**Status:** Approved in brainstorming; pending user review of this file  
**Scope:** Architecture (module boundaries) + code quality (ruff, tests, CI)  
**Scale:** Medium  
**Entrypoints:** May change; documentation and Docker will be updated

## Goal

Restructure the Flask + ML diabetes prediction app into a clear `diabetes/` package with strict layer boundaries, without changing HTTP/UI behavior. Add ruff to CI alongside pytest.

## Non-goals

- New ML models or training algorithms
- UI redesign
- Authentication / authorization
- Changes to JSON API contracts (`/health`, `/api/predict`, `/api/explain`) or form field names
- `src/` layout
- Browser e2e tests

## Chosen approach

Package layout at repo root (`diabetes/`), not `src/`. Thin new entrypoints (`run.py`, `train.py`). Old root modules (`app.py`, `train_diabetes_model.py`, etc.) are removed after the move.

## Target structure

```
diabetes/
  __init__.py
  core/
    __init__.py
    config.py
    exceptions.py
    validators.py
    scoring.py
  ml/
    __init__.py
    registry.py          # from model_registry.py
    train.py             # from train_diabetes_model.py
    predict.py           # from predict_diabetes.py
    bootstrap.py         # from bootstrap_models.py
    explainability.py
  web/
    __init__.py
    app.py               # Flask create_app() + routes
    forms.py             # form/threshold/context helpers
templates/
static/
tests/
docs/
run.py                   # production/dev server entry
train.py                 # CLI training entry
Dockerfile
docker-compose.yml
requirements.txt
requirements-docker.txt
```

Artifacts stay at repo root: `diabetes_models.joblib`, `diabetes_prediction_dataset.csv`, `model_metrics.json`, `feature_importance.json`.

## Dependency rules

| Layer | May import |
|-------|------------|
| `diabetes.core` | stdlib + third-party only (no `ml`, no `web`) |
| `diabetes.ml` | `diabetes.core` + ML libs |
| `diabetes.web` | `diabetes.ml` + `diabetes.core` + Flask |

Violations are treated as bugs; keep imports one-way.

## Components

| Module | Responsibility |
|--------|----------------|
| `core.config` | Paths (`BASE_DIR` = repo root), constants, env |
| `core.exceptions` | Domain errors |
| `core.validators` | Patient + threshold normalization |
| `core.scoring` | `compute_selection_score` / `get_selection_score` |
| `ml.registry` | Classifiers, pipelines, SMOTE flags |
| `ml.train` | Load data, 3-way split, train, tune, metrics, save |
| `ml.predict` | Bundle load/cache, inference, summary |
| `ml.bootstrap` | Cold-start training when bundle missing |
| `ml.explainability` | Feature importance for `/api/explain` |
| `web.forms` | Parse HTML form, threshold %, Jinja context helpers |
| `web.app` | `create_app()`, routes, CSRF, error handlers |

## Data flow (unchanged contracts)

1. HTML form or JSON body  
2. `core.validators.validate_person_data`  
3. `ml.predict.predict_with_summary` (mode `all` \| `best`)  
4. Response: Jinja HTML or JSON  

Endpoints remain:

- `GET /` / `POST /` — UI  
- `GET /health` — `{ status, models_ready }`  
- `POST /api/predict` — prediction JSON  
- `GET /api/explain` — feature importance  

## Error handling

Keep current semantics:

- `InvalidPatientDataError` → HTTP 400 / UI alert  
- `ModelNotFoundError` → HTTP 503 / train message  
- `PredictionError` and unexpected errors → logged; user-facing message, no traceback in HTML  
- try/except at I/O boundaries (disk, joblib, template render)

## Code quality

- Add **ruff** (check; format check optional but preferred in CI)  
- CI order: `ruff check` → `pytest` → Docker build  
- Update all test imports to `diabetes.*`  
- Add smoke test for `create_app()`  
- Remove duplicate / obsolete tests if found during move  
- No intentional behavior changes

## `BASE_DIR` and paths

`core.config.BASE_DIR` must resolve to the **repository root** (parent of the `diabetes/` package), so existing relative artifact paths keep working with Docker and local runs.

Flask templates/static: configure `Flask(..., template_folder=..., static_folder=...)` to repo-root `templates/` and `static/`, or keep equivalent path resolution via `BASE_DIR`.

## Entrypoints and ops

| Old | New |
|-----|-----|
| `python app.py` | `python run.py` |
| `python train_diabetes_model.py` | `python train.py` |
| Docker `CMD` / compose | Point at `run.py` |
| `.devcontainer` | Point at `run.py` |
| README | Document new commands |

Backward-compatible root shims are **not** required (explicitly out of compatibility scope).

## Migration order

1. Create `diabetes/{core,ml,web}` and move modules with import updates  
2. Add `run.py` / `train.py`; wire Docker, README, devcontainer  
3. Delete obsolete root Python modules  
4. Add ruff config + CI step  
5. Fix tests until green; confirm ruff clean  

## Testing strategy

- Keep `tests/` at repo root  
- Fixtures continue to use sample person data and temp bundles as today  
- Success criteria: full pytest suite green + ruff check clean on CI Python version  

## Success criteria

- [ ] Code lives under `diabetes/{core,ml,web}` with one-way dependencies  
- [ ] Same HTTP routes and response shapes  
- [ ] `run.py` / `train.py` documented and used by Docker  
- [ ] Ruff + pytest in CI  
- [ ] README / Docker / devcontainer updated  
- [ ] Old root modules removed  

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Broken artifact paths after package move | Fix `BASE_DIR` to repo root; smoke-test load metrics/models |
| Flask can't find templates | Explicit `template_folder` / `static_folder` from `BASE_DIR` |
| Import cycles | Enforce layer rules; forms helpers stay in `web`, not `core` |
| Large `train.py` still bulky | Accept for this pass; further split only if needed later |
