# Package Restructure + Quality Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the app into `diabetes/{core,ml,web}` with one-way imports, new `run.py`/`train.py` entrypoints, ruff in CI, and unchanged HTTP behavior.

**Architecture:** Repo-root package `diabetes/`. `core` has no project deps; `ml` imports only `core`; `web` imports `ml`+`core`. Artifacts and `templates/`/`static/` stay at repo root. `BASE_DIR` in config points at repo root via `Path(__file__).resolve().parents[2]`.

**Tech Stack:** Python 3.10+, Flask, Waitress, scikit-learn, XGBoost, pytest, ruff, Docker.

**Spec:** `docs/superpowers/specs/2026-08-04-package-restructure-quality-design.md`

## Global Constraints

- HTTP routes and JSON shapes unchanged: `/`, `/health`, `/api/predict`, `/api/explain`
- No new ML features, UI redesign, or auth
- Layer rule: `core` ↛ `ml`/`web`; `ml` ↛ `web`
- Old root modules deleted after move; no permanent shims required
- Entrypoints become `python run.py` and `python train.py`
- CI: `ruff check` then `pytest` then Docker build
- Keep Ukrainian user-facing error strings as they are today

## File Structure (target)

| Path | Responsibility |
|------|----------------|
| `diabetes/__init__.py` | Package marker (empty or version string) |
| `diabetes/core/config.py` | Constants + `BASE_DIR` = repo root |
| `diabetes/core/exceptions.py` | Domain exceptions |
| `diabetes/core/validators.py` | Patient / threshold validation |
| `diabetes/core/scoring.py` | Selection score helpers |
| `diabetes/ml/registry.py` | Classifiers / pipelines (ex-`model_registry.py`) |
| `diabetes/ml/train.py` | Training CLI logic (ex-`train_diabetes_model.py`) |
| `diabetes/ml/predict.py` | Inference (ex-`predict_diabetes.py`) |
| `diabetes/ml/bootstrap.py` | Cold-start (ex-`bootstrap_models.py`) |
| `diabetes/ml/explainability.py` | `/api/explain` data |
| `diabetes/web/forms.py` | Form parse, metrics display, Jinja context |
| `diabetes/web/app.py` | `create_app()`, routes, CSRF |
| `run.py` | Server entry |
| `train.py` | Training CLI entry |
| `pyproject.toml` or `ruff.toml` | Ruff config |
| `tests/test_package_layout.py` | BASE_DIR + layer smoke |
| `tests/test_create_app.py` | `create_app()` smoke |

**Delete after migration:** root `app.py`, `config.py`, `exceptions.py`, `validators.py`, `scoring.py`, `model_registry.py`, `train_diabetes_model.py`, `predict_diabetes.py`, `bootstrap_models.py`, `explainability.py`.

---

### Task 1: Package skeleton + BASE_DIR contract

**Files:**
- Create: `diabetes/__init__.py`
- Create: `diabetes/core/__init__.py`
- Create: `diabetes/ml/__init__.py`
- Create: `diabetes/web/__init__.py`
- Create: `diabetes/core/config.py` (moved from root `config.py` with BASE_DIR fix)
- Create: `tests/test_package_layout.py`
- Keep root `config.py` temporarily until Task 2 finishes import cutover (or update all imports in Task 2 — prefer: create new config first, leave old until Task 6 delete)

**Interfaces:**
- Consumes: existing root `config.py` content
- Produces: `diabetes.core.config.BASE_DIR: Path` → repository root; same public names (`DATA_PATH`, `MODELS_BUNDLE_PATH`, `FEATURES`, …)

- [ ] **Step 1: Write the failing test**

Create `tests/test_package_layout.py`:

```python
"""Smoke: package layout and BASE_DIR point at repo root."""

from pathlib import Path


def test_base_dir_is_repo_root():
    from diabetes.core.config import BASE_DIR, DATA_PATH, MODELS_BUNDLE_PATH

    repo_root = Path(__file__).resolve().parents[1]
    assert BASE_DIR == repo_root
    assert DATA_PATH == repo_root / "diabetes_prediction_dataset.csv"
    assert MODELS_BUNDLE_PATH == repo_root / "diabetes_models.joblib"


def test_core_package_importable():
    import diabetes.core  # noqa: F401
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_package_layout.py -v`

Expected: FAIL (`ModuleNotFoundError: No module named 'diabetes'`)

- [ ] **Step 3: Create packages and config**

`diabetes/__init__.py`, `diabetes/core/__init__.py`, `diabetes/ml/__init__.py`, `diabetes/web/__init__.py` — empty files.

Copy root `config.py` → `diabetes/core/config.py` and change BASE_DIR:

```python
# Was: Path(__file__).parent  (root config)
# Now: diabetes/core/config.py → parents[2] = repo root
BASE_DIR = Path(__file__).resolve().parents[2]
```

Leave all other constants identical.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_package_layout.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add diabetes/ tests/test_package_layout.py
git commit -m "Add diabetes package skeleton with repo-root BASE_DIR."
```

---

### Task 2: Move `core` modules (exceptions, validators, scoring)

**Files:**
- Create: `diabetes/core/exceptions.py` (from `exceptions.py`)
- Create: `diabetes/core/validators.py` (from `validators.py`, imports → `diabetes.core.*`)
- Create: `diabetes/core/scoring.py` (from `scoring.py`, imports → `diabetes.core.config`)
- Modify: `tests/test_exceptions.py`, `tests/test_validators.py`, `tests/test_scoring.py`, `tests/test_config.py`
- Modify: still-root consumers later; for now update only core tests + keep root files importing from new locations OR duplicate until Task 6

**Preferred cutover for this task:** move files into package, update their internal imports, point **all** existing root modules and tests at `diabetes.core.*` so nothing imports root `config`/`exceptions`/`validators`/`scoring` anymore. Leave empty-deleted for Task 6.

**Interfaces:**
- Consumes: `diabetes.core.config`
- Produces:
  - `validate_person_data(data: dict) -> dict`
  - `parse_prediction_threshold(value, default: float | None = None) -> float`
  - `compute_selection_score(metrics: dict | None) -> float`
  - `get_selection_score(metrics: dict | None) -> float`
  - Exception classes unchanged

- [ ] **Step 1: Write failing import assertions in existing tests**

In `tests/test_validators.py` change imports to:

```python
from diabetes.core.exceptions import InvalidPatientDataError
from diabetes.core.validators import validate_person_data, parse_prediction_threshold
```

(same pattern for scoring/exceptions/config tests)

- [ ] **Step 2: Run one test file to verify fail**

Run: `python -m pytest tests/test_validators.py::test_validate_person_data_success -v`

Expected: FAIL (`ModuleNotFoundError` or import error for `diabetes.core.validators`)

- [ ] **Step 3: Move modules and fix imports**

1. Copy `exceptions.py` → `diabetes/core/exceptions.py` (no project imports).
2. Copy `scoring.py` → `diabetes/core/scoring.py`; replace `from config import` with `from diabetes.core.config import`.
3. Copy `validators.py` → `diabetes/core/validators.py`; replace `from config import` / `from exceptions import` with `diabetes.core.*`.
4. Update root `app.py`, `predict_diabetes.py`, `train_diabetes_model.py`, `model_registry.py`, `bootstrap_models.py`, `explainability.py` to import from `diabetes.core.*` temporarily (still at root until later tasks).
5. Update all tests that imported `config`, `exceptions`, `validators`, `scoring`.

- [ ] **Step 4: Run core-related tests**

Run: `python -m pytest tests/test_config.py tests/test_exceptions.py tests/test_validators.py tests/test_scoring.py -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add diabetes/core app.py predict_diabetes.py train_diabetes_model.py model_registry.py bootstrap_models.py explainability.py scoring.py validators.py exceptions.py config.py tests/
git commit -m "Move core config/exceptions/validators/scoring into diabetes.core."
```

Note: root duplicates may still exist; Task 6 deletes them. If you already deleted root copies here, that is fine if nothing imports them.

---

### Task 3: Move `ml` modules

**Files:**
- Create: `diabetes/ml/registry.py` (from `model_registry.py`)
- Create: `diabetes/ml/predict.py` (from `predict_diabetes.py`)
- Create: `diabetes/ml/train.py` (from `train_diabetes_model.py`)
- Create: `diabetes/ml/bootstrap.py` (from `bootstrap_models.py`)
- Create: `diabetes/ml/explainability.py` (from `explainability.py`)
- Modify: `tests/test_model_registry.py`, `tests/test_predict_diabetes.py`, `tests/test_train_diabetes_model.py`, `tests/test_bootstrap_models.py`, `tests/test_explainability.py`, `tests/conftest.py`
- Modify: root `app.py` imports → `diabetes.ml.*`

**Interfaces:**
- Consumes: `diabetes.core.config`, `diabetes.core.exceptions`, `diabetes.core.validators`, `diabetes.core.scoring`
- Produces (must keep names):
  - `predict_with_summary(person, threshold=..., mode=...) -> dict`
  - `get_bundle_optimal_threshold(default=...) -> float`
  - `get_training_metrics() -> dict`
  - `get_feature_importance() -> list[dict]`
  - `reset_pipeline_cache() -> None`
  - `ensure_models_ready(*, enable_tuning: bool = False) -> bool`
  - `get_explanation() -> list[dict]`
  - `main(argv: list[str] | None = None) -> int` in `diabetes.ml.train`
  - `train_all_models(...)`, `build_pipeline(...)`, etc. as today

- [ ] **Step 1: Point one test at new import path**

In `tests/test_model_registry.py`:

```python
from diabetes.ml.registry import (
    build_pipeline,
    create_smote,
    get_classifiers,
    get_model_pipelines,
)
```

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest tests/test_model_registry.py::test_get_classifiers_contains_expected_algorithms -v`

Expected: FAIL (`ModuleNotFoundError: diabetes.ml.registry`)

- [ ] **Step 3: Move ML modules and rewrite imports**

For each file, replace:

| Old | New |
|-----|-----|
| `from config import ...` | `from diabetes.core.config import ...` |
| `from exceptions import ...` | `from diabetes.core.exceptions import ...` |
| `from validators import ...` | `from diabetes.core.validators import ...` |
| `from scoring import ...` | `from diabetes.core.scoring import ...` |
| `from model_registry import ...` | `from diabetes.ml.registry import ...` |
| `from predict_diabetes import ...` | `from diabetes.ml.predict import ...` |
| `from train_diabetes_model import ...` | `from diabetes.ml.train import ...` |
| `import bootstrap_models` | `from diabetes.ml import bootstrap as bootstrap_models` or `from diabetes.ml.bootstrap import ...` |
| `from explainability import ...` | `from diabetes.ml.explainability import ...` |

Update `tests/conftest.py`:

```python
import diabetes.ml.predict as predict_module
import diabetes.ml.train as train_module
```

Update patches in tests: strings like `"app.predict_with_summary"` stay until web move; `"predict_diabetes.X"` → `"diabetes.ml.predict.X"`; `"train_diabetes_model.X"` → `"diabetes.ml.train.X"`; `"bootstrap_models.X"` → `"diabetes.ml.bootstrap.X"`.

- [ ] **Step 4: Run ML tests**

Run: `python -m pytest tests/test_model_registry.py tests/test_predict_diabetes.py tests/test_train_diabetes_model.py tests/test_bootstrap_models.py tests/test_explainability.py -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add diabetes/ml app.py tests/ conftest.py
git commit -m "Move ML registry/predict/train/bootstrap/explain into diabetes.ml."
```

---

### Task 4: Move web layer (`create_app`, forms, routes)

**Files:**
- Create: `diabetes/web/forms.py`
- Create: `diabetes/web/app.py`
- Create: `tests/test_create_app.py`
- Modify: `tests/test_app.py`, `tests/test_api.py`, `tests/test_e2e_smoke.py`
- Delete or stop using root `app.py` after cutover

**Interfaces:**
- Consumes: `diabetes.ml.predict`, `diabetes.ml.bootstrap`, `diabetes.ml.explainability`, `diabetes.core.*`
- Produces:
  - `create_app() -> Flask`
  - Module-level `app = create_app()` for Waitress (`diabetes.web.app:app`)
  - Helpers in `forms.py`: `parse_form`, `parse_threshold_from_form`, `parse_threshold_from_payload`, `get_error_message`, `format_metrics_for_display`, `load_metrics_rows`, `load_feature_importance`, `get_default_threshold`, `build_index_context`

- [ ] **Step 1: Write failing create_app smoke test**

`tests/test_create_app.py`:

```python
"""Smoke: Flask application factory."""

from unittest.mock import patch


def test_create_app_returns_flask_with_routes():
    with patch("diabetes.ml.bootstrap.ensure_models_ready", return_value=True):
        from diabetes.web.app import create_app

        application = create_app()
        application.config["TESTING"] = True
        application.config["WTF_CSRF_ENABLED"] = False

        client = application.test_client()
        response = client.get("/health")
        assert response.status_code == 200
        payload = response.get_json()
        assert payload["status"] == "ok"
        assert "models_ready" in payload
```

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest tests/test_create_app.py -v`

Expected: FAIL (`ModuleNotFoundError: diabetes.web.app`)

- [ ] **Step 3: Implement `forms.py` and `app.py`**

1. Move pure helpers from root `app.py` into `diabetes/web/forms.py` with updated imports.
2. Implement `create_app()` in `diabetes/web/app.py`:

```python
def create_app() -> Flask:
    application = Flask(
        __name__,
        template_folder=str(BASE_DIR / "templates"),
        static_folder=str(BASE_DIR / "static"),
    )
    application.secret_key = FLASK_SECRET_KEY
    # CSRF setup (same try/except as today)
    # register routes (health, api_predict, api_explain, index)
    # register error handlers
    return application


app = create_app()
```

3. Move `__main__` block out of `app.py` into Task 5 `run.py` (do not keep server loop inside package module if `run.py` owns it).

4. Update tests to import from `diabetes.web.app` / `diabetes.web.forms` and fix patch paths:
   - `"app.predict_with_summary"` → `"diabetes.web.app.predict_with_summary"` (or wherever imported)
   - `"app.bootstrap_models.ensure_models_ready"` → `"diabetes.web.app.bootstrap_models.ensure_models_ready"` or patch `diabetes.ml.bootstrap.ensure_models_ready`

- [ ] **Step 4: Run web tests**

Run: `python -m pytest tests/test_create_app.py tests/test_app.py tests/test_api.py tests/test_e2e_smoke.py -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add diabetes/web tests/test_create_app.py tests/test_app.py tests/test_api.py tests/test_e2e_smoke.py
git commit -m "Move Flask app into diabetes.web with create_app factory."
```

---

### Task 5: Entrypoints + Docker + docs

**Files:**
- Create: `run.py`
- Create: `train.py`
- Modify: `Dockerfile`
- Modify: `docker-compose.yml` (if CMD implied only via Dockerfile — usually Dockerfile enough)
- Modify: `.devcontainer/devcontainer.json`
- Modify: `README.md`

**Interfaces:**
- Consumes: `diabetes.web.app.create_app` / `app`; `diabetes.ml.train.main`
- Produces: CLI commands documented in README

- [ ] **Step 1: Add entrypoints**

`run.py`:

```python
"""Server entrypoint for Flask + Waitress."""

from __future__ import annotations

import logging
import os

from diabetes.web.app import app, create_app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    host = os.environ.get("HOST", "127.0.0.1")
    try:
        port = int(os.environ.get("PORT", "5000"))
    except ValueError:
        logger.warning("Некоректний PORT у середовищі, використано 5000")
        port = 5000

    application = create_app() if debug_mode else app

    if not debug_mode:
        try:
            from waitress import serve
        except ImportError:
            logger.warning("waitress не встановлено; використано Flask dev server.")
        else:
            serve(application, host=host, port=port)
            return

    application.run(debug=debug_mode, host=host, port=port)


if __name__ == "__main__":
    main()
```

`train.py`:

```python
"""CLI entrypoint for model training."""

from __future__ import annotations

import sys

from diabetes.ml.train import main


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Update Dockerfile CMD**

Replace:

```dockerfile
CMD ["waitress-serve", "--host=0.0.0.0", "--port=5000", "app:app"]
```

with:

```dockerfile
CMD ["waitress-serve", "--host=0.0.0.0", "--port=5000", "diabetes.web.app:app"]
```

- [ ] **Step 3: Update README + devcontainer**

README quick start:

```bash
python run.py
python train.py
python train.py --no-tune --sample 20000
```

`.devcontainer/devcontainer.json`: `"server": "python run.py"`

- [ ] **Step 4: Smoke import check**

Run:

```bash
python -c "from diabetes.web.app import app; print(app.url_map)"
python -c "from diabetes.ml.train import main; print(callable(main))"
```

Expected: prints URL map / `True` without errors

- [ ] **Step 5: Commit**

```bash
git add run.py train.py Dockerfile .devcontainer/devcontainer.json README.md
git commit -m "Add run/train entrypoints and update Docker and docs."
```

---

### Task 6: Delete obsolete root modules

**Files:**
- Delete: `app.py`, `config.py`, `exceptions.py`, `validators.py`, `scoring.py`, `model_registry.py`, `train_diabetes_model.py`, `predict_diabetes.py`, `bootstrap_models.py`, `explainability.py`
- Modify: any remaining imports (grep)

- [ ] **Step 1: Grep for old imports**

Run:

```bash
rg -n "^(from|import) (app|config|exceptions|validators|scoring|model_registry|train_diabetes_model|predict_diabetes|bootstrap_models|explainability)\b" --glob '!diabetes/**' --glob '!.git/**'
```

Expected: only hits inside files about to be deleted (or zero)

- [ ] **Step 2: Delete root modules**

Delete the ten files listed above.

- [ ] **Step 3: Full test suite**

Run: `python -m pytest tests/ -q`

Expected: all PASS (same count ± new tests from Tasks 1/4)

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "Remove obsolete root modules after package migration."
```

---

### Task 7: Ruff + CI quality gate

**Files:**
- Create: `ruff.toml` (or `[tool.ruff]` in `pyproject.toml`)
- Modify: `requirements.txt` (add `ruff>=0.8.0`)
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: migrated package tree
- Produces: CI fails on lint errors

- [ ] **Step 1: Add ruff config**

`ruff.toml`:

```toml
target-version = "py310"
line-length = 88

[lint]
select = ["E", "F", "I", "UP", "B"]
ignore = ["E501"]  # long Ukrainian strings / URLs OK

[lint.per-file-ignores]
"tests/**" = ["B011"]
```

Add to `requirements.txt`:

```
ruff>=0.8.0
```

- [ ] **Step 2: Run ruff and fix violations**

Run: `python -m ruff check diabetes run.py train.py tests`

Fix real issues (unused imports, bare excepts if any). Do not weaken rules to silence everything without cause.

Expected after fixes: exit code 0

- [ ] **Step 3: Update CI**

In `.github/workflows/ci.yml` after pip install:

```yaml
      - name: Ruff
        run: python -m ruff check diabetes run.py train.py tests

      - name: Run unit tests
        run: python -m pytest tests/ -q
```

- [ ] **Step 4: Full local verification**

Run:

```bash
python -m ruff check diabetes run.py train.py tests
python -m pytest tests/ -q
```

Expected: both succeed

- [ ] **Step 5: Commit**

```bash
git add ruff.toml requirements.txt .github/workflows/ci.yml diabetes/ tests/ run.py train.py
git commit -m "Add ruff linting and enforce it in CI."
```

---

## Self-Review (plan vs spec)

| Spec requirement | Task |
|------------------|------|
| `diabetes/{core,ml,web}` structure | 1–4 |
| One-way dependencies | 2–4 (enforced by imports; optional grep in Task 6) |
| `BASE_DIR` = repo root | 1 |
| Flask templates/static from repo root | 4 |
| Same HTTP contracts | 4 (no route/shape changes) |
| `run.py` / `train.py` | 5 |
| Docker / README / devcontainer | 5 |
| Delete old root modules | 6 |
| Ruff + pytest in CI | 7 |
| `create_app` smoke test | 4 |
| No `src/` layout / no auth / no UI redesign | respected |

Placeholder scan: none intentional.  
Type/name consistency: `create_app() -> Flask`, `predict_with_summary`, `ensure_models_ready`, `BASE_DIR` used consistently.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-04-package-restructure-quality.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks  
2. **Inline Execution** — execute in this session with `executing-plans`, batch checkpoints  

Which approach?
