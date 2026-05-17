# Tests + CI/CD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the failing test placeholder with a real test suite, add a GitHub Actions CI workflow that runs tests on every push/PR, and add a manual simulate workflow that runs the Monte Carlo simulation and uploads results as an artifact.

**Architecture:** Three independent deliverables committed separately to the `dev` branch. Tests use pytest fixtures and monkeypatching (no mocks of external libraries). The two GitHub Actions workflows share the same dependency installation pattern (pip + caching).

**Tech Stack:** pytest, typer.testing.CliRunner, xgboost, joblib, pandas, numpy, GitHub Actions (actions/checkout@v4, actions/setup-python@v5, actions/cache@v4, actions/upload-artifact@v4)

---

## File Map

| Action | Path | Purpose |
|--------|------|---------|
| Modify | `tests/test_data.py` | Replace placeholder with 6 real tests |
| Create | `.github/workflows/ci.yml` | CI on push + pull_request |
| Create | `.github/workflows/simulate.yml` | Manual simulate + artifact upload |

---

## Task 1: Replace `tests/test_data.py` with real tests

**Files:**
- Modify: `tests/test_data.py`

**Context:**
- `data/processed/master_features.parquet` → 9796 rows × 99 cols (committed, available in CI)
- `models/model_features.json` → list of 92 feature names
- `models/xgb_match_predictor.pkl` → XGBClassifier, 3-class (away/draw/home), expects 92 features
- `world_cup_2026.modeling.train` → typer app; `BEST_PARAMS` is a module-level dict with `n_estimators: 414`
- `world_cup_2026.data_ingestion.normalize.normalize_team_name` → maps aliases to canonical names; canonical names pass through unchanged

---

- [ ] **Step 1: Confirm the placeholder test currently fails**

  ```
  cd C:\Users\feder\Documents\data_repos\world-cup-2026-predictor
  venv\Scripts\activate
  pytest tests/test_data.py -v
  ```

  Expected output includes:
  ```
  FAILED tests/test_data.py::test_code_is_tested - assert False
  ```

- [ ] **Step 2: Write the new `tests/test_data.py`**

  Replace the entire file with:

  ```python
  """
  Tests for data integrity, model loading, and pipeline correctness.
  """

  import json

  import joblib
  import numpy as np
  import pandas as pd
  import pytest

  from pathlib import Path


  # ---------------------------------------------------------------------------
  # Fixtures
  # ---------------------------------------------------------------------------

  @pytest.fixture(scope="session")
  def proj_root():
      return Path(__file__).resolve().parents[1]


  @pytest.fixture(scope="session")
  def processed_dir(proj_root):
      return proj_root / "data" / "processed"


  @pytest.fixture(scope="session")
  def models_dir(proj_root):
      return proj_root / "models"


  # ---------------------------------------------------------------------------
  # Data integrity
  # ---------------------------------------------------------------------------

  def test_master_features_shape(processed_dir):
      df = pd.read_parquet(processed_dir / "master_features.parquet")
      assert df.shape == (9796, 99), f"Expected (9796, 99), got {df.shape}"


  def test_model_features_count(models_dir):
      with open(models_dir / "model_features.json") as f:
          features = json.load(f)
      assert len(features) == 92, f"Expected 92 features, got {len(features)}"


  # ---------------------------------------------------------------------------
  # Model loading and inference
  # ---------------------------------------------------------------------------

  def test_model_predict_proba(models_dir):
      with open(models_dir / "model_features.json") as f:
          features = json.load(f)
      model = joblib.load(models_dir / "xgb_match_predictor.pkl")
      X = np.zeros((1, len(features)))
      proba = model.predict_proba(X)
      assert proba.shape == (1, 3), f"Expected shape (1, 3), got {proba.shape}"
      np.testing.assert_allclose(proba.sum(axis=1), [1.0], rtol=1e-5)


  # ---------------------------------------------------------------------------
  # Pipeline smoke test
  # ---------------------------------------------------------------------------

  def test_train_smoke(monkeypatch, tmp_path):
      import world_cup_2026.modeling.train as train_module
      from typer.testing import CliRunner

      fast_params = {**train_module.BEST_PARAMS, "n_estimators": 5}
      monkeypatch.setattr(train_module, "BEST_PARAMS", fast_params)

      runner = CliRunner()
      result = runner.invoke(train_module.app, [
          "--model-out", str(tmp_path / "model_smoke.pkl"),
          "--features-out", str(tmp_path / "features_smoke.json"),
      ])
      assert result.exit_code == 0, result.output


  # ---------------------------------------------------------------------------
  # normalize_team_name — 48 WC2026 teams (canonical, pass-through)
  # ---------------------------------------------------------------------------

  WC2026_TEAMS = [
      "Mexico", "South Africa", "South Korea", "Czechia",
      "Canada", "Bosnia-Herzegovina", "Qatar", "Switzerland",
      "Brazil", "Morocco", "Haiti", "Scotland",
      "USA", "Paraguay", "Australia", "Turkey",
      "Germany", "Curacao", "Côte d'Ivoire", "Ecuador",
      "Netherlands", "Japan", "Sweden", "Tunisia",
      "Belgium", "Egypt", "Iran", "New Zealand",
      "Spain", "Cape Verde", "Saudi Arabia", "Uruguay",
      "France", "Senegal", "Iraq", "Norway",
      "Argentina", "Algeria", "Austria", "Jordan",
      "Portugal", "DR Congo", "Uzbekistan", "Colombia",
      "England", "Croatia", "Ghana", "Panama",
  ]


  @pytest.mark.parametrize("team", WC2026_TEAMS)
  def test_normalize_wc2026_teams(team):
      from world_cup_2026.data_ingestion.normalize import normalize_team_name
      assert normalize_team_name(team) == team


  # ---------------------------------------------------------------------------
  # normalize_team_name — known aliases
  # ---------------------------------------------------------------------------

  ALIASES = [
      ("korea republic", "South Korea"),
      ("united states", "USA"),
      ("ir iran", "Iran"),
      ("ivory coast", "Côte d'Ivoire"),
      ("türkiye", "Turkey"),
      ("democratic republic of the congo", "DR Congo"),
  ]


  @pytest.mark.parametrize("alias,expected", ALIASES)
  def test_normalize_aliases(alias, expected):
      from world_cup_2026.data_ingestion.normalize import normalize_team_name
      assert normalize_team_name(alias) == expected
  ```

- [ ] **Step 3: Run the full test suite and confirm all tests pass**

  ```
  pytest tests/test_data.py -v
  ```

  Expected: all tests pass. The parametrized normalize tests appear as 48 + 6 = 54 individual cases.

  If `test_train_smoke` hangs, it means `n_estimators=5` is still reading 414 — confirm monkeypatch line targets `train_module.BEST_PARAMS`, not a local copy.

- [ ] **Step 4: Commit**

  ```
  git add tests/test_data.py
  git commit -m "test: add real tests for data integrity, model loading, and normalize"
  ```

---

## Task 2: Create `.github/workflows/ci.yml`

**Files:**
- Create: `.github/workflows/ci.yml`

**Context:** The CI runs on every push and every PR to any branch. It installs from `requirements.txt` (which includes `-e .` for editable install of `world_cup_2026`). Heavy packages (torch, mlflow, jupyterlab) are in `requirements.txt` — pip caching is critical to keep subsequent runs fast.

---

- [ ] **Step 1: Create the `.github/workflows/` directory**

  ```
  mkdir .github
  mkdir .github\workflows
  ```

  (Skip if the directory already exists — no error if it does.)

- [ ] **Step 2: Write `.github/workflows/ci.yml`**

  Create the file with this exact content:

  ```yaml
  name: CI

  on:
    push:
    pull_request:

  jobs:
    test:
      runs-on: ubuntu-latest

      steps:
        - uses: actions/checkout@v4

        - name: Set up Python 3.10
          uses: actions/setup-python@v5
          with:
            python-version: "3.10"

        - name: Cache pip
          uses: actions/cache@v4
          with:
            path: ~/.cache/pip
            key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
            restore-keys: |
              ${{ runner.os }}-pip-

        - name: Install dependencies
          run: pip install -r requirements.txt

        - name: Run tests
          run: pytest tests/ -v

        - name: Check train.py import
          run: python -c "import world_cup_2026.modeling.train"
  ```

- [ ] **Step 3: Commit**

  ```
  git add .github/workflows/ci.yml
  git commit -m "ci: add GitHub Actions CI workflow (push + pull_request)"
  ```

---

## Task 3: Create `.github/workflows/simulate.yml`

**Files:**
- Create: `.github/workflows/simulate.yml`

**Context:** Manual trigger only. `simulate.py` default output path is `outputs/predictions/simulation_results.csv` — that's the path the artifact step must point to. The `n_sims` input is a string (GitHub Actions inputs are always strings); typer's `--n-sims` accepts an integer via CLI argument so this works fine.

---

- [ ] **Step 1: Write `.github/workflows/simulate.yml`**

  Create the file with this exact content:

  ```yaml
  name: Simulate WC2026

  on:
    workflow_dispatch:
      inputs:
        n_sims:
          description: "Number of Monte Carlo simulations"
          required: false
          default: "1000"
          type: string

  jobs:
    simulate:
      runs-on: ubuntu-latest

      steps:
        - uses: actions/checkout@v4

        - name: Set up Python 3.10
          uses: actions/setup-python@v5
          with:
            python-version: "3.10"

        - name: Cache pip
          uses: actions/cache@v4
          with:
            path: ~/.cache/pip
            key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
            restore-keys: |
              ${{ runner.os }}-pip-

        - name: Install dependencies
          run: pip install -r requirements.txt

        - name: Run simulation
          run: python -m world_cup_2026.simulation.simulate --n-sims ${{ inputs.n_sims }}

        - name: Upload simulation results
          uses: actions/upload-artifact@v4
          with:
            name: simulation-results
            path: outputs/predictions/simulation_results.csv
  ```

- [ ] **Step 2: Commit**

  ```
  git add .github/workflows/simulate.yml
  git commit -m "ci: add manual simulate workflow with artifact upload"
  ```

---

## Self-Review

**Spec coverage:**

| Requirement | Task |
|-------------|------|
| master_features.parquet shape (9796, 99) | Task 1 → `test_master_features_shape` |
| model_features.json has 92 features | Task 1 → `test_model_features_count` |
| pkl loads + predict_proba on 92 features | Task 1 → `test_model_predict_proba` |
| train.py smoke test, pocos estimadores | Task 1 → `test_train_smoke` (n_estimators=5) |
| normalize_team_name for 48 WC2026 teams | Task 1 → `test_normalize_wc2026_teams` ×48 |
| CI: push + PR to any branch | Task 2 → `on: push:` + `pull_request:` (no branch filter) |
| CI: Python 3.10 | Task 2 → `python-version: "3.10"` |
| CI: install from requirements.txt | Task 2 → `pip install -r requirements.txt` |
| CI: pytest tests/ | Task 2 → `pytest tests/ -v` |
| CI: verify train.py import | Task 2 → `python -c "import world_cup_2026.modeling.train"` |
| Simulate: workflow_dispatch + n_sims input | Task 3 → `on: workflow_dispatch` with `n_sims` input |
| Simulate: Python 3.10 | Task 3 → `python-version: "3.10"` |
| Simulate: run simulation | Task 3 → `python -m world_cup_2026.simulation.simulate --n-sims ${{ inputs.n_sims }}` |
| Simulate: upload CSV artifact | Task 3 → `actions/upload-artifact@v4` |

All requirements covered. No gaps.
