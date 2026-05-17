# Tests + CI/CD Design — WC2026 Predictor

**Date:** 2026-05-17  
**Branch:** dev  
**Scope:** pytest test suite, GitHub Actions CI workflow, GitHub Actions simulate workflow

---

## Context

The project is a production ML pipeline (XGBoost) that predicts WC2026 match results. Key artifacts committed to git:

- `data/processed/master_features.parquet` — 9796 rows × 99 columns
- `data/processed/team_snapshot_clustered.parquet` — team snapshot with cluster labels
- `models/xgb_match_predictor.pkl` — serialized XGBoost classifier (92 features)
- `models/model_features.json` — ordered list of 92 feature names
- `data/raw/areezvisram12_fixture/teams.csv` — 48 WC2026 teams with group assignments

All data files required by tests are available in CI after `git checkout` because they are tracked in the repo (`.gitignore` has `!data/processed/*.parquet` exception and `areezvisram12_fixture/` is not ignored).

---

## Task 1 — `tests/test_data.py`

### Structure

Single file, function-based tests with pytest fixtures. No class-based grouping needed for this scope.

**Fixtures:**
- `proj_root` — `Path(__file__).resolve().parents[1]`
- `processed_dir` — `proj_root / "data" / "processed"`
- `models_dir` — `proj_root / "models"`

**Tests:**

| Test | Assertion |
|------|-----------|
| `test_master_features_shape` | `df.shape == (9796, 99)` |
| `test_model_features_count` | `len(features) == 92` |
| `test_model_predict_proba` | `proba.shape == (1, 3)` and `sum ≈ 1.0` |
| `test_train_smoke(tmp_path)` | exit_code == 0 with monkeypatched BEST_PARAMS |
| `test_normalize_wc2026_teams[team]` | parametrized × 48 teams, result == input |
| `test_normalize_aliases[alias-expected]` | parametrized × 6 known aliases |

### Train Smoke Test Detail

Approach: **monkeypatch `BEST_PARAMS`** in the test (user-selected). No modification to production code.

```python
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
```

### WC2026 Teams (48)

From `data/raw/areezvisram12_fixture/teams.csv`:

Mexico, South Africa, South Korea, Czechia, Canada, Bosnia-Herzegovina, Qatar, Switzerland, Brazil, Morocco, Haiti, Scotland, USA, Paraguay, Australia, Turkey, Germany, Curacao, Côte d'Ivoire, Ecuador, Netherlands, Japan, Sweden, Tunisia, Belgium, Egypt, Iran, New Zealand, Spain, Cape Verde, Saudi Arabia, Uruguay, France, Senegal, Iraq, Norway, Argentina, Algeria, Austria, Jordan, Portugal, DR Congo, Uzbekistan, Colombia, England, Croatia, Ghana, Panama

All pass through `normalize_team_name` unchanged (verified against `_PASSTHROUGH_NAMES` and `_TEAM_ALIASES` in `normalize.py`).

### Alias Tests (6)

| Input | Expected |
|-------|----------|
| `"korea republic"` | `"South Korea"` |
| `"united states"` | `"USA"` |
| `"ir iran"` | `"Iran"` |
| `"ivory coast"` | `"Côte d'Ivoire"` |
| `"türkiye"` | `"Turkey"` |
| `"democratic republic of the congo"` | `"DR Congo"` |

---

## Task 2 — `.github/workflows/ci.yml`

### Triggers
- `push` to any branch
- `pull_request` to any branch

### Runner
`ubuntu-latest`

### Steps

1. `actions/checkout@v4`
2. `actions/setup-python@v5` — Python `"3.10"`
3. `actions/cache@v4` — pip cache keyed on `hashFiles('requirements.txt')`
4. `pip install -r requirements.txt`
5. `pytest tests/ -v`
6. `python -c "import world_cup_2026.modeling.train"` — import smoke check

### Notes

- `requirements.txt` includes `torch`, which is large. Pip caching is critical for subsequent runs.
- The `-e .` entry in `requirements.txt` handles editable install of the `world_cup_2026` package — no separate install step needed.

---

## Task 3 — `.github/workflows/simulate.yml`

### Trigger
`workflow_dispatch` with input:
- `n_sims`: string, description "Number of Monte Carlo simulations", default `"1000"`

### Runner
`ubuntu-latest`

### Steps

1. `actions/checkout@v4`
2. `actions/setup-python@v5` — Python `"3.10"`
3. `actions/cache@v4` — pip cache (same key as CI)
4. `pip install -r requirements.txt`
5. `python -m world_cup_2026.simulation.simulate --n-sims ${{ inputs.n_sims }}`
6. `actions/upload-artifact@v4`:
   - name: `simulation-results`
   - path: `outputs/predictions/simulation_results.csv`

### Notes

- Simulation reads `data/raw/areezvisram12_fixture/teams.csv` and processed parquets — all available after checkout.
- Default 1000 sims (vs 10,000 in the script default) keeps runtime manageable in CI (~1-2 min).

---

## Out of Scope

- Discrepancy between `train.py`'s `MODEL_FEATURES` (90 features, missing `rest_days_home`/`rest_days_away`) and `model_features.json` (92 features): noted but not addressed here.
- CI matrix testing across multiple Python versions.
- Linting or type-checking steps in CI.
