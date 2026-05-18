"""
Tests for data integrity, model loading, and pipeline correctness.
"""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from pathlib import Path

_PARQUET = Path(__file__).resolve().parents[1] / "data" / "processed" / "master_features.parquet"
_SKIP_IF_NO_DATA = pytest.mark.skipif(
    not _PARQUET.exists(), reason="master_features.parquet not available in CI"
)

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
@_SKIP_IF_NO_DATA
def test_master_features_shape(processed_dir):
    df = pd.read_parquet(processed_dir / "master_features.parquet")
    assert df.shape == (9796, 97), f"Expected (9796, 97), got {df.shape}"


def test_model_features_count(models_dir):
    with open(models_dir / "model_features.json") as f:
        features = json.load(f)
    assert len(features) == 93, f"Expected 93 features, got {len(features)}"


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
@_SKIP_IF_NO_DATA
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
