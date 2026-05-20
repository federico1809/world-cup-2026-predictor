"""Tests for streamlit_app/app.py — pure data functions only."""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Stub streamlit before importing app so @st.cache_* decorators don't fail
sys.modules.setdefault("streamlit", MagicMock())

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def snapshot():
    from streamlit_app.app import _build_snapshot
    return _build_snapshot()


@pytest.fixture(scope="module")
def model_features():
    with open(ROOT / "models" / "model_features.json") as f:
        return json.load(f)


def test_snapshot_row_count(snapshot):
    assert len(snapshot) == 48


def test_snapshot_required_columns(snapshot):
    required = {
        "team", "elo", "cluster_name", "cluster_enc",
        "ranking", "squad_value",
        "form_10_win_rate", "form_10_goals_scored_avg",
    }
    assert required.issubset(set(snapshot.columns))


def test_snapshot_no_nulls_in_key_cols(snapshot):
    for col in ["elo", "ranking", "squad_value", "cluster_enc"]:
        assert snapshot[col].isna().sum() == 0, f"NaNs found in column: {col}"


def test_snapshot_cluster_enc_range(snapshot):
    assert snapshot["cluster_enc"].between(0, 4).all()


def test_build_match_features_shape(snapshot, model_features):
    from streamlit_app.app import _build_match_features_vec
    teams = snapshot["team"].tolist()
    vec = _build_match_features_vec(snapshot, model_features, teams[0], teams[1])
    assert vec.shape == (1, 93)


def test_build_match_features_no_nan(snapshot, model_features):
    from streamlit_app.app import _build_match_features_vec
    teams = snapshot["team"].tolist()
    vec = _build_match_features_vec(snapshot, model_features, teams[0], teams[1])
    assert not np.isnan(vec).any(), "Feature vector contains NaN values"
