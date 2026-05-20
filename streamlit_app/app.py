"""WC2026 Match Predictor — Streamlit Dashboard"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[1]
MODEL_PATH    = ROOT / "models" / "xgb_match_predictor.pkl"
FEATURES_PATH = ROOT / "models" / "model_features.json"
SIM_PATH      = ROOT / "outputs" / "predictions" / "simulation_results.csv"
SNAPSHOT_PATH = ROOT / "data" / "processed" / "team_snapshot_clustered.parquet"
MASTER_PATH   = ROOT / "data" / "processed" / "master_features.parquet"
RANKING_PATH  = ROOT / "data" / "raw" / "cashncarry_rankings" / "fifa_ranking-2024-04-04.csv"

# ── Style constants ────────────────────────────────────────────────────────────
CLUSTER_COLORS = {
    "Elite":                 "#1f4e79",
    "Consolidated Mid-Tier": "#2e7d32",
    "Dynamic Mid-Tier":      "#e65100",
    "Underdogs":             "#6a1b9a",
}
CLUSTER_ORDER = [
    "Consolidated Mid-Tier", "Dynamic Mid-Tier", "Elite", "Non-WC2026", "Underdogs"
]

# ── Feature constants (must match simulate.py exactly) ────────────────────────
FORM_WINDOWS = [5, 10, 20]
FORM_STATS = [
    "win_rate", "draw_rate", "loss_rate",
    "goals_scored_avg", "goals_conceded_avg", "goal_diff_avg",
    "clean_sheet_rate", "failed_score_rate", "points_avg",
    "matches_played", "weighted_points",
]
SNAPSHOT_DATE = pd.Timestamp("2026-03-31")


# ── Pure data functions (no Streamlit — importable in tests) ──────────────────

def _build_snapshot() -> pd.DataFrame:
    """
    Load 48-team snapshot. Adds ranking + squad_value from master_features.parquet
    if present; falls back to raw FIFA ranking CSV + 1.0 default for squad_value.
    Also adds cluster_enc via LabelEncoder (same fit order as simulate.py/train.py).
    """
    df = pd.read_parquet(SNAPSHOT_PATH)

    le = LabelEncoder()
    le.fit(CLUSTER_ORDER)
    df["cluster_enc"] = le.transform(df["cluster_name"])

    if MASTER_PATH.exists():
        master = pd.read_parquet(MASTER_PATH)
        master = master[master["date"] <= SNAPSHOT_DATE]

        latest_rank = (
            master.sort_values("date")
            .groupby("home_team").last()["ranking_home"]
            .reset_index()
            .rename(columns={"home_team": "team", "ranking_home": "ranking"})
        )
        df = df.merge(latest_rank, on="team", how="left")
        df["ranking"] = df["ranking"].fillna(latest_rank["ranking"].median())

        latest_sq = (
            master.sort_values("date")
            .groupby("home_team").last()["squad_value_home"]
            .reset_index()
            .rename(columns={"home_team": "team", "squad_value_home": "squad_value"})
        )
        df = df.merge(latest_sq, on="team", how="left")
        df["squad_value"] = df["squad_value"].fillna(latest_sq["squad_value"].median())
    else:
        raw_rank = pd.read_csv(RANKING_PATH)
        raw_rank = (
            raw_rank.sort_values("rank_date")
            .groupby("country_full").last()
            .reset_index()[["country_full", "rank"]]
            .rename(columns={"country_full": "team", "rank": "ranking"})
        )
        df = df.merge(raw_rank, on="team", how="left")
        df["ranking"] = df["ranking"].fillna(df["ranking"].median())
        df["squad_value"] = 1.0

    return df.reset_index(drop=True)


def _build_match_features_vec(
    df_snap: pd.DataFrame,
    model_features: list,
    home_team: str,
    away_team: str,
) -> np.ndarray:
    """
    Build the 93-feature vector for one matchup.
    Fixed values: neutral=True, rest_days=30, match_importance=3, H2H zeroed.
    Returns shape (1, 93).
    """
    h = df_snap[df_snap["team"] == home_team].iloc[0]
    a = df_snap[df_snap["team"] == away_team].iloc[0]

    elo_diff      = float(h["elo"]) - float(a["elo"])
    win_prob_home = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))

    features: dict = {
        "elo_pre_home":     float(h["elo"]),
        "elo_pre_away":     float(a["elo"]),
        "elo_diff":         elo_diff,
        "win_prob_home":    win_prob_home,
        "neutral":          1.0,
        "ranking_home":     float(h["ranking"]),
        "ranking_away":     float(a["ranking"]),
        "ranking_diff":     float(h["ranking"]) - float(a["ranking"]),
        "squad_value_home": float(h["squad_value"]),
        "squad_value_away": float(a["squad_value"]),
        "squad_value_diff": float(h["squad_value"]) - float(a["squad_value"]),
        "rest_days_home":   30.0,
        "rest_days_away":   30.0,
        "match_importance": 3.0,
    }

    for w in FORM_WINDOWS:
        for stat in FORM_STATS:
            features[f"home_form_{w}_{stat}"] = float(h[f"form_{w}_{stat}"])
            features[f"away_form_{w}_{stat}"] = float(a[f"form_{w}_{stat}"])

    features.update({
        "h2h_matches":               0.0,
        "h2h_win_rate_a":            0.5,
        "h2h_goal_diff_a":           0.0,
        "h2h_elo_edge_a":            0.0,
        "h2h_weighted_edge_a":       0.0,
        "h2h_decay_weight":          0.0,
        "h2h_reliable":              0.0,
        "transitive_common_rivals":  0.0,
        "transitive_edge_a":         0.0,
        "transitive_goal_diff_edge": 0.0,
        "transitive_reliable":       0.0,
    })

    features["home_cluster_enc"] = float(h["cluster_enc"])
    features["away_cluster_enc"] = float(a["cluster_enc"])

    return np.array([features[f] for f in model_features], dtype=float).reshape(1, -1)


# ── Cached loaders ─────────────────────────────────────────────────────────────

@st.cache_data
def load_snapshot() -> pd.DataFrame:
    return _build_snapshot()


@st.cache_resource
def load_model_and_features():
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH) as f:
        features = json.load(f)
    return model, features


@st.cache_data
def load_simulation() -> pd.DataFrame:
    return pd.read_csv(SIM_PATH)


# ── Page stubs (replaced in Tasks 5–7) ────────────────────────────────────────

def page_overview(df_sim: pd.DataFrame, df_snap: pd.DataFrame) -> None:
    st.header("Tournament Overview")
    st.write("Coming soon.")


def page_deep_dive(df_sim: pd.DataFrame, df_snap: pd.DataFrame) -> None:
    st.header("Team Deep Dive")
    st.write("Coming soon.")


def page_predictor(df_snap: pd.DataFrame, model, model_features: list) -> None:
    st.header("Match Predictor")
    st.write("Coming soon.")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="⚽ WC2026 Predictor", layout="wide", page_icon="⚽")
    st.title("⚽ WC2026 Predictor")

    page = st.sidebar.radio(
        "Navigation",
        ["Tournament Overview", "Team Deep Dive", "Match Predictor"],
    )

    df_snap               = load_snapshot()
    df_sim                = load_simulation()
    model, model_features = load_model_and_features()

    if page == "Tournament Overview":
        page_overview(df_sim, df_snap)
    elif page == "Team Deep Dive":
        page_deep_dive(df_sim, df_snap)
    else:
        page_predictor(df_snap, model, model_features)


if __name__ == "__main__":
    main()
