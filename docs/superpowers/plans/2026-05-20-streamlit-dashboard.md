# WC2026 Streamlit Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `streamlit_app/app.py` — a single-file Streamlit dashboard with three pages: Tournament Overview, Team Deep Dive, and Match Predictor.

**Architecture:** Pure data-transformation functions (`_build_snapshot`, `_build_match_features_vec`) live at module level so they're importable and testable without Streamlit. Thin `@st.cache_*` wrappers call them at runtime. Page renderers are plain functions called from `main()`.

**Tech Stack:** Streamlit, Pandas, NumPy, XGBoost (via joblib), Plotly Express + Graph Objects, Pathlib, scikit-learn LabelEncoder

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `streamlit_app/__init__.py` | Package marker (empty) |
| Create | `streamlit_app/app.py` | Entire dashboard (~350 lines) |
| Create | `tests/test_streamlit_app.py` | Unit tests for pure data functions |
| Modify | `requirements.txt` | Add `streamlit` entry |

---

## Background: Data Constraints

`data/processed/master_features.parquet` **does not exist** in this repo. `_build_snapshot` tries it first for `ranking` and `squad_value`; if absent it falls back to the raw FIFA ranking CSV for `ranking` and sets `squad_value = 1.0` for all teams (neutral default — `squad_value_diff` = 0 in every matchup, which is acceptable).

`model.classes_` = `['away', 'draw', 'home']` (alphabetical). `predict_proba` column order: `[0]` = P(away win), `[1]` = P(draw), `[2]` = P(home win). The code uses `dict(zip(model.classes_, probs))` to stay safe regardless of order.

---

## Task 1: Install Streamlit and create directory structure

**Files:**
- Modify: `requirements.txt`
- Create: `streamlit_app/__init__.py`

- [ ] **Step 1: Install streamlit in the venv**

```powershell
.\venv\Scripts\pip.exe install streamlit
```

Expected: `Successfully installed streamlit-<version>`

- [ ] **Step 2: Add streamlit to requirements.txt**

Open `requirements.txt` and add `streamlit` after the `# --- Visualization ---` block:

```
# --- Visualization ---
matplotlib
seaborn
plotly
streamlit
```

- [ ] **Step 3: Create the package directory and marker**

```powershell
New-Item -ItemType Directory -Force streamlit_app
New-Item -ItemType File streamlit_app\__init__.py
```

- [ ] **Step 4: Commit**

```powershell
git add requirements.txt streamlit_app\__init__.py
git commit -m "chore: add streamlit dependency and create streamlit_app package"
```

---

## Task 2: Write failing tests for pure data functions

**Files:**
- Create: `tests/test_streamlit_app.py`

- [ ] **Step 1: Create the test file**

Create `tests/test_streamlit_app.py` with this exact content:

```python
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
```

- [ ] **Step 2: Run tests — expect ImportError (app.py doesn't exist yet)**

```powershell
.\venv\Scripts\pytest.exe tests/test_streamlit_app.py -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name '_build_snapshot' from 'streamlit_app.app'` (or `ModuleNotFoundError`).

---

## Task 3: Implement `_build_snapshot` and `_build_match_features_vec`

**Files:**
- Create: `streamlit_app/app.py` (first half — imports through pure functions)

- [ ] **Step 1: Create `streamlit_app/app.py` with imports, path constants, module constants, and pure functions**

Create `streamlit_app/app.py` with this exact content (stop before the `@st.cache_*` wrappers — those come in Task 4):

```python
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
        df["ranking"] = df["ranking"].fillna(df["ranking"].median())

        latest_sq = (
            master.sort_values("date")
            .groupby("home_team").last()["squad_value_home"]
            .reset_index()
            .rename(columns={"home_team": "team", "squad_value_home": "squad_value"})
        )
        df = df.merge(latest_sq, on="team", how="left")
        df["squad_value"] = df["squad_value"].fillna(df["squad_value"].median())
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
```

- [ ] **Step 2: Run the tests**

```powershell
.\venv\Scripts\pytest.exe tests/test_streamlit_app.py -v
```

Expected output:

```
tests/test_streamlit_app.py::test_snapshot_row_count PASSED
tests/test_streamlit_app.py::test_snapshot_required_columns PASSED
tests/test_streamlit_app.py::test_snapshot_no_nulls_in_key_cols PASSED
tests/test_streamlit_app.py::test_snapshot_cluster_enc_range PASSED
tests/test_streamlit_app.py::test_build_match_features_shape PASSED
tests/test_streamlit_app.py::test_build_match_features_no_nan PASSED

6 passed
```

If any test fails, diagnose before continuing. Common issues:
- `KeyError: 'cluster_name'` → check parquet column names match `"cluster_name"` (not `"cluster"`)
- `ValueError in LabelEncoder.transform` → a team has a cluster not in `CLUSTER_ORDER`; print `df["cluster_name"].unique()` to check

- [ ] **Step 3: Commit**

```powershell
git add streamlit_app/app.py tests/test_streamlit_app.py
git commit -m "feat(dashboard): add pure data functions _build_snapshot + _build_match_features_vec with tests"
```

---

## Task 4: Add caching wrappers, main(), and verify the app starts

**Files:**
- Modify: `streamlit_app/app.py` (append caching wrappers + stub pages + main)

- [ ] **Step 1: Append caching wrappers and a stub main() to app.py**

Add the following block at the end of `streamlit_app/app.py` (after `_build_match_features_vec`):

```python
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
```

- [ ] **Step 2: Verify the app starts**

```powershell
.\venv\Scripts\streamlit.exe run streamlit_app/app.py
```

Open `http://localhost:8501` in a browser. Expected: title "⚽ WC2026 Predictor" with sidebar showing three navigation options, each page showing "Coming soon." No errors in terminal.

Stop the server with Ctrl+C.

- [ ] **Step 3: Commit**

```powershell
git add streamlit_app/app.py
git commit -m "feat(dashboard): add caching wrappers and navigation skeleton"
```

---

## Task 5: Implement `page_overview`

**Files:**
- Modify: `streamlit_app/app.py` — replace `page_overview` stub

- [ ] **Step 1: Replace the `page_overview` stub with the full implementation**

Replace the existing `page_overview` function body in `app.py` with:

```python
def page_overview(df_sim: pd.DataFrame, df_snap: pd.DataFrame) -> None:
    st.header("Tournament Overview")

    display_cols = ["team", "p_champion", "p_final", "p_sf", "p_r16"]
    df = (
        df_sim[display_cols]
        .merge(df_snap[["team", "cluster_name"]], on="team", how="left")
        .sort_values("p_champion", ascending=False)
        .reset_index(drop=True)
    )

    # ── Bar chart: top 15 by P(Champion) ─────────────────────────────────────
    top15 = df.head(15).copy()
    top15["P(Champion) %"] = (top15["p_champion"] * 100).round(1)
    fig_bar = px.bar(
        top15.sort_values("p_champion"),
        x="P(Champion) %",
        y="team",
        orientation="h",
        color="cluster_name",
        color_discrete_map=CLUSTER_COLORS,
        labels={"team": ""},
        title="Top 15 Teams — P(Champion)",
    )
    fig_bar.update_layout(legend_title="Cluster", height=450)
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Styled table ─────────────────────────────────────────────────────────
    st.subheader("All 48 Teams")

    # Keep floats for sorting; use Styler.format for display, hide cluster column
    def _row_style(row):
        color = CLUSTER_COLORS.get(row["cluster_name"], "transparent")
        return [f"background-color: {color}55; color: inherit"] * len(row)

    pct_cols = ["p_champion", "p_final", "p_sf", "p_r16"]
    styled = (
        df.style
        .apply(_row_style, axis=1)
        .format({col: "{:.1%}" for col in pct_cols})
        .hide(axis="columns", subset=["cluster_name"])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Cluster legend ────────────────────────────────────────────────────────
    cols = st.columns(len(CLUSTER_COLORS))
    for i, (name, color) in enumerate(CLUSTER_COLORS.items()):
        cols[i].markdown(
            f'<span style="background:{color}55; padding:2px 8px; border-radius:4px;">{name}</span>',
            unsafe_allow_html=True,
        )
```

- [ ] **Step 2: Run the app and verify the Overview page**

```powershell
.\venv\Scripts\streamlit.exe run streamlit_app/app.py
```

Check:
- Bar chart shows top 15 teams, bars colored by cluster
- Table shows all 48 teams with percentage formatting (e.g., `32.0%`)
- Row colors match clusters
- Columns are sortable in the Streamlit dataframe widget

Stop the server with Ctrl+C.

- [ ] **Step 3: Commit**

```powershell
git add streamlit_app/app.py
git commit -m "feat(dashboard): implement Tournament Overview page"
```

---

## Task 6: Implement `page_deep_dive`

**Files:**
- Modify: `streamlit_app/app.py` — replace `page_deep_dive` stub

- [ ] **Step 1: Replace the `page_deep_dive` stub**

```python
def page_deep_dive(df_sim: pd.DataFrame, df_snap: pd.DataFrame) -> None:
    st.header("Team Deep Dive")

    team_order = df_sim.sort_values("p_champion", ascending=False)["team"].tolist()
    selected   = st.selectbox("Select a team", team_order)

    snap = df_snap[df_snap["team"] == selected].iloc[0]
    sim  = df_sim[df_sim["team"]   == selected].iloc[0]

    # ── Metric cards ──────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Elo", f"{snap['elo']:.0f}")
    c2.metric("FIFA Ranking", f"{int(snap['ranking'])}")
    sv = float(snap["squad_value"])
    sv_label = f"€{sv:.0f}M" if sv > 1.0 else "N/A"
    c3.metric("Squad Value", sv_label)
    c4.metric("Cluster", snap["cluster_name"])
    c5.metric("Form Win Rate (10)", f"{snap['form_10_win_rate']:.1%}")

    # ── Comparison chart: selected team vs tournament average ─────────────────
    st.subheader("vs. Tournament Average (normalized 0–1)")

    metric_cols = {
        "Elo":                "elo",
        "Win Rate (10)":      "form_10_win_rate",
        "Goals Scored (10)":  "form_10_goals_scored_avg",
        "Goal Diff (10)":     "form_10_goal_diff_avg",
    }
    rows = []
    for label, col in metric_cols.items():
        vals = df_snap[col].astype(float)
        lo, hi = vals.min(), vals.max()
        denom = (hi - lo) or 1e-9
        rows.append({
            "Metric":            label,
            "Selected Team":     float((snap[col] - lo) / denom),
            "Tournament Average": float((vals.mean() - lo) / denom),
        })

    # Ranking is inverted: rank 1 = best, so normalize then flip
    ranks = df_snap["ranking"].astype(float)
    lo, hi = ranks.min(), ranks.max()
    denom = (hi - lo) or 1e-9
    rows.append({
        "Metric":            "FIFA Ranking (inv.)",
        "Selected Team":     float(1.0 - (snap["ranking"] - lo) / denom),
        "Tournament Average": float(1.0 - (ranks.mean() - lo) / denom),
    })

    df_cmp = pd.DataFrame(rows).melt(
        id_vars="Metric", var_name="Source", value_name="Score"
    )
    fig_cmp = px.bar(
        df_cmp,
        x="Score", y="Metric", color="Source",
        orientation="h", barmode="group",
        title=f"{selected} vs. Tournament Average",
        range_x=[0, 1],
        color_discrete_sequence=["#1f77b4", "#aec7e8"],
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    # ── Probability funnel ────────────────────────────────────────────────────
    st.subheader("Tournament Probabilities")
    prob_data = {
        "Round of 16": float(sim["p_r16"]),
        "Semi-Final":  float(sim["p_sf"]),
        "Final":       float(sim["p_final"]),
        "Champion":    float(sim["p_champion"]),
    }
    df_prob = pd.DataFrame({
        "Round":       list(prob_data.keys()),
        "Probability": list(prob_data.values()),
    })
    fig_prob = px.bar(
        df_prob, x="Round", y="Probability",
        text=[f"{v:.1%}" for v in prob_data.values()],
        color="Probability",
        color_continuous_scale="Blues",
        title=f"{selected} — Path to the Title",
    )
    fig_prob.update_traces(textposition="outside")
    fig_prob.update_layout(showlegend=False, yaxis_tickformat=".0%")
    st.plotly_chart(fig_prob, use_container_width=True)
```

- [ ] **Step 2: Run the app and verify the Deep Dive page**

```powershell
.\venv\Scripts\streamlit.exe run streamlit_app/app.py
```

Check:
- Team selector defaults to the top-ranked team by P(Champion)
- Five metric cards render correctly (Squad Value shows "N/A" when squad_value ≤ 1.0)
- Comparison chart shows two grouped bars per metric
- Probability chart decreases from P(R16) down to P(Champion)

Stop the server.

- [ ] **Step 3: Commit**

```powershell
git add streamlit_app/app.py
git commit -m "feat(dashboard): implement Team Deep Dive page"
```

---

## Task 7: Implement `page_predictor`

**Files:**
- Modify: `streamlit_app/app.py` — replace `page_predictor` stub

- [ ] **Step 1: Replace the `page_predictor` stub**

```python
def page_predictor(df_snap: pd.DataFrame, model, model_features: list) -> None:
    st.header("Match Predictor")
    st.caption(
        "Assumes neutral venue · 30 rest days per side · World Cup importance (tier 3) · "
        "H2H features zeroed (all WC2026 group-stage matchups are novel)."
    )

    teams = sorted(df_snap["team"].tolist())
    col1, col2 = st.columns(2)
    home = col1.selectbox("Home Team", teams, index=0,  key="pred_home")
    away = col2.selectbox("Away Team", teams, index=1,  key="pred_away")

    if st.button("Predict", type="primary"):
        if home == away:
            st.warning("Please select two different teams.")
            return

        vec   = _build_match_features_vec(df_snap, model_features, home, away)
        probs = model.predict_proba(vec)[0]

        # model.classes_ = ['away', 'draw', 'home'] — use dict to be order-safe
        class_map = dict(zip(model.classes_, probs))
        p_home = float(class_map.get("home", probs[2]))
        p_draw = float(class_map.get("draw", probs[1]))
        p_away = float(class_map.get("away", probs[0]))

        st.subheader(f"{home} vs. {away}")

        c1, c2, c3 = st.columns(3)
        c1.metric(f"{home} Win", f"{p_home:.1%}")
        c2.metric("Draw",        f"{p_draw:.1%}")
        c3.metric(f"{away} Win", f"{p_away:.1%}")

        # Stacked horizontal bar
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[p_home], y=[""], orientation="h",
            name=f"{home} Win",
            marker_color="#1f77b4",
            text=f" {p_home:.1%}",
            textposition="inside",
            insidetextanchor="start",
        ))
        fig.add_trace(go.Bar(
            x=[p_draw], y=[""], orientation="h",
            name="Draw",
            marker_color="#aec7e8",
            text=f" Draw {p_draw:.1%}",
            textposition="inside",
            insidetextanchor="middle",
        ))
        fig.add_trace(go.Bar(
            x=[p_away], y=[""], orientation="h",
            name=f"{away} Win",
            marker_color="#ff7f0e",
            text=f" {p_away:.1%}",
            textposition="inside",
            insidetextanchor="end",
        ))
        fig.update_layout(
            barmode="stack",
            height=120,
            xaxis=dict(range=[0, 1], tickformat=".0%", title=""),
            yaxis=dict(showticklabels=False),
            margin=dict(l=0, r=0, t=10, b=30),
            legend=dict(orientation="h", y=-0.8),
        )
        st.plotly_chart(fig, use_container_width=True)
```

- [ ] **Step 2: Run the app and verify the Match Predictor page**

```powershell
.\venv\Scripts\streamlit.exe run streamlit_app/app.py
```

Check:
- Select two different teams and click Predict — three metric cards and stacked bar appear
- Probabilities sum to ~1.0 (minor float rounding is fine)
- Select the same team for both dropdowns and click Predict — warning message shown, no crash
- Try a strong vs. weak team (e.g., France vs. San Marino equivalent) — home win probability should dominate

Stop the server.

- [ ] **Step 3: Commit**

```powershell
git add streamlit_app/app.py
git commit -m "feat(dashboard): implement Match Predictor page"
```

---

## Task 8: Final regression pass and tests

**Files:**
- No file changes — verification only

- [ ] **Step 1: Run the full test suite**

```powershell
.\venv\Scripts\pytest.exe tests/test_streamlit_app.py -v
```

Expected: 6 tests pass, 0 failures.

- [ ] **Step 2: Smoke-test all three pages**

```powershell
.\venv\Scripts\streamlit.exe run streamlit_app/app.py
```

Walkthrough:
1. **Overview**: table shows 48 rows, bar chart shows 15 bars, percentages display correctly
2. **Deep Dive**: switch to 3 different teams — metrics, chart, funnel all update
3. **Predictor**: run 3 predictions including one same-team guard

Stop the server.

- [ ] **Step 3: Verify run instructions are clear**

The command to run the app is:

```powershell
# Activate venv first (if not already active):
.\venv\Scripts\Activate.ps1

# Then run:
streamlit run streamlit_app/app.py
```

App opens at `http://localhost:8501`.
