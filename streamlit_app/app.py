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
        if RANKING_PATH.exists():
            raw_rank = (
                pd.read_csv(RANKING_PATH)
                .sort_values("rank_date")
                .groupby("country_full").last()
                .reset_index()[["country_full", "rank"]]
                .rename(columns={"country_full": "team", "rank": "ranking"})
            )
            df = df.merge(raw_rank, on="team", how="left")
            df["ranking"] = df["ranking"].fillna(200)
        else:
            df["ranking"] = 200
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

    has_blend = "p_champion_blended" in df_sim.columns
    rank_col = "p_champion_blended" if has_blend else "p_champion"

    display_cols = ["team", "p_champion", "p_final", "p_sf", "p_qf", "p_r16", "p_r32"]
    if has_blend:
        display_cols = ["team", "p_champion_blended", "p_champion",
                        "p_final", "p_sf", "p_qf", "p_r16", "p_r32"]
    df = (
        df_sim[display_cols]
        .merge(df_snap[["team", "cluster_name"]], on="team", how="left")
        .sort_values(rank_col, ascending=False)
        .reset_index(drop=True)
    )

    # ── Bar chart: top 15 by P(Champion) ─────────────────────────────────────
    top15 = df.head(15).copy()
    top15["P(Champion) %"] = (top15[rank_col] * 100).round(1)
    top15_sorted = top15.sort_values(rank_col, ascending=True)
    fig_bar = go.Figure()
    for cluster, color in CLUSTER_COLORS.items():
        subset = top15_sorted[top15_sorted["cluster_name"] == cluster]
        if subset.empty:
            continue
        fig_bar.add_trace(go.Bar(
            x=subset["P(Champion) %"],
            y=subset["team"],
            orientation="h",
            name=cluster,
            marker_color=color,
        ))
    title = "Top 15 Teams — P(Champion)"
    if has_blend:
        title += " (Blended with Expert Consensus)"
    fig_bar.update_layout(
        title=title,
        legend_title="Cluster",
        height=450,
        yaxis=dict(
            categoryorder="array",
            categoryarray=top15_sorted["team"].tolist(),
        ),
    )
    fig_bar.update_xaxes(tickformat=".1f", title_text="P(Champion) %", rangemode="tozero")
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Styled table ─────────────────────────────────────────────────────────
    st.subheader("All 48 Teams")

    RENAME_MAP = {
        "p_champion_blended": "P(Champion) Blended",
        "p_champion": "P(Champion) Model",
        "p_final": "P(Final)",
        "p_sf": "P(SF)",
        "p_qf": "P(QF)",
        "p_r16": "P(R16)",
        "p_r32": "P(R32)",
    }
    cluster_series = df["cluster_name"]
    df_display = (
        df.drop(columns=["cluster_name"])
        .sort_values(rank_col, ascending=False)
        .reset_index(drop=True)
        .rename(columns=RENAME_MAP)
    )
    pct_cols = [RENAME_MAP[c] for c in RENAME_MAP if RENAME_MAP[c] in df_display.columns]

    def _row_style(row):
        hex_color = CLUSTER_COLORS.get(cluster_series.iloc[row.name], "")
        if not hex_color:
            return [""] * len(row)
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        return [f"background-color: rgba({r},{g},{b},0.33)"] * len(row)

    styled = (
        df_display.style
        .apply(_row_style, axis=1)
        .format({col: "{:.2%}" for col in pct_cols})
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Cluster legend ────────────────────────────────────────────────────────
    cols = st.columns(len(CLUSTER_COLORS))
    for i, (name, color) in enumerate(CLUSTER_COLORS.items()):
        cols[i].markdown(
            f'<span style="background:{color}55; padding:2px 8px; border-radius:4px;">{name}</span>',
            unsafe_allow_html=True,
        )


def page_deep_dive(df_sim: pd.DataFrame, df_snap: pd.DataFrame) -> None:
    st.header("Team Deep Dive")

    has_blend = "p_champion_blended" in df_sim.columns
    rank_col = "p_champion_blended" if has_blend else "p_champion"
    team_order = df_sim.sort_values(rank_col, ascending=False)["team"].tolist()
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
        denom = max(float(hi - lo), 1e-9)
        rows.append({
            "Metric":            label,
            "Selected Team":     float((snap[col] - lo) / denom),
            "Tournament Average": float((vals.mean() - lo) / denom),
        })

    # Ranking is inverted: rank 1 = best, so normalize then flip
    ranks = df_snap["ranking"].astype(float)
    lo, hi = ranks.min(), ranks.max()
    denom = max(float(hi - lo), 1e-9)
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
    # Cumulative "reach at least X" probabilities — monotone across all teams
    p_r16_reach   = float(sim["p_r16"])   + float(sim["p_qf"]) + float(sim["p_sf"]) + float(sim["p_third"]) + float(sim["p_final"]) + float(sim["p_champion"])
    p_qf_reach    = float(sim["p_qf"])    + float(sim["p_sf"]) + float(sim["p_third"]) + float(sim["p_final"]) + float(sim["p_champion"])
    p_sf_reach    = float(sim["p_sf"])    + float(sim["p_third"]) + float(sim["p_final"]) + float(sim["p_champion"])
    p_final_reach = float(sim["p_final"]) + float(sim["p_champion"])
    prob_data = {
        "Reach R16":   p_r16_reach,
        "Reach QF":    p_qf_reach,
        "Reach SF":    p_sf_reach,
        "Reach Final": p_final_reach,
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
    fig_prob.update_layout(showlegend=False, yaxis_tickformat=".0%", margin=dict(t=40))
    st.plotly_chart(fig_prob, use_container_width=True)

    if "p_champion_blended" in sim.index:
        st.caption(
            f"Blended P(Champion) incl. expert consensus: "
            f"{float(sim['p_champion_blended']):.1%}"
        )


def page_predictor(df_snap: pd.DataFrame, model, model_features: list) -> None:
    st.header("Match Predictor")
    st.write(
        "Select any two teams to get a match outcome probability. "
        "The model uses 93 features built from historical and current team stats."
    )
    st.caption(
        "Assumes 30 rest days per side · World Cup importance (tier 3) · "
        "H2H features zeroed (all WC2026 group-stage matchups are novel)."
    )

    teams = sorted(df_snap["team"].tolist())
    col1, col2 = st.columns(2)
    team1 = col1.selectbox("Team 1", teams, index=0, key="pred_team1")
    team2 = col2.selectbox("Team 2", teams, index=1, key="pred_team2")
    st.caption("⚽ Neutral ground prediction — result is independent of team order.")

    if st.button("Predict", type="primary"):
        if team1 == team2:
            st.warning("Please select two different teams.")
            return

        # Run model in both orderings and average for a symmetric, order-independent result.
        # Trained targets: 0=away win, 1=draw, 2=home win
        vec1 = _build_match_features_vec(df_snap, model_features, team1, team2)
        vec2 = _build_match_features_vec(df_snap, model_features, team2, team1)
        p1   = model.predict_proba(vec1)[0]   # [team2 win, draw, team1 win]
        p2   = model.predict_proba(vec2)[0]   # [team1 win, draw, team2 win]

        p_team1 = (float(p1[2]) + float(p2[0])) / 2
        p_draw  = (float(p1[1]) + float(p2[1])) / 2
        p_team2 = (float(p1[0]) + float(p2[2])) / 2

        st.subheader(f"{team1} vs. {team2}")

        c1, c2, c3 = st.columns(3)
        c1.metric(f"{team1} Win", f"{p_team1:.1%}")
        c2.metric("Draw",         f"{p_draw:.1%}")
        c3.metric(f"{team2} Win", f"{p_team2:.1%}")

        # Stacked horizontal bar
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[p_team1], y=[""], orientation="h",
            name=f"{team1} Win",
            marker_color="#1f77b4",
            text=f"{p_team1:.1%}",
            textposition="inside",
            insidetextanchor="middle",
        ))
        fig.add_trace(go.Bar(
            x=[p_draw], y=[""], orientation="h",
            name="Draw",
            marker_color="#aec7e8",
            text=f"Draw {p_draw:.1%}",
            textposition="inside",
            insidetextanchor="middle",
        ))
        fig.add_trace(go.Bar(
            x=[p_team2], y=[""], orientation="h",
            name=f"{team2} Win",
            marker_color="#ff7f0e",
            text=f"{p_team2:.1%}",
            textposition="inside",
            insidetextanchor="middle",
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
