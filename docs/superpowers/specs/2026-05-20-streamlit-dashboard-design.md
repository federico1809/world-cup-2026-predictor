# WC2026 Streamlit Dashboard — Design Spec

**Date:** 2026-05-20  
**Branch:** dev  
**Scope:** Single-file Streamlit app (`streamlit_app/app.py`) with three pages.

---

## 1. Architecture

```
streamlit_app/
└── app.py          # ~350 lines, self-contained
```

Three pages served via `st.sidebar` radio navigation. All heavy assets loaded once at startup and cached.

### Caching strategy

| Decorator | Content |
|---|---|
| `@st.cache_resource` | XGBoost model (`joblib.load`) + feature list (`json.load`) |
| `@st.cache_data` | `df_snapshot` — 48-team feature snapshot |
| `@st.cache_data` | `df_sim` — simulation results CSV |

### Path resolution

All paths derived relative to `app.py` using `pathlib`:

```python
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH    = ROOT / "models/xgb_match_predictor.pkl"
FEATURES_PATH = ROOT / "models/model_features.json"
SIM_PATH      = ROOT / "outputs/predictions/simulation_results.csv"
SNAPSHOT_PATH = ROOT / "data/processed/team_snapshot_clustered.parquet"
MASTER_PATH   = ROOT / "data/processed/master_features.parquet"
RANKING_PATH  = ROOT / "data/raw/cashncarry_rankings/fifa_ranking-2024-04-04.csv"
```

---

## 2. df_snapshot Rebuild

`team_snapshot_clustered.parquet` (48 rows) provides Elo, 33 form features, and `cluster_name`. Two additional fields are required by `build_match_features`:

- **`ranking`**: Try `master_features.parquet` first (column `ranking_home`, latest date ≤ 2026-03-31, grouped by `home_team`). Fallback: load `fifa_ranking-2024-04-04.csv`, match on `country_full`, use `total_points` ranking column (field `rank`). Remaining nulls filled with median.
- **`squad_value`**: Try `master_features.parquet` first (column `squad_value_home`). Fallback: fill all with median of available values (or 1.0 if completely unavailable).
- **`cluster_enc`**: `LabelEncoder` fit on the fixed order `["Consolidated Mid-Tier", "Dynamic Mid-Tier", "Elite", "Non-WC2026", "Underdogs"]` — identical to `simulate.py`.

---

## 3. Page 1 — Tournament Overview

**Data source:** `simulation_results.csv`  
**Displayed columns:** `team, p_champion, p_final, p_sf, p_r16` (formatted as `X.X%`)  
**Default sort:** `p_champion` descending  
**Sortable:** yes, via `st.dataframe` with `column_config`  
**Row coloring:** `pandas Styler` background color mapped to `cluster_name` merged from df_snapshot:

| Cluster | Color |
|---|---|
| Elite | `#1f4e79` (dark blue) |
| Consolidated Mid-Tier | `#2e7d32` (dark green) |
| Dynamic Mid-Tier | `#e65100` (deep orange) |
| Underdogs | `#6a1b9a` (purple) |

**Chart:** Plotly horizontal bar chart, top 15 teams by P(Champion). X-axis as percentage. Color bars by cluster.

---

## 4. Page 2 — Team Deep Dive

**Team selector:** `st.selectbox`, options sorted by `p_champion` descending (most likely champion first).

**Metric cards (top row):** Elo · FIFA Ranking · Squad Value · Cluster · Form Win Rate (last 10)

**Comparison chart:** Plotly horizontal bar chart with normalized values (0–1 min-max across all 48 teams). Metrics compared: Elo, ranking (inverted — lower rank = better), `form_10_win_rate`, `form_10_goals_scored_avg`, `form_10_goal_diff_avg`. Two bars per metric: selected team vs. tournament mean.

**Probability funnel:** Plotly bar chart showing `p_r16 → p_sf → p_final → p_champion` for the selected team, labeled with raw percentages.

---

## 5. Page 3 — Match Predictor

**Inputs:** `st.selectbox` for Home team, `st.selectbox` for Away team (all 48 WC2026 teams). `st.button("Predict")`.

**Feature construction:** Inline replication of `build_match_features` from `simulate.py`, using `df_snapshot` and `MODEL_FEATURES` globals. Fixed values: `neutral=True`, `rest_days_home=rest_days_away=30`, `match_importance=3`. H2H features all zeroed/defaulted as in simulate.py.

**Output:** `model.predict_proba(feature_vector)` → three probabilities. Displayed as:
- Three `st.metric` cards: Home Win / Draw / Away Win
- Plotly stacked horizontal bar (full width, single row) showing the three probabilities

**Guard:** If home == away, show `st.warning` and skip prediction.

---

## 6. Style

- Page title: `"⚽ WC2026 Predictor"`
- Sidebar navigation via `st.sidebar.radio`
- No custom CSS — use Streamlit defaults
- Charts use Plotly (already in requirements.txt)

---

## 7. Dependencies

`streamlit` is not in `requirements.txt`. User must run:

```bash
pip install streamlit
```

All other dependencies (`pandas`, `numpy`, `xgboost`, `joblib`, `plotly`, `pyarrow`) are already present.

---

## 8. Run command

```bash
streamlit run streamlit_app/app.py
```
