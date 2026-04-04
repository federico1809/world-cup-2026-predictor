# ⚽ FIFA World Cup 2026 Predictor
> End-to-end machine learning pipeline for predicting FIFA World Cup 2026 outcomes
> using ensemble models, Monte Carlo simulation, and unsupervised clustering.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-In%20Development-orange)

---

## 🎯 What This Project Predicts

| Level | Output |
|-------|--------|
| Match | Win / Draw / Loss probability per match |
| Group stage | Final standings + classification probabilities |
| Knockout bracket | How R16 fixture assembles from group results |
| Phase advancement | P(reach R16 / QF / SF / Final / Win) per team |
| Tournament winner | Full probability distribution for all 48 teams |

### Adaptive retraining strategy
The model is designed for real-time use during the tournament:
- Pre-tournament: predict all 104 matches end-to-end via Monte Carlo
- After group stage: retrain on actual group results, predict knockout phase
- After each round: incorporate new results, update remaining predictions

Final presentation: FiveThirtyEight-style probability table + Streamlit interactive
dashboard + static notebook visualizations.

---

## 🧠 Key Design Principles

- Recent form over historical reputation — Brazil's 1970 title does not predict 2026.
  All features computed with strict temporal awareness.
- No data leakage — features computed strictly as of each match date.
- Strict temporal split — train/val/test by date, never random.
- Calibrated probabilities — outputs calibrated for reliable Monte Carlo input.
- Unsupervised methods inform supervised — clustering of 48 qualified teams generates
  categorical features; PCA detects multicollinearity.
- Adaptive pipeline — model retrained mid-tournament with real results.

---

## 📁 Project Structure

    world-cup-2026-predictor/
    ├── configs/                  Global parameters (seeds, paths, hyperparameters)
    ├── data/
    │   ├── raw/                  Downloaded datasets (never modified)
    │   ├── interim/              Intermediate transformations
    │   ├── processed/            Model-ready feature matrices (*.parquet tracked)
    │   └── external/             StatsBomb events, third-party sources
    ├── models/                   Serialized trained models and encoders
    ├── notebooks/
    │   ├── 01_eda/               Exploratory Data Analysis
    │   ├── 02_features/          Feature Engineering
    │   ├── 03_unsupervised/      Clustering and PCA
    │   ├── 04_modeling/          Supervised modeling and evaluation
    │   └── 05_simulation/        Monte Carlo tournament simulation
    ├── outputs/
    │   ├── figures/              Generated plots (gitignored)
    │   └── predictions/          Tournament simulation results
    ├── world_cup_2026/
    │   ├── data_ingestion/       Download pipeline and normalization
    │   ├── features/             Elo, H2H, form feature modules
    │   ├── modeling/             Training and inference
    │   └── simulation/           Monte Carlo engine
    └── tests/

---

## 📊 Datasets

| Source | Description | Period | Size |
|--------|-------------|--------|------|
| martj42 | All international results | 1872-2026 | 49,071 matches |
| patateriedata | Daily updated results incl. qualifiers | 1872-2026 | 51,384 matches |
| lchikry | Pre-calculated Elo, form, FIFA ratings | 1872-2025 | 43,364 x 35 features |
| joshfjelstul | Relational World Cup database | 1930-2022 | 900 matches |
| cashncarry | Monthly FIFA rankings | 1992-2024 | 67,472 records |
| sarazahran1 | WC2026 Elo probability baseline | 2026 | 72 matches |
| areezvisram12 | Complete 104-match fixture | 2026 | 104 matches |
| StatsBomb Open Data | Match events xG, passes, shots | 2018-2022 WC | JSON |

---

## 🔬 Feature Engineering

### Implemented (master_features.parquet — 9,796 x 94)

| Module | File | Description |
|--------|------|-------------|
| Elo rating | features/elo.py | Recalculated from 150yr history, dynamic K-factor (WC=60, Friendly=20) |
| H2H + Transitive | features/h2h.py | Direct H2H edge + transitive rival + temporal decay |
| Recent form | features/form.py | Win rate, goals, points over 5/10/20 matches + exp decay |
| FIFA Rankings | cashncarry dataset | ranking_home, ranking_away, ranking_diff — as-of join per match date |
| Neutral venue | results dataset | Binary flag — reduces home advantage ~3.5pp |
| Cluster label | notebook 03 | KMeans cluster assignment (Elite / Mid-Tier / Underdogs) |

### Planned

| Feature | Source | Signal strength |
|---------|--------|-----------------|
| Squad market value | Transfermarkt | HIGH |
| Average squad age | Transfermarkt | HIGH |
| Coach tenure months | Transfermarkt | MEDIUM |
| Squad continuity since 2022 WC | Transfermarkt | HIGH |
| Key player injuries/suspensions | Press scraping | HIGH |
| Venue altitude | Sedes data | MEDIUM |
| Match day weather | Weather API | LOW-MEDIUM |

---

## 🤖 Modeling Pipeline

### Supervised models

| Model | Val Accuracy | Val F1-macro | Val Log-loss | Status |
|-------|-------------|--------------|--------------|--------|
| Logistic Regression (baseline) | 0.4113 | 0.3262 | 1.0948 | Done |
| XGBoost + Optuna (87 features) | 0.3969 | 0.3667 | 1.0886 | Done ✓ selected |
| Random Forest (87 features) | 0.3846 | 0.3566 | 1.0895 | Done |
| MLP (PyTorch) | — | — | — | Pending |
| Stacking Ensemble | — | — | — | Pending |

**Selected model:** XGBoost — best F1-macro and log-loss on validation set.

**Feature set (87):** Elo (4) + Neutral (1) + FIFA Rankings (3) + Form 5/10/20 (66) + H2H (11) + Cluster (2)

**Top features by gain:** neutral, elo_diff, h2h_win_rate_a, win_prob_home, ranking_diff

### Unsupervised methods

| Method | Output | Status |
|--------|--------|--------|
| K-Means (k=4) | Cluster labels for 48 WC2026 teams | Done |
| PCA 2D | Visualization + variance analysis | Done |
| Anomaly detection | Distance to centroid — Ecuador, Qatar flagged | Done |

**Cluster results:**
| Cluster | Name | n | Avg Elo | Form WR |
|---------|------|---|---------|---------|
| 1 | Elite | 16 | 1983 | 0.72 |
| 0 | Consolidated Mid-Tier | 12 | 1847 | 0.38 |
| 2 | Dynamic Mid-Tier | 14 | 1828 | 0.54 |
| 3 | Underdogs | 6 | 1679 | 0.32 |

### Monte Carlo simulation
- 10,000 full tournament simulations
- Calibrated probabilities from XGBoost
- Penalty shootout modeled for knockout rounds
- Adaptive: re-run after each round with real results

---

## 📈 EDA Key Findings

| Finding | Value |
|---------|-------|
| Elo diff correlation with goal diff | 0.515 |
| Home advantage — all internationals | 49% HW / 23% D / 28% AW |
| Home advantage — World Cup neutral | 45.5% HW / 22% D / 32.3% AW |
| Away win rate trend 2021-2026 | 28% to 33% |
| Top Elo WC2026 team | Spain (2209) |
| Brazil current form last 10 | 0.50 win rate |
| England current form last 10 | 0.90 win rate |

---

## 🚀 Quickstart

    git clone https://github.com/federico1809/world-cup-2026-predictor.git
    cd world-cup-2026-predictor
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    python -m world_cup_2026.data_ingestion.download

---

## 🗂️ Development Status

| Phase | Status |
|-------|--------|
| Project scaffold | ✅ Done |
| Data ingestion pipeline | ✅ Done |
| Team name normalization 42/42 | ✅ Done |
| Exploratory Data Analysis | ✅ Done |
| Elo calculator | ✅ Done |
| H2H + transitive rival features | ✅ Done |
| Recent form features 5/10/20 | ✅ Done |
| FIFA Rankings feature join | ✅ Done |
| Unsupervised clustering (k=4) | ✅ Done |
| Supervised modeling (XGBoost) | ✅ Done |
| Transfermarkt squad features | ⏳ Pending |
| Monte Carlo simulation | ⏳ Pending |
| Streamlit dashboard | ⏳ Pending |

---

## 👤 Author

Federico Ceballos Torres — Data Scientist
GitHub: https://github.com/federico1809

---

## 📄 License

MIT — see LICENSE for details.