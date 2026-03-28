# ⚽ FIFA World Cup 2026 Predictor

> End-to-end machine learning pipeline for predicting FIFA World Cup 2026 match outcomes and final standings using ensemble models, Monte Carlo simulation, and unsupervised clustering.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-In%20Development-orange)

---

## 🎯 Project Goal

Predict the FIFA World Cup 2026 results at three levels of granularity:

1. **Match level** — Win / Draw / Loss probability for each of the 104 matches
2. **Group stage** — Classification probabilities for all 12 groups
3. **Tournament** — Final standings with confidence intervals via 10,000 Monte Carlo simulations

---

## 🧠 Key Design Principles

- **Recent form over historical reputation** — a team's last 20 matches matter infinitely more than their 1970 World Cup title. All features are computed with temporal awareness.
- **No data leakage** — features are computed strictly as of each match date. No future information bleeds into training.
- **Strict temporal split** — train/val/test split by date, never random.
- **Calibrated probabilities** — model outputs are probability-calibrated to feed Monte Carlo simulation reliably.

---

## 📁 Project Structure
`
world-cup-2026-predictor/
│
├── configs/              # Global parameters (seeds, paths, hyperparameters)
├── data/
│   ├── raw/              # Downloaded datasets (never modified)
│   ├── interim/          # Intermediate transformations
│   ├── processed/        # Model-ready feature matrices
│   └── external/         # StatsBomb events, third-party sources
│
├── models/               # Serialized trained models and encoders
├── notebooks/
│   ├── 01_eda/           # Exploratory Data Analysis
│   ├── 02_features/      # Feature Engineering
│   ├── 03_unsupervised/  # Clustering and PCA
│   ├── 04_modeling/      # Supervised modeling and evaluation
│   └── 05_simulation/    # Monte Carlo tournament simulation
│
├── outputs/
│   ├── figures/          # Generated plots
│   └── predictions/      # Tournament simulation results
│
├── world_cup_2026/       # Source package
│   ├── data_ingestion/   # Download pipeline and normalization
│   ├── features/         # Feature engineering modules
│   ├── modeling/         # Training and inference
│   └── simulation/       # Monte Carlo engine
│
└── tests/                # Unit tests
`

---

## 📊 Datasets

| Source | Description | Period | Size |
|--------|-------------|--------|------|
| [martj42 — International Results](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) | All international match results | 1872–2026 | 49,071 matches |
| [patateriedata — Daily Updates](https://www.kaggle.com/datasets/patateriedata/all-international-football-results) | Results with daily updates incl. qualifiers | 1872–2026 | 51,384 matches |
| [lchikry — Match Features](https://www.kaggle.com/datasets/lchikry/international-football-match-features-and-statistics) | Pre-calculated Elo, form, FIFA ratings | 1872–2025 | 43,364 × 35 features |
| [joshfjelstul — World Cup DB](https://www.kaggle.com/datasets/joshfjelstul/world-cup-database) | Relational WC database | 1930–2022 | 900 matches |
| [cashncarry — FIFA Rankings](https://www.kaggle.com/datasets/cashncarry/fifaworldranking) | Monthly FIFA rankings | 1992–2024 | 67,472 records |
| [sarazahran1 — Elo Baseline](https://www.kaggle.com/datasets/sarazahran1/wc2026-match-probability-baseline-dataset) | WC2026 Elo probability baseline | 2026 | 72 matches |
| [areezvisram12 — WC2026 Fixture](https://www.kaggle.com/datasets/areezvisram12/fifa-world-cup-2026-match-data-unofficial) | Complete 104-match fixture | 2026 | 104 matches |
| [StatsBomb Open Data](https://github.com/statsbomb/open-data) | Match events (xG, passes, shots) | 2018, 2022 WC | JSON |

---

## 🔬 Methodology

### Feature Engineering
- **Elo rating** — recalculated from scratch on 150+ years of results with dynamic K-factor
- **Recent form** — win rate, goals scored/conceded over last 5/10/20 matches, computed as of match date
- **FIFA ranking difference** — as of the month of each match
- **Neutral venue flag** — reduces home advantage ~3.5pp in World Cup context
- **Sample weights** — exponential decay favoring recent matches; friendlies penalized

### Unsupervised Learning
- **K-Means + DBSCAN** clustering of 48 WC2026 teams → cluster label as categorical feature
- **PCA** for multicollinearity detection and dimensionality reduction

### Supervised Models
| Model | Role |
|-------|------|
| Logistic Regression | Baseline |
| Random Forest | Ensemble component |
| XGBoost + SHAP | Primary model + interpretability |
| MLP (PyTorch) | Deep learning benchmark |
| Stacking Ensemble | Final predictor |

### Monte Carlo Simulation
- 10,000 full tournament simulations
- Calibrated probabilities per match from stacking ensemble
- Penalty shootout modeled explicitly for knockout rounds
- Output: probability distribution over final standings for all 48 teams

---

## 📈 EDA Key Findings

- Elo difference correlates 0.515 with goal difference — strongest single feature
- Home advantage drops 3.5pp in neutral World Cup venues
- Away win rate trending upward (28% → 33% over 2021–2026)
- Recent form (last 10 matches) shows 0.10 win rate differential between match winners and losers

---

## 🚀 Quickstart
`ash
# Clone and setup
git clone https://github.com/federico1809/world-cup-2026-predictor.git
cd world-cup-2026-predictor
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Download datasets (requires Kaggle API key)
python -m world_cup_2026.data_ingestion.download

# Run EDA notebook
jupyter lab notebooks/01_eda/01_eda_results.ipynb
`

---

## 🗂️ Development Status

| Phase | Status |
|-------|--------|
| Project scaffold | ✅ Complete |
| Data ingestion pipeline | ✅ Complete |
| Exploratory Data Analysis | ✅ Complete |
| Feature Engineering | 🔄 In progress |
| Unsupervised clustering | ⏳ Pending |
| Supervised modeling | ⏳ Pending |
| Monte Carlo simulation | ⏳ Pending |

---

## 👤 Author

**Federico Ceballos Torres**
Data Scientist — [GitHub](https://github.com/federico1809) · [LinkedIn](https://linkedin.com/in/tu-perfil)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
