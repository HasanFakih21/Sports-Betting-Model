# scripts/

The three pipeline scripts that form the end-to-end workflow.  Run them in
order from the project root (or from any directory — all paths are resolved
relative to the script file).

## `data_preprocessing_full.py` — Step 1: Data preparation

Merges raw season CSVs with FIFA team ratings and Understat xG stats, applies
canonical team-name normalisation, and writes the cleaned combined dataset.

```bash
python scripts/data_preprocessing_full.py
```

**Input:** `data/raw/*.csv`  
**Output:** `data/processed/combined_dataset_clean.csv`

---

## `model_training.py` — Step 2: Model training

Trains a deep neural network match-outcome classifier using an
expanding-window cross-validation scheme, then calibrates it with temperature
scaling and saves all artefacts.

```bash
python scripts/model_training.py
```

**Input:** `data/processed/combined_dataset_clean.csv`  
**Output:** `models/match_outcome_model.keras`, `models/scaler.pkl`,
`models/optimal_temperature.txt`

---

## `portfolio_optimization.py` — Step 3: Evaluation & simulation

Evaluates the trained model on a held-out test season, applies isotonic
regression calibration, plots a reliability diagram, and simulates a
restricted fractional Kelly betting strategy.

```bash
python scripts/portfolio_optimization.py [--season 2024-2025]
```

**Input:** `data/processed/combined_dataset_clean.csv`, `models/` artefacts  
**Output:** `results/figures/reliability_home.png`,
`results/logs/portfolio_simulation_history_restricted_calibrated.csv`,
`results/logs/sensitivity_analysis.csv`
