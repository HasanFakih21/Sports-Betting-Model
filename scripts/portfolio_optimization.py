"""
portfolio_optimization.py

Evaluates the trained match-outcome model on a held-out test season and
simulates a restricted fractional Kelly betting strategy.

Outputs (written to results/):
    figures/reliability_home.png                              – reliability diagram
    logs/portfolio_simulation_history_restricted_calibrated.csv
    logs/sensitivity_analysis.csv

Usage (run from the project root):
    python scripts/portfolio_optimization.py [--season 2024-2025]
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss
from tensorflow.keras.models import load_model

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = _PROJECT_ROOT / "data" / "processed"
MODELS_PATH = _PROJECT_ROOT / "models"
RESULTS_FIGURES_PATH = _PROJECT_ROOT / "results" / "figures"
RESULTS_LOGS_PATH = _PROJECT_ROOT / "results" / "logs"

OUTCOME_LABELS = np.array(["H", "D", "A"])

FEATURE_COLS = [
    "PSH", "PSD", "PSA",
    "HomeRating_Overall", "HomeRating_Attack", "HomeRating_Midfield",
    "HomeRating_Defence", "HomeRating_Players", "HomeRating_StartingXI_AvgAge",
    "AwayRating_Overall", "AwayRating_Attack", "AwayRating_Midfield",
    "AwayRating_Defence", "AwayRating_Players", "AwayRating_StartingXI_AvgAge",
    "HomeUnderstat_M", "HomeUnderstat_W", "HomeUnderstat_D", "HomeUnderstat_L",
    "HomeUnderstat_G", "HomeUnderstat_GA", "HomeUnderstat_PTS",
    "HomeUnderstat_xG", "HomeUnderstat_xGA", "HomeUnderstat_xPTS",
    "AwayUnderstat_M", "AwayUnderstat_W", "AwayUnderstat_D", "AwayUnderstat_L",
    "AwayUnderstat_G", "AwayUnderstat_GA", "AwayUnderstat_PTS",
    "AwayUnderstat_xG", "AwayUnderstat_xGA", "AwayUnderstat_xPTS",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def calibrate_multiclass(
    preds: np.ndarray,
    calibrators: dict[str, IsotonicRegression],
) -> np.ndarray:
    """Apply per-class isotonic calibrators and renormalise."""
    calibrated = np.zeros_like(preds)
    for i, label in enumerate(OUTCOME_LABELS):
        calibrated[:, i] = calibrators[label].predict(preds[:, i])
    row_sums = calibrated.sum(axis=1, keepdims=True)
    # Avoid division by zero for degenerate rows
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return calibrated / row_sums


def multiclass_brier(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    return {
        label: brier_score_loss((y_true == label).astype(int), y_prob[:, i])
        for i, label in enumerate(OUTCOME_LABELS)
    }


def reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    accuracy, confidence = [], []
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if np.sum(mask) > 0:
            accuracy.append(np.mean(y_true[mask]))
            confidence.append(np.mean(y_prob[mask]))
        else:
            accuracy.append(np.nan)
            confidence.append(np.nan)
    return bin_centers, np.array(confidence), np.array(accuracy)


def simulate_restricted_portfolio(
    preds: np.ndarray,
    odds: np.ndarray,
    true_labels: np.ndarray,
    frac_mult: float,
) -> tuple[float, float, float, float, list[dict]]:
    """Simulate the restricted fractional Kelly strategy.

    Returns:
        final_wealth, log_growth, max_drawdown, volatility, history
    """
    def kelly_fraction(p: float, o: float) -> float:
        if o <= 1:
            return 0.0
        base = (p - (1.0 / o)) / (o - 1.0)
        return frac_mult * base if base > 0 else 0.0

    wealth = 1.0
    wealth_series: list[float] = []
    history: list[dict] = []

    for i in range(len(preds)):
        current_odds = odds[i]
        p_preds = preds[i]
        kellys = np.array([kelly_fraction(p_preds[j], current_odds[j]) for j in range(3)])

        if kellys.max() > 0:
            bet_idx = int(np.argmax(kellys))
            bet_fraction = float(kellys[bet_idx])
            chosen_outcome = OUTCOME_LABELS[bet_idx]
        else:
            bet_fraction = 0.0
            chosen_outcome = None

        if bet_fraction > 0:
            if chosen_outcome == true_labels[i]:
                wealth *= 1 - bet_fraction + bet_fraction * current_odds[bet_idx]
            else:
                wealth *= 1 - bet_fraction

        wealth_series.append(wealth)
        history.append({
            "match_index": i,
            "bet_outcome": chosen_outcome,
            "true_outcome": true_labels[i],
            "bet_fraction": bet_fraction,
            "odds": current_odds.tolist() if bet_fraction > 0 else None,
            "wealth": wealth,
        })

    log_growth = float(np.log(wealth))
    ws = np.array(wealth_series)
    cum_max = np.maximum.accumulate(ws)
    drawdowns = (ws - cum_max) / cum_max
    max_dd = float(np.min(drawdowns))
    vol = float(np.std(np.diff(np.log(ws)))) if len(ws) > 1 else 0.0
    return wealth, log_growth, max_dd, vol, history


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------

def main(test_season: str = "2024-2025") -> None:
    set_seeds(42)
    RESULTS_FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    RESULTS_LOGS_PATH.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load test data & artefacts
    # ------------------------------------------------------------------
    data = pd.read_csv(PROCESSED_DATA_PATH / "combined_dataset_clean.csv")
    test_data = data[data["Season"] == test_season].dropna(subset=["FTR"]).copy()
    if test_data.empty:
        raise ValueError(
            f"No data found for season '{test_season}'. "
            f"Available seasons: {sorted(data['Season'].unique())}"
        )

    X_test = test_data[FEATURE_COLS].copy()
    true_results = test_data["FTR"].values

    scaler = joblib.load(MODELS_PATH / "scaler.pkl")
    X_test_scaled = scaler.transform(X_test)

    # Support both .keras (new) and .h5 (legacy) model files
    keras_path = MODELS_PATH / "match_outcome_model.keras"
    h5_path = MODELS_PATH / "match_outcome_model.h5"
    model_path = keras_path if keras_path.exists() else h5_path
    model = load_model(model_path)

    optimal_temperature = float(
        (MODELS_PATH / "optimal_temperature.txt").read_text().strip()
    )
    print(f"Optimal temperature loaded: {optimal_temperature:.4f}")

    # ------------------------------------------------------------------
    # 2. MC Dropout ensemble predictions
    # ------------------------------------------------------------------
    n_iter = 20
    mc_preds = np.array([
        model(X_test_scaled, training=True).numpy()
        for _ in range(n_iter)
    ])
    ensemble_preds = mc_preds.mean(axis=0)

    # Split into calibration (30%) and evaluation (70%) subsets
    calib_size = int(0.3 * len(test_data))
    all_indices = np.arange(len(test_data))
    calib_indices = np.random.choice(all_indices, size=calib_size, replace=False)
    eval_indices = np.setdiff1d(all_indices, calib_indices)

    calib_preds = ensemble_preds[calib_indices]
    eval_preds = ensemble_preds[eval_indices]
    calib_true = true_results[calib_indices]
    eval_true = true_results[eval_indices]

    # ------------------------------------------------------------------
    # 3. Isotonic regression calibration (one-vs-all)
    # ------------------------------------------------------------------
    calibrators: dict[str, IsotonicRegression] = {}
    for i, label in enumerate(OUTCOME_LABELS):
        y_binary = (calib_true == label).astype(int)
        iso_reg = IsotonicRegression(out_of_bounds="clip")
        iso_reg.fit(calib_preds[:, i], y_binary)
        calibrators[label] = iso_reg

    calibrated_eval_preds = calibrate_multiclass(eval_preds, calibrators)

    # ------------------------------------------------------------------
    # 4. Calibration evaluation & reliability diagram
    # ------------------------------------------------------------------
    raw_brier = multiclass_brier(eval_true, eval_preds)
    cal_brier = multiclass_brier(eval_true, calibrated_eval_preds)
    print("Raw Brier scores:", raw_brier)
    print("Calibrated Brier scores:", cal_brier)

    y_true_home = (eval_true == "H").astype(int)
    bins, conf_raw, acc_raw = reliability_diagram(y_true_home, eval_preds[:, 0])
    _, conf_cal, acc_cal = reliability_diagram(y_true_home, calibrated_eval_preds[:, 0])

    plt.figure(figsize=(8, 6))
    plt.plot(bins, conf_raw, "o-", label="Raw Predicted")
    plt.plot(bins, acc_raw, "o-", label="Raw Accuracy")
    plt.plot(bins, conf_cal, "s-", label="Calibrated Predicted")
    plt.plot(bins, acc_cal, "s-", label="Calibrated Accuracy")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(f"Reliability Diagram – Home Wins ({test_season})")
    plt.legend()
    reliability_plot_path = RESULTS_FIGURES_PATH / "reliability_home.png"
    plt.savefig(reliability_plot_path)
    plt.close()
    print(f"Reliability diagram saved to {reliability_plot_path}")

    # ------------------------------------------------------------------
    # 5. Restricted fractional Kelly simulation
    # ------------------------------------------------------------------
    odds_eval = test_data.iloc[eval_indices][["PSH", "PSD", "PSA"]].astype(float).values
    default_frac = 0.25

    final_wealth, log_growth, max_dd, volatility, sim_history = simulate_restricted_portfolio(
        calibrated_eval_preds, odds_eval, eval_true, default_frac
    )
    print(f"\nRestricted strategy (calibrated, multiplier={default_frac})")
    print(f"  Final wealth : {final_wealth:.4f}")
    print(f"  Log growth   : {log_growth:.4f}")
    print(f"  Max drawdown : {max_dd:.4f}")
    print(f"  Volatility   : {volatility:.4f}")

    sim_log_path = RESULTS_LOGS_PATH / "portfolio_simulation_history_restricted_calibrated.csv"
    pd.DataFrame(sim_history).to_csv(sim_log_path, index=False)
    print(f"Simulation history saved to {sim_log_path}")

    # ------------------------------------------------------------------
    # 6. Sensitivity analysis – varying fractional multipliers
    # ------------------------------------------------------------------
    multipliers = [0.10, 0.15, 0.20, 0.25]
    sens_rows = []
    for fm in multipliers:
        fw, lg, mdd, volat, _ = simulate_restricted_portfolio(
            calibrated_eval_preds, odds_eval, eval_true, fm
        )
        sens_rows.append({
            "fractional_multiplier": fm,
            "final_wealth": fw,
            "log_growth": lg,
            "max_drawdown": mdd,
            "volatility": volat,
        })
    sens_df = pd.DataFrame(sens_rows)
    print("\nSensitivity analysis (fractional multiplier):")
    print(sens_df.to_string(index=False))
    sensitivity_path = RESULTS_LOGS_PATH / "sensitivity_analysis.csv"
    sens_df.to_csv(sensitivity_path, index=False)
    print(f"Sensitivity analysis saved to {sensitivity_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run portfolio optimisation for a given season.")
    parser.add_argument(
        "--season",
        default="2024-2025",
        help="Season label used as the test set (default: 2024-2025)",
    )
    args = parser.parse_args()
    main(test_season=args.season)
