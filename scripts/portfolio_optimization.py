import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Model
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss
import tensorflow as tf
import random

# --------------------------
# Set Random Seeds for Reproducibility
# --------------------------
seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

# --------------------------
# Define File Paths (based on new structure)
# --------------------------
PROCESSED_DATA_PATH = os.path.join("data", "processed")
MODELS_PATH = os.path.join("models")
RESULTS_FIGURES_PATH = os.path.join("results", "figures")
RESULTS_LOGS_PATH = os.path.join("results", "logs")

# Ensure output directories exist
os.makedirs(RESULTS_FIGURES_PATH, exist_ok=True)
os.makedirs(RESULTS_LOGS_PATH, exist_ok=True)

# --------------------------
# 1. Load Test Data & Artifacts
# --------------------------
data_file = os.path.join(PROCESSED_DATA_PATH, "combined_dataset_clean.csv")
data = pd.read_csv(data_file)
# Select test data for season "2024-2025"
test_data = data[data["Season"] == "2024-2025"].dropna(subset=["FTR"]).copy()

feature_cols = [
    "PSH", "PSD", "PSA",
    "HomeRating_Overall", "HomeRating_Attack", "HomeRating_Midfield", "HomeRating_Defence", "HomeRating_Players", "HomeRating_StartingXI_AvgAge",
    "AwayRating_Overall", "AwayRating_Attack", "AwayRating_Midfield", "AwayRating_Defence", "AwayRating_Players", "AwayRating_StartingXI_AvgAge",
    "HomeUnderstat_M", "HomeUnderstat_W", "HomeUnderstat_D", "HomeUnderstat_L", "HomeUnderstat_G", "HomeUnderstat_GA", "HomeUnderstat_PTS", "HomeUnderstat_xG", "HomeUnderstat_xGA", "HomeUnderstat_xPTS",
    "AwayUnderstat_M", "AwayUnderstat_W", "AwayUnderstat_D", "AwayUnderstat_L", "AwayUnderstat_G", "AwayUnderstat_GA", "AwayUnderstat_PTS", "AwayUnderstat_xG", "AwayUnderstat_xGA", "AwayUnderstat_xPTS"
]
X_test = test_data[feature_cols].copy()
true_results = test_data["FTR"].values
outcome_labels = np.array(["H", "D", "A"])

scaler_path = os.path.join(MODELS_PATH, "scaler.pkl")
scaler = joblib.load(scaler_path)
X_test_scaled = scaler.transform(X_test)

model_path = os.path.join(MODELS_PATH, "match_outcome_model.h5")
model = load_model(model_path)

optimal_temp_path = os.path.join(MODELS_PATH, "optimal_temperature.txt")
with open(optimal_temp_path, "r") as f:
    optimal_temperature = float(f.read().strip())
print(f"Optimal temperature loaded: {optimal_temperature:.4f}")

# --------------------------
# 2. Obtain Ensemble Predictions via MC Dropout and Calibration Split
# --------------------------
n_iter = 20
mc_preds = []
for i in range(n_iter):
    # Use dropout during inference by calling the model with training=True
    preds = model(X_test_scaled, training=True).numpy()
    mc_preds.append(preds)
mc_preds = np.array(mc_preds)
ensemble_preds = np.mean(mc_preds, axis=0)  # Raw ensemble predictions

# Split test data into calibration and evaluation sets (30% calibration)
calib_frac = 0.3
calib_size = int(calib_frac * len(test_data))
all_indices = np.arange(len(test_data))
calib_indices = np.random.choice(all_indices, size=calib_size, replace=False)
eval_indices = np.setdiff1d(all_indices, calib_indices)

calib_preds = ensemble_preds[calib_indices]
eval_preds = ensemble_preds[eval_indices]
calib_true = test_data.iloc[calib_indices]["FTR"].values
eval_true = test_data.iloc[eval_indices]["FTR"].values

# --------------------------
# 3. Calibrate Predictions Using Isotonic Regression (One-vs-All)
# --------------------------
calibrators = {}
for i, label in enumerate(outcome_labels):
    y_binary = (calib_true == label).astype(int)
    p_pred = calib_preds[:, i]
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(p_pred, y_binary)
    calibrators[label] = iso_reg

def calibrate_multiclass(preds):
    calibrated = np.zeros_like(preds)
    for i, label in enumerate(outcome_labels):
        calibrated[:, i] = calibrators[label].predict(preds[:, i])
    calibrated /= calibrated.sum(axis=1, keepdims=True)
    return calibrated

calibrated_eval_preds = calibrate_multiclass(eval_preds)

# --------------------------
# 4. Evaluate Calibration: Brier Scores and Reliability Diagram for Home Wins
# --------------------------
def multiclass_brier(y_true, y_prob):
    brier_scores = {}
    for i, label in enumerate(outcome_labels):
        y_bin = (y_true == label).astype(int)
        brier_scores[label] = brier_score_loss(y_bin, y_prob[:, i])
    return brier_scores

raw_brier = multiclass_brier(eval_true, eval_preds)
cal_brier = multiclass_brier(eval_true, calibrated_eval_preds)
print("Raw Brier scores:", raw_brier)
print("Calibrated Brier scores:", cal_brier)

def reliability_diagram(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    accuracy = []
    confidence = []
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if np.sum(mask) > 0:
            accuracy.append(np.mean(y_true[mask]))
            confidence.append(np.mean(y_prob[mask]))
        else:
            accuracy.append(np.nan)
            confidence.append(np.nan)
    return bin_centers, np.array(confidence), np.array(accuracy)

y_true_home = (eval_true == "H").astype(int)
y_prob_home_raw = eval_preds[:, 0]
y_prob_home_cal = calibrated_eval_preds[:, 0]

bins, conf_raw, acc_raw = reliability_diagram(y_true_home, y_prob_home_raw)
_, conf_cal, acc_cal = reliability_diagram(y_true_home, y_prob_home_cal)

plt.figure(figsize=(8,6))
plt.plot(bins, conf_raw, 'o-', label="Raw Predicted")
plt.plot(bins, acc_raw, 'o-', label="Raw Accuracy")
plt.plot(bins, conf_cal, 's-', label="Calibrated Predicted")
plt.plot(bins, acc_cal, 's-', label="Calibrated Accuracy")
plt.plot([0,1], [0,1], 'k--', label="Perfect Calibration")
plt.xlabel("Predicted Probability")
plt.ylabel("Fraction of Positives")
plt.title("Reliability Diagram for Home Wins (Evaluation)")
plt.legend()
reliability_plot_path = os.path.join(RESULTS_FIGURES_PATH, "reliability_home.png")
plt.savefig(reliability_plot_path)
plt.close()
print(f"Reliability diagram for Home wins saved as {reliability_plot_path}")

# --------------------------
# 5. Restricted Fractional Kelly Simulation
# --------------------------
# For the evaluation set, assume bookmaker odds are in columns "PSH", "PSD", "PSA"
odds_eval = test_data.iloc[eval_indices][["PSH", "PSD", "PSA"]].astype(float).values

# Define a default fractional multiplier for the restricted strategy
default_fractional_multiplier = 0.25

def simulate_restricted_portfolio(preds, odds, true_labels, frac_mult):
    def kelly_fraction(p, o):
        if o <= 1:
            return 0.0
        base = (p - (1.0 / o)) / (o - 1.0)
        return frac_mult * base if base > 0 else 0.0

    wealth = 1.0
    wealth_series = []
    history = []
    for i in range(len(preds)):
        current_odds = odds[i]
        p_preds = preds[i]
        kellys = np.array([kelly_fraction(p_preds[j], current_odds[j]) for j in range(3)])
        if kellys.max() > 0:
            bet_idx = np.argmax(kellys)
            bet_fraction = kellys[bet_idx]
            chosen_outcome = outcome_labels[bet_idx]
        else:
            bet_fraction = 0.0
            chosen_outcome = None

        if bet_fraction > 0:
            if chosen_outcome == true_labels[i]:
                wealth *= (1 - bet_fraction + bet_fraction * current_odds[bet_idx])
            else:
                wealth *= (1 - bet_fraction)
        wealth_series.append(wealth)
        history.append({
            "match_index": i,
            "bet_outcome": chosen_outcome,
            "true_outcome": true_labels[i],
            "bet_fraction": bet_fraction,
            "odds": current_odds.tolist() if bet_fraction > 0 else None,
            "wealth": wealth
        })
    log_growth = np.log(wealth / 1.0)
    cum_max = np.maximum.accumulate(np.array(wealth_series))
    drawdowns = (np.array(wealth_series) - cum_max) / cum_max
    max_dd = np.min(drawdowns)
    vol = np.std(np.diff(np.log(wealth_series))) if len(wealth_series) > 1 else 0.0
    return wealth, log_growth, max_dd, vol, history

final_wealth, log_growth, max_dd, volatility, sim_history = simulate_restricted_portfolio(
    calibrated_eval_preds, odds_eval, eval_true, default_fractional_multiplier
)
print(f"Restricted Strategy (Calibrated, multiplier={default_fractional_multiplier}) Final Wealth: {final_wealth:.4f}")
print(f"Log Growth: {log_growth:.4f}")
print(f"Maximum Drawdown: {max_dd:.4f}")
print(f"Volatility of Log Returns: {volatility:.4f}")

simulation_log_path = os.path.join(RESULTS_LOGS_PATH, "portfolio_simulation_history_restricted_calibrated.csv")
pd.DataFrame(sim_history).to_csv(simulation_log_path, index=False)
print(f"Portfolio simulation history saved to {simulation_log_path}")

# --------------------------
# 6. Sensitivity Analysis: Varying Fractional Multipliers
# --------------------------
multipliers = [0.10, 0.15, 0.20, 0.25]
results = []
for fm in multipliers:
    fw, lg, mdd, volat, _ = simulate_restricted_portfolio(calibrated_eval_preds, odds_eval, eval_true, fm)
    results.append({
        "fractional_multiplier": fm,
        "final_wealth": fw,
        "log_growth": lg,
        "max_drawdown": mdd,
        "volatility": volat
    })
results_df = pd.DataFrame(results)
print("Sensitivity Analysis (Fractional Multiplier):")
print(results_df)
sensitivity_path = os.path.join(RESULTS_LOGS_PATH, "sensitivity_analysis.csv")
results_df.to_csv(sensitivity_path, index=False)
print(f"Sensitivity analysis saved to {sensitivity_path}")
