"""
model_training.py

Trains a deep-learning match-outcome classifier on the processed Premier League
dataset and saves the artefacts needed for inference and calibration:

    models/match_outcome_model.keras   – trained Keras model
    models/scaler.pkl                  – fitted StandardScaler
    models/optimal_temperature.txt     – best temperature for post-hoc scaling

Usage (run from the project root):
    python scripts/model_training.py
"""

from __future__ import annotations

import os
import random
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import callbacks, layers, models, optimizers

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = _PROJECT_ROOT / "data" / "processed"
MODELS_PATH = _PROJECT_ROOT / "models"

# Canonical label order: Home win, Draw, Away win
OUTCOME_LABELS = ["H", "D", "A"]

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
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def encode_labels(series: pd.Series) -> np.ndarray:
    """One-hot encode FTR column using the canonical OUTCOME_LABELS order."""
    return pd.get_dummies(
        pd.Categorical(series, categories=OUTCOME_LABELS)
    ).values.astype(np.float32)


def drop_first_week(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the first 7 days of each season (early-season instability)."""
    return df[df["Date"] > (df["Date"].min() + timedelta(days=7))]


def temperature_scale(logits: np.ndarray, T: float) -> np.ndarray:
    exp_logits = np.exp(logits / T)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def build_cv_model(n_features: int) -> models.Sequential:
    return models.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(3, activation="softmax"),
    ])


def build_final_model(n_features: int) -> models.Sequential:
    return models.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(3, activation="softmax"),
    ])


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    set_seeds(42)
    MODELS_PATH.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load & preprocess data
    # ------------------------------------------------------------------
    data_path = PROCESSED_DATA_PATH / "combined_dataset_clean.csv"
    data = pd.read_csv(data_path)
    data = data.dropna(subset=["FTR"])
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.groupby("Season", group_keys=False).apply(drop_first_week)

    X_full = data[FEATURE_COLS].copy()
    y_full = encode_labels(data["FTR"])

    # ------------------------------------------------------------------
    # 2. Scale features
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_full_scaled = scaler.fit_transform(X_full)
    scaler_path = MODELS_PATH / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    # ------------------------------------------------------------------
    # 3. L1 logistic regression for feature-importance insight
    # ------------------------------------------------------------------
    lr_model = LogisticRegression(
        penalty="l1", solver="saga", max_iter=5000, random_state=42
    )
    lr_model.fit(X_full_scaled, y_full.argmax(axis=1))
    nonzero_mask = np.any(lr_model.coef_ != 0, axis=0)
    selected_features = np.array(FEATURE_COLS)[nonzero_mask].tolist()
    print("Features selected by L1 logistic regression:")
    print(selected_features)

    # ------------------------------------------------------------------
    # 4. Expanding-window (rolling) cross-validation
    # ------------------------------------------------------------------
    data_sorted = data.sort_values("Date").reset_index(drop=True)
    N = len(data_sorted)
    fold_size = int(0.1 * N)
    cv_losses, cv_accuracies = [], []
    print("\nRolling-window CV results:")
    start, fold = 0, 1
    while start + fold_size < N:
        train_cv = data_sorted.iloc[: start + fold_size]
        val_cv = data_sorted.iloc[start + fold_size : start + 2 * fold_size]
        if len(val_cv) < 10:
            break

        X_train_cv = scaler.transform(train_cv[FEATURE_COLS])
        y_train_cv = encode_labels(train_cv["FTR"])
        X_val_cv = scaler.transform(val_cv[FEATURE_COLS])
        y_val_cv = encode_labels(val_cv["FTR"])

        cv_model = build_cv_model(X_train_cv.shape[1])
        cv_model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0005),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        cv_model.fit(
            X_train_cv, y_train_cv,
            validation_data=(X_val_cv, y_val_cv),
            epochs=20, batch_size=32, verbose=0,
            callbacks=[callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            )],
        )
        loss, acc = cv_model.evaluate(X_val_cv, y_val_cv, verbose=0)
        cv_losses.append(loss)
        cv_accuracies.append(acc)
        print(f"  Fold {fold}: val_loss={loss:.4f}, val_acc={acc:.4f}")
        start += fold_size
        fold += 1

    print(f"Average CV loss: {np.mean(cv_losses):.4f}")
    print(f"Average CV accuracy: {np.mean(cv_accuracies):.4f}\n")

    # ------------------------------------------------------------------
    # 5. Final model – train on all data
    # ------------------------------------------------------------------
    final_model = build_final_model(X_full_scaled.shape[1])
    final_model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0005),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    final_model.fit(
        X_full_scaled, y_full,
        validation_split=0.1,
        epochs=100,
        batch_size=32,
        callbacks=[
            callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
            ),
        ],
        verbose=1,
    )

    # ------------------------------------------------------------------
    # 6. Temperature scaling calibration
    # ------------------------------------------------------------------
    # Use the last 10% of chronologically sorted data as calibration set
    cal_val = data_sorted.iloc[int(0.9 * N):]
    X_cal_scaled = scaler.transform(cal_val[FEATURE_COLS])
    y_cal = encode_labels(cal_val["FTR"])

    # Extract penultimate-layer output to compute raw logits
    penultimate_model = models.Model(
        inputs=final_model.input,
        outputs=final_model.layers[-2].output,
    )
    W, b = final_model.layers[-1].get_weights()

    def get_logits(X: np.ndarray) -> np.ndarray:
        Z = penultimate_model.predict(X, verbose=0)
        return np.dot(Z, W) + b

    logits_val = get_logits(X_cal_scaled)
    best_T, best_nll = 1.0, np.inf
    for T in np.linspace(0.5, 5.0, 50):
        calibrated = temperature_scale(logits_val, T)
        nll = log_loss(y_cal, calibrated)
        if nll < best_nll:
            best_nll, best_T = nll, T
    print(f"Optimal temperature: {best_T:.4f}  (calibration NLL: {best_nll:.4f})")

    final_loss, final_acc = final_model.evaluate(X_full_scaled, y_full, verbose=0)
    print(f"Final model – loss: {final_loss:.4f}, accuracy: {final_acc:.4f}")

    # ------------------------------------------------------------------
    # 7. Save artefacts
    # ------------------------------------------------------------------
    model_path = MODELS_PATH / "match_outcome_model.keras"
    final_model.save(model_path)
    joblib.dump(scaler, MODELS_PATH / "scaler.pkl")
    (MODELS_PATH / "optimal_temperature.txt").write_text(f"{best_T:.4f}\n")
    print(f"Artefacts saved to {MODELS_PATH}")


if __name__ == "__main__":
    main()
