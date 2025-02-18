import os
import random
import numpy as np
import pandas as pd
import joblib
import datetime
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from tensorflow.keras import layers, models, callbacks, optimizers
import tensorflow as tf

# --------------------------
# Set Random Seeds for Reproducibility
# --------------------------
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# --------------------------
# Define File Paths based on Folder Structure
# --------------------------
PROCESSED_DATA_PATH = os.path.join("data", "processed")
MODELS_PATH = os.path.join("models")
os.makedirs(MODELS_PATH, exist_ok=True)

# --------------------------
# 1. Data Loading & Preprocessing
# --------------------------
data_path = os.path.join(PROCESSED_DATA_PATH, "combined_dataset_clean.csv")
data = pd.read_csv(data_path)
data = data.dropna(subset=["FTR"])

# Convert Date column to datetime
data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

# Drop the first week of each season (to avoid early-season instability)
def drop_first_week(df):
    min_date = df["Date"].min()
    return df[df["Date"] > (min_date + timedelta(days=7))]
data = data.groupby("Season", group_keys=False).apply(drop_first_week)

# --------------------------
# 2. Define Features & Target
# --------------------------
feature_cols = [
    "PSH", "PSD", "PSA",
    "HomeRating_Overall", "HomeRating_Attack", "HomeRating_Midfield", "HomeRating_Defence", "HomeRating_Players", "HomeRating_StartingXI_AvgAge",
    "AwayRating_Overall", "AwayRating_Attack", "AwayRating_Midfield", "AwayRating_Defence", "AwayRating_Players", "AwayRating_StartingXI_AvgAge",
    "HomeUnderstat_M", "HomeUnderstat_W", "HomeUnderstat_D", "HomeUnderstat_L", "HomeUnderstat_G", "HomeUnderstat_GA", "HomeUnderstat_PTS", "HomeUnderstat_xG", "HomeUnderstat_xGA", "HomeUnderstat_xPTS",
    "AwayUnderstat_M", "AwayUnderstat_W", "AwayUnderstat_D", "AwayUnderstat_L", "AwayUnderstat_G", "AwayUnderstat_GA", "AwayUnderstat_PTS", "AwayUnderstat_xG", "AwayUnderstat_xGA", "AwayUnderstat_xPTS"
]
X_full = data[feature_cols].copy()
y_full = pd.get_dummies(data["FTR"])

# --------------------------
# 3. Scale Features
# --------------------------
scaler = StandardScaler()
X_full_scaled = scaler.fit_transform(X_full)
scaler_path = os.path.join(MODELS_PATH, "scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to {scaler_path}")

# --------------------------
# 4. Feature Selection using L1 Logistic Regression (for informational purposes)
# --------------------------
lr_model = LogisticRegression(penalty="l1", solver="saga", multi_class="multinomial", max_iter=5000, random_state=seed_value)
lr_model.fit(X_full_scaled, y_full.values.argmax(axis=1))
nonzero_mask = np.any(lr_model.coef_ != 0, axis=0)
selected_features = np.array(feature_cols)[nonzero_mask].tolist()
print("Selected features from L1 logistic regression:")
print(selected_features)
# (We'll continue using the full set.)

# --------------------------
# 5. Expanding-Window (Rolling) Cross-Validation (STCV)
# --------------------------
data_sorted = data.sort_values("Date").reset_index(drop=True)
N = len(data_sorted)
fold_size = int(0.1 * N)  # 10% of data per fold
cv_losses = []
cv_accuracies = []
print("Rolling Window CV Results:")
start = 0
fold = 1
while start + fold_size < N:
    train_cv = data_sorted.iloc[:start + fold_size]
    val_cv = data_sorted.iloc[start + fold_size : start + 2 * fold_size]
    
    if len(val_cv) < 10:
        break

    X_train_cv = train_cv[feature_cols]
    y_train_cv = pd.get_dummies(train_cv["FTR"])
    X_val_cv = val_cv[feature_cols]
    y_val_cv = pd.get_dummies(val_cv["FTR"])
    
    X_train_cv_scaled = scaler.transform(X_train_cv)
    X_val_cv_scaled = scaler.transform(X_val_cv)
    
    cv_model = models.Sequential([
        layers.Input(shape=(X_train_cv_scaled.shape[1],)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(3, activation="softmax")
    ])
    cv_model.compile(optimizer=optimizers.Adam(learning_rate=0.0005),
                     loss="categorical_crossentropy", metrics=["accuracy"])
    
    cv_model.fit(X_train_cv_scaled, y_train_cv, 
                 validation_data=(X_val_cv_scaled, y_val_cv),
                 epochs=20, batch_size=32, verbose=0,
                 callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)])
    
    loss, acc = cv_model.evaluate(X_val_cv_scaled, y_val_cv, verbose=0)
    cv_losses.append(loss)
    cv_accuracies.append(acc)
    print(f"Fold {fold}: Val Loss: {loss:.4f}, Val Accuracy: {acc:.4f}")
    
    start += fold_size
    fold += 1

print(f"Average CV Loss: {np.mean(cv_losses):.4f}")
print(f"Average CV Accuracy: {np.mean(cv_accuracies):.4f}")

# --------------------------
# 6. Final Model Training on All Data (or on most recent data)
# --------------------------
final_model = models.Sequential([
    layers.Input(shape=(X_full_scaled.shape[1],)),
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
    layers.Dense(3, activation="softmax")
])
optimizer = optimizers.Adam(learning_rate=0.0005)
final_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

final_history = final_model.fit(
    X_full_scaled, y_full,
    validation_split=0.1,
    epochs=100,
    batch_size=32,
    callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
               callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1)],
    verbose=1
)

# --------------------------
# 7. Force Model Build for Calibration
# --------------------------
dummy = np.zeros((1, X_full_scaled.shape[1]))
_ = final_model.predict(dummy)

# --------------------------
# 8. Temperature Scaling Calibration
# --------------------------
# Use final 10% of sorted data as calibration set.
cal_val = data_sorted.iloc[int(0.9 * N):]
X_cal_val = cal_val[feature_cols]
y_cal_val = pd.get_dummies(cal_val["FTR"])
X_cal_val_scaled = scaler.transform(X_cal_val)

# Build penultimate model from the final model.
input_tensor = final_model.layers[0].input
penultimate_output = final_model.layers[-2].output
# Note: Use models.Model from tensorflow.keras.models
penultimate_model = models.Model(inputs=input_tensor, outputs=penultimate_output)
W, b = final_model.layers[-1].get_weights()

def get_logits(X):
    Z = penultimate_model.predict(X)
    return np.dot(Z, W) + b

def temperature_scale(logits, T):
    exp_logits = np.exp(logits / T)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

logits_val = get_logits(X_cal_val_scaled)
Ts = np.linspace(0.5, 5.0, 50)
best_T = None
best_nll = np.inf
for T in Ts:
    calibrated = temperature_scale(logits_val, T)
    nll = log_loss(y_cal_val.values, calibrated)
    if nll < best_nll:
        best_nll = nll
        best_T = T
print(f"Optimal temperature: {best_T:.4f}, Calibration NLL: {best_nll:.4f}")

final_loss, final_acc = final_model.evaluate(X_full_scaled, y_full, verbose=0)
print(f"Final Model Loss (all data): {final_loss:.4f}, Final Model Accuracy (all data): {final_acc:.4f}")

# --------------------------
# 9. Save Final Artifacts
# --------------------------
model_path = os.path.join(MODELS_PATH, "match_outcome_model.h5")
final_model.save(model_path)
joblib.dump(scaler, os.path.join(MODELS_PATH, "scaler.pkl"))
with open(os.path.join(MODELS_PATH, "optimal_temperature.txt"), "w") as f:
    f.write(f"{best_T:.4f}")
print(f"Final model and artifacts saved to {MODELS_PATH}")

if __name__ == "__main__":
    pass  # This script runs on import.
