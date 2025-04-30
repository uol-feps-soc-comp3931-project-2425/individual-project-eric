import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

# --- Configuration ---
INPUT_FILE = "fullFeaturesCryptoDataUnscaled.csv" 
OUTPUT_DIR = ""
SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler.joblib")
FEATURE_LIST_PATH = os.path.join(OUTPUT_DIR, "featureColumns.npy")
SEQUENCE_DATA_PATH = os.path.join(OUTPUT_DIR, "sequenceData.npz")

SEQUENCE_LENGTH = 15 
PREDICTION_HORIZON = 5
TRAIN_SPLIT_RATIO = 0.8
TARGET_ASSET_PREFIX = "btc" 

print(f"Loading data with indicators from {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE, index_col="open_time", parse_dates=True)
print(f"Loaded data shape: {df.shape}")

df["target"] = df[f"{TARGET_ASSET_PREFIX}_close"].shift(-PREDICTION_HORIZON)


exclude_cols = [col for col in df.columns if any(x in col for x in ["_open", "_high", "_low", "_volume"]) and not col.endswith("_OBV")]
exclude_cols.append("target") 
exclude_cols.extend([f"{asset}_close" for asset in ["btc", "eth", "sol"]])

feature_columns = [col for col in df.columns if col not in exclude_cols]
print(f"Selected {len(feature_columns)} features:")
# print(feature_columns) 

# Drop rows with NaNs created by shifting or indicator calculation
df.dropna(inplace=True)
print(f"Data shape after dropping NaNs: {df.shape}")

# Separate features (X) and target (y)
X = df[feature_columns].astype(np.float32)
y = df["target"].values.astype(np.float32)

# Split data into training and testing sets
split_index = int(len(df) * TRAIN_SPLIT_RATIO)
X_train_raw, X_test_raw = X[:split_index], X[split_index:]
y_train_raw, y_test_raw = y[:split_index], y[split_index:]

print(f"Training set size: {len(X_train_raw)}")
print(f"Test set size: {len(X_test_raw)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# Save the scaler and feature list
joblib.dump(scaler, SCALER_PATH)
np.save(FEATURE_LIST_PATH, np.array(feature_columns))
print(f"Scaler saved to {SCALER_PATH}")
print(f"Feature list saved to {FEATURE_LIST_PATH}")

# Create sequences
def create_sequences(X, y, seq_len, horizon):
    Xs, ys = [], []
    for i in range(len(X) - seq_len - horizon + 1):
        Xs.append(X[i:(i + seq_len)])
        ys.append(y[i + seq_len + horizon - 1]) 
    return np.array(Xs), np.array(ys)

print(f"Creating sequences with length {SEQUENCE_LENGTH}...")
X_train, y_train = create_sequences(X_train_scaled, y_train_raw, SEQUENCE_LENGTH, PREDICTION_HORIZON)
X_test, y_test = create_sequences(X_test_scaled, y_test_raw, SEQUENCE_LENGTH, PREDICTION_HORIZON)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

np.savez_compressed(SEQUENCE_DATA_PATH, 
                    X_train=X_train, y_train=y_train, 
                    X_test=X_test, y_test=y_test)
print(f"Sequence data saved to {SEQUENCE_DATA_PATH}")

print("Sequence data preparation complete.")

