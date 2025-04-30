import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List

# Assume df_final is loaded or available here
if 'df_final' not in locals() or not isinstance(df_final, pd.DataFrame) or df_final.empty:
    raise ValueError("Multivariate DataFrame 'df_final' not found, not a DataFrame, or empty. Please run the prerequisite steps.")

ASSETS: List[str] = ['BTC', 'ETH', 'SOL']
SEQUENCE_LENGTH: int = 60
TRAIN_SPLIT_PCT: float = 0.70
VALIDATION_SPLIT_PCT: float = 0.15

TARGET_COLS: List[str] = [f'log_return_capped_{asset}' for asset in ASSETS]
if not all(col in df_final.columns for col in TARGET_COLS):
    raise ValueError(f"Error: One or more target columns not found in df_final: {TARGET_COLS}")

FEATURE_COLS: List[str] = df_final.columns.tolist()
N_FEATURES: int = len(FEATURE_COLS)
print(f"Using {N_FEATURES} features: {FEATURE_COLS}")
N_TARGETS: int = len(TARGET_COLS)

def create_multivariate_sequences(
    data_features: pd.DataFrame,
    data_targets: pd.DataFrame,
    sequence_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates sequences and corresponding multivariate targets from aligned data.

    Args:
        data_features (pd.DataFrame): DataFrame containing all features (already aligned).
                                        Shape: (n_aligned_samples, n_features).
        data_targets (pd.DataFrame): DataFrame containing target columns (already aligned).
                                      Shape: (n_aligned_samples, n_targets).
        sequence_length (int): The number of time steps for each input sequence.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Input sequences (X) with shape (n_sequences, sequence_length, n_features).
            - np.ndarray: Target values (y) with shape (n_sequences, n_targets).
    """
    X, y = [], []
    feature_values = data_features.values
    target_values = data_targets.values
    n_aligned_samples = len(feature_values)

    if n_aligned_samples < sequence_length:
        print(f"Warning: Not enough aligned data ({n_aligned_samples}) to create sequences of length {sequence_length}.")
        n_features = feature_values.shape[1]
        n_targets = target_values.shape[1]
        return np.empty((0, sequence_length, n_features)), np.empty((0, n_targets))

    print(f"Creating sequences from aligned feature data shape: {feature_values.shape}")

    for i in range(n_aligned_samples - sequence_length):
        sequence_end_index = i + sequence_length
        target_index = sequence_end_index

        X.append(feature_values[i:sequence_end_index, :])
        y.append(target_values[target_index, :])

    return np.array(X), np.array(y)

print("Aligning features (t) and targets (t+1)...")
X_data = df_final[FEATURE_COLS].copy()
y_shifted = df_final[TARGET_COLS].shift(-1)

combined_data = pd.concat([X_data, y_shifted], axis=1)
shifted_target_cols = [f"{col}_target_shifted" for col in TARGET_COLS]
combined_data.columns = FEATURE_COLS + shifted_target_cols
initial_len = len(combined_data)
combined_data.dropna(inplace=True)
final_len = len(combined_data)
print(f"Dropped {initial_len - final_len} rows due to NaN after target shift.")

if final_len == 0:
    raise ValueError("Error: No data remaining after aligning features and targets.")

X_data_aligned = combined_data[FEATURE_COLS]
y_data_aligned = combined_data[shifted_target_cols]

print(f"Aligned feature data shape: {X_data_aligned.shape}")
print(f"Aligned target data shape: {y_data_aligned.shape}")

X, y = create_multivariate_sequences(X_data_aligned, y_data_aligned, SEQUENCE_LENGTH)

if X.size == 0 or y.size == 0:
    raise RuntimeError("Error: No sequences were created. Check data length and sequence length.")

print(f"Created multivariate sequences: X shape {X.shape}, y shape {y.shape}")
if y.shape[1] != N_TARGETS:
     print(f"Warning: y shape is {y.shape}, but expected {N_TARGETS} target columns.")
if X.shape[2] != N_FEATURES:
     print(f"Warning: X shape is {X.shape}, but expected {N_FEATURES} feature columns.")


total_samples = X.shape[0]
if total_samples == 0:
    raise ValueError("Cannot split data, no samples available.")

train_end_index = int(total_samples * TRAIN_SPLIT_PCT)
validation_end_index = train_end_index + int(total_samples * VALIDATION_SPLIT_PCT)
validation_end_index = min(validation_end_index, total_samples)
if train_end_index >= validation_end_index and VALIDATION_SPLIT_PCT > 0:
     print(f"Warning: Training split ({TRAIN_SPLIT_PCT}) and validation split ({VALIDATION_SPLIT_PCT}) result in overlapping or empty validation set. Adjust percentages.")
     validation_end_index = train_end_index

X_train, y_train = X[:train_end_index], y[:train_end_index]
X_val, y_val = X[train_end_index:validation_end_index], y[train_end_index:validation_end_index]
X_test, y_test = X[validation_end_index:], y[validation_end_index:]

print(f"Total samples: {total_samples}")
print(f"Train split: {len(X_train)} samples (Index 0 to {train_end_index-1})")
print(f"Validation split: {len(X_val)} samples (Index {train_end_index} to {validation_end_index-1})")
print(f"Test split: {len(X_test)} samples (Index {validation_end_index} to {total_samples-1})")

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val:   {X_val.shape}, y_val:   {y_val.shape}")
print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")

if len(X_train) > 0 and (y_train.ndim != 2 or y_train.shape[1] != N_TARGETS):
     print(f"Warning: y_train shape is {y_train.shape}, expected 2D with {N_TARGETS} columns.")
if len(X_val) > 0 and (y_val.ndim != 2 or y_val.shape[1] != N_TARGETS):
     print(f"Warning: y_val shape is {y_val.shape}, expected 2D with {N_TARGETS} columns.")
if len(X_test) > 0 and (y_test.ndim != 2 or y_test.shape[1] != N_TARGETS):
     print(f"Warning: y_test shape is {y_test.shape}, expected 2D with {N_TARGETS} columns.")


if X_train.shape[0] == 0:
    print("Warning: X_train is empty. Skipping scaling. Scaled data will be empty.")
    X_train_scaled = np.empty_like(X_train)
    X_val_scaled = np.empty_like(X_val)
    X_test_scaled = np.empty_like(X_test)
    scaler = None
else:
    n_samples_train, n_timesteps, n_features_check = X_train.shape
    if n_features_check != N_FEATURES:
         raise ValueError(f"Mismatch in feature count. Expected {N_FEATURES}, found {n_features_check} in X_train.")

    n_samples_val = X_val.shape[0]
    n_samples_test = X_test.shape[0]

    X_train_reshaped = X_train.reshape(-1, N_FEATURES)
    X_val_reshaped = X_val.reshape(-1, N_FEATURES) if n_samples_val > 0 else np.empty((0, N_FEATURES))
    X_test_reshaped = X_test.reshape(-1, N_FEATURES) if n_samples_test > 0 else np.empty((0, N_FEATURES))

    scaler = StandardScaler()
    print(f"Fitting StandardScaler on training data shape: {X_train_reshaped.shape}")
    scaler.fit(X_train_reshaped)
    # import joblib
    # joblib.dump(scaler, 'multivariate_scaler.joblib')
    # print("Scaler saved to multivariate_scaler.joblib") # Example of saving scaler (commented out)

    print("Transforming datasets...")
    X_train_scaled_reshaped = scaler.transform(X_train_reshaped)
    X_val_scaled_reshaped = scaler.transform(X_val_reshaped) if n_samples_val > 0 else np.empty((0, N_FEATURES))
    X_test_scaled_reshaped = scaler.transform(X_test_reshaped) if n_samples_test > 0 else np.empty((0, N_FEATURES))

    X_train_scaled = X_train_scaled_reshaped.reshape(n_samples_train, n_timesteps, N_FEATURES)

    if n_samples_val > 0:
        X_val_scaled = X_val_scaled_reshaped.reshape(n_samples_val, n_timesteps, N_FEATURES)
    else:
        X_val_scaled = np.empty((0, n_timesteps, N_FEATURES))

    if n_samples_test > 0:
        X_test_scaled = X_test_scaled_reshaped.reshape(n_samples_test, n_timesteps, N_FEATURES)
    else:
        X_test_scaled = np.empty((0, n_timesteps, N_FEATURES))

print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"X_val_scaled shape:   {X_val_scaled.shape}")
print(f"X_test_scaled shape:  {X_test_scaled.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_val shape:   {y_val.shape}")
print(f"y_test shape:  {y_test.shape}")

print("\nMultivariate data preparation process finished.")
print("Data (X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test) is now ready.")
if scaler:
    print("Feature scaler (StandardScaler) has been fitted on the training data.")
else:
    print("Feature scaling was skipped as the training set was empty.")