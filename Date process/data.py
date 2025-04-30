import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional

FILE_PATH: str = 'btcusdt_1m_data.csv'
TIMESTAMP_COL_NAME: str = 'open_time'
REQUIRED_OHLCV_COLS: List[str] = ['open', 'high', 'low', 'close', 'volume']
EPSILON: float = 1e-10

IQR_WINDOW: int = 60
IQR_MULTIPLIER: float = 3.0
VOL_NORM_WINDOW: int = 24 * 60
MULTI_RETURN_WINDOWS: List[int] = [5, 15]
VOLATILITY_WINDOWS: List[int] = [15, 60]
RSI_PERIOD: int = 14
LAGGED_RETURN_PERIODS: List[int] = [1, 5, 10]

SEQUENCE_LENGTH: int = 60
TARGET_COLUMN: str = 'log_return_capped'
TRAIN_SPLIT_PCT: float = 0.70
VALIDATION_SPLIT_PCT: float = 0.15

try:
    from ta.momentum import RSIIndicator
    print("Successfully imported 'ta' library for RSI calculation.")
except ImportError:
    print("Warning: 'ta' library not found. RSI feature will be skipped (or use default).")
    RSIIndicator = None


def create_sequences(
    data: pd.DataFrame,
    sequence_length: int,
    feature_columns: List[str],
    target_column_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates sequences and corresponding targets from time series data.

    Args:
        data (pd.DataFrame): DataFrame containing features AND the target column.
        sequence_length (int): The number of time steps for each input sequence.
        feature_columns (List[str]): List of column names to use as input features for X.
        target_column_name (str): The name of the column to be used as the target (y).

    Returns:
        tuple: A tuple containing:
            - np.array: Input sequences (X) with shape (n_samples, sequence_length, n_features).
            - np.array: Target values (y) with shape (n_samples,).
    """
    X, y = [], []
    if target_column_name not in data.columns:
         print(f"Error in create_sequences: Target column '{target_column_name}' not in data.")
         return np.array(X), np.array(y)
    if not all(col in data.columns for col in feature_columns):
        print(f"Error in create_sequences: One or more feature columns not found in data.")
        missing = [col for col in feature_columns if col not in data.columns]
        print(f"Missing columns: {missing}")
        return np.array(X), np.array(y)

    data_copy = data[feature_columns + [target_column_name]].copy()
    data_copy['target_shifted'] = data_copy[target_column_name].shift(-1)
    data_seq = data_copy.dropna(subset=['target_shifted'])

    feature_data = data_seq[feature_columns].values
    target_data = data_seq['target_shifted'].values

    if len(feature_data) < sequence_length:
        print(f"Warning in create_sequences: Not enough data ({len(feature_data)} rows) "
              f"to create sequences of length {sequence_length}.")
        return np.array(X), np.array(y)

    print(f"Creating sequences from data shape: {feature_data.shape}")
    for i in range(len(feature_data) - sequence_length + 1):
        sequence_end_index = i + sequence_length
        X.append(feature_data[i:sequence_end_index])
        y.append(target_data[sequence_end_index - 1])

    return np.array(X), np.array(y)


def run_data_processing():
    """Runs the entire data loading, processing, and preparation pipeline."""

    try:
        df = pd.read_csv(FILE_PATH)
        print(f"Successfully loaded data from {FILE_PATH}")
        print(f"Initial shape: {df.shape}")
        print("Initial data sample:")
        print(df.head())
    except FileNotFoundError:
        print(f"Error: File not found at {FILE_PATH}")
        return None
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

    if TIMESTAMP_COL_NAME not in df.columns:
        print(f"Error: Timestamp column '{TIMESTAMP_COL_NAME}' not found.")
        print(f"Available columns: {df.columns.tolist()}")
        return None

    try:
        df['datetime'] = pd.to_datetime(df[TIMESTAMP_COL_NAME], errors='coerce', utc=True)
        if df['datetime'].isnull().any():
             print("Standard datetime parsing failed for some/all values, trying epoch...")
             df['datetime_ms'] = pd.to_datetime(df[TIMESTAMP_COL_NAME], unit='ms', errors='coerce', utc=True)
             df['datetime_s'] = pd.to_datetime(df[TIMESTAMP_COL_NAME], unit='s', errors='coerce', utc=True)
             df['datetime'] = df['datetime_ms'].combine_first(df['datetime_s']).combine_first(df['datetime'])
             df.drop(columns=['datetime_ms', 'datetime_s'], inplace=True)

        if df['datetime'].isnull().any():
            print(f"Warning: Could not convert all timestamps in '{TIMESTAMP_COL_NAME}'. Dropping rows with invalid timestamps.")
            print(f"Rows before timestamp drop: {len(df)}")
            df.dropna(subset=['datetime'], inplace=True)
            print(f"Rows after timestamp drop: {len(df)}")
            if len(df) == 0:
                print("Error: No valid timestamp data remaining.")
                return None

    except Exception as e:
        print(f"Error converting timestamp column '{TIMESTAMP_COL_NAME}': {e}")
        return None

    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)

    missing_cols = [col for col in REQUIRED_OHLCV_COLS if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing one or more required columns: {missing_cols}")
        print(f"Available columns: {df.columns.tolist()}")
        return None

    df = df[REQUIRED_OHLCV_COLS].copy()

    for col in REQUIRED_OHLCV_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    initial_nans = df.isnull().sum().sum()
    df.dropna(inplace=True)
    if initial_nans > 0:
         print(f"Dropped {initial_nans} NaN values during initial cleaning/conversion.")

    if len(df) == 0:
        print("Error: No data remaining after initial cleaning and NaN drop.")
        return None

    print(f"DataFrame shape after initial cleaning: {df.shape}")
    print(df.head())

    if len(df) < 2:
        print("Error: Not enough data (< 2 rows) to calculate returns.")
        return None

    df['prev_close'] = df['close'].shift(1)
    df['log_return'] = np.log((df['close'] + EPSILON) / (df['prev_close'] + EPSILON))
    df.drop(columns=['prev_close'], inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    rows_before_drop = len(df)
    df.dropna(subset=['log_return'], inplace=True)
    print(f"Calculated log returns. Dropped {rows_before_drop - len(df)} rows (incl. first row).")

    if len(df) == 0:
        print("Error: No data remaining after calculating log returns.")
        return None

    if len(df) >= IQR_WINDOW:
        q1 = df['log_return'].rolling(window=IQR_WINDOW, min_periods=IQR_WINDOW // 2).quantile(0.25)
        q3 = df['log_return'].rolling(window=IQR_WINDOW, min_periods=IQR_WINDOW // 2).quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - IQR_MULTIPLIER * iqr
        upper_bound = q3 + IQR_MULTIPLIER * iqr
        df['log_return_capped'] = np.clip(df['log_return'], lower_bound, upper_bound)
        df['log_return_capped'].fillna(df['log_return'], inplace=True)
        print(f"Applied rolling IQR capping (window={IQR_WINDOW}) to log returns.")
    else:
        print(f"Warning: Not enough data ({len(df)} rows) for IQR window {IQR_WINDOW}. Using uncapped log returns.")
        df['log_return_capped'] = df['log_return']

    if df['log_return_capped'].isnull().any():
         print("Warning: NaNs found in log_return_capped after processing. Dropping rows.")
         rows_before_drop = len(df)
         df.dropna(subset=['log_return_capped'], inplace=True)
         print(f"Dropped {rows_before_drop - len(df)} rows with NaNs in capped returns.")

    if len(df) == 0:
        print("Error: No data remaining after outlier capping.")
        return None

    print(f"DataFrame shape after return calculation & capping: {df.shape}")
    print(df[['close', 'log_return', 'log_return_capped']].head())

    if len(df) >= VOL_NORM_WINDOW // 10:
        vol_mean = df['volume'].rolling(window=VOL_NORM_WINDOW, min_periods=VOL_NORM_WINDOW // 10).mean()
        vol_std = df['volume'].rolling(window=VOL_NORM_WINDOW, min_periods=VOL_NORM_WINDOW // 10).std()
        df['volume_zscore'] = (df['volume'] - vol_mean) / (vol_std + EPSILON)
        df['volume_zscore'].fillna(0, inplace=True)
        print(f"Calculated rolling Volume Z-score (window={VOL_NORM_WINDOW}).")
    else:
        print(f"Warning: Not enough data ({len(df)} rows) for Volume Z-score window {VOL_NORM_WINDOW}. Setting Z-score to 0.")
        df['volume_zscore'] = 0.0

    for window in MULTI_RETURN_WINDOWS:
        col_name = f'log_return_{window}m'
        if len(df) >= window:
            df[col_name] = df['log_return_capped'].rolling(window=window, min_periods=window // 2).sum()
            df[col_name].fillna(0, inplace=True)
        else:
            print(f"Warning: Not enough data for {window}m return. Setting to 0.")
            df[col_name] = 0.0
    print(f"Calculated multi-scale returns ({MULTI_RETURN_WINDOWS}).")

    for window in VOLATILITY_WINDOWS:
        col_name = f'volatility_{window}m'
        if len(df) >= window:
            df[col_name] = df['log_return_capped'].rolling(window=window, min_periods=window // 2).std()
            df[col_name].fillna(0, inplace=True)
        else:
            print(f"Warning: Not enough data for {window}m volatility. Setting to 0.")
            df[col_name] = 0.0
    print(f"Calculated realized volatility ({VOLATILITY_WINDOWS}).")

    print(f"DataFrame shape after volume norm & feature engineering: {df.shape}")
    feature_cols_p3 = ['volume_zscore'] + \
                      [f'log_return_{w}m' for w in MULTI_RETURN_WINDOWS] + \
                      [f'volatility_{w}m' for w in VOLATILITY_WINDOWS]
    print(df[feature_cols_p3].head())

    if RSIIndicator:
        if len(df) > RSI_PERIOD:
            close_for_rsi = df['close'].copy()
            if close_for_rsi.isnull().any():
                print("Warning: NaNs found in 'close' column before RSI calculation. Filling with ffill.")
                close_for_rsi.fillna(method='ffill', inplace=True)
                close_for_rsi.dropna(inplace=True)

            if len(close_for_rsi) > RSI_PERIOD:
                rsi_indicator = RSIIndicator(close=close_for_rsi, window=RSI_PERIOD)
                df['rsi'] = rsi_indicator.rsi().reindex(df.index)
                df['rsi'].fillna(method='ffill', inplace=True)
                df['rsi'].fillna(50, inplace=True)
                print(f"Calculated RSI (period={RSI_PERIOD}).")
            else:
                print(f"Warning: Not enough valid 'close' data ({len(close_for_rsi)} rows) for RSI calculation after NaN handling. Setting RSI to 50.")
                df['rsi'] = 50.0
        else:
            print(f"Warning: Not enough data ({len(df)} rows) for RSI period {RSI_PERIOD}. Setting RSI to 50.")
            df['rsi'] = 50.0
    else:
        print("Skipping RSI calculation as 'ta' library is not available. Setting RSI to 50.")
        df['rsi'] = 50.0

    for lag in LAGGED_RETURN_PERIODS:
        col_name = f'log_return_lag_{lag}'
        df[col_name] = df['log_return_capped'].shift(lag)
        df[col_name].fillna(0, inplace=True)
    print(f"Calculated Lagged Returns ({LAGGED_RETURN_PERIODS}).")

    potential_features = [
        'log_return_capped',
        'volume_zscore',
        'rsi',
    ] + [f'log_return_{w}m' for w in MULTI_RETURN_WINDOWS] \
      + [f'volatility_{w}m' for w in VOLATILITY_WINDOWS] \
      + [f'log_return_lag_{lag}' for lag in LAGGED_RETURN_PERIODS]

    final_feature_columns = []
    print("\nSelecting and verifying final feature columns...")
    for f in potential_features:
        if f in df.columns:
            if df[f].isnull().all():
                print(f"Warning: Feature '{f}' is all NaN. Excluding.")
            else:
                final_feature_columns.append(f)
        else:
            print(f"Warning: Expected feature '{f}' not found in DataFrame. Skipping.")

    if TARGET_COLUMN not in final_feature_columns:
         print(f"Error: Target column '{TARGET_COLUMN}' is missing or all NaN in the final features.")
         return None

    df_final = df[final_feature_columns].copy()

    if df_final.isnull().values.any():
        print("\nWarning: NaNs found in final feature set before sequencing.")
        print("Columns with NaNs and count:")
        print(df_final.isnull().sum()[df_final.isnull().sum() > 0])
        rows_before_final_drop = len(df_final)
        df_final.dropna(inplace=True)
        print(f"Dropped {rows_before_final_drop - len(df_final)} more rows with NaNs.")
    else:
        print("\nNo NaNs found in the final feature set.")

    if len(df_final) == 0:
        print("Error: No data remaining after final NaN checks. Check intermediate steps.")
        return None

    print("Final processed data sample (features selected):")
    print(df_final.head())
    print(f"\nFinal data shape: {df_final.shape}")
    print(f"Selected feature columns (incl. target): {df_final.columns.tolist()}")

    input_feature_names = [col for col in df_final.columns if col != TARGET_COLUMN]

    if not input_feature_names:
         print("Error: No input features remain after excluding the target column.")
         return None

    print(f"Using features for X: {input_feature_names}")
    print(f"Using target for y: {TARGET_COLUMN}")

    X, y = create_sequences(df_final, SEQUENCE_LENGTH, input_feature_names, TARGET_COLUMN)

    if X.size == 0 or y.size == 0:
        print("Error: Sequence creation resulted in empty arrays. Check data length vs sequence length.")
        return None

    print(f"\nCreated sequences:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Number of features in X: {X.shape[2]}")

    total_samples = X.shape[0]
    if total_samples < 3:
         print("Error: Not enough sequences generated to split into Train/Val/Test.")
         return None

    train_end_index = int(total_samples * TRAIN_SPLIT_PCT)
    validation_end_index = train_end_index + int(total_samples * VALIDATION_SPLIT_PCT)

    validation_end_index = min(validation_end_index, total_samples)
    train_end_index = min(train_end_index, validation_end_index)

    X_train, y_train = X[:train_end_index], y[:train_end_index]
    X_val, y_val = X[train_end_index:validation_end_index], y[train_end_index:validation_end_index]
    X_test, y_test = X[validation_end_index:], y[validation_end_index:]

    print("\nDataset shapes after splitting:")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape:   {X_val.shape}, y_val shape:   {y_val.shape}")
    print(f"X_test shape:  {X_test.shape}, y_test shape:  {y_test.shape}")

    if X_train.shape[0] == 0 or X_val.shape[0] == 0 or X_test.shape[0] == 0:
         print("Warning: One or more splits (Train/Val/Test) resulted in 0 samples.")
         print("Consider adjusting split percentages or acquiring more data.")

    if X_train.shape[0] == 0:
        print("Error: X_train is empty, cannot fit scaler.")
        return None

    n_samples_train, n_timesteps, n_features = X_train.shape
    n_samples_val = X_val.shape[0]
    n_samples_test = X_test.shape[0]

    X_train_reshaped = X_train.reshape(-1, n_features)
    X_val_reshaped = X_val.reshape(-1, n_features) if n_samples_val > 0 else np.empty((0, n_features))
    X_test_reshaped = X_test.reshape(-1, n_features) if n_samples_test > 0 else np.empty((0, n_features))

    scaler = StandardScaler()
    print("Fitting StandardScaler on training data...")
    scaler.fit(X_train_reshaped)
    print("Scaler fitted.")

    print("Transforming datasets...")
    X_train_scaled_reshaped = scaler.transform(X_train_reshaped)
    X_val_scaled_reshaped = scaler.transform(X_val_reshaped) if n_samples_val > 0 else np.empty((0, n_features))
    X_test_scaled_reshaped = scaler.transform(X_test_reshaped) if n_samples_test > 0 else np.empty((0, n_features))
    print("Transformation complete.")

    X_train_scaled = X_train_scaled_reshaped.reshape(n_samples_train, n_timesteps, n_features)
    X_val_scaled = X_val_scaled_reshaped.reshape(n_samples_val, n_timesteps, n_features) if n_samples_val > 0 else np.empty((0, n_timesteps, n_features))
    X_test_scaled = X_test_scaled_reshaped.reshape(n_samples_test, n_timesteps, n_features) if n_samples_test > 0 else np.empty((0, n_timesteps, n_features))

    print("\nDataset shapes after scaling and reshaping:")
    print(f"X_train_scaled shape: {X_train_scaled.shape}")
    print(f"X_val_scaled shape:   {X_val_scaled.shape}")
    print(f"X_test_scaled shape:  {X_test_scaled.shape}")

    print("Feature scaling complete. Data is ready for model input.")
    print("Returning scaled data (X_train, y_train, X_val, y_val, X_test, y_test) and the scaler.")

    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, scaler, input_feature_names

if __name__ == "__main__":
    print("Starting data processing pipeline...")
    results = run_data_processing()

    if results:
        X_train_final, y_train_final, X_val_final, y_val_final, X_test_final, y_test_final, fitted_scaler, final_features = results
        print("\nPipeline completed successfully.")
        print(f"Final features used for model input: {final_features}")
    else:
        print("\nData processing pipeline failed or returned no data.")