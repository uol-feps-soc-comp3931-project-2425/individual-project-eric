import pandas as pd
import numpy as np

try:
    from ta.momentum import RSIIndicator
    print("TA library found. RSI will be calculated.")
    ta_available = True
except ImportError:
    print("TA library not found. RSI calculation will be skipped (set to 50).")
    RSIIndicator = None
    ta_available = False

print("Imports complete.")

assets = ['BTC', 'ETH', 'SOL']
files = {
    'BTC': 'btcusdt_1m_data.csv',
    'ETH': 'ethusdt_1m_data.csv',
    'SOL': 'solusdt_1m_data.csv'
}
timestamp_col_name = 'open_time' 
required_cols_load = ['open', 'high', 'low', 'close', 'volume'] 
epsilon = 1e-10

interpolation_limit = 5
log_return_iqr_window = 60
log_return_iqr_multiplier = 3.0
volume_norm_window = 24 * 60
volatility_window_15m = 15
volatility_window_60m = 60
rsi_period = 14
lag_periods = [1, 5, 10]


print("\n--- Starting Load, Clean, Merge Assets ---")

dfs = {}

def load_and_clean_asset(asset_name, file_path, ts_col):
    print(f"Loading {asset_name} from {file_path}...")
    try:
        df_asset = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}"); return None
    except Exception as e:
        print(f"Error loading CSV for {asset_name}: {e}"); return None

    if ts_col not in df_asset.columns:
        print(f"Error: Timestamp column '{ts_col}' not found in {file_path}. Available: {df_asset.columns.tolist()}"); return None
    try:
        try: df_asset['datetime'] = pd.to_datetime(df_asset[ts_col], utc=True)
        except (ValueError, TypeError):
            try: df_asset['datetime'] = pd.to_datetime(df_asset[ts_col], unit='ms', utc=True)
            except (ValueError, TypeError):
                 df_asset['datetime'] = pd.to_datetime(df_asset[ts_col], unit='s', utc=True)
    except Exception as e:
        print(f"Error converting timestamp for {asset_name}: {e}"); return None

    df_asset.set_index('datetime', inplace=True)
    df_asset.sort_index(inplace=True)

    df_asset.columns = df_asset.columns.str.lower().str.replace(' ', '_') 
    current_required = [col for col in required_cols_load if col in df_asset.columns]
    if len(current_required) != len(required_cols_load):
         print(f"Warning: Missing some required columns in {asset_name}. Found: {current_required}")
    df_asset = df_asset[current_required].copy()

    df_asset.columns = [f"{col}_{asset_name}" for col in df_asset.columns]

    for col in df_asset.columns:
        df_asset[col] = pd.to_numeric(df_asset[col], errors='coerce')

    print(f"Finished loading and initial cleaning for {asset_name}. Shape: {df_asset.shape}")
    return df_asset

for asset, f_path in files.items():
    dfs[asset] = load_and_clean_asset(asset, f_path, timestamp_col_name)

if any(df is None for df in dfs.values()):
    print("Error: Failed to load one or more asset files. Exiting.")
    exit()

print("\nMerging DataFrames using inner join...")
df_merged = pd.concat(dfs.values(), axis=1, join='inner')
print(f"Merged DataFrame shape: {df_merged.shape}")
if len(df_merged) == 0:
    print("Error: Merged DataFrame is empty. Check if CSV files have overlapping time periods.")
    exit()

print("--- Finished ---")


print("\n--- Starting Multi-Asset Preprocessing & Features ---")

all_feature_cols = [] 

for asset in assets:
    print(f"\nProcessing features for {asset}...")
    close_col = f'close_{asset}'
    volume_col = f'volume_{asset}'

    asset_cols = [col for col in df_merged.columns if col.endswith(f'_{asset}')]
    df_merged[asset_cols] = df_merged[asset_cols].interpolate(method='linear', limit=interpolation_limit)
    df_merged.dropna(subset=asset_cols, inplace=True)
    print(f"  Interpolated/Dropped NaNs for {asset}. Current shape: {df_merged.shape}")
    if len(df_merged) == 0: print("Error: Data empty after NaN handling."); exit()


    log_return_col = f'log_return_{asset}'
    log_return_capped_col = f'log_return_capped_{asset}'

    prev_close_col = f'prev_{close_col}'
    df_merged[prev_close_col] = df_merged[close_col].shift(1)
    df_merged.dropna(subset=[prev_close_col], inplace=True)
    if len(df_merged) == 0: print(f"Error: Data empty after shift for {asset}."); exit()

    df_merged[log_return_col] = np.log((df_merged[close_col] + epsilon) / (df_merged[prev_close_col] + epsilon))
    df_merged.replace([np.inf, -np.inf], 0, inplace=True)
    df_merged.drop(columns=[prev_close_col], inplace=True)
    df_merged.dropna(subset=[log_return_col], inplace=True)

    if len(df_merged) >= log_return_iqr_window:
        q1 = df_merged[log_return_col].rolling(window=log_return_iqr_window, min_periods=1).quantile(0.25)
        q3 = df_merged[log_return_col].rolling(window=log_return_iqr_window, min_periods=1).quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - log_return_iqr_multiplier * iqr
        upper_bound = q3 + log_return_iqr_multiplier * iqr
        df_merged[log_return_capped_col] = np.clip(df_merged[log_return_col], lower_bound, upper_bound)
        df_merged[log_return_capped_col].fillna(df_merged[log_return_col], inplace=True)
    else:
        df_merged[log_return_capped_col] = df_merged[log_return_col]
    df_merged.dropna(subset=[log_return_capped_col], inplace=True)
    print(f"  Calculated/Capped Log Returns for {asset}.")
    all_feature_cols.append(log_return_capped_col)


    volume_zscore_col = f'volume_zscore_{asset}'
    if len(df_merged) >= volume_norm_window :
        vol_mean = df_merged[volume_col].rolling(window=volume_norm_window, min_periods=1).mean()
        vol_std = df_merged[volume_col].rolling(window=volume_norm_window, min_periods=1).std()
        df_merged[volume_zscore_col] = (df_merged[volume_col] - vol_mean) / (vol_std + epsilon)
        df_merged[volume_zscore_col].fillna(0, inplace=True)
    else:
        df_merged[volume_zscore_col] = 0
    print(f"  Calculated Volume Z-score for {asset}.")
    all_feature_cols.append(volume_zscore_col)


    log_return_5m_col = f'log_return_5m_{asset}'
    log_return_15m_col = f'log_return_15m_{asset}'
    df_merged[log_return_5m_col] = df_merged[log_return_capped_col].rolling(window=5, min_periods=1).sum()
    df_merged[log_return_15m_col] = df_merged[log_return_capped_col].rolling(window=15, min_periods=1).sum()
    df_merged[log_return_5m_col].fillna(0, inplace=True)
    df_merged[log_return_15m_col].fillna(0, inplace=True)
    all_feature_cols.extend([log_return_5m_col, log_return_15m_col])

    volatility_15m_col = f'volatility_15m_{asset}'
    volatility_60m_col = f'volatility_60m_{asset}'
    df_merged[volatility_15m_col] = df_merged[log_return_capped_col].rolling(window=volatility_window_15m, min_periods=1).std()
    df_merged[volatility_60m_col] = df_merged[log_return_capped_col].rolling(window=volatility_window_60m, min_periods=1).std()
    df_merged[volatility_15m_col].fillna(0, inplace=True)
    df_merged[volatility_60m_col].fillna(0, inplace=True)
    all_feature_cols.extend([volatility_15m_col, volatility_60m_col])


    # RSI
    rsi_col = f'rsi_{asset}'
    if ta_available and len(df_merged) > rsi_period:
        if df_merged[close_col].isnull().any():
            close_filled = df_merged[close_col].fillna(method='ffill')
            close_filled.dropna(inplace=True)
        else: close_filled = df_merged[close_col]
        if len(close_filled) > rsi_period:
            rsi_indicator = RSIIndicator(close=close_filled, window=rsi_period)
            df_merged[rsi_col] = rsi_indicator.rsi()
            df_merged[rsi_col].fillna(method='ffill', inplace=True)
            df_merged[rsi_col].fillna(50, inplace=True)
        else: df_merged[rsi_col] = 50
    else: df_merged[rsi_col] = 50
    print(f"  Calculated/Set RSI for {asset}.")
    all_feature_cols.append(rsi_col)


    for lag in lag_periods:
        lag_col = f'log_return_lag_{lag}_{asset}'
        df_merged[lag_col] = df_merged[log_return_capped_col].shift(lag)
        df_merged[lag_col].fillna(0, inplace=True)
        all_feature_cols.append(lag_col)
    print(f"  Calculated Lagged Returns for {asset}.")


print("\n--- Starting Final DataFrame Creation ---")

initial_rows_part5 = len(df_merged)
df_merged.dropna(inplace=True)
print(f"Dropped {initial_rows_part5 - len(df_merged)} rows with NaNs after all feature calculations.")

if len(df_merged) == 0:
    print("Error: DataFrame empty after final NaN drop. Check data quality or processing steps.")
    exit()

df_final = df_merged[all_feature_cols].copy() 

print(f"\nFinal combined DataFrame shape: {df_final.shape}")
print(f"Total number of features: {len(df_final.columns)}")
print("\nFinal DataFrame columns:")
print(df_final.columns.tolist())
print("\nFinal DataFrame head:")
print(df_final.head())
print("\nFinal DataFrame info:")
df_final.info()

print("\n--- Finished ---")
print("Multivariate data loaded, merged, processed, and features engineered.")
print("The DataFrame 'df_final' is now ready for sequence creation and multivariate baseline models.")