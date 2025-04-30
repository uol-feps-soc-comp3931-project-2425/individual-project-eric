import pandas as pd
import numpy as np
import ta
import os

# Configuration
INPUT_FILE = "preprocessed_crypto_data.csv"
OUTPUT_DIR = "result/"
ASSETS = ["btc", "eth", "sol"]

def add_indicators(df, asset_prefix):
    """Adds technical indicators to the dataframe for a specific asset."""
    close_col = f"{asset_prefix}_close"
    high_col = f"{asset_prefix}_high"
    low_col = f"{asset_prefix}_low"
    volume_col = f"{asset_prefix}_volume"

    required_cols = [close_col, high_col, low_col, volume_col]
    if not all(col in df.columns for col in required_cols):
        print(f"Warning: Missing required columns for {asset_prefix}. Skipping some indicators.")
        df[f"{asset_prefix}_SMA_10"] = ta.trend.sma_indicator(df[close_col], window=10)
        df[f"{asset_prefix}_SMA_50"] = ta.trend.sma_indicator(df[close_col], window=50)
        df[f"{asset_prefix}_EMA_10"] = ta.trend.ema_indicator(df[close_col], window=10)
        df[f"{asset_prefix}_EMA_50"] = ta.trend.ema_indicator(df[close_col], window=50)
        df[f"{asset_prefix}_RSI"] = ta.momentum.rsi(df[close_col], window=14)
        # MACD
        macd = ta.trend.MACD(df[close_col])
        df[f"{asset_prefix}_MACD"] = macd.macd()
        df[f"{asset_prefix}_MACD_signal"] = macd.macd_signal()
        df[f"{asset_prefix}_MACD_hist"] = macd.macd_diff()
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df[close_col])
        df[f"{asset_prefix}_Bollinger_High"] = bollinger.bollinger_hband()
        df[f"{asset_prefix}_Bollinger_Low"] = bollinger.bollinger_lband()
        return df

    df[f"{asset_prefix}_SMA_10"] = ta.trend.sma_indicator(df[close_col], window=10)
    df[f"{asset_prefix}_SMA_50"] = ta.trend.sma_indicator(df[close_col], window=50)
    df[f"{asset_prefix}_EMA_10"] = ta.trend.ema_indicator(df[close_col], window=10)
    df[f"{asset_prefix}_EMA_50"] = ta.trend.ema_indicator(df[close_col], window=50)
    df[f"{asset_prefix}_RSI"] = ta.momentum.rsi(df[close_col], window=14)
    
    # MACD
    macd = ta.trend.MACD(df[close_col])
    df[f"{asset_prefix}_MACD"] = macd.macd()
    df[f"{asset_prefix}_MACD_signal"] = macd.macd_signal()
    df[f"{asset_prefix}_MACD_hist"] = macd.macd_diff()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df[close_col])
    df[f"{asset_prefix}_Bollinger_High"] = bollinger.bollinger_hband()
    df[f"{asset_prefix}_Bollinger_Low"] = bollinger.bollinger_lband()
    
    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df[high_col], df[low_col], df[close_col])
    df[f"{asset_prefix}_Stoch_k"] = stoch.stoch()
    df[f"{asset_prefix}_Stoch_d"] = stoch.stoch_signal()
    
    # Average True Range (ATR)
    df[f"{asset_prefix}_ATR"] = ta.volatility.average_true_range(df[high_col], df[low_col], df[close_col], window=14)
    df[f"{asset_prefix}_OBV"] = ta.volume.on_balance_volume(df[close_col], df[volume_col])
    df[f"{asset_prefix}_ROC"] = ta.momentum.roc(df[close_col], window=12)

    return df

# Main Execution
print(f"Loading preprocessed data from {INPUT_FILE}...")
df_merged = pd.read_csv(INPUT_FILE, index_col="open_time", parse_dates=True)
print(f"Loaded data shape: {df_merged.shape}")
asset_dfs = {}

print("Calculating indicators for each asset...")
for asset in ASSETS:
    print(f"Calculating indicators for {asset.upper()}...")
    asset_cols = [col for col in df_merged.columns if col.startswith(asset + "_")]
    time_cols = [col for col in ["hour", "day_of_week", "day_of_month", "month"] if col in df_merged.columns]
    asset_df = df_merged[asset_cols + time_cols].copy()
    
    rename_map = {f"{asset}_open": "open", f"{asset}_high": "high", f"{asset}_low": "low", f"{asset}_close": "close", f"{asset}_volume": "volume"}
    asset_df_renamed = asset_df.rename(columns=rename_map)
    asset_df_with_indicators = add_indicators(asset_df_renamed, asset)
    
    reverse_rename_map = {v: k for k, v in rename_map.items()}
    asset_df_with_indicators.rename(columns=reverse_rename_map, inplace=True)
    asset_df_with_indicators.dropna(inplace=True)
    
    asset_dfs[asset] = asset_df_with_indicators
    output_filename = os.path.join(OUTPUT_DIR, f"{asset}_data_with_indicators.csv")
    asset_df_with_indicators.to_csv(output_filename)
    print(f"Saved {asset.upper()} data with indicators to {output_filename}. Shape: {asset_df_with_indicators.shape}")

print("Merging all assets with indicators...")
final_df = pd.concat(asset_dfs.values(), axis=1, join=\"inner\")
final_df = final_df.loc[:,~final_df.columns.duplicated()]
final_output_file = os.path.join(OUTPUT_DIR, "full_features_crypto_data.csv")
final_df.to_csv(final_output_file)
print(f"Final combined data with all features saved to {final_output_file}. Shape: {final_df.shape}")

print("Indicator calculation complete.")
