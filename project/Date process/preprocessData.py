import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

# --- Configuration ---
DATA_FILES = {
    "btc": "btcusdt_1m_data.csv",
    "eth": "ethusdt_1m_data.csv",
    "sol": "solusdt_1m_data.csv"
}
OUTPUT_DIR = ""
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "preprocessedCryptoData.csv")

# --- Feature Engineering Functions ---
def calculate_rsi(data, window=14):
    delta = data["close"].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    exp1 = data["close"].ewm(span=fast_period, adjust=False).mean()
    exp2 = data["close"].ewm(span=slow_period, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

def calculate_bollinger_bands(data, window=20, num_std_dev=2):
    rolling_mean = data["close"].rolling(window=window).mean()
    rolling_std = data["close"].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    return upper_band, lower_band

def calculate_atr(data, window=14):
    high_low = data["high"] - data["low"]
    high_close = np.abs(data["high"] - data["close"].shift())
    low_close = np.abs(data["low"] - data["close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr

# --- Main Preprocessing Logic ---
all_data = {}

print("Starting data preprocessing...")

for symbol, filepath in DATA_FILES.items():
    print(f"Processing {symbol} data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        # Convert timestamp to datetime and set as index
        df["open_time"] = pd.to_datetime(df["open_time"], unit=\'ms
        df.set_index("open_time", inplace=True)
        df.sort_index(inplace=True)

        # Keep original OHLCV columns
        df[f"{symbol}_open"] = df["open"]
        df[f"{symbol}_high"] = df["high"]
        df[f"{symbol}_low"] = df["low"]
        df[f"{symbol}_close"] = df["close"]
        df[f"{symbol}_volume"] = df["volume"]

        # Calculate basic features
        df[f"{symbol}_log_return"] = np.log(df["close"] / df["close"].shift(1))
        df[f"{symbol}_volatility"] = df[f"{symbol}_log_return"].rolling(window=20).std() * np.sqrt(20) 

        # Calculate technical indicators (using original OHLC)
        df[f"{symbol}_rsi"] = calculate_rsi(df)
        df[f"{symbol}_macd"], df[f"{symbol}_macd_signal"], df[f"{symbol}_macd_hist"] = calculate_macd(df)
        df[f"{symbol}_bollinger_upper"], df[f"{symbol}_bollinger_lower"] = calculate_bollinger_bands(df)
        df[f"{symbol}_atr"] = calculate_atr(df)

        # Calculate moving averages
        df[f"{symbol}_sma_10"] = df["close"].rolling(window=10).mean()
        df[f"{symbol}_sma_50"] = df["close"].rolling(window=50).mean()
        df[f"{symbol}_ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
        df[f"{symbol}_ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
        
        # Add time-based features
        df["hour"] = df.index.hour
        df["day_of_week"] = df.index.dayofweek
        df["day_of_month"] = df.index.day
        df["month"] = df.index.month

        # Select relevant columns for this asset
        cols_to_keep = [col for col in df.columns if col.startswith(symbol + "_") or col in ["hour", "day_of_week", "day_of_month", "month"]]
        all_data[symbol] = df[cols_to_keep]
        print(f"Finished processing {symbol}. Shape: {all_data[symbol].shape}")

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

# Merge dataframes
if not all_data:
    print("No data processed. Exiting.")
else:
    print("Merging dataframes...")
    merged_df = pd.concat(all_data.values(), axis=1, join=\'inner\') # Use inner join to align timestamps
    
    # Handle potential duplicate time-based columns (keep only one set)
    merged_df = merged_df.loc[:,~merged_df.columns.duplicated()] 

    # Drop rows with NaN values resulting from calculations
    merged_df.dropna(inplace=True)

    print(f"Final merged dataframe shape: {merged_df.shape}")
    print("Saving preprocessed data...")
    merged_df.to_csv(OUTPUT_FILE)
    print(f"Preprocessed data saved to {OUTPUT_FILE}")

print("Preprocessing complete.")

