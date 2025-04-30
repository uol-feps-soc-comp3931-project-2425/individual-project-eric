import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Set display and chart styles
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False 
sns.set_style("whitegrid")

def load_crypto_data(file_path):
    """Load and process cryptocurrency data"""
    # First, try to determine the file format
    with open(file_path, 'r') as f:
        first_few_lines = [f.readline() for _ in range(3)]

    print(f"Previewing file {file_path}:")
    for line in first_few_lines:
        print(line.strip())

    # Load data, specify date format
    df = pd.read_csv(file_path, parse_dates=True)

    # Check if the first or second row contains header information
    if 'Ticker' in df.columns or 'Price' in df.columns or 'Datetime' in df.columns:
        try:
            df = pd.read_csv(file_path, header=1)
            if 'Datetime' in df.columns:
                df['Datetime'] = pd.to_datetime(df['Datetime'])
                df.set_index('Datetime', inplace=True)
        except Exception as e:
            print(f"Error occurred while trying to process headers: {e}")

    print(f"Data shape: {df.shape}")
    print(f"Column names: {df.columns.tolist()}")

    return df

# Replace with your actual file paths
btc_data = load_crypto_data('BTC_USD_data.csv')
eth_data = load_crypto_data('ETH_USD_data.csv')
sol_data = load_crypto_data('SOL_USD_data.csv')

print("\nData Preview:")
print("BTC Data:")
print(btc_data.head())


column_mapping = {
    'BTC-USD': 'btc_Close',   
    'BTC-USD.1': 'btc_High',  
    'BTC-USD.2': 'btc_Low',   
    'BTC-USD.3': 'btc_Open',  
    'BTC-USD.4': 'btc_Volume',

    'ETH-USD': 'eth_Close',
    'ETH-USD.1': 'eth_High',
    'ETH-USD.2': 'eth_Low',
    'ETH-USD.3': 'eth_Open',
    'ETH-USD.4': 'eth_Volume',

    'SOL-USD': 'sol_Close',
    'SOL-USD.1': 'sol_High',
    'SOL-USD.2': 'sol_Low',
    'SOL-USD.3': 'sol_Open',
    'SOL-USD.4': 'sol_Volume'
}

# Apply column mapping to each DataFrame
btc_data = btc_data.rename(columns={col: column_mapping.get(col, col) for col in btc_data.columns})
eth_data = eth_data.rename(columns={col: column_mapping.get(col, col) for col in eth_data.columns})
sol_data = sol_data.rename(columns={col: column_mapping.get(col, col) for col in sol_data.columns})

# If the index is in datetime format, we can merge by index
merged_data = pd.concat([btc_data, eth_data, sol_data], axis=1)
print(f"\nMerged data shape: {merged_data.shape}")
print(f"Merged column names: {merged_data.columns.tolist()}")

print(f"\nMissing value statistics:\n{merged_data.isnull().sum()}")

# Use forward fill and backward fill to handle missing values
merged_data = merged_data.ffill() # Forward fill
merged_data = merged_data.bfill() # Backward fill

# Check if there are still missing values
missing_values = merged_data.isnull().sum().sum()
print(f"Total missing values after processing: {missing_values}")

# Calculate hourly returns for each cryptocurrency
try:
    merged_data['btc_return'] = merged_data['btc_Close'].pct_change() * 100
    merged_data['eth_return'] = merged_data['eth_Close'].pct_change() * 100
    merged_data['sol_return'] = merged_data['sol_Close'].pct_change() * 100
    print("Successfully calculated returns")
except KeyError as e:
    print(f"Error calculating returns: {e}")
    print("Attempting to determine the correct price columns...")
    # List all columns to find the actual price columns
    print("Available columns:")
    for col in merged_data.columns:
        print(f" - {col}")

    # Guess which columns might be Close price columns based on names
    possible_close_cols = [col for col in merged_data.columns if 'close' in col.lower() or 'price' in col.lower()]
    print(f"Possible Close price columns: {possible_close_cols}")

    # If possible Close price columns are found, try using them
    if possible_close_cols:
        for coin, col in zip(['btc', 'eth', 'sol'], possible_close_cols[:3]):
            merged_data[f'{coin}_return'] = merged_data[col].pct_change() * 100
    else:
        # If no clear Close price columns are found, use the first column as price data
        btc_cols = [col for col in merged_data.columns if col.startswith('btc_')]
        eth_cols = [col for col in merged_data.columns if col.startswith('eth_')]
        sol_cols = [col for col in merged_data.columns if col.startswith('sol_')]

        if btc_cols: merged_data['btc_return'] = merged_data[btc_cols[0]].pct_change() * 100
        if eth_cols: merged_data['eth_return'] = merged_data[eth_cols[0]].pct_change() * 100
        if sol_cols: merged_data['sol_return'] = merged_data[sol_cols[0]].pct_change() * 100

merged_data = merged_data.dropna()

merged_data.to_csv('processed_crypto_data.csv')

print("\nData preprocessing complete!")
print(f"Processed data size: {merged_data.shape}")
print(f"Data saved to 'processed_crypto_data.csv'")