# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from statsmodels.tsa.stattools import ccf
import warnings
warnings.filterwarnings('ignore')  

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['savefig.dpi'] = 300  

try:
    crypto_data = pd.read_csv('processed_crypto_data.csv', index_col=0, parse_dates=True)
    print("Successfully loaded processed data!")
except FileNotFoundError:
    print("Processed data file not found, trying to load raw data.")
    try:
        btc_data = pd.read_csv('BTC-USD_data.csv', index_col=0, parse_dates=True)
        eth_data = pd.read_csv('ETH-USD_data.csv', index_col=0, parse_dates=True)
        sol_data = pd.read_csv('SOL-USD_data.csv', index_col=0, parse_dates=True)
        
        if 'Ticker' in btc_data.index:
            btc_data = pd.read_csv('BTC-USD_data.csv', skiprows=1, index_col=0, parse_dates=True)
            eth_data = pd.read_csv('ETH-USD_data.csv', skiprows=1, index_col=0, parse_dates=True)
            sol_data = pd.read_csv('SOL-USD_data.csv', skiprows=1, index_col=0, parse_dates=True)
        
        btc_data = btc_data.add_prefix('btc_')
        eth_data = eth_data.add_prefix('eth_')
        sol_data = sol_data.add_prefix('sol_')
        

        crypto_data = pd.concat([btc_data, eth_data, sol_data], axis=1)

        crypto_data = crypto_data.fillna(method='ffill').fillna(method='bfill')
        
        crypto_data['btc_return'] = crypto_data['btc_Close'].pct_change() * 100
        crypto_data['eth_return'] = crypto_data['eth_Close'].pct_change() * 100
        crypto_data['sol_return'] = crypto_data['sol_Close'].pct_change() * 100

        crypto_data = crypto_data.dropna()
        
        print("Successfully loaded and processed raw data!")
    except Exception as e:
        print(f"Error loading data: {e}")
        import sys
        sys.exit()

print("\nData Information:")
print(f"Date Range: {crypto_data.index.min()} to {crypto_data.index.max()}")
print(f"Total Data Points: {len(crypto_data)}")

print("\n--- 2.1 Statistical Description ---")

price_stats = crypto_data[['btc_Close', 'eth_Close', 'sol_Close']].describe()
print("\nCryptocurrency Price Statistics:")
print(price_stats)

returns_stats = crypto_data[['btc_return', 'eth_return', 'sol_return']].describe()
print("\nCryptocurrency Return Statistics:")
print(returns_stats)


print("\n--- 2.2 Price Trend Visualization ---")
print("Generating price trend charts...")

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

axes[0].plot(crypto_data.index, crypto_data['btc_Close'], color='orange')
axes[0].set_title('Bitcoin (BTC) Price Trend', fontsize=14)
axes[0].set_ylabel('Price (USD)', fontsize=12)
axes[0].grid(True)
axes[1].plot(crypto_data.index, crypto_data['eth_Close'], color='blue')
axes[1].set_title('Ethereum (ETH) Price Trend', fontsize=14)
axes[1].set_ylabel('Price (USD)', fontsize=12)
axes[1].grid(True)
axes[2].plot(crypto_data.index, crypto_data['sol_Close'], color='purple')
axes[2].set_title('Solana (SOL) Price Trend', fontsize=14)
axes[2].set_xlabel('Date', fontsize=12)
axes[2].set_ylabel('Price (USD)', fontsize=12)
axes[2].grid(True)

plt.tight_layout()
plt.savefig('crypto_price_trends.png')
plt.close()

print("\n===== 2.3 Normalized Price Comparison =====")
print("Generating normalized price comparison chart...")

plt.figure(figsize=(14, 7))

for coin, color, label in zip(
    ['btc_Close', 'eth_Close', 'sol_Close'], 
    ['orange', 'blue', 'purple'],
    ['BTC', 'ETH', 'SOL']
):
    normalized = crypto_data[coin] / crypto_data[coin].iloc[0] * 100
    plt.plot(crypto_data.index, normalized, label=label, color=color)

plt.title('Cryptocurrency Relative Price Comparison (Base=100)', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Relative Price (%)', fontsize=12)
plt.legend()
plt.grid(True)
plt.savefig('normalized_prices.png')
plt.close()

print("\n--- 2.4 Return Distribution Analysis ---")
print("Generating return distribution charts...")

plt.figure(figsize=(14, 6))
for coin, color, label in zip(
    ['btc_return', 'eth_return', 'sol_return'], 
    ['orange', 'blue', 'purple'],
    ['BTC', 'ETH', 'SOL']
):
    sns.histplot(crypto_data[coin], kde=True, color=color, alpha=0.3, label=label)

plt.title('Returns Distribution Comparison', fontsize=14)
plt.xlabel('Return (%)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.grid(True)
plt.savefig('return_distribution.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.boxplot(data=crypto_data[['btc_return', 'eth_return', 'sol_return']])
plt.title('Cryptocurrency Return Volatility Comparison', fontsize=14)
plt.ylabel('Return (%)', fontsize=12)
plt.grid(True, axis='y')
plt.xticks([0, 1, 2], ['BTC', 'ETH', 'SOL'])
plt.savefig('return_volatility_boxplot.png')
plt.close()

print("\n===== 2.5 Correlation Analysis =====")
print("Generating correlation matrices...")

returns_corr = crypto_data[['btc_return', 'eth_return', 'sol_return']].corr()
print("\nReturn Correlation Matrix:")
print(returns_corr)

plt.figure(figsize=(10, 8))
sns.heatmap(returns_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.4f')
plt.title('Cryptocurrency Returns Correlation Matrix', fontsize=14)
plt.savefig('returns_correlation.png')
plt.close()

price_corr = crypto_data[['btc_Close', 'eth_Close', 'sol_Close']].corr()
print("\nPrice Correlation Matrix:")
print(price_corr)

plt.figure(figsize=(10, 8))
sns.heatmap(price_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.4f')
plt.title('Cryptocurrency Price Correlation Matrix', fontsize=14)
plt.savefig('price_correlation.png')
plt.close()

print("Generating return scatter plot matrix...")
returns_data = crypto_data[['btc_return', 'eth_return', 'sol_return']].copy()
returns_data.columns = ['BTC', 'ETH', 'SOL']  
sns.pairplot(returns_data)
plt.suptitle('Cryptocurrency Returns Scatter Matrix', y=1.02, fontsize=16)
plt.savefig('returns_pairplot.png')
plt.close()

print("\n--- 2.6 Rolling Correlation Analysis ---")
print("Generating rolling correlation charts...")

windows = [24*1, 24*7, 24*30]  
window_labels = ['1 Day', '7 Days', '30 Days']

for window, label in zip(windows, window_labels):
    rolling_corr_btc_eth = crypto_data['btc_return'].rolling(window=window).corr(crypto_data['eth_return'])
    rolling_corr_btc_sol = crypto_data['btc_return'].rolling(window=window).corr(crypto_data['sol_return'])
    rolling_corr_eth_sol = crypto_data['eth_return'].rolling(window=window).corr(crypto_data['sol_return'])
    
    plt.figure(figsize=(14, 7))
    plt.plot(crypto_data.index, rolling_corr_btc_eth, label='BTC-ETH', color='green')
    plt.plot(crypto_data.index, rolling_corr_btc_sol, label='BTC-SOL', color='red')
    plt.plot(crypto_data.index, rolling_corr_eth_sol, label='ETH-SOL', color='purple')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    plt.title(f'Rolling Correlation of Crypto Returns (Window={label})', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Correlation Coefficient', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.ylim(-1, 1)  
    

    save_label = label.replace(' ', '_')
    plt.savefig(f'rolling_correlation_{save_label}.png')
    plt.close()


print("\n--- 2.7 Cross-Correlation Analysis ---")
print("Performing cross-correlation analysis...")

def plot_cross_correlation(returns1, returns2, coin1, coin2, max_lags=24):
    data = pd.concat([returns1, returns2], axis=1).dropna()
    returns1_clean = data.iloc[:, 0]
    returns2_clean = data.iloc[:, 1]
    

    cross_corr = ccf(returns1_clean, returns2_clean, adjusted=True)
    

    plt.figure(figsize=(14, 6))
    lags = range(-max_lags, max_lags+1)
    plt.stem(lags, cross_corr[max_lags-max_lags:max_lags+max_lags+1])
    plt.axhline(y=0, linestyle='-', color='black', alpha=0.3)
    

    conf_level = 1.96 / np.sqrt(len(returns1_clean))
    plt.axhline(y=conf_level, linestyle='--', color='red', alpha=0.5)
    plt.axhline(y=-conf_level, linestyle='--', color='red', alpha=0.5)
    
    plt.title(f'Cross-Correlation Analysis between {coin1} and {coin2}\nPositive lag: {coin1} leads, Negative lag: {coin2} leads', fontsize=14)
    plt.xlabel('Time Lag (Hours)', fontsize=12)
    plt.ylabel('Correlation Coefficient', fontsize=12)
    plt.grid(True)
    plt.savefig(f'cross_corr_{coin1.lower()}_{coin2.lower()}.png')
    plt.close()
    

    max_corr_idx = np.argmax(np.abs(cross_corr))
    max_corr = cross_corr[max_corr_idx]
    lag = max_corr_idx - max_lags
    

    significant_lags = []
    for i, corr in enumerate(cross_corr):
        if abs(corr) > conf_level:
            lag_value = i - max_lags
            significant_lags.append((lag_value, corr))
    
    print(f"{coin1}-{coin2} Cross-Correlation Results:")
    print(f"  Max Correlation: {max_corr:.4f} at lag {lag} hours")
    print(f"  Significant lags: {[lag for lag, corr in significant_lags]}")
    
    return lag, max_corr, significant_lags


eth_btc_lag, eth_btc_corr, eth_btc_sig = plot_cross_correlation(
    crypto_data['eth_return'], 
    crypto_data['btc_return'], 
    'ETH', 'BTC', 
    max_lags=24
)


sol_btc_lag, sol_btc_corr, sol_btc_sig = plot_cross_correlation(
    crypto_data['sol_return'], 
    crypto_data['btc_return'], 
    'SOL', 'BTC', 
    max_lags=24
)

sol_eth_lag, sol_eth_corr, sol_eth_sig = plot_cross_correlation(
    crypto_data['sol_return'], 
    crypto_data['eth_return'], 
    'SOL', 'ETH', 
    max_lags=24
)

print("\n--- 2.8 Time Series Characteristics Analysis ---")

from statsmodels.graphics.tsaplots import plot_acf

for coin, color, label in zip(
    ['btc_return', 'eth_return', 'sol_return'], 
    ['orange', 'blue', 'purple'],
    ['BTC', 'ETH', 'SOL']
):
    plt.figure(figsize=(12, 6))
    plot_acf(crypto_data[coin].dropna(), lags=48, alpha=0.05)
    plt.title(f'Autocorrelation Function of {label} Returns')
    plt.grid(True)
    plt.savefig(f'{label.lower()}_acf.png')
    plt.close()

print("\n--- 2.9 Volatility Clustering Analysis ---")
print("Analyzing volatility clustering...")

volatility_window = 24*3  
for coin, label in zip(
    ['btc_return', 'eth_return', 'sol_return'],
    ['BTC', 'ETH', 'SOL']
):
    crypto_data[f'{coin}_volatility'] = crypto_data[coin].rolling(window=volatility_window).std()

plt.figure(figsize=(14, 7))
for coin, color, label in zip(
    ['btc_return_volatility', 'eth_return_volatility', 'sol_return_volatility'], 
    ['orange', 'blue', 'purple'],
    ['BTC', 'ETH', 'SOL']
):
    plt.plot(crypto_data.index, crypto_data[coin], label=label, color=color)

plt.title(f'Cryptocurrency Return Volatility (Rolling Window={volatility_window//24} Days)', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Volatility (Std Dev)', fontsize=12)
plt.legend()
plt.grid(True)
plt.savefig('volatility_comparison.png')
plt.close()

print("\n--- Advanced Relationship Analysis ---")

for coin in ['btc', 'eth', 'sol']:
    crypto_data[f'{coin}_direction'] = np.sign(crypto_data[f'{coin}_return'])

btc_up = crypto_data[crypto_data['btc_direction'] > 0]
btc_down = crypto_data[crypto_data['btc_direction'] < 0]

eth_up_when_btc_up = (btc_up['eth_direction'] > 0).mean()
sol_up_when_btc_up = (btc_up['sol_direction'] > 0).mean()

eth_down_when_btc_down = (btc_down['eth_direction'] < 0).mean()
sol_down_when_btc_down = (btc_down['sol_direction'] < 0).mean()

print("\nConditional Probability Analysis:")
print(f"Probability of ETH rising when BTC rises: {eth_up_when_btc_up:.4f}")
print(f"Probability of SOL rising when BTC rises: {sol_up_when_btc_up:.4f}")
print(f"Probability of ETH falling when BTC falls: {eth_down_when_btc_down:.4f}")
print(f"Probability of SOL falling when BTC falls: {sol_down_when_btc_down:.4f}")

prob_data = pd.DataFrame({
    'Prob of Rise when BTC Rises': [eth_up_when_btc_up, sol_up_when_btc_up],
    'Prob of Fall when BTC Falls': [eth_down_when_btc_down, sol_down_when_btc_down]
}, index=['ETH', 'SOL'])

plt.figure(figsize=(10, 6))
prob_data.plot(kind='bar', figsize=(10, 6))
plt.title('Conditional Probability: Crypto Price Movement Linkage', fontsize=14)
plt.ylabel('Probability', fontsize=12)
plt.ylim(0, 1)
plt.grid(True, axis='y')
plt.savefig('conditional_probability.png')
plt.close()

report = f"""
# Cryptocurrency Price Relationship Exploratory Analysis Report

## 1. Basic Statistics

### Price Range:
- BTC: ${price_stats.loc['min', 'btc_Close']:.2f} - ${price_stats.loc['max', 'btc_Close']:.2f} USD
- ETH: ${price_stats.loc['min', 'eth_Close']:.2f} - ${price_stats.loc['max', 'eth_Close']:.2f} USD
- SOL: ${price_stats.loc['min', 'sol_Close']:.2f} - ${price_stats.loc['max', 'sol_Close']:.2f} USD

### Return Statistics:
- BTC average return: {returns_stats.loc['mean', 'btc_return']:.4f}%
- ETH average return: {returns_stats.loc['mean', 'eth_return']:.4f}%
- SOL average return: {returns_stats.loc['mean', 'sol_return']:.4f}%

### Volatility (Standard Deviation):
- BTC: {returns_stats.loc['std', 'btc_return']:.4f}%
- ETH: {returns_stats.loc['std', 'eth_return']:.4f}%
- SOL: {returns_stats.loc['std', 'sol_return']:.4f}%

## 2. Correlation Analysis

### Return Correlations:
- BTC-ETH: {returns_corr.loc['btc_return', 'eth_return']:.4f}
- BTC-SOL: {returns_corr.loc['btc_return', 'sol_return']:.4f}
- ETH-SOL: {returns_corr.loc['eth_return', 'sol_return']:.4f}

### Cross-Correlation Analysis (Lead-Lag Relationships):
- ETH and BTC: Max correlation {eth_btc_corr:.4f} at lag {eth_btc_lag} hours
- SOL and BTC: Max correlation {sol_btc_corr:.4f} at lag {sol_btc_lag} hours
- SOL and ETH: Max correlation {sol_eth_corr:.4f} at lag {sol_eth_lag} hours

## 3. Conditional Probability Analysis

- Probability of ETH rising when BTC rises: {eth_up_when_btc_up:.4f}
- Probability of SOL rising when BTC rises: {sol_up_when_btc_up:.4f}
- Probability of ETH falling when BTC falls: {eth_down_when_btc_down:.4f}
- Probability of SOL falling when BTC falls: {sol_down_when_btc_down:.4f}

## 4. Key Findings

1. The three cryptocurrencies' price movements are highly correlated, especially between BTC and ETH
2. Based on cross-correlation analysis, {"ETH appears to lead BTC" if eth_btc_lag < 0 else "BTC appears to lead ETH" if eth_btc_lag > 0 else "ETH and BTC move almost simultaneously"}
3. {"SOL appears to lead BTC" if sol_btc_lag < 0 else "BTC appears to lead SOL" if sol_btc_lag > 0 else "SOL and BTC move almost simultaneously"}
4. {"SOL appears to lead ETH" if sol_eth_lag < 0 else "ETH appears to lead SOL" if sol_eth_lag > 0 else "SOL and ETH move almost simultaneously"}

## 5. Next Steps

1. Build predictive models based on the discovered lead-lag relationships
2. Explore relationship changes under different market conditions (bull/bear markets)
3. Consider incorporating additional factors such as trading volume into the analysis
"""

with open('eda_summary_report.md', 'w') as f:
    f.write(report)

print("\nExploratory analysis complete! All charts and analysis report have been saved.")
print("Summary report saved to 'eda_summary_report.md'")

