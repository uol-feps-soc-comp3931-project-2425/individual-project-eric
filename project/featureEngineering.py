import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib
import warnings
warnings.filterwarnings('ignore')

print("Starting the feature engineering phase...")

try:
    crypto_data = pd.read_csv('processed_crypto_data.csv', index_col=0, parse_dates=True)
    print(f"Successfully loaded data, {len(crypto_data)} rows")
except FileNotFoundError:
    print("Processed data file not found. Please run the data preprocessing stage code first.")
    exit()

def add_technical_indicators(df):
    """Add technical analysis indicators for each cryptocurrency"""
    print("Adding technical indicators...")

    # List coins for which to add technical indicators
    coins = ['btc', 'eth', 'sol']

    for coin in coins:
        print(f"Adding technical indicators for {coin.upper()}...")

        # Ensure necessary columns exist
        required_cols = [f'{coin}_Close', f'{coin}_High', f'{coin}_Low', f'{coin}_Volume']
        if not all(col in df.columns for col in required_cols):
            print(f"Warning: {coin} is missing necessary price data columns, attempting to use available columns...")

            # If High/Low columns are missing, approximate using Close
            if f'{coin}_High' not in df.columns and f'{coin}_Close' in df.columns:
                df[f'{coin}_High'] = df[f'{coin}_Close']
                print(f"- Using Close price to substitute missing High price")

            if f'{coin}_Low' not in df.columns and f'{coin}_Close' in df.columns:
                df[f'{coin}_Low'] = df[f'{coin}_Close']
                print(f"- Using Close price to substitute missing Low price")

            if f'{coin}_Volume' not in df.columns:
                df[f'{coin}_Volume'] = 0
                print(f"- Missing Volume data, filling with 0")

        price_col = f'{coin}_Close'
        high_col = f'{coin}_High'
        low_col = f'{coin}_Low'
        volume_col = f'{coin}_Volume'

        try:
            # Moving Averages - Short, Medium, and Long term
            df[f'{coin}_ma7'] = df[price_col].rolling(window=7).mean()   
            df[f'{coin}_ma24'] = df[price_col].rolling(window=24).mean() 
            df[f'{coin}_ma72'] = df[price_col].rolling(window=72).mean() 
            df[f'{coin}_ma168'] = df[price_col].rolling(window=168).mean()

            # Percentage distance of price from moving averages
            df[f'{coin}_ma7_dist'] = (df[price_col] - df[f'{coin}_ma7']) / df[f'{coin}_ma7'] * 100
            df[f'{coin}_ma24_dist'] = (df[price_col] - df[f'{coin}_ma24']) / df[f'{coin}_ma24'] * 100
            df[f'{coin}_ma72_dist'] = (df[price_col] - df[f'{coin}_ma72']) / df[f'{coin}_ma72'] * 100

            # Exponential Moving Average (EMA)
            df[f'{coin}_ema12'] = df[price_col].ewm(span=12).mean()
            df[f'{coin}_ema26'] = df[price_col].ewm(span=26).mean()

            # Price Momentum (Price Change)
            df[f'{coin}_momentum_1h'] = df[price_col].diff(1)
            df[f'{coin}_momentum_3h'] = df[price_col].diff(3)
            df[f'{coin}_momentum_6h'] = df[price_col].diff(6)
            df[f'{coin}_momentum_12h'] = df[price_col].diff(12)
            df[f'{coin}_momentum_24h'] = df[price_col].diff(24)

            # Rate of Change (%)
            df[f'{coin}_roc_1h'] = df[price_col].pct_change(1) * 100
            df[f'{coin}_roc_3h'] = df[price_col].pct_change(3) * 100
            df[f'{coin}_roc_6h'] = df[price_col].pct_change(6) * 100
            df[f'{coin}_roc_12h'] = df[price_col].pct_change(12) * 100
            df[f'{coin}_roc_24h'] = df[price_col].pct_change(24) * 100

            # Volatility indicators
            df[f'{coin}_volatility_24h'] = df[price_col].rolling(window=24).std()
            df[f'{coin}_volatility_72h'] = df[price_col].rolling(window=72).std()

            # Volume change
            df[f'{coin}_volume_change'] = df[volume_col].pct_change()
            df[f'{coin}_volume_ma24'] = df[volume_col].rolling(window=24).mean()
            df[f'{coin}_volume_ratio'] = df[volume_col] / df[f'{coin}_volume_ma24']

            # Add advanced technical indicators using TA-Lib
            # MACD Indicator
            try:
                df[f'{coin}_macd'], df[f'{coin}_macd_signal'], df[f'{coin}_macd_hist'] = talib.MACD(
                    df[price_col], fastperiod=12, slowperiod=26, signalperiod=9
                )
                print(f"- Successfully added MACD indicator")
            except Exception as e:
                print(f"- Could not add MACD indicator: {e}")

            # RSI Indicator (Relative Strength Index)
            try:
                df[f'{coin}_rsi_6'] = talib.RSI(df[price_col], timeperiod=6)
                df[f'{coin}_rsi_12'] = talib.RSI(df[price_col], timeperiod=12)
                df[f'{coin}_rsi_24'] = talib.RSI(df[price_col], timeperiod=24)
                print(f"- Successfully added RSI indicator")
            except Exception as e:
                print(f"- Could not add RSI indicator: {e}")

            # Bollinger Bands
            try:
                df[f'{coin}_bb_upper'], df[f'{coin}_bb_middle'], df[f'{coin}_bb_lower'] = talib.BBANDS(
                    df[price_col], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
                )
                # Calculate relative position of price within Bollinger Bands (between 0-1)
                df[f'{coin}_bb_pos'] = (df[price_col] - df[f'{coin}_bb_lower']) / (df[f'{coin}_bb_upper'] - df[f'{coin}_bb_lower'])
                print(f"- Successfully added Bollinger Bands indicator")
            except Exception as e:
                print(f"- Could not add Bollinger Bands indicator: {e}")

            # ATR (Average True Range)
            try:
                df[f'{coin}_atr'] = talib.ATR(
                    df[high_col], df[low_col], df[price_col], timeperiod=14
                )
                print(f"- Successfully added ATR indicator")
            except Exception as e:
                print(f"- Could not add ATR indicator: {e}")

            # Stochastic Oscillator
            try:
                df[f'{coin}_slowk'], df[f'{coin}_slowd'] = talib.STOCH(
                    df[high_col], df[low_col], df[price_col],
                    fastk_period=14, slowk_period=3, slowd_period=3
                )
                print(f"- Successfully added Stochastic Oscillator indicator")
            except Exception as e:
                print(f"- Could not add Stochastic Oscillator indicator: {e}")

            # CCI (Commodity Channel Index)
            try:
                df[f'{coin}_cci'] = talib.CCI(
                    df[high_col], df[low_col], df[price_col], timeperiod=14
                )
                print(f"- Successfully added CCI indicator")
            except Exception as e:
                print(f"- Could not add CCI indicator: {e}")

            # OBV (On-Balance Volume)
            try:
                df[f'{coin}_obv'] = talib.OBV(df[price_col], df[volume_col])
                print(f"- Successfully added OBV indicator")
            except Exception as e:
                print(f"- Could not add OBV indicator: {e}")

            print(f"Successfully added technical indicators for {coin.upper()}")

        except Exception as e:
            print(f"Error adding technical indicators for {coin}: {e}")

    return df

def create_lag_features(df, lags_config):
    """
    Create lead-lag features based on configuration

    Parameters:
    df - DataFrame
    lags_config - Dictionary, format {'source_target': [lag1, lag2, ...]}
                  e.g., {'eth_btc': [1, 2, 3]} means create 1, 2, 3 hour lag features of ETH leading BTC
    """
    print("\nCreating lead-lag features...")

    for pair, lags in lags_config.items():
        source, target = pair.split('_')
        print(f"Creating lag features for {source.upper()}-{target.upper()} pair: {lags}")

        # Check if source and target coin columns exist
        if f'{source}_Close' not in df.columns or f'{source}_return' not in df.columns:
            print(f"Warning: Data columns for {source} not found, skipping this pair")
            continue

        for lag in lags:
            # Create lagged return feature (past returns of the source coin)
            df[f'{source}_{target}_return_lag{lag}'] = df[f'{source}_return'].shift(lag)

            # Create lagged price feature
            df[f'{source}_{target}_close_lag{lag}'] = df[f'{source}_Close'].shift(lag)

            # Price change (ratio of current price to lagged price)
            df[f'{source}_{target}_price_change_lag{lag}'] = df[f'{source}_Close'] / df[f'{source}_Close'].shift(lag) - 1

            # If volume data exists, add lagged volume feature
            if f'{source}_Volume' in df.columns:
                df[f'{source}_{target}_volume_lag{lag}'] = df[f'{source}_Volume'].shift(lag)


            for indicator in ['rsi_12', 'rsi_24', 'macd', 'bb_pos']:
                col_name = f'{source}_{indicator}'
                if col_name in df.columns:
                    df[f'{source}_{target}_{indicator}_lag{lag}'] = df[col_name].shift(lag)

    print("Lead-lag feature creation complete")
    return df

def create_target_variables(df, forecast_horizons=[1, 6, 12, 24]):
    """
    Create target variables for different forecast horizons

    Parameters:
    df - DataFrame
    forecast_horizons - List of forecast horizons (hours)
    """
    print("\nCreating target variables...")

    for coin in ['btc', 'eth', 'sol']:
        if f'{coin}_return' not in df.columns:
            print(f"Warning: Column {coin}_return not found, skipping target variable creation for this coin")
            continue

        print(f"Creating target variables for {coin.upper()}...")

        for horizon in forecast_horizons:
            # Future return
            df[f'{coin}_future_return_{horizon}h'] = df[f'{coin}_return'].shift(-horizon)

            # Future price movement direction (up/down: 1 if > 0, 0 otherwise)
            df[f'{coin}_future_direction_{horizon}h'] = (df[f'{coin}_future_return_{horizon}h'] > 0).astype(int)

            # Whether future price will have a significant change (more than 1% absolute return)
            df[f'{coin}_future_significant_{horizon}h'] = (df[f'{coin}_future_return_{horizon}h'].abs() > 1.0).astype(int)

            # Classify magnitude of future price change (-2: large drop, -1: small drop, 0: sideways, 1: small rise, 2: large rise)
            conditions = [
                (df[f'{coin}_future_return_{horizon}h'] <= -2.0), # Large drop (more than 2%)
                (df[f'{coin}_future_return_{horizon}h'] > -2.0) & (df[f'{coin}_future_return_{horizon}h'] <= -0.5), # Small drop
                (df[f'{coin}_future_return_{horizon}h'] > -0.5) & (df[f'{coin}_future_return_{horizon}h'] < 0.5),  # Sideways
                (df[f'{coin}_future_return_{horizon}h'] >= 0.5) & (df[f'{coin}_future_return_{horizon}h'] < 2.0),  # Small rise
                (df[f'{coin}_future_return_{horizon}h'] >= 2.0)   # Large rise (more than 2%)
            ]
            choices = [-2, -1, 0, 1, 2]
            df[f'{coin}_future_movement_{horizon}h'] = np.select(conditions, choices, default=0)

    print("Target variable creation complete")
    return df

def add_correlation_features(df, window_sizes=[24, 72, 168]):
    """
    Add rolling correlation features between coins

    Parameters:
    df - DataFrame
    window_sizes - List of rolling window sizes (hours)
    """
    print("\nAdding inter-coin correlation features...")

    # Check if all necessary return columns exist
    required_cols = ['btc_return', 'eth_return', 'sol_return']
    if not all(col in df.columns for col in required_cols):
        print("Warning: Missing necessary return columns, skipping correlation feature addition")
        return df

    # Create list of coin pairs
    pairs = [
        ('btc', 'eth'),
        ('btc', 'sol'),
        ('eth', 'sol')
    ]

    for window in window_sizes:
        print(f"Calculating rolling correlation for window size {window} hours...")

        for coin1, coin2 in pairs:
            # Rolling correlation coefficient
            col_name = f'{coin1}_{coin2}_corr_{window}h'
            df[col_name] = df[f'{coin1}_return'].rolling(window=window).corr(df[f'{coin2}_return'])


            if window >= 72:
                 # Calculate rolling standard deviations if they don't exist
                std_dev1_col = f'{coin1}_rolling_std_{window}h'
                std_dev2_col = f'{coin2}_rolling_std_{window}h'
                if std_dev1_col not in df.columns:
                    df[std_dev1_col] = df[f'{coin1}_return'].rolling(window).std()
                if std_dev2_col not in df.columns:
                     df[std_dev2_col] = df[f'{coin2}_return'].rolling(window).std()

                # Calculate Beta, handle division by zero
                df[f'{coin1}_{coin2}_beta_{window}h'] = df[col_name] * (df[std_dev2_col] / df[std_dev1_col].replace(0, np.nan))


    print("Inter-coin correlation feature addition complete")
    return df

def add_market_state_features(df):
    """Add features describing the overall market state"""
    print("\nAdding market state features...")

    # Check if necessary columns exist (using BTC as market proxy)
    if 'btc_Close' not in df.columns:
        print("Warning: BTC price data not found, skipping market state feature addition")
        return df

    for period in [24, 72, 168]: # 1 day, 3 days, 7 days
        ma_col = f'btc_ma{period}'
        if ma_col not in df.columns:
            print(f"Calculating missing {ma_col} for market state...")
            df[ma_col] = df['btc_Close'].rolling(window=period).mean()

        # Position of price relative to the moving average (1 if above, 0 if below)
        df[f'market_trend_{period}h'] = (df['btc_Close'] > df[ma_col]).astype(int)


    df['market_strong_uptrend'] = (
        (df['btc_Close'] > df['btc_ma24']) &
        (df['btc_ma24'] > df['btc_ma72']) &
        (df['btc_ma72'] > df['btc_ma168']) &
        (df['btc_Close'] > 1.05 * df['btc_ma24']) 
    ).astype(int)

    # Strong downtrend: Price is significantly below all moving averages (MA order: short < mid < long)
    df['market_strong_downtrend'] = (
        (df['btc_Close'] < df['btc_ma24']) &
        (df['btc_ma24'] < df['btc_ma72']) &
        (df['btc_ma72'] < df['btc_ma168']) &
        (df['btc_Close'] < 0.95 * df['btc_ma24']) # Price is at least 5% below the 24-hour MA
    ).astype(int)

    # Sideways/Consolidation: Price fluctuates around a medium-term moving average (e.g., 72h MA)
    df['market_sideways'] = (
        (df['btc_Close'] > 0.98 * df['btc_ma72']) &
        (df['btc_Close'] < 1.02 * df['btc_ma72'])
    ).astype(int)

    # Market volatility (based on 24-hour standard deviation of BTC returns)
    btc_return_col = 'btc_return'
    btc_vol_col = 'btc_volatility_24h_return' # Use volatility of returns, not price
    if btc_return_col in df.columns:
         if btc_vol_col not in df.columns:
             print(f"Calculating missing {btc_vol_col} for market state...")
             df[btc_vol_col] = df[btc_return_col].rolling(window=24).std()

         if btc_vol_col in df.columns: # Check again after potential calculation
            # Determine thresholds for high and low volatility based on quantiles
            volatility_low_threshold = df[btc_vol_col].quantile(0.25)
            volatility_high_threshold = df[btc_vol_col].quantile(0.75)

            df['market_high_volatility'] = (df[btc_vol_col] > volatility_high_threshold).astype(int)
            df['market_low_volatility'] = (df[btc_vol_col] < volatility_low_threshold).astype(int)
    else:
        print("Warning: BTC return column not found, cannot calculate market volatility state.")


    print("Market state feature addition complete")
    return df

def add_cyclical_features(df):
    """Add time-related cyclical features"""
    print("\nAdding cyclical features...")

    # Ensure the index is of DatetimeIndex type
    if not isinstance(df.index, pd.DatetimeIndex):
        print("Warning: Index is not DatetimeIndex type, attempting conversion...")
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            print(f"Error: Cannot convert index to DatetimeIndex type: {e}. Skipping cyclical feature addition.")
            return df

    # Hour feature (cycle within a day)
    df['hour'] = df.index.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Day of week feature (cycle within a week)
    df['day_of_week'] = df.index.dayofweek # Monday=0, Sunday=6
    # Convert day of week to cyclical feature
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Month feature (cycle within a year)
    df['month'] = df.index.month
    # Convert month to cyclical feature
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Is weekend (Saturday or Sunday)
    df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)


    london_hours = (df.index.hour >= 8) & (df.index.hour < 16)
    df['is_london_hours'] = london_hours.astype(int)

    print("Cyclical feature addition complete")
    return df

lags_config = {
    'eth_btc': [1, 2, 3, 6, 12, 24], 
    'sol_btc': [1, 2, 3, 6, 12, 24], 
    'sol_eth': [1, 2, 3, 6, 12],     
    'btc_eth': [1, 3, 6],            
    'btc_sol': [1, 3, 6],            
    'eth_sol': [1, 3, 6]             
}

# Target prediction time horizons
forecast_horizons = [1, 3, 6, 12, 24] 

# Execute feature engineering steps
crypto_data = add_technical_indicators(crypto_data)
crypto_data = create_lag_features(crypto_data, lags_config)
crypto_data = create_target_variables(crypto_data, forecast_horizons)
crypto_data = add_correlation_features(crypto_data)
crypto_data = add_market_state_features(crypto_data)
crypto_data = add_cyclical_features(crypto_data)

nan_count_before = crypto_data.isna().sum().sum()
print(f"\nTotal NaN values before final cleaning: {nan_count_before}")

# Drop NaN values
crypto_data = crypto_data.dropna()

# Check data size after dropping NaN
print(f"Number of data rows after feature engineering and cleaning: {len(crypto_data)}")
print(f"Number of features: {len(crypto_data.columns)}")

output_file = 'crypto_features.csv'
crypto_data.to_csv(output_file)
print(f"\nFeature engineering complete! Data saved to {output_file}")

# Display stats for target variables
key_features = [col for col in crypto_data.columns if 'future_return' in col or 'future_direction' in col or 'future_movement' in col]
if key_features:
    print("\nTarget variable statistics:")
    print(crypto_data[key_features].describe().transpose()[['mean', 'std', 'min', 'max']])

# Show feature count by category (approximation based on column name patterns)
feature_categories = {
    'Technical Indicators': ['ma', 'ema', 'rsi', 'macd', 'bb_', 'atr', 'cci', 'slowk', 'slowd', 'obv', 'momentum', 'volatility', 'roc', 'dist'],
    'Lead-Lag Features': ['_lag'],
    'Target Variables': ['future_'],
    'Correlation Features': ['_corr_', '_beta_'],
    'Market State': ['market_'],
    'Cyclical Features': ['hour', 'day_of_week', 'month', 'is_weekend', 'is_london_hours', '_sin', '_cos'],
    'Base Data': ['_Close', '_High', '_Low', '_Open', '_Volume', '_return'] 
}

print("\nFeature Category Statistics (Approximate):")
total_features = 0
for category, patterns in feature_categories.items():
    count = sum(1 for col in crypto_data.columns if any(pattern in col for pattern in patterns))
    print(f"- {category}: {count} features")
    total_features += count 

print(f"Total columns counted (may include overlaps): {total_features}")
print(f"Actual total columns in DataFrame: {len(crypto_data.columns)}")