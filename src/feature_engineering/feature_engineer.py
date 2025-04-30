"""
Module for feature engineering in cryptocurrency price prediction.
"""
import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta

from ..utils.config import (
    LOG_RETURN_LAGS, VOLATILITY_WINDOWS, RSI_PERIOD, LAGGED_RETURNS
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Engineers features for cryptocurrency price prediction.
    
    As specified in Appendix D.2 of the paper:
    1. Log Returns at different time intervals
    2. Realized Volatility
    3. Relative Strength Index (RSI)
    4. Volume Z-Score (calculated in preprocessing)
    5. Lagged Log Returns
    """
    
    def __init__(self, processed_dir: str = '../../data/processed', 
                features_dir: str = '../../data/features',
                log_return_lags: List[int] = LOG_RETURN_LAGS,
                volatility_windows: List[int] = VOLATILITY_WINDOWS,
                rsi_period: int = RSI_PERIOD,
                lagged_returns: List[int] = LAGGED_RETURNS):
        """
        Initialize the FeatureEngineer.
        
        Args:
            processed_dir: Directory with preprocessed data
            features_dir: Directory to save engineered features
            log_return_lags: Time lags for log returns (in minutes)
            volatility_windows: Window sizes for realized volatility (in minutes)
            rsi_period: Period for RSI calculation
            lagged_returns: Lagged log returns (in minutes)
        """
        self.processed_dir = processed_dir
        self.features_dir = features_dir
        self.log_return_lags = log_return_lags
        self.volatility_windows = volatility_windows
        self.rsi_period = rsi_period
        self.lagged_returns = lagged_returns
        
        os.makedirs(self.features_dir, exist_ok=True)
    
    def load_preprocessed_data(self, symbol: str, quote: str = 'USDT') -> pd.DataFrame:
        """
        Load preprocessed data from disk.
        
        Args:
            symbol: Cryptocurrency symbol
            quote: Quote currency
            
        Returns:
            DataFrame with preprocessed data
        """
        file_path = os.path.join(self.processed_dir, f"{symbol}{quote}_preprocessed.csv")
        
        if not os.path.exists(file_path):
            logger.error(f"Preprocessed data file not found: {file_path}")
            return pd.DataFrame()
        
        df = pd.read_csv(file_path)
        
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        return df
    
    def calculate_log_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate log returns at different time intervals.
        
        As specified in the paper:
        - Log Returns (r_t^(τ)): Calculated as r_t^(τ) = ln(P_t / P_{t-τ}) for time lags
          τ ∈ {1, 5, 15} minutes for the Close price (P_t).
        
        Args:
            df: DataFrame with preprocessed data
            
        Returns:
            DataFrame with log returns features
        """
        if df.empty or len(df) <= max(self.log_return_lags):
            return df
        
        df_features = df.copy()
        
        for lag in self.log_return_lags:
            col_name = f'log_return_{lag}m'
            df_features[col_name] = np.log(df_features['close'] / df_features['close'].shift(lag))
        
        return df_features
    
    def calculate_realized_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate realized volatility over rolling windows.
        
        As specified in the paper:
        - Realized Volatility (σ_t^(n)): Calculated as the standard deviation of the 1-minute
          log returns over rolling windows of n ∈ {15, 60} minutes.
        
        Args:
            df: DataFrame with log returns
            
        Returns:
            DataFrame with realized volatility features
        """
        if df.empty or 'log_return_1m' not in df.columns or len(df) <= max(self.volatility_windows):
            return df
        
        df_features = df.copy()
        
        for window in self.volatility_windows:
            col_name = f'realized_vol_{window}m'
            df_features[col_name] = df_features['log_return_1m'].rolling(
                window=window, min_periods=1
            ).std()
        
        return df_features
    
    def calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Relative Strength Index (RSI).
        
        As specified in the paper:
        - RSI: Standard 14-period RSI calculated on the 1-minute Close prices.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with RSI feature
        """
        if df.empty or len(df) <= self.rsi_period:
            return df
        
        df_features = df.copy()
        
        delta = df_features['close'].diff()
        
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=self.rsi_period, min_periods=1).mean()
        avg_loss = loss.rolling(window=self.rsi_period, min_periods=1).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        
        df_features['rsi'] = 100 - (100 / (1 + rs)).fillna(50)  # Default to 50 for NaN values
        
        return df_features
    
    def calculate_lagged_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate lagged log returns.
        
        As specified in the paper:
        - Lagged Log Returns: Previous 1-minute log returns were included as features:
          r_{t-k}^(1) for k ∈ {1, 5, 10}.
        
        Args:
            df: DataFrame with log returns
            
        Returns:
            DataFrame with lagged log returns features
        """
        if df.empty or 'log_return_1m' not in df.columns or len(df) <= max(self.lagged_returns):
            return df
        
        df_features = df.copy()
        
        for lag in self.lagged_returns:
            col_name = f'lagged_return_t-{lag}'
            df_features[col_name] = df_features['log_return_1m'].shift(lag)
        
        return df_features
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer all features for a single DataFrame.
        
        Args:
            df: DataFrame with preprocessed data
            
        Returns:
            DataFrame with all engineered features
        """
        if df.empty:
            return df
        
        df = self.calculate_log_returns(df)
        df = self.calculate_realized_volatility(df)
        df = self.calculate_rsi(df)
        df = self.calculate_lagged_returns(df)
        
        df = df.dropna()
        
        return df
    
    def engineer_features_for_symbol(self, symbol: str, quote: str = 'USDT') -> pd.DataFrame:
        """
        Engineer features for a single cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol
            quote: Quote currency
            
        Returns:
            DataFrame with engineered features
        """
        df = self.load_preprocessed_data(symbol, quote)
        
        if df.empty:
            logger.error(f"No preprocessed data to engineer features for {symbol}{quote}")
            return df
        
        df_features = self.engineer_features(df)
        
        output_path = os.path.join(self.features_dir, f"{symbol}{quote}_features.csv")
        df_features.to_csv(output_path, index=False)
        logger.info(f"Saved engineered features to {output_path}")
        
        return df_features
    
    def engineer_features_for_all_symbols(self, symbols: List[str], 
                                        quote: str = 'USDT') -> Dict[str, pd.DataFrame]:
        """
        Engineer features for multiple cryptocurrencies.
        
        Args:
            symbols: List of cryptocurrency symbols
            quote: Quote currency
            
        Returns:
            Dictionary mapping symbols to DataFrames with engineered features
        """
        features_dict = {}
        
        for symbol in symbols:
            df_features = self.engineer_features_for_symbol(symbol, quote)
            
            if not df_features.empty:
                features_dict[symbol] = df_features
        
        return features_dict

if __name__ == "__main__":
    from ..utils.config import CRYPTOCURRENCIES, QUOTE_CURRENCY
    
    feature_engineer = FeatureEngineer(
        processed_dir='../../data/processed',
        features_dir='../../data/features'
    )
    
    features_dict = feature_engineer.engineer_features_for_all_symbols(
        symbols=CRYPTOCURRENCIES,
        quote=QUOTE_CURRENCY
    )
