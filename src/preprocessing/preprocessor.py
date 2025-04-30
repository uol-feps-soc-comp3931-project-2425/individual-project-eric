"""
Module for preprocessing cryptocurrency data.
"""
import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta

from ..utils.config import (
    MAX_GAP_MINUTES, IQR_MULTIPLIER, VOLUME_ZSCORE_WINDOW
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Preprocesses cryptocurrency OHLCV data.
    
    As specified in Appendix D.1.2 of the paper:
    1. Timestamp Standardization
    2. Missing Value Handling
    3. Outlier Detection and Treatment
    4. Data Normalization
    """
    
    def __init__(self, data_dir: str = '../../data/raw', 
                processed_dir: str = '../../data/processed',
                max_gap_minutes: int = MAX_GAP_MINUTES,
                iqr_multiplier: float = IQR_MULTIPLIER,
                volume_zscore_window: int = VOLUME_ZSCORE_WINDOW):
        """
        Initialize the DataPreprocessor.
        
        Args:
            data_dir: Directory with raw data
            processed_dir: Directory to save processed data
            max_gap_minutes: Maximum gap in minutes for linear interpolation
            iqr_multiplier: Multiplier for IQR outlier detection
            volume_zscore_window: Window size for volume Z-score calculation (in minutes)
        """
        self.data_dir = data_dir
        self.processed_dir = processed_dir
        self.max_gap_minutes = max_gap_minutes
        self.iqr_multiplier = iqr_multiplier
        self.volume_zscore_window = volume_zscore_window
        
        os.makedirs(self.processed_dir, exist_ok=True)
    
    def load_data(self, symbol: str, quote: str = 'USDT', 
                 timeframe: str = '1m') -> pd.DataFrame:
        """
        Load raw data from disk.
        
        Args:
            symbol: Cryptocurrency symbol
            quote: Quote currency
            timeframe: Timeframe of the data
            
        Returns:
            DataFrame with raw data
        """
        file_path = os.path.join(self.data_dir, f"{symbol}{quote}_{timeframe}.csv")
        
        if not os.path.exists(file_path):
            logger.error(f"Data file not found: {file_path}")
            return pd.DataFrame()
        
        df = pd.read_csv(file_path)
        
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        return df
    
    def standardize_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize timestamps to UTC and align to minute boundaries.
        
        Args:
            df: DataFrame with raw data
            
        Returns:
            DataFrame with standardized timestamps
        """
        if df.empty:
            return df
        
        if df['datetime'].dt.tz is None:
            df['datetime'] = df['datetime'].dt.tz_localize('UTC')
        else:
            df['datetime'] = df['datetime'].dt.tz_convert('UTC')
        
        df['datetime'] = df['datetime'].dt.floor('min')
        
        df = df.sort_values('datetime')
        
        df = df.drop_duplicates(subset=['datetime'])
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the data.
        
        As specified in the paper:
        - For gaps up to 5 consecutive minutes, missing OHLC price values and Volume
          are filled using linear interpolation.
        - Hourly data blocks containing any gap longer than 5 minutes are removed.
        
        Args:
            df: DataFrame with standardized timestamps
            
        Returns:
            DataFrame with missing values handled
        """
        if df.empty:
            return df
        
        min_datetime = df['datetime'].min()
        max_datetime = df['datetime'].max()
        full_range = pd.date_range(start=min_datetime, end=max_datetime, freq='min', tz='UTC')
        
        df_reindexed = df.set_index('datetime').reindex(full_range)
        
        df_reindexed['gap_size'] = df_reindexed['open'].isnull().astype(int).groupby(
            df_reindexed['open'].notnull().cumsum()
        ).cumsum()
        
        rows_to_keep = df_reindexed['gap_size'] <= self.max_gap_minutes
        
        df_filled = df_reindexed[rows_to_keep].copy()
        
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df_filled[numeric_cols] = df_filled[numeric_cols].interpolate(method='linear', limit=self.max_gap_minutes)
        
        df_filled = df_filled.reset_index().rename(columns={'index': 'datetime'})
        df_filled = df_filled.drop(columns=['gap_size'])
        
        df_filled['hour'] = df_filled['datetime'].dt.floor('H')
        
        hourly_missing = df_reindexed.reset_index().rename(columns={'index': 'datetime'})
        hourly_missing['hour'] = hourly_missing['datetime'].dt.floor('H')
        hourly_missing['large_gap'] = hourly_missing['gap_size'] > self.max_gap_minutes
        
        hours_with_large_gaps = hourly_missing.groupby('hour')['large_gap'].any()
        hours_to_remove = hours_with_large_gaps[hours_with_large_gaps].index
        
        df_filled = df_filled[~df_filled['hour'].isin(hours_to_remove)]
        
        df_filled = df_filled.drop(columns=['hour'])
        
        return df_filled
    
    def detect_and_treat_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and treat outliers in the data.
        
        As specified in the paper:
        - Log returns (rt = ln(Pt/Pt-1)) are calculated.
        - Outliers are detected within rolling 1-hour windows using the IQR method.
        - Any return falling outside [Q25 - 3.0 * IQR, Q75 + 3.0 * IQR] is capped.
        
        Args:
            df: DataFrame with missing values handled
            
        Returns:
            DataFrame with outliers treated
        """
        if df.empty or len(df) < 2:
            return df
        
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        df_treated = df.copy()
        
        window_size = 60
        
        for i in range(window_size, len(df_treated)):
            window = df_treated['log_return'].iloc[i-window_size:i]
            
            q25 = window.quantile(0.25)
            q75 = window.quantile(0.75)
            iqr = q75 - q25
            
            lower_bound = q25 - self.iqr_multiplier * iqr
            upper_bound = q75 + self.iqr_multiplier * iqr
            
            current_return = df_treated['log_return'].iloc[i]
            if current_return < lower_bound:
                df_treated.loc[df_treated.index[i], 'log_return'] = lower_bound
            elif current_return > upper_bound:
                df_treated.loc[df_treated.index[i], 'log_return'] = upper_bound
        
        
        return df_treated
    
    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the data.
        
        As specified in the paper:
        - Log returns are used as the primary input for price change modeling.
        - Volume data is normalized using a rolling 24-hour z-score:
          Vnorm,t = (Vt - μroll,24h) / σroll,24h
        
        Args:
            df: DataFrame with outliers treated
            
        Returns:
            DataFrame with normalized data
        """
        if df.empty or len(df) < self.volume_zscore_window:
            return df
        
        df_normalized = df.copy()
        
        df_normalized['volume_mean'] = df_normalized['volume'].rolling(
            window=self.volume_zscore_window, min_periods=1
        ).mean()
        
        df_normalized['volume_std'] = df_normalized['volume'].rolling(
            window=self.volume_zscore_window, min_periods=1
        ).std()
        
        df_normalized['volume_zscore'] = (
            (df_normalized['volume'] - df_normalized['volume_mean']) / 
            df_normalized['volume_std'].replace(0, np.nan)
        ).fillna(0)  # Handle division by zero
        
        df_normalized = df_normalized.drop(columns=['volume_mean', 'volume_std'])
        
        return df_normalized
    
    def preprocess_data(self, symbol: str, quote: str = 'USDT', 
                       timeframe: str = '1m') -> pd.DataFrame:
        """
        Preprocess data for a single cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol
            quote: Quote currency
            timeframe: Timeframe of the data
            
        Returns:
            DataFrame with preprocessed data
        """
        df = self.load_data(symbol, quote, timeframe)
        
        if df.empty:
            logger.error(f"No data to preprocess for {symbol}{quote}")
            return df
        
        df = self.standardize_timestamps(df)
        df = self.handle_missing_values(df)
        df = self.detect_and_treat_outliers(df)
        df = self.normalize_data(df)
        
        output_path = os.path.join(self.processed_dir, f"{symbol}{quote}_preprocessed.csv")
        df.to_csv(output_path, index=False)
        logger.info(f"Saved preprocessed data to {output_path}")
        
        return df
    
    def preprocess_all_data(self, symbols: List[str], quote: str = 'USDT', 
                          timeframe: str = '1m') -> Dict[str, pd.DataFrame]:
        """
        Preprocess data for multiple cryptocurrencies.
        
        Args:
            symbols: List of cryptocurrency symbols
            quote: Quote currency
            timeframe: Timeframe of the data
            
        Returns:
            Dictionary mapping symbols to DataFrames with preprocessed data
        """
        preprocessed_data = {}
        
        for symbol in symbols:
            df = self.preprocess_data(symbol, quote, timeframe)
            
            if not df.empty:
                preprocessed_data[symbol] = df
        
        return preprocessed_data

if __name__ == "__main__":
    from ..utils.config import CRYPTOCURRENCIES, QUOTE_CURRENCY, TIMEFRAME
    
    preprocessor = DataPreprocessor(
        data_dir='../../data/raw',
        processed_dir='../../data/processed'
    )
    
    preprocessed_data = preprocessor.preprocess_all_data(
        symbols=CRYPTOCURRENCIES,
        quote=QUOTE_CURRENCY,
        timeframe=TIMEFRAME
    )
