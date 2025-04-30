"""
Module for dataset preparation in cryptocurrency price prediction.
"""
import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

from ..utils.config import (
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, SEQUENCE_LENGTH
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetPreparation:
    """
    Prepares datasets for cryptocurrency price prediction models.
    
    As specified in the paper:
    1. Data partitioning: 70% training, 15% validation, 15% test
    2. Sequence generation: 60-minute time steps
    3. Feature selection and scaling
    """
    
    def __init__(self, features_dir: str = '../../data/features', 
                dataset_dir: str = '../../data/dataset',
                train_split: float = TRAIN_SPLIT,
                val_split: float = VAL_SPLIT,
                test_split: float = TEST_SPLIT,
                sequence_length: int = SEQUENCE_LENGTH):
        """
        Initialize the DatasetPreparation.
        
        Args:
            features_dir: Directory with engineered features
            dataset_dir: Directory to save prepared datasets
            train_split: Proportion of data for training
            val_split: Proportion of data for validation
            test_split: Proportion of data for testing
            sequence_length: Length of input sequences (in minutes)
        """
        self.features_dir = features_dir
        self.dataset_dir = dataset_dir
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.sequence_length = sequence_length
        
        os.makedirs(self.dataset_dir, exist_ok=True)
    
    def load_features(self, symbol: str, quote: str = 'USDT') -> pd.DataFrame:
        """
        Load engineered features from disk.
        
        Args:
            symbol: Cryptocurrency symbol
            quote: Quote currency
            
        Returns:
            DataFrame with engineered features
        """
        file_path = os.path.join(self.features_dir, f"{symbol}{quote}_features.csv")
        
        if not os.path.exists(file_path):
            logger.error(f"Features file not found: {file_path}")
            return pd.DataFrame()
        
        df = pd.read_csv(file_path)
        
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        return df
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into training, validation, and test sets.
        
        As specified in the paper:
        - 70% training, 15% validation, 15% test
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if df.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        df = df.sort_values('datetime')
        
        n = len(df)
        train_idx = int(n * self.train_split)
        val_idx = train_idx + int(n * self.val_split)
        
        train_df = df.iloc[:train_idx].copy()
        val_df = df.iloc[train_idx:val_idx].copy()
        test_df = df.iloc[val_idx:].copy()
        
        logger.info(f"Data split: {len(train_df)} training, {len(val_df)} validation, {len(test_df)} test")
        
        return train_df, val_df, test_df
    
    def scale_features(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                      test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
        """
        Scale features using StandardScaler.
        
        Args:
            train_df: Training data
            val_df: Validation data
            test_df: Test data
            
        Returns:
            Tuple of (scaled_train_df, scaled_val_df, scaled_test_df, scaler)
        """
        if train_df.empty or val_df.empty or test_df.empty:
            return train_df, val_df, test_df, None
        
        feature_cols = [col for col in train_df.columns if col not in ['datetime', 'symbol']]
        
        scaler = StandardScaler()
        
        scaler.fit(train_df[feature_cols])
        
        scaled_train = train_df.copy()
        scaled_val = val_df.copy()
        scaled_test = test_df.copy()
        
        scaled_train[feature_cols] = scaler.transform(train_df[feature_cols])
        scaled_val[feature_cols] = scaler.transform(val_df[feature_cols])
        scaled_test[feature_cols] = scaler.transform(test_df[feature_cols])
        
        return scaled_train, scaled_val, scaled_test, scaler
    
    def create_sequences(self, df: pd.DataFrame, target_col: str = 'log_return_1m', 
                        target_steps_ahead: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction.
        
        As specified in the paper:
        - Sequence length: 60 minutes
        
        Args:
            df: DataFrame with scaled features
            target_col: Column to predict
            target_steps_ahead: Number of steps ahead to predict
            
        Returns:
            Tuple of (X, y) where X is the input sequences and y is the target values
        """
        if df.empty:
            return np.array([]), np.array([])
        
        feature_cols = [col for col in df.columns if col not in ['datetime', 'symbol']]
        
        data = df[feature_cols].values
        
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length - target_steps_ahead + 1):
            X.append(data[i:i+self.sequence_length])
            
            target_idx = i + self.sequence_length + target_steps_ahead - 1
            target_col_idx = feature_cols.index(target_col)
            y.append(data[target_idx, target_col_idx])
        
        return np.array(X), np.array(y)
    
    def create_multivariate_sequences(self, dfs: Dict[str, pd.DataFrame], 
                                    target_symbols: List[str],
                                    target_col: str = 'log_return_1m', 
                                    target_steps_ahead: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create multivariate sequences for time series prediction.
        
        Args:
            dfs: Dictionary mapping symbols to DataFrames with scaled features
            target_symbols: List of symbols to predict
            target_col: Column to predict
            target_steps_ahead: Number of steps ahead to predict
            
        Returns:
            Tuple of (X, y) where X is the input sequences and y is the target values
        """
        if not dfs or not all(dfs.values()):
            return np.array([]), np.array([])
        
        aligned_dfs = {}
        for symbol, df in dfs.items():
            aligned_dfs[symbol] = df.set_index('datetime')
        
        common_index = aligned_dfs[list(aligned_dfs.keys())[0]].index
        for df in aligned_dfs.values():
            common_index = common_index.intersection(df.index)
        
        for symbol in aligned_dfs:
            aligned_dfs[symbol] = aligned_dfs[symbol].loc[common_index].reset_index()
        
        feature_cols = {}
        for symbol, df in aligned_dfs.items():
            feature_cols[symbol] = [col for col in df.columns if col not in ['datetime', 'symbol']]
        
        X, y = [], []
        
        min_length = min(len(df) for df in aligned_dfs.values())
        
        for i in range(min_length - self.sequence_length - target_steps_ahead + 1):
            seq = []
            for symbol, df in aligned_dfs.items():
                data = df[feature_cols[symbol]].values
                seq.append(data[i:i+self.sequence_length])
            
            seq = np.concatenate(seq, axis=1)
            X.append(seq)
            
            targets = []
            for symbol in target_symbols:
                data = aligned_dfs[symbol][feature_cols[symbol]].values
                target_idx = i + self.sequence_length + target_steps_ahead - 1
                target_col_idx = feature_cols[symbol].index(target_col)
                targets.append(data[target_idx, target_col_idx])
            
            y.append(targets)
        
        return np.array(X), np.array(y)
    
    def prepare_datasets(self, symbols: List[str], quote: str = 'USDT',
                        target_steps_ahead: List[int] = [1, 5, 15, 30]) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Prepare datasets for all cryptocurrencies and prediction horizons.
        
        Args:
            symbols: List of cryptocurrency symbols
            quote: Quote currency
            target_steps_ahead: List of prediction horizons (in minutes)
            
        Returns:
            Dictionary mapping prediction horizons to dictionaries of datasets
        """
        features_dict = {}
        for symbol in symbols:
            df = self.load_features(symbol, quote)
            if not df.empty:
                features_dict[symbol] = df
        
        if not features_dict:
            logger.error("No features data available")
            return {}
        
        split_data_dict = {}
        for symbol, df in features_dict.items():
            train_df, val_df, test_df = self.split_data(df)
            split_data_dict[symbol] = {
                'train': train_df,
                'val': val_df,
                'test': test_df
            }
        
        scaled_data_dict = {}
        scalers_dict = {}
        for symbol, data_dict in split_data_dict.items():
            scaled_train, scaled_val, scaled_test, scaler = self.scale_features(
                data_dict['train'], data_dict['val'], data_dict['test']
            )
            scaled_data_dict[symbol] = {
                'train': scaled_train,
                'val': scaled_val,
                'test': scaled_test
            }
            scalers_dict[symbol] = scaler
        
        datasets_dict = {}
        
        for steps_ahead in target_steps_ahead:
            train_X, train_y = self.create_multivariate_sequences(
                {s: scaled_data_dict[s]['train'] for s in symbols},
                target_symbols=symbols,
                target_col='log_return_1m',
                target_steps_ahead=steps_ahead
            )
            
            val_X, val_y = self.create_multivariate_sequences(
                {s: scaled_data_dict[s]['val'] for s in symbols},
                target_symbols=symbols,
                target_col='log_return_1m',
                target_steps_ahead=steps_ahead
            )
            
            test_X, test_y = self.create_multivariate_sequences(
                {s: scaled_data_dict[s]['test'] for s in symbols},
                target_symbols=symbols,
                target_col='log_return_1m',
                target_steps_ahead=steps_ahead
            )
            
            datasets_dict[f't+{steps_ahead}'] = {
                'train_X': train_X,
                'train_y': train_y,
                'val_X': val_X,
                'val_y': val_y,
                'test_X': test_X,
                'test_y': test_y
            }
            
            np.save(os.path.join(self.dataset_dir, f'train_X_t+{steps_ahead}.npy'), train_X)
            np.save(os.path.join(self.dataset_dir, f'train_y_t+{steps_ahead}.npy'), train_y)
            np.save(os.path.join(self.dataset_dir, f'val_X_t+{steps_ahead}.npy'), val_X)
            np.save(os.path.join(self.dataset_dir, f'val_y_t+{steps_ahead}.npy'), val_y)
            np.save(os.path.join(self.dataset_dir, f'test_X_t+{steps_ahead}.npy'), test_X)
            np.save(os.path.join(self.dataset_dir, f'test_y_t+{steps_ahead}.npy'), test_y)
            
            logger.info(f"Created datasets for t+{steps_ahead} prediction horizon")
            logger.info(f"  Train: X shape {train_X.shape}, y shape {train_y.shape}")
            logger.info(f"  Val: X shape {val_X.shape}, y shape {val_y.shape}")
            logger.info(f"  Test: X shape {test_X.shape}, y shape {test_y.shape}")
        
        return datasets_dict

class TensorflowDataGenerator:
    """
    Data generator for TensorFlow models.
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int = 64, shuffle: bool = True):
        """
        Initialize the TensorflowDataGenerator.
        
        Args:
            X: Input sequences
            y: Target values
            batch_size: Batch size
            shuffle: Whether to shuffle the data
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = X.shape[0]
        self.indices = np.arange(self.n_samples)
        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        """
        Get the number of batches per epoch.
        
        Returns:
            Number of batches
        """
        return int(np.ceil(self.n_samples / self.batch_size))
    
    def __getitem__(self, idx):
        """
        Get a batch of data.
        
        Args:
            idx: Batch index
            
        Returns:
            Tuple of (X_batch, y_batch)
        """
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, self.n_samples)
        batch_indices = self.indices[start_idx:end_idx]
        
        X_batch = self.X[batch_indices]
        y_batch = self.y[batch_indices]
        
        return X_batch, y_batch
    
    def on_epoch_end(self):
        """
        Called at the end of each epoch.
        """
        if self.shuffle:
            np.random.shuffle(self.indices)

class PyTorchDataset:
    """
    Dataset class for PyTorch models.
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize the PyTorchDataset.
        
        Args:
            X: Input sequences
            y: Target values
        """
        self.X = X
        self.y = y
    
    def __len__(self):
        """
        Get the number of samples.
        
        Returns:
            Number of samples
        """
        return len(self.X)
    
    def __getitem__(self, idx):
        """
        Get a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (X_sample, y_sample)
        """
        return self.X[idx], self.y[idx]

if __name__ == "__main__":
    from ..utils.config import CRYPTOCURRENCIES, QUOTE_CURRENCY
    
    dataset_preparation = DatasetPreparation(
        features_dir='../../data/features',
        dataset_dir='../../data/dataset'
    )
    
    datasets_dict = dataset_preparation.prepare_datasets(
        symbols=CRYPTOCURRENCIES,
        quote=QUOTE_CURRENCY,
        target_steps_ahead=[1, 5, 15, 30]
    )
