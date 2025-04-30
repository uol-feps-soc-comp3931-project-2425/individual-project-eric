"""
Main script for training and evaluating cryptocurrency price prediction models.
"""
import os
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import tensorflow as tf
import torch
from typing import Dict, List, Optional, Union, Tuple

from src.models.lstm.lstm_model import LSTMModel
from src.models.transformer.transformer_model import TransformerModel, TransformerTrainer
from src.evaluation.metrics import (
    calculate_metrics, plot_predictions, plot_metrics_comparison, save_metrics_to_csv
)
from src.utils.config import (
    CRYPTOCURRENCIES, SEQUENCE_LENGTH, 
    LSTM_BATCH_SIZE, LSTM_MAX_EPOCHS,
    TRANSFORMER_BATCH_SIZE, TRANSFORMER_MAX_EPOCHS
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_dataset(dataset_dir: str, horizon: str = 't+1') -> tuple:
    """
    Load dataset for a specific prediction horizon.
    
    Args:
        dataset_dir: Directory with datasets
        horizon: Prediction horizon (e.g., 't+1', 't+5')
        
    Returns:
        Tuple of (train_X, train_y, val_X, val_y, test_X, test_y)
    """
    train_X = np.load(os.path.join(dataset_dir, f'train_X_{horizon}.npy'))
    train_y = np.load(os.path.join(dataset_dir, f'train_y_{horizon}.npy'))
    val_X = np.load(os.path.join(dataset_dir, f'val_X_{horizon}.npy'))
    val_y = np.load(os.path.join(dataset_dir, f'val_y_{horizon}.npy'))
    test_X = np.load(os.path.join(dataset_dir, f'test_X_{horizon}.npy'))
    test_y = np.load(os.path.join(dataset_dir, f'test_y_{horizon}.npy'))
    
    return train_X, train_y, val_X, val_y, test_X, test_y

def train_lstm_model(train_X: np.ndarray, train_y: np.ndarray,
                   val_X: np.ndarray, val_y: np.ndarray,
                   test_X: np.ndarray, test_y: np.ndarray,
                   model_dir: str, horizon: str) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Train and evaluate LSTM model.
    
    Args:
        train_X: Training input sequences
        train_y: Training target values
        val_X: Validation input sequences
        val_y: Validation target values
        test_X: Test input sequences
        test_y: Test target values
        model_dir: Directory to save model
        horizon: Prediction horizon
        
    Returns:
        Tuple of (metrics, predictions)
    """
    _, sequence_length, n_features = train_X.shape
    n_outputs = train_y.shape[1]
    
    model = LSTMModel(
        model_dir=model_dir,
        sequence_length=sequence_length,
        n_features=n_features,
        n_outputs=n_outputs
    )
    
    model.summary()
    
    history = model.train(
        train_X=train_X,
        train_y=train_y,
        val_X=val_X,
        val_y=val_y,
        batch_size=LSTM_BATCH_SIZE,
        epochs=LSTM_MAX_EPOCHS
    )
    
    metrics = model.evaluate(
        test_X=test_X,
        test_y=test_y
    )
    
    y_pred = model.predict(test_X)
    
    model.save()
    
    metrics_with_horizon = {}
    for key, value in metrics.items():
        metrics_with_horizon[f'{horizon}_{key}'] = value
    
    return metrics_with_horizon, y_pred

def train_transformer_model(train_X: np.ndarray, train_y: np.ndarray,
                          val_X: np.ndarray, val_y: np.ndarray,
                          test_X: np.ndarray, test_y: np.ndarray,
                          model_dir: str, horizon: str) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Train and evaluate Transformer model.
    
    Args:
        train_X: Training input sequences
        train_y: Training target values
        val_X: Validation input sequences
        val_y: Validation target values
        test_X: Test input sequences
        test_y: Test target values
        model_dir: Directory to save model
        horizon: Prediction horizon
        
    Returns:
        Tuple of (metrics, predictions)
    """
    _, sequence_length, n_features = train_X.shape
    n_outputs = train_y.shape[1]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = TransformerModel(
        n_features=n_features,
        n_outputs=n_outputs
    )
    
    logger.info(f"Transformer model architecture:")
    logger.info(f"{model}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    trainer = TransformerTrainer(
        model_dir=model_dir,
        device=device
    )
    
    train_loader, val_loader, test_loader = trainer.create_dataloaders(
        train_X=train_X,
        train_y=train_y,
        val_X=val_X,
        val_y=val_y,
        test_X=test_X,
        test_y=test_y
    )
    
    history = trainer.train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    metrics, y_pred = trainer.evaluate(
        model=model,
        test_loader=test_loader
    )
    
    trainer.save_model(
        model=model,
        filepath=os.path.join(model_dir, 'transformer_model.pth')
    )
    
    metrics_with_horizon = {}
    for key, value in metrics.items():
        metrics_with_horizon[f'{horizon}_{key}'] = value
    
    return metrics_with_horizon, y_pred

def main():
    """
    Main function to train and evaluate cryptocurrency price prediction models.
    """
    os.makedirs('data/dataset', exist_ok=True)
    os.makedirs('models/lstm', exist_ok=True)
    os.makedirs('models/transformer', exist_ok=True)
    os.makedirs('results/lstm', exist_ok=True)
    os.makedirs('results/transformer', exist_ok=True)
    os.makedirs('results/comparison', exist_ok=True)
    
    horizons = ['t+1', 't+5', 't+15', 't+30']
    
    lstm_metrics = {}
    transformer_metrics = {}
    
    for horizon in horizons:
        logger.info(f"Processing {horizon} prediction horizon")
        
        try:
            train_X, train_y, val_X, val_y, test_X, test_y = load_dataset(
                dataset_dir='data/dataset',
                horizon=horizon
            )
            
            logger.info(f"Dataset shapes for {horizon}:")
            logger.info(f"  Train: X {train_X.shape}, y {train_y.shape}")
            logger.info(f"  Val: X {val_X.shape}, y {val_y.shape}")
            logger.info(f"  Test: X {test_X.shape}, y {test_y.shape}")
        except FileNotFoundError as e:
            logger.error(f"Dataset not found for {horizon}: {e}")
            continue
        
        logger.info(f"Training LSTM model for {horizon}")
        lstm_model_dir = f'models/lstm/{horizon}'
        lstm_horizon_metrics, lstm_predictions = train_lstm_model(
            train_X=train_X,
            train_y=train_y,
            val_X=val_X,
            val_y=val_y,
            test_X=test_X,
            test_y=test_y,
            model_dir=lstm_model_dir,
            horizon=horizon
        )
        
        lstm_metrics.update(lstm_horizon_metrics)
        
        plot_predictions(
            y_true=test_y,
            y_pred=lstm_predictions,
            crypto_names=CRYPTOCURRENCIES,
            output_dir=f'results/lstm/{horizon}',
            horizon=horizon,
            model_name='LSTM'
        )
        
        logger.info(f"Training Transformer model for {horizon}")
        transformer_model_dir = f'models/transformer/{horizon}'
        transformer_horizon_metrics, transformer_predictions = train_transformer_model(
            train_X=train_X,
            train_y=train_y,
            val_X=val_X,
            val_y=val_y,
            test_X=test_X,
            test_y=test_y,
            model_dir=transformer_model_dir,
            horizon=horizon
        )
        
        transformer_metrics.update(transformer_horizon_metrics)
        
        plot_predictions(
            y_true=test_y,
            y_pred=transformer_predictions,
            crypto_names=CRYPTOCURRENCIES,
            output_dir=f'results/transformer/{horizon}',
            horizon=horizon,
            model_name='Transformer'
        )
        
        logger.info(f"Completed {horizon} prediction horizon")
    
    metrics_dict = {
        'LSTM': lstm_metrics,
        'Transformer': transformer_metrics
    }
    
    plot_metrics_comparison(
        metrics_dict=metrics_dict,
        output_dir='results/comparison',
        horizons=horizons
    )
    
    save_metrics_to_csv(
        metrics_dict=metrics_dict,
        output_dir='results/comparison'
    )
    
    logger.info("Model training and evaluation completed")

if __name__ == "__main__":
    main()
