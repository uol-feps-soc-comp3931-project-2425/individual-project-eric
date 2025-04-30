"""
Module for evaluation metrics in cryptocurrency price prediction.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error (MAE).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MAE value
    """
    return np.mean(np.abs(y_true - y_pred))

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error (RMSE).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        RMSE value
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Directional Accuracy (DA).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        DA value
    """
    direction_actual = np.sign(y_true)
    direction_pred = np.sign(y_pred)
    
    direction_actual[direction_actual == 0] = 1
    direction_pred[direction_pred == 0] = 1
    
    return np.mean(direction_actual == direction_pred)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     crypto_names: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Calculate all evaluation metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        crypto_names: List of cryptocurrency names
        
    Returns:
        Dictionary with evaluation metrics
    """
    mae = calculate_mae(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    
    metrics = {
        'mae': mae,
        'rmse': rmse
    }
    
    n_outputs = y_true.shape[1]
    
    if crypto_names is None:
        crypto_names = [f'crypto_{i}' for i in range(n_outputs)]
    
    for i in range(n_outputs):
        metrics[f'mae_{crypto_names[i]}'] = calculate_mae(y_true[:, i], y_pred[:, i])
        metrics[f'rmse_{crypto_names[i]}'] = calculate_rmse(y_true[:, i], y_pred[:, i])
        
        metrics[f'da_{crypto_names[i]}'] = calculate_directional_accuracy(y_true[:, i], y_pred[:, i])
    
    metrics['directional_accuracy'] = np.mean([metrics[f'da_{crypto}'] for crypto in crypto_names])
    
    return metrics

def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, 
                    crypto_names: List[str], output_dir: str, 
                    horizon: str, model_name: str) -> None:
    """
    Plot predictions vs actual values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        crypto_names: List of cryptocurrency names
        output_dir: Directory to save plots
        horizon: Prediction horizon
        model_name: Name of the model
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i, crypto in enumerate(crypto_names):
        plt.figure(figsize=(12, 6))
        
        n_points = min(200, len(y_true))
        indices = np.arange(len(y_true) - n_points, len(y_true))
        
        plt.plot(indices, y_true[indices, i], label='Actual')
        plt.plot(indices, y_pred[indices, i], label='Predicted')
        
        plt.title(f'{crypto} Log Return Predictions - {model_name} ({horizon})')
        plt.xlabel('Time Step')
        plt.ylabel('Log Return')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{crypto}_{horizon}_{model_name}_predictions.png'))
        plt.close()

def plot_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]], 
                          output_dir: str, horizons: List[str]) -> None:
    """
    Plot metrics comparison between models.
    
    Args:
        metrics_dict: Dictionary mapping model names to dictionaries of metrics
        output_dir: Directory to save plots
        horizons: List of prediction horizons
    """
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = ['mae', 'rmse', 'directional_accuracy']
    
    for metric in metrics:
        data = []
        
        for model_name, model_metrics in metrics_dict.items():
            for horizon in horizons:
                if f'{horizon}_{metric}' in model_metrics:
                    data.append({
                        'Model': model_name,
                        'Horizon': horizon,
                        metric.upper(): model_metrics[f'{horizon}_{metric}']
                    })
        
        df = pd.DataFrame(data)
        
        if not df.empty:
            plt.figure(figsize=(12, 6))
            
            sns.barplot(x='Horizon', y=metric.upper(), hue='Model', data=df)
            
            plt.title(f'{metric.upper()} Comparison')
            plt.xlabel('Prediction Horizon')
            plt.ylabel(metric.upper())
            plt.legend(title='Model')
            plt.grid(True, axis='y')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{metric}_comparison.png'))
            plt.close()

def save_metrics_to_csv(metrics_dict: Dict[str, Dict[str, float]], 
                      output_dir: str, filename: str = 'metrics_comparison.csv') -> None:
    """
    Save metrics comparison to CSV file.
    
    Args:
        metrics_dict: Dictionary mapping model names to dictionaries of metrics
        output_dir: Directory to save CSV file
        filename: Name of the CSV file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    data = []
    
    for model_name, model_metrics in metrics_dict.items():
        for metric_name, metric_value in model_metrics.items():
            data.append({
                'Model': model_name,
                'Metric': metric_name,
                'Value': metric_value
            })
    
    df = pd.DataFrame(data)
    
    csv_path = os.path.join(output_dir, filename)
    df.to_csv(csv_path, index=False)
    
    logger.info(f"Metrics saved to {csv_path}")
