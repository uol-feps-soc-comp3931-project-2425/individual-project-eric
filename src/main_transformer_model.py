"""
Main script for Transformer model training and evaluation.
"""
import os
import numpy as np
import logging
import matplotlib.pyplot as plt
from datetime import datetime
import torch

from src.models.transformer.transformer_model import TransformerModel, TransformerTrainer
from src.utils.config import (
    CRYPTOCURRENCIES, SEQUENCE_LENGTH, TRANSFORMER_BATCH_SIZE, TRANSFORMER_MAX_EPOCHS,
    TRANSFORMER_DIM, TRANSFORMER_HEADS, TRANSFORMER_LAYERS, TRANSFORMER_FF_DIM
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("transformer_model.log"),
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

def plot_training_history(history, output_dir: str):
    """
    Plot training history.
    
    Args:
        history: Training history
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_mae'])
    plt.plot(history['val_mae'])
    plt.title('Model MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()

def plot_predictions(y_true, y_pred, cryptocurrencies, output_dir: str, horizon: str):
    """
    Plot predictions vs actual values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        cryptocurrencies: List of cryptocurrency symbols
        output_dir: Directory to save plots
        horizon: Prediction horizon
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i, crypto in enumerate(cryptocurrencies):
        plt.figure(figsize=(12, 6))
        
        n_points = min(200, len(y_true))
        indices = np.arange(len(y_true) - n_points, len(y_true))
        
        plt.plot(indices, y_true[indices, i], label='Actual')
        plt.plot(indices, y_pred[indices, i], label='Predicted')
        
        plt.title(f'{crypto} Log Return Predictions ({horizon})')
        plt.xlabel('Time Step')
        plt.ylabel('Log Return')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{crypto}_{horizon}_predictions.png'))
        plt.close()

def main():
    """
    Main function to train and evaluate the Transformer model.
    """
    os.makedirs('data/dataset', exist_ok=True)
    os.makedirs('models/transformer', exist_ok=True)
    os.makedirs('results/transformer', exist_ok=True)
    
    horizons = ['t+1', 't+5', 't+15', 't+30']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
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
        
        _, sequence_length, n_features = train_X.shape
        n_outputs = train_y.shape[1]
        
        model = TransformerModel(
            n_features=n_features,
            n_outputs=n_outputs,
            d_model=TRANSFORMER_DIM,
            nhead=TRANSFORMER_HEADS,
            num_layers=TRANSFORMER_LAYERS,
            dim_feedforward=TRANSFORMER_FF_DIM
        )
        
        logger.info(f"Transformer model architecture:")
        logger.info(f"{model}")
        logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
        
        trainer = TransformerTrainer(
            model_dir=f'models/transformer/{horizon}',
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
        
        plot_training_history(
            history=history,
            output_dir=f'results/transformer/{horizon}'
        )
        
        metrics, y_pred = trainer.evaluate(
            model=model,
            test_loader=test_loader
        )
        
        with open(f'results/transformer/{horizon}/metrics.txt', 'w') as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
        
        plot_predictions(
            y_true=test_y,
            y_pred=y_pred,
            cryptocurrencies=CRYPTOCURRENCIES,
            output_dir=f'results/transformer/{horizon}',
            horizon=horizon
        )
        
        trainer.save_model(
            model=model,
            filepath=os.path.join(f'models/transformer/{horizon}', 'transformer_model.pth')
        )
        
        logger.info(f"Completed {horizon} prediction horizon")
    
    logger.info("Transformer model training and evaluation completed")

if __name__ == "__main__":
    main()
