"""
Transformer model implementation for cryptocurrency price prediction.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
import math
from typing import Dict, List, Optional, Union, Tuple

from ...utils.config import (
    SEQUENCE_LENGTH, TRANSFORMER_DIM, TRANSFORMER_HEADS, TRANSFORMER_LAYERS,
    TRANSFORMER_FF_DIM, TRANSFORMER_LEARNING_RATE, TRANSFORMER_WEIGHT_DECAY,
    TRANSFORMER_BATCH_SIZE, TRANSFORMER_MAX_EPOCHS
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding as specified in the paper.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize the positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]

class TransformerModel(nn.Module):
    """
    Transformer model for cryptocurrency price prediction.
    
    As specified in Appendix D.5 of the paper:
    - Input Embedding: Dense layer projecting 15 input features to model dimension (d = 64)
    - Positional Encoding: Standard sinusoidal positional encoding
    - Encoder Blocks: 2 stacked standard Transformer encoder blocks
    - Multi-Head Self-Attention: 4 attention heads, dimension per head dk = d/4 = 16
    - Position-wise Feed-Forward Network: Two dense layers (256 → 64) with ReLU activation
    - Output Generation: Global average pooling followed by a dense layer with 3 units
    """
    
    def __init__(self, n_features: int = 15, n_outputs: int = 3,
                d_model: int = TRANSFORMER_DIM, nhead: int = TRANSFORMER_HEADS,
                num_layers: int = TRANSFORMER_LAYERS, dim_feedforward: int = TRANSFORMER_FF_DIM,
                dropout: float = 0.1):
        """
        Initialize the Transformer model.
        
        Args:
            n_features: Number of input features
            n_outputs: Number of output units
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
        """
        super(TransformerModel, self).__init__()
        
        self.embedding = nn.Linear(n_features, d_model)
        
        self.positional_encoding = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        self.output_layer = nn.Linear(d_model, n_outputs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Transformer model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_features)
            
        Returns:
            Output tensor of shape (batch_size, n_outputs)
        """
        x = self.embedding(x)
        
        x = self.positional_encoding(x)
        
        x = self.transformer_encoder(x)
        
        x = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
        x = self.global_avg_pool(x)  # (batch_size, d_model, 1)
        x = x.squeeze(-1)  # (batch_size, d_model)
        
        x = self.output_layer(x)
        
        return x

class TransformerTrainer:
    """
    Trainer for the Transformer model.
    
    As specified in Appendix D.5.2 of the paper:
    - Loss Function: Mean Squared Error (MSE)
    - Optimizer: AdamW with learning rate 10^-4 and weight decay 10^-2
    - Learning Rate Schedule: ReduceLROnPlateau
    - Early Stopping: Patience=10
    - Batch Size: 64
    - Epochs: Maximum 100
    - Gradient Clipping: Applied at global norm 1.0
    """
    
    def __init__(self, model_dir: str = '../../models/transformer',
                learning_rate: float = TRANSFORMER_LEARNING_RATE,
                weight_decay: float = TRANSFORMER_WEIGHT_DECAY,
                batch_size: int = TRANSFORMER_BATCH_SIZE,
                max_epochs: int = TRANSFORMER_MAX_EPOCHS,
                device: Optional[str] = None):
        """
        Initialize the Transformer trainer.
        
        Args:
            model_dir: Directory to save model checkpoints
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            batch_size: Batch size
            max_epochs: Maximum number of epochs
            device: Device to use for training ('cuda' or 'cpu')
        """
        self.model_dir = model_dir
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        os.makedirs(self.model_dir, exist_ok=True)
    
    def create_dataloaders(self, train_X: np.ndarray, train_y: np.ndarray,
                         val_X: np.ndarray, val_y: np.ndarray,
                         test_X: np.ndarray, test_y: np.ndarray) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders for training, validation, and testing.
        
        Args:
            train_X: Training input sequences
            train_y: Training target values
            val_X: Validation input sequences
            val_y: Validation target values
            test_X: Test input sequences
            test_y: Test target values
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        train_X_tensor = torch.tensor(train_X, dtype=torch.float32)
        train_y_tensor = torch.tensor(train_y, dtype=torch.float32)
        val_X_tensor = torch.tensor(val_X, dtype=torch.float32)
        val_y_tensor = torch.tensor(val_y, dtype=torch.float32)
        test_X_tensor = torch.tensor(test_X, dtype=torch.float32)
        test_y_tensor = torch.tensor(test_y, dtype=torch.float32)
        
        train_dataset = TensorDataset(train_X_tensor, train_y_tensor)
        val_dataset = TensorDataset(val_X_tensor, val_y_tensor)
        test_dataset = TensorDataset(test_X_tensor, test_y_tensor)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        return train_loader, val_loader, test_loader
    
    def train(self, model: TransformerModel, train_loader: DataLoader,
             val_loader: DataLoader) -> Dict[str, List[float]]:
        """
        Train the Transformer model.
        
        Args:
            model: Transformer model
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Dictionary with training history
        """
        model = model.to(self.device)
        
        criterion = nn.MSELoss()
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.2,
            patience=5,
            verbose=True
        )
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': []
        }
        
        logger.info("Training Transformer model...")
        
        for epoch in range(self.max_epochs):
            model.train()
            train_loss = 0.0
            train_mae = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = model(batch_X)
                
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
                train_mae += torch.mean(torch.abs(outputs - batch_y)).item() * batch_X.size(0)
            
            train_loss /= len(train_loader.dataset)
            train_mae /= len(train_loader.dataset)
            
            model.eval()
            val_loss = 0.0
            val_mae = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = model(batch_X)
                    
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item() * batch_X.size(0)
                    val_mae += torch.mean(torch.abs(outputs - batch_y)).item() * batch_X.size(0)
            
            val_loss /= len(val_loader.dataset)
            val_mae /= len(val_loader.dataset)
            
            scheduler.step(val_loss)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_mae'].append(train_mae)
            history['val_mae'].append(val_mae)
            
            logger.info(f"Epoch {epoch+1}/{self.max_epochs} - "
                      f"Train Loss: {train_loss:.6f}, Train MAE: {train_mae:.6f}, "
                      f"Val Loss: {val_loss:.6f}, Val MAE: {val_mae:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                self.save_model(model, os.path.join(self.model_dir, 'best_model.pth'))
                logger.info("Saved best model checkpoint")
            else:
                patience_counter += 1
                
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        logger.info("Transformer model training completed")
        
        self.load_model(model, os.path.join(self.model_dir, 'best_model.pth'))
        
        return history
    
    def evaluate(self, model: TransformerModel, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the Transformer model.
        
        Args:
            model: Transformer model
            test_loader: Test data loader
            
        Returns:
            Dictionary with evaluation metrics
        """
        model = model.to(self.device)
        
        model.eval()
        
        test_loss = 0.0
        test_mae = 0.0
        
        criterion = nn.MSELoss()
        
        all_preds = []
        all_targets = []
        
        logger.info("Evaluating Transformer model...")
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = model(batch_X)
                
                loss = criterion(outputs, batch_y)
                
                test_loss += loss.item() * batch_X.size(0)
                test_mae += torch.mean(torch.abs(outputs - batch_y)).item() * batch_X.size(0)
                
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
        
        test_loss /= len(test_loader.dataset)
        test_mae /= len(test_loader.dataset)
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
        
        direction_actual = np.sign(all_targets)
        direction_pred = np.sign(all_preds)
        
        direction_actual[direction_actual == 0] = 1
        direction_pred[direction_pred == 0] = 1
        
        da_per_crypto = []
        for i in range(all_targets.shape[1]):
            da = np.mean(direction_actual[:, i] == direction_pred[:, i])
            da_per_crypto.append(da)
        
        da_overall = np.mean(da_per_crypto)
        
        metrics = {
            'loss': test_loss,
            'mae': test_mae,
            'rmse': rmse,
            'directional_accuracy': da_overall
        }
        
        for i in range(all_targets.shape[1]):
            metrics[f'da_crypto_{i}'] = da_per_crypto[i]
        
        logger.info(f"Evaluation metrics: {metrics}")
        
        return metrics, all_preds
    
    def save_model(self, model: TransformerModel, filepath: str) -> None:
        """
        Save the Transformer model.
        
        Args:
            model: Transformer model
            filepath: Path to save the model
        """
        torch.save(model.state_dict(), filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, model: TransformerModel, filepath: str) -> None:
        """
        Load the Transformer model.
        
        Args:
            model: Transformer model
            filepath: Path to load the model from
        """
        if not os.path.exists(filepath):
            logger.error(f"Model file not found: {filepath}")
            return
        
        model.load_state_dict(torch.load(filepath, map_location=self.device))
        logger.info(f"Model loaded from {filepath}")

if __name__ == "__main__":
    model = TransformerModel()
    print(model)
