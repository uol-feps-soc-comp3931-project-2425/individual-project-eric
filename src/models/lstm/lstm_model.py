"""
LSTM model implementation for cryptocurrency price prediction.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Bidirectional, Dense, Dropout, LayerNormalization,
    Concatenate, Attention, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
)
from tensorflow.keras.optimizers import Adam
import logging
from typing import Dict, List, Optional, Union, Tuple

from ...utils.config import (
    SEQUENCE_LENGTH, LSTM_UNITS, LSTM_LAYERS, LSTM_DROPOUT,
    LSTM_DENSE_UNITS, LSTM_LEARNING_RATE, LSTM_BATCH_SIZE, LSTM_MAX_EPOCHS
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BahdanauAttention(tf.keras.layers.Layer):
    """
    Bahdanau-style attention mechanism as specified in the paper.
    """
    
    def __init__(self, units):
        """
        Initialize the attention layer.
        
        Args:
            units: Number of attention units
        """
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
    
    def call(self, query, values):
        """
        Apply attention mechanism.
        
        Args:
            query: Query tensor
            values: Values tensor
            
        Returns:
            Context vector and attention weights
        """
        
        hidden_with_time_axis = tf.expand_dims(query, 1)
        
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)
        ))
        
        attention_weights = tf.nn.softmax(score, axis=1)
        
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights

class LSTMModel:
    """
    LSTM model for cryptocurrency price prediction.
    
    As specified in Appendix D.3 of the paper:
    - Input Layer: Accepts sequences of length 60 (minutes) with 15 features
    - Bidirectional LSTM Layers: Two stacked BiLSTM layers, each with 64 units
    - Attention Layer: Bahdanau-style additive attention mechanism
    - Feedforward Network: Two dense layers (64 units, then 32 units) with ReLU activation
    - Output Layer: A final dense layer with 3 units (linear activation)
    """
    
    def __init__(self, model_dir: str = '../../models/lstm',
                sequence_length: int = SEQUENCE_LENGTH,
                n_features: int = 15,
                n_outputs: int = 3,
                lstm_units: int = LSTM_UNITS,
                lstm_layers: int = LSTM_LAYERS,
                dropout_rate: float = LSTM_DROPOUT,
                dense_units: List[int] = LSTM_DENSE_UNITS,
                learning_rate: float = LSTM_LEARNING_RATE):
        """
        Initialize the LSTM model.
        
        Args:
            model_dir: Directory to save model checkpoints
            sequence_length: Length of input sequences
            n_features: Number of input features
            n_outputs: Number of output units
            lstm_units: Number of LSTM units per layer
            lstm_layers: Number of LSTM layers
            dropout_rate: Dropout rate
            dense_units: List of units for dense layers
            learning_rate: Learning rate for optimizer
        """
        self.model_dir = model_dir
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.lstm_units = lstm_units
        self.lstm_layers = lstm_layers
        self.dropout_rate = dropout_rate
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.model = self._build_model()
    
    def _build_model(self) -> Model:
        """
        Build the LSTM model architecture.
        
        Returns:
            Compiled Keras model
        """
        inputs = Input(shape=(self.sequence_length, self.n_features))
        
        x = inputs
        lstm_outputs = []
        
        for i in range(self.lstm_layers):
            x = Bidirectional(
                LSTM(self.lstm_units, return_sequences=True),
                name=f'bidirectional_lstm_{i+1}'
            )(x)
            x = LayerNormalization(name=f'layer_norm_{i+1}')(x)
            x = Dropout(self.dropout_rate, name=f'dropout_{i+1}')(x)
            lstm_outputs.append(x)
        
        lstm_output = lstm_outputs[-1]
        
        query = GlobalAveragePooling1D()(lstm_output)
        
        attention = BahdanauAttention(self.lstm_units)
        context_vector, attention_weights = attention(query, lstm_output)
        
        x = context_vector
        for i, units in enumerate(self.dense_units):
            x = Dense(units, activation='relu', name=f'dense_{i+1}')(x)
            x = Dropout(self.dropout_rate, name=f'dense_dropout_{i+1}')(x)
        
        outputs = Dense(self.n_outputs, activation='linear', name='output')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def summary(self) -> None:
        """
        Print model summary.
        """
        self.model.summary()
    
    def train(self, train_X: np.ndarray, train_y: np.ndarray,
             val_X: np.ndarray, val_y: np.ndarray,
             batch_size: int = LSTM_BATCH_SIZE,
             epochs: int = LSTM_MAX_EPOCHS) -> tf.keras.callbacks.History:
        """
        Train the LSTM model.
        
        Args:
            train_X: Training input sequences
            train_y: Training target values
            val_X: Validation input sequences
            val_y: Validation target values
            batch_size: Batch size
            epochs: Maximum number of epochs
            
        Returns:
            Training history
        """
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(self.model_dir, 'best_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            TensorBoard(
                log_dir=os.path.join(self.model_dir, 'logs'),
                histogram_freq=1
            )
        ]
        
        logger.info("Training LSTM model...")
        history = self.model.fit(
            train_X, train_y,
            validation_data=(val_X, val_y),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("LSTM model training completed")
        
        return history
    
    def evaluate(self, test_X: np.ndarray, test_y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the LSTM model.
        
        Args:
            test_X: Test input sequences
            test_y: Test target values
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating LSTM model...")
        loss, mae = self.model.evaluate(test_X, test_y, verbose=1)
        
        y_pred = self.model.predict(test_X)
        
        rmse = np.sqrt(np.mean((y_pred - test_y) ** 2))
        
        direction_actual = np.sign(test_y)
        direction_pred = np.sign(y_pred)
        
        direction_actual[direction_actual == 0] = 1
        direction_pred[direction_pred == 0] = 1
        
        da_per_crypto = []
        for i in range(self.n_outputs):
            da = np.mean(direction_actual[:, i] == direction_pred[:, i])
            da_per_crypto.append(da)
        
        da_overall = np.mean(da_per_crypto)
        
        metrics = {
            'loss': loss,
            'mae': mae,
            'rmse': rmse,
            'directional_accuracy': da_overall
        }
        
        for i in range(self.n_outputs):
            metrics[f'da_crypto_{i}'] = da_per_crypto[i]
        
        logger.info(f"Evaluation metrics: {metrics}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the LSTM model.
        
        Args:
            X: Input sequences
            
        Returns:
            Predicted values
        """
        return self.model.predict(X)
    
    def save(self, filepath: Optional[str] = None) -> None:
        """
        Save the LSTM model.
        
        Args:
            filepath: Path to save the model (default: model_dir/lstm_model.h5)
        """
        if filepath is None:
            filepath = os.path.join(self.model_dir, 'lstm_model.h5')
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: Optional[str] = None) -> None:
        """
        Load the LSTM model.
        
        Args:
            filepath: Path to load the model from (default: model_dir/lstm_model.h5)
        """
        if filepath is None:
            filepath = os.path.join(self.model_dir, 'lstm_model.h5')
        
        if not os.path.exists(filepath):
            logger.error(f"Model file not found: {filepath}")
            return
        
        self.model = tf.keras.models.load_model(
            filepath,
            custom_objects={'BahdanauAttention': BahdanauAttention}
        )
        logger.info(f"Model loaded from {filepath}")

if __name__ == "__main__":
    model = LSTMModel(model_dir='../../models/lstm')
    model.summary()
