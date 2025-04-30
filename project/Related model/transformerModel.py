import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import os

# --- Configuration ---
SEQUENCE_DATA_PATH = "sequence_data.npz"
MODEL_SAVE_PATH = "transformerModel.keras" 
HISTORY_PLOT_PATH = "transformerTrainingHistory.png"


SEQUENCE_LENGTH = 15 
NUM_FEATURES = None 
HEAD_SIZE = 128     
NUM_HEADS = 4       
FF_DIM = 128        
NUM_TRANSFORMER_BLOCKS = 2 
MLP_UNITS = [64]    
DROPOUT = 0.1
MLP_DROPOUT = 0.2

EPOCHS = 50 
BATCH_SIZE = 32 
VALIDATION_SPLIT = 0.2

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return x + res

def build_transformer_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
    """Builds the Transformer model."""
    inputs = Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = GlobalAveragePooling1D(data_format="channels_last")(x)
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)
    outputs = Dense(1)(x) 

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")
    print("Transformer Model Summary:")
    model.summary()
    return model

def train_model():
    """Loads data, trains the Transformer model, and saves it."""
    print(f"Loading sequence data from {SEQUENCE_DATA_PATH}...")
    try:
        data = np.load(SEQUENCE_DATA_PATH)
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_test = data["X_test"]
        y_test = data["y_test"]
        print(f"Data loaded. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    except FileNotFoundError:
        print(f"Error: {SEQUENCE_DATA_PATH} not found. Please run prepareSequenceData.py first.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if len(X_train) == 0 or len(y_train) == 0:
        print("Error: Training data is empty. Check the preprocessing steps.")
        return
        
    global NUM_FEATURES
    NUM_FEATURES = X_train.shape[2]
    input_shape = (SEQUENCE_LENGTH, NUM_FEATURES)

    model = build_transformer_model(
        input_shape,
        head_size=HEAD_SIZE,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_transformer_blocks=NUM_TRANSFORMER_BLOCKS,
        mlp_units=MLP_UNITS,
        dropout=DROPOUT,
        mlp_dropout=MLP_DROPOUT,
    )

    early_stopping = EarlyStopping(monitor=\'val_loss\', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor=\'val_loss\', mode=\'min\')

    print("Starting Transformer model training...")
    print("NOTE: Training this model with full data might require significant memory.")
    print(f"Using Sequence Length: {SEQUENCE_LENGTH}, Batch Size: {BATCH_SIZE}")
    
    subset_percentage = 0.2 
    subset_size = int(len(X_train) * subset_percentage)
    X_train_subset = X_train[:subset_size]
    y_train_subset = y_train[:subset_size]
    print(f"Training on a subset of {subset_percentage*100}% of the training data ({subset_size} samples) due to potential memory constraints.")

    history = model.fit(X_train_subset, y_train_subset, 
                        epochs=EPOCHS, 
                        batch_size=BATCH_SIZE, 
                        validation_split=VALIDATION_SPLIT, 
                        callbacks=[early_stopping, model_checkpoint],
                        verbose=1)

    print(f"Model training finished. Best model (potentially from subset) saved to {MODEL_SAVE_PATH}")

    print("Evaluating model on test data...")
    try:
        best_model = load_model(MODEL_SAVE_PATH)
        test_loss = best_model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss (MSE) using best saved model: {test_loss}")
    except Exception as e:
        print(f"Could not load or evaluate the saved model: {e}")
        print("Evaluating with the model state at the end of training (might not be the best).")
        test_loss = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss (MSE) using final model state: {test_loss}")


    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label=\'Train Loss\')
    plt.plot(history.history["val_loss"], label=\'Validation Loss\')
    plt.title(\'Transformer Model Loss During Training (Subset)\
    plt.ylabel(\'Loss (MSE)\
    plt.xlabel(\'Epoch\')
    plt.legend(loc=\'upper right\')
    plt.grid(True)
    plt.savefig(HISTORY_PLOT_PATH)
    print(f"Training history plot saved to {HISTORY_PLOT_PATH}")
    plt.close()

if __name__ == "__main__":
    train_model()

