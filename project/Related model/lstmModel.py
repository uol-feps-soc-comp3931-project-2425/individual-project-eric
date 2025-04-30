import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import os

# --- Configuration ---
SEQUENCE_DATA_PATH = "sequenceData.npz"
MODEL_SAVE_PATH = "lstmModel.keras" 
HISTORY_PLOT_PATH = "lstmTrainingHistory.png"


LSTM_UNITS = 50 
DROPOUT_RATE = 0.2
EPOCHS = 50 
BATCH_SIZE = 32 
VALIDATION_SPLIT = 0.2 
def build_lstm_model(input_shape):
    """Builds the LSTM model structure."""
    model = Sequential([
        Input(shape=input_shape),
        LSTM(LSTM_UNITS, return_sequences=False), 
        Dropout(DROPOUT_RATE),
        Dense(1) 
    ])
    model.compile(optimizer=\'adam\', loss=\'mean_squared_error\')
    print("LSTM Model Summary:")
    model.summary()
    return model

def train_model():
    """Loads data, trains the LSTM model, and saves it."""
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


    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape)

    # Callbacks
    early_stopping = EarlyStopping(monitor=\'val_loss\
                                 patience=10, 
                                 restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, 
                                     save_best_only=True, 
                                     monitor=\'val_loss\
                                     mode=\'min\')

    print("Starting model training...")

    history = model.fit(X_train, y_train, 
                        epochs=EPOCHS, 
                        batch_size=BATCH_SIZE, 
                        validation_split=VALIDATION_SPLIT, 
                        callbacks=[early_stopping, model_checkpoint],
                        verbose=1)

    print(f"Model training finished. Best model saved to {MODEL_SAVE_PATH}")

    print("Evaluating model on test data...")
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss (MSE): {test_loss}")

    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label=\'Train Loss\
    plt.plot(history.history["val_loss"], label=\'Validation Loss\
    plt.title(\'Model Loss During Training\
    plt.ylabel(\'Loss (MSE)\
    plt.xlabel(\'Epoch\
    plt.legend(loc=\'upper right\
    plt.grid(True)
    plt.savefig(HISTORY_PLOT_PATH)
    print(f"Training history plot saved to {HISTORY_PLOT_PATH}")
    plt.close()

if __name__ == "__main__":
    # os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    # Consider setting memory limits if running on resource-constrained environments
    # gpus = tf.config.experimental.list_physical_devices(\'GPU\
    # if gpus:
    #     try:
    #         # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    #         tf.config.experimental.set_virtual_device_configuration(
    #             gpus[0],
    #             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]) # Adjust memory limit as needed
    #         logical_gpus = tf.config.experimental.list_logical_devices(\'GPU\
    #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #     except RuntimeError as e:
    #         # Virtual devices must be set before GPUs have been initialized
    #         print(e)
            
    train_model()

