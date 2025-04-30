import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
import tensorflow as tf 

try:
    from ta.momentum import RSIIndicator
    print("TA library found. RSI will be calculated.")
except ImportError:
    print("TA library not found. RSI calculation will be skipped (set to 50).")
    RSIIndicator = None

if tf:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout, Attention, LayerNormalization
    from tensorflow.keras.models import Model 
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau 
else:
    print("Warning: TensorFlow not found. LSTM model definition and training will be skipped.")
    Model = object
    EarlyStopping = object
    ReduceLROnPlateau = object

print("Imports complete.")

file_path = 'btcusdt_1m_data.csv'
timestamp_col_name = 'open_time' 
required_cols = ['open', 'high', 'low', 'close', 'volume']
epsilon = 1e-10 

# Preprocessing & Feature Params
interpolation_limit = 5
log_return_iqr_window = 60
log_return_iqr_multiplier = 3.0
volume_norm_window = 24 * 60 
volatility_window_15m = 15
volatility_window_60m = 60
rsi_period = 14
lag_periods = [1, 5, 10]

# Sequence & Split Params
sequence_length = 60
train_split_pct = 0.70
validation_split_pct = 0.15

# Model & Training Params
n_outputs = 3 
lstm_units = 64
dropout_rate = 0.2
learning_rate = 1e-3
batch_size = 64
epochs = 50 
early_stopping_patience = 10
lr_reduction_patience = 5


print("\n--- Starting Data Loading & Cleaning ---")
try:
    df = pd.read_csv(file_path)
    print(f"Successfully loaded data from {file_path}")
except FileNotFoundError: print(f"Error: File not found at {file_path}"); exit()
except Exception as e: print(f"Error loading CSV: {e}"); exit()

if timestamp_col_name not in df.columns: print(f"Error: Timestamp column '{timestamp_col_name}' not found."); exit()
try: df['datetime'] = pd.to_datetime(df[timestamp_col_name], utc=True)
except Exception as e: print(f"Error converting timestamp: {e}"); exit()

df.set_index('datetime', inplace=True); df.sort_index(inplace=True)
if not all(col in df.columns for col in required_cols): print(f"Error: Missing columns: {required_cols}"); exit()
df = df[required_cols].copy()
for col in required_cols: df[col] = pd.to_numeric(df[col], errors='coerce')

print("\n--- Starting Preprocessing (Interpolation, Log Returns, Outliers) ---")
df.interpolate(method='linear', limit=interpolation_limit, inplace=True)
initial_rows_part2 = len(df); df.dropna(inplace=True)
print(f"Dropped {initial_rows_part2 - len(df)} rows with persistent NaNs.")
if len(df) == 0: print("Error: No data after NaN drop."); exit()

df['prev_close'] = df['close'].shift(1); df.dropna(subset=['prev_close'], inplace=True)
if len(df) == 0: print("Error: No data after shift for log return."); exit()
df['log_return'] = np.log((df['close'] + epsilon) / (df['prev_close'] + epsilon))
df.replace([np.inf, -np.inf], 0, inplace=True); df.drop(columns=['prev_close'], inplace=True)
df.dropna(subset=['log_return'], inplace=True)

if len(df) >= log_return_iqr_window:
    q1 = df['log_return'].rolling(window=log_return_iqr_window, min_periods=1).quantile(0.25)
    q3 = df['log_return'].rolling(window=log_return_iqr_window, min_periods=1).quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - log_return_iqr_multiplier * iqr; upper_bound = q3 + log_return_iqr_multiplier * iqr
    df['log_return_capped'] = np.clip(df['log_return'], lower_bound, upper_bound)
    df['log_return_capped'].fillna(df['log_return'], inplace=True)
else: df['log_return_capped'] = df['log_return']
df.dropna(subset=['log_return_capped'], inplace=True)


print("\n--- Starting Preprocessing (Volume Norm) & Features (Returns, Volatility) ---")
if len(df) >= volume_norm_window :
    vol_mean = df['volume'].rolling(window=volume_norm_window, min_periods=1).mean()
    vol_std = df['volume'].rolling(window=volume_norm_window, min_periods=1).std()
    df['volume_zscore'] = (df['volume'] - vol_mean) / (vol_std + epsilon)
    df['volume_zscore'].fillna(0, inplace=True)
else: df['volume_zscore'] = 0

df['log_return_5m'] = df['log_return_capped'].rolling(window=5, min_periods=1).sum(); df['log_return_5m'].fillna(0, inplace=True)
df['log_return_15m'] = df['log_return_capped'].rolling(window=15, min_periods=1).sum(); df['log_return_15m'].fillna(0, inplace=True)
df['volatility_15m'] = df['log_return_capped'].rolling(window=volatility_window_15m, min_periods=1).std(); df['volatility_15m'].fillna(0, inplace=True)
df['volatility_60m'] = df['log_return_capped'].rolling(window=volatility_window_60m, min_periods=1).std(); df['volatility_60m'].fillna(0, inplace=True)


print("\n--- Starting Features (RSI, Lags) ---")
if RSIIndicator and len(df) > rsi_period:
    if df['close'].isnull().any(): close_filled = df['close'].fillna(method='ffill'); close_filled.dropna(inplace=True)
    else: close_filled = df['close']
    if len(close_filled) > rsi_period:
         rsi_indicator = RSIIndicator(close=close_filled, window=rsi_period)
         df['rsi'] = rsi_indicator.rsi()
         df['rsi'].fillna(method='ffill', inplace=True); df['rsi'].fillna(50, inplace=True)
    else: df['rsi'] = 50
elif not RSIIndicator: df['rsi'] = 50
else: df['rsi'] = 50

for lag in lag_periods:
    df[f'log_return_lag_{lag}'] = df['log_return_capped'].shift(lag)
    df[f'log_return_lag_{lag}'].fillna(0, inplace=True)

model_features = ['log_return_capped', 'log_return_5m', 'log_return_15m', 'volatility_15m', 'volatility_60m', 'rsi', 'volume_zscore', 'log_return_lag_1', 'log_return_lag_5', 'log_return_lag_10']
final_feature_columns = [f for f in model_features if f in df.columns]
df_final = df[final_feature_columns].copy()

if df_final.isnull().values.any():
    print("Warning: NaNs found before sequencing. Dropping rows."); rows_before_final_drop = len(df_final)
    df_final.dropna(inplace=True); print(f"Dropped {rows_before_final_drop - len(df_final)} rows.")
if len(df_final) == 0: print("Error: No data left."); exit()
print(f"Final selected features ({len(df_final.columns)}): {df_final.columns.tolist()}")


print("\n--- Starting Sequence Preparation ---")
def create_sequences(data, sequence_length, target_column_name):
    X, y = [], []
    if target_column_name not in data.columns: print(f"Error: Target '{target_column_name}' not in data."); return np.array(X), np.array(y)
    data_copy = data.copy()
    data_copy['target_shifted'] = data_copy[target_column_name].shift(-1)
    data_seq = data_copy.dropna(subset=['target_shifted'])
    feature_columns = [col for col in data_seq.columns if col not in [target_column_name, 'target_shifted']]
    if not feature_columns: print("Error: No features for sequencing."); return np.array(X), np.array(y)
    feature_data = data_seq[feature_columns].values
    target_data = data_seq['target_shifted'].values
    if len(feature_data) < sequence_length: print(f"Warn: Not enough data ({len(feature_data)}) for sequences ({sequence_length})."); return np.array(X), np.array(y)
    for i in range(len(feature_data) - sequence_length):
        X.append(feature_data[i:(i + sequence_length)])
        y.append(target_data[i + sequence_length -1]) 
    return np.array(X), np.array(y)

target_col = 'log_return_capped'
if target_col not in df_final.columns: print(f"Error: Target '{target_col}' not available."); exit()
X, y = create_sequences(df_final, sequence_length, target_col)

if X.size == 0 or y.size == 0: print("Error: No sequences created."); exit()
print(f"Created sequences: X shape {X.shape}, y shape {y.shape}")


print("\n--- Starting Train/Val/Test Split ---")
total_samples = X.shape[0]
train_end_index = int(total_samples * train_split_pct)
validation_end_index = train_end_index + int(total_samples * validation_split_pct)
X_train, y_train = X[:train_end_index], y[:train_end_index]
X_val, y_val = X[train_end_index:validation_end_index], y[train_end_index:validation_end_index]
X_test, y_test = X[validation_end_index:], y[validation_end_index:]
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val:   {X_val.shape}, y_val:   {y_val.shape}")
print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")
print("--- Part 6 Finished ---")


print("\n--- Starting Feature Scaling ---")
if X_train.shape[0] == 0: print("Error: X_train empty."); exit()
n_samples_train, n_timesteps, n_features = X_train.shape
n_samples_val, n_samples_test = X_val.shape[0], X_test.shape[0]
X_train_reshaped = X_train.reshape(-1, n_features); X_val_reshaped = X_val.reshape(-1, n_features); X_test_reshaped = X_test.reshape(-1, n_features)
scaler = StandardScaler()
print("Fitting StandardScaler...")
scaler.fit(X_train_reshaped)
print("Transforming datasets...")
X_train_scaled_reshaped = scaler.transform(X_train_reshaped); X_val_scaled_reshaped = scaler.transform(X_val_reshaped); X_test_scaled_reshaped = scaler.transform(X_test_reshaped)
X_train_scaled = X_train_scaled_reshaped.reshape(n_samples_train, n_timesteps, n_features)
X_val_scaled = X_val_scaled_reshaped.reshape(n_samples_val, n_timesteps, n_features)
X_test_scaled = X_test_scaled_reshaped.reshape(n_samples_test, n_timesteps, n_features)
print(f"X_train_scaled: {X_train_scaled.shape}, X_val_scaled: {X_val_scaled.shape}, X_test_scaled: {X_test_scaled.shape}")


print("\n--- Starting Defining LSTM Model ---")
lstm_model = None # Initialize
if tf and X_train_scaled.size > 0:
    n_timesteps = X_train_scaled.shape[1]
    n_features = X_train_scaled.shape[2]
    print(f"Input shape: ({n_timesteps}, {n_features}), Output units: {n_outputs}")
    input_layer = Input(shape=(n_timesteps, n_features), name='input_sequence')
    lstm_out1 = Bidirectional(LSTM(units=lstm_units, return_sequences=True), name='bilstm_1')(input_layer)
    lstm_out2 = Bidirectional(LSTM(units=lstm_units, return_sequences=True), name='bilstm_2')(lstm_out1)
    dropout1 = Dropout(dropout_rate, name='dropout_1')(lstm_out2)
    attention_out = Attention(name='attention_layer')([dropout1, dropout1])
    dense1 = Dense(64, activation='relu', name='dense_1')(attention_out)
    dropout2 = Dropout(dropout_rate, name='dropout_2')(dense1)
    output_layer = Dense(n_outputs, activation='linear', name='output_layer')(dropout2)
    lstm_model = Model(inputs=input_layer, outputs=output_layer, name='LSTM_with_Attention')
    print("LSTM model structure defined.")
    lstm_model.summary() 
else: print("Skipping LSTM definition (TensorFlow not found or no training data).")


print("\n--- Starting Compile and Train LSTM ---")
if lstm_model and tf: 
    print("Compiling LSTM model...")
    lstm_model.compile(optimizer=Adam(learning_rate=learning_rate),
                       loss='mean_squared_error', 
                       metrics=['mae']) 
    print("Model compiled.")


    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=early_stopping_patience,
                                   restore_best_weights=True) 

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.5, 
                                  patience=lr_reduction_patience)

    # model_checkpoint = ModelCheckpoint(filepath='best_lstm_model.keras', # Keras format
    #                                    monitor='val_loss',
    #                                    save_best_only=True)
    # callbacks_list = [early_stopping, reduce_lr, model_checkpoint]
    callbacks_list = [early_stopping, reduce_lr]

    # Train the model
    print("\nStarting LSTM model training")


    # Check y shapes (important if n_outputs > 1)
    print(f"y_train shape: {y_train.shape}, y_val shape: {y_val.shape}")
    if n_outputs > 1 and y_train.ndim == 1:
        print(f"Warning: n_outputs={n_outputs} but y_train is 1D. Training might error or behave unexpectedly.")
        # y_train = np.repeat(y_train[:, np.newaxis], n_outputs, axis=1)
        # y_val = np.repeat(y_val[:, np.newaxis], n_outputs, axis=1)
        # print(f"Reshaped y_train: {y_train.shape}, y_val: {y_val.shape}")



    history = lstm_model.fit(X_train_scaled, y_train,
                             epochs=epochs,
                             batch_size=batch_size,
                             validation_data=(X_val_scaled, y_val),
                             callbacks=callbacks_list,
                             verbose=1) 

    print("\nModel training finished.")
    # pd.DataFrame(history.history).plot(figsize=(8, 5))
    # plt.grid(True)
    # plt.gca().set_ylim(0, max(history.history['loss'] + history.history['val_loss'])) 
    # plt.show()

else:
    print("Skipping model compilation and training (Model not defined).")



print("Data loaded, processed, features engineered, split, scaled, LSTM model defined, compiled, and training example executed.")
print("Next steps: Evaluate the trained model on X_test_scaled, y_test. Implement other models (Transformer, Baselines).")