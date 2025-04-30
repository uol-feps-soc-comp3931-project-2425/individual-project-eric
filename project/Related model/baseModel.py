import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    from statsmodels.tsa.api import VAR
except ImportError:
    print("Error: statsmodels library not found. Please install using 'pip install statsmodels'.")
    print("VAR model section will be skipped.")
    VAR = None

if 'df_final' not in locals() or not isinstance(df_final, pd.DataFrame) or df_final.empty:
    print("Error: Multivariate DataFrame 'df_final' not found, is not a DataFrame, or is empty.")
    print("Please ensure you have run the script that generates 'df_final' first.")
    exit()

assets = ['BTC', 'ETH', 'SOL']
target_cols = [f'log_return_capped_{asset}' for asset in assets]

missing_cols = [col for col in target_cols if col not in df_final.columns]
if missing_cols:
    print(f"Error: One or more target columns not found in df_final: {missing_cols}")
    print(f"Available columns: {df_final.columns.tolist()}")
    exit()

print("\n--- Preparing Baseline Model 1: VAR Model ---")

if VAR is None:
    print("Skipping VAR model setup because statsmodels is not installed.")
else:
    print("Note: VAR model assumes the input log return series are stationary. Stationarity testing (e.g., ADF test) is recommended.")
    var_data = df_final[target_cols].copy()

    if var_data.isnull().values.any():
        print(f"Warning: NaN values detected in VAR input data. Attempting to fill using forward fill (ffill).")
        var_data.ffill(inplace=True)
        if var_data.isnull().values.any():
            print(f"Warning: NaN values still present after ffill. Attempting to fill using backward fill (bfill).")
            var_data.bfill(inplace=True)
        if var_data.isnull().values.any():
            print(f"Error: NaN values still present in VAR data after filling. Please check the original data.")
            exit()

    total_len_var = len(var_data)
    train_split_pct = 0.70
    validation_split_pct = 0.15
    test_split_pct = 1.0 - train_split_pct - validation_split_pct

    if not (train_split_pct > 0 and validation_split_pct > 0 and test_split_pct > 0):
        print("Error: Train, validation, and test split percentages must all be greater than 0.")
        exit()

    train_end_index_var = int(total_len_var * train_split_pct)
    validation_end_index_var = train_end_index_var + int(total_len_var * validation_split_pct)

    var_train = var_data[:train_end_index_var]
    var_val = var_data[train_end_index_var:validation_end_index_var]
    var_test = var_data[validation_end_index_var:]

    print(f"VAR data split (Total {total_len_var} rows):")
    print(f"  Training set: {var_train.shape} (Index 0 to {train_end_index_var-1})")
    print(f"  Validation set: {var_val.shape} (Index {train_end_index_var} to {validation_end_index_var-1})")
    print(f"  Test set: {var_test.shape} (Index {validation_end_index_var} to {total_len_var-1})")

    min_train_size = 20
    if len(var_train) < min_train_size:
        print(f"Error: VAR training set is too small (only {len(var_train)} samples) to reliably fit the model.")
        print("Skipping VAR model fitting and prediction.")
    else:
        var_model = VAR(var_train)

        max_lags_to_test = min(15, len(var_train) // 2 - 1)
        print(f"Selecting VAR lag order (Testing up to {max_lags_to_test} lags, this may take time)...")
        best_lag = 5

        if max_lags_to_test > 0:
            try:
                selected_order = var_model.select_order(maxlags=max_lags_to_test, ic='bic')
                best_lag = selected_order.selected_lags
                print(f"Best VAR lag order selected by BIC: {best_lag}")
                if best_lag == 0:
                    print("Warning: BIC selected lag order is 0, which might indicate no significant autocorrelation. Using default lag 1.")
                    best_lag = 1
            except Exception as e:
                print(f"Warning: VAR lag order selection failed ({e}). Using default lag={best_lag}.")
        else:
             print(f"Warning: Training data too small to automatically select lag order. Using default lag={best_lag}.")

        if best_lag >= len(var_train):
             print(f"Error: Selected lag order ({best_lag}) is greater than or equal to the number of training samples ({len(var_train)}). Cannot fit model.")
             print("Please check data or lag selection process.")
        else:
            print(f"Fitting VAR({best_lag}) model...")
            var_results = var_model.fit(best_lag)

            lag_order = var_results.k_ar
            print(f"\nPerforming VAR rolling forecast on validation set (steps=1)... Using lag order: {lag_order}")

            if len(var_val) == 0:
                print("Warning: VAR validation set is empty. Cannot perform prediction and evaluation.")
                var_val_pred = pd.DataFrame(columns=[f'{col}_pred' for col in target_cols])
            else:
                history = var_train.values[-lag_order:]

                var_val_pred_list = []
                for i in range(len(var_val)):
                    pred = var_results.forecast(y=history, steps=1)
                    var_val_pred_list.append(pred[0])
                    actual_obs = var_val.iloc[[i]].values
                    history = np.vstack((history[1:], actual_obs))

                var_val_pred = pd.DataFrame(var_val_pred_list, index=var_val.index, columns=[f'{col}_pred' for col in target_cols])
                print(f"VAR validation predictions shape: {var_val_pred.shape}")

                print("\nVAR Validation Set Evaluation:")
                try:
                    var_val_mse = {}
                    for asset in assets:
                        target_col_name = f'log_return_capped_{asset}'
                        pred_col_name = f'log_return_capped_{asset}_pred'
                        var_val_mse[asset] = mean_squared_error(var_val[target_col_name], var_val_pred[pred_col_name])
                        print(f"  MSE ({asset}): {var_val_mse[asset]:.8f}")

                    avg_var_val_mse = np.mean(list(var_val_mse.values()))
                    print(f"  Average MSE (all assets): {avg_var_val_mse:.8f}")

                    var_val_mae_btc = mean_absolute_error(var_val[f'log_return_capped_BTC'], var_val_pred[f'log_return_capped_BTC_pred'])
                    print(f"  MAE (BTC example): {var_val_mae_btc:.8f}")

                except Exception as e:
                    print(f"Error calculating VAR validation metrics: {e}")

print("\n--- Preparing Baseline Model 2: Random Forest Regressor ---")

if df_final.isnull().values.any():
    print("Warning: NaN values found in the original df_final data.")
    print(f"NaN counts:\n{df_final.isnull().sum()}")
    print("Attempting to handle NaNs using forward fill (ffill) and backward fill (bfill)...")
    df_final.ffill(inplace=True)
    df_final.bfill(inplace=True)
    if df_final.isnull().values.any():
         print("Error: NaN values still present in df_final after filling. Please handle data in earlier steps.")
         exit()

feature_cols_rf = [col for col in df_final.columns if col not in target_cols]
if not feature_cols_rf:
     print("Error: No feature columns found for Random Forest. Please check df_final columns.")
     exit()

print(f"Random Forest model will use the following {len(feature_cols_rf)} features:")

X_rf = df_final[feature_cols_rf]
y_rf = df_final[target_cols].shift(-1)

X_rf = X_rf[:-1]
y_rf = y_rf[:-1]

if X_rf.isnull().values.any() or y_rf.isnull().values.any():
     print("Warning: NaN values found after aligning features and target. This should not happen; check data preprocessing. Attempting to drop rows with NaNs.")
     combined_rf = pd.concat([X_rf, y_rf], axis=1)
     combined_rf.dropna(inplace=True)
     X_rf = combined_rf[feature_cols_rf]
     y_rf = combined_rf[target_cols]
     print("Rows with NaNs dropped.")

print(f"Prepared Random Forest input shapes: X_rf={X_rf.shape}, y_rf={y_rf.shape}")

if len(X_rf) == 0:
    print("Error: No data remaining for Random Forest model after preparing features and target.")
    print("Skipping Random Forest setup.")
else:
    total_len_rf = len(X_rf)
    train_end_index_rf = int(total_len_rf * train_split_pct)
    validation_end_index_rf = train_end_index_rf + int(total_len_rf * validation_split_pct)

    X_train_rf, y_train_rf = X_rf[:train_end_index_rf], y_rf[:train_end_index_rf]
    X_val_rf, y_val_rf = X_rf[train_end_index_rf:validation_end_index_rf], y_rf[train_end_index_rf:validation_end_index_rf]
    X_test_rf, y_test_rf = X_rf[validation_end_index_rf:], y_rf[validation_end_index_rf:]

    print(f"Random Forest data split (Total {total_len_rf} aligned X/y pairs):")
    print(f"  Training set: X={X_train_rf.shape}, y={y_train_rf.shape}")
    print(f"  Validation set: X={X_val_rf.shape}, y={y_val_rf.shape}")
    print(f"  Test set: X={X_test_rf.shape}, y={y_test_rf.shape}")

    if len(X_train_rf) == 0:
        print("Warning: Random Forest training set is empty. Cannot train model.")
        print("Skipping RF model fitting and prediction.")
    else:
        print("Fitting Random Forest Regressor...")
        rf_model = RandomForestRegressor(n_estimators=100,
                                         max_depth=10,
                                         min_samples_leaf=5,
                                         max_features=0.8,
                                         random_state=42,
                                         n_jobs=-1)

        rf_model.fit(X_train_rf, y_train_rf)
        print("Random Forest model fitting complete.")

        print("\nPerforming RF prediction on validation set...")
        if len(X_val_rf) == 0:
             print("Warning: RF validation set is empty. Cannot perform prediction and evaluation.")
             rf_val_pred = np.array([])
        else:
            rf_val_pred = rf_model.predict(X_val_rf)
            print(f"RF validation predictions shape: {rf_val_pred.shape}")

            print("\nRF Validation Set Evaluation:")
            try:
                rf_val_mse_all = mean_squared_error(y_val_rf, rf_val_pred)
                print(f"  Average MSE (all assets): {rf_val_mse_all:.8f}")

                rf_val_mse = {}
                for i, asset in enumerate(assets):
                    target_col_name = f'log_return_capped_{asset}'
                    rf_val_mse[asset] = mean_squared_error(y_val_rf.iloc[:, i], rf_val_pred[:, i])
                    print(f"  MSE ({asset}): {rf_val_mse[asset]:.8f}")

                btc_target_index = target_cols.index(f'log_return_capped_BTC')
                rf_val_mae_btc = mean_absolute_error(y_val_rf.iloc[:, btc_target_index], rf_val_pred[:, btc_target_index])
                print(f"  MAE (BTC example): {rf_val_mae_btc:.8f}")

            except Exception as e:
                 print(f"Error calculating RF validation metrics: {e}")

print("\nMultivariate baseline models (VAR and Random Forest) setup complete.")
print("Example predictions and evaluations performed on the validation set.")
print("Next step: Model Evaluation. Compare performance of LSTM, Transformer, VAR, RF, etc., on the test set.")