import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.dates as mdates
from datetime import datetime
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

np.random.seed(42)
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

os.makedirs('outputs', exist_ok=True)
os.makedirs('outputs/models', exist_ok=True)
os.makedirs('outputs/figures', exist_ok=True)
os.makedirs('outputs/trading_simulation', exist_ok=True)

def load_and_clean_data(file_path):
    print("Loading data...")
    
    preview = pd.read_csv(file_path, nrows=5)
    
    date_cols = []
    for col in preview.columns:
        if 'Ticker' in col:
            try:
                test_date = pd.to_datetime(preview[col].iloc[0], errors='coerce')
                if not pd.isna(test_date):
                    date_cols.append(col)
                    print(f"Found date column: {col}")
            except:
                pass
    
    if date_cols:
        print("Using first valid Ticker column as date index")
        df = pd.read_csv(file_path)
        
        df['proper_date'] = pd.to_datetime(df[date_cols[0]], errors='coerce')
        df.set_index('proper_date', inplace=True)
    else:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    if not isinstance(df.index, pd.DatetimeIndex):
        print("Warning: Index is not datetime type, trying to force conversion...")
        try:
            df.index = pd.to_datetime(df.index)
        except:
            print("Cannot convert index to datetime, will create new date index")
            df = df.reset_index()
            df.index = pd.date_range(start='2023-01-01', periods=len(df), freq='H')
            
    df['original_index'] = df.index
    
    print("Removing non-numeric columns...")
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower() or 'ticker' in col.lower()]
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    
    cols_to_drop = list(set(non_numeric_cols) - set(['original_index']))
    if cols_to_drop:
        print(f"Columns removed: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
        
    print("Processing infinite values and missing data...")
    for col in df.columns:
        if col == 'original_index':
            continue
            
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            print(f"  Column '{col}' has {nan_count} missing values")
            
            if nan_count > 0.5 * len(df):
                print(f"  Column '{col}' has too many missing values, will be removed")
                df = df.drop(columns=[col])
            else:
                df[col] = df[col].fillna(df[col].median())

    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    print(f"Cleaned data shape: {df.shape}")
    print(f"Data time range: {df.index.min()} to {df.index.max()}")
    
    df = df.sort_index()
    
    return df

def create_target_variables(data, horizons=[24, 100]):
    for coin in ['btc', 'eth', 'sol']:
        price_col = f'{coin}_Close'
        if price_col in data.columns:
            for horizon in horizons:
                target_col = f'{coin}_future_direction_{horizon}h'
                if target_col not in data.columns:
                    print(f"Creating target variable: {target_col}")
                    data[target_col] = (data[price_col].shift(-horizon) > data[price_col]).astype(int)
                    
    for coin in ['btc', 'eth', 'sol']:
        for horizon in horizons:
            target_col = f'{coin}_future_direction_{horizon}h'
            if target_col in data.columns:
                data = data.dropna(subset=[target_col])
                break
                
    return data

def detect_market_condition(data, coin, window=24*7):
    price_col = f'{coin}_Close'
    if price_col not in data.columns:
        return 'unknown'
        
    price_series = data[price_col].copy()
    pct_change = price_series.pct_change(window).iloc[-1] * 100 
    
    volatility = price_series.pct_change().rolling(window).std().iloc[-1] * 100
    
    daily_changes = price_series.pct_change().rolling(24).mean()
    direction_changes = ((daily_changes > 0) != (daily_changes.shift(1) > 0)).rolling(window).sum().iloc[-1]
    
    if volatility > 4: 
        return 'volatile'
    elif direction_changes > window/3: 
        return 'ranging'
    elif pct_change > 5: 
        return 'trending_up'
    elif pct_change < -5: 
        return 'trending_down'
    else:
        return 'ranging'

def select_best_features(X_train, y_train, source, target, max_features=40):
    print("Selecting best features...")
    
    if source.lower() == 'sol' and target.lower() == 'eth':
        priority_features = [col for col in X_train.columns if any(
            term in col for term in [
                'price', 'Close', 'return', 'volatility', 'momentum', 
                'rsi', 'macd', 'volume', 'ma7', 'ma24'
            ]
        )]
        
        priority_features = priority_features[:max_features]
        
        if len(priority_features) < max_features:
            remaining_features = [col for col in X_train.columns if col not in priority_features]
            if remaining_features:
                selector = RandomForestClassifier(n_estimators=100, random_state=42)
                X_remaining = X_train[remaining_features]
                selector.fit(X_remaining, y_train)
                
                importances = selector.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                top_remaining = [remaining_features[i] for i in indices[:max_features-len(priority_features)]]
                selected_features = priority_features + top_remaining
                
                print(f"For SOL→ETH: Selected {len(priority_features)} priority features and {len(top_remaining)} supplementary features")
                return selected_features, None
                
        print(f"For SOL→ETH: Selected {len(priority_features)} priority features")
        return priority_features, None
        
    selector = RandomForestClassifier(n_estimators=100, random_state=42)
    selector.fit(X_train, y_train)
    
    importances = selector.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    top_indices = indices[:max_features]
    selected_features = X_train.columns[top_indices]
    
    return selected_features, importances

def optimize_model_parameters(source, target, horizon):
    if source.lower() == 'sol' and target.lower() == 'eth' and horizon == 24:
        xgb_params = {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'random_state': 42
        }
        rf_params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
        lgbm_params = {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
    elif source.lower() == 'eth' and target.lower() == 'btc' and horizon == 24:
        rf_params = {
            'n_estimators': 200,
            'max_depth': 20,
            'min_samples_split': 4,
            'min_samples_leaf': 1,
            'random_state': 42
        }
        xgb_params = {
            'n_estimators': 150,
            'learning_rate': 0.05,
            'max_depth': 5,
            'random_state': 42
        }
        lgbm_params = {
            'n_estimators': 150,
            'learning_rate': 0.05,
            'random_state': 42
        }
    else:
        rf_params = {
            'n_estimators': 100,
            'random_state': 42
        }
        xgb_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'random_state': 42
        }
        lgbm_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'random_state': 42
        }
        
    return {
        'RandomForest': rf_params,
        'XGBoost': xgb_params,
        'LightGBM': lgbm_params
    }

def cross_coin_prediction(source, target, horizon, data, feature_selection=True):
    print(f"\n{'='*60}")
    print(f"Predicting {source.upper()} → {target.upper()} {horizon}h price direction")
    print(f"{'='*60}")
    
    market_condition = detect_market_condition(data, target)
    print(f"Current market condition: {market_condition}")
    
    target_col = f'{target}_future_direction_{horizon}h'
    if target_col not in data.columns:
        print(f"Target variable {target_col} not found, trying to create")
        try:
            price_col = f'{target}_Close'
            if price_col in data.columns:
                data[target_col] = (data[price_col].shift(-horizon) > data[price_col]).astype(int)
                print(f"✓ Successfully created target variable")
            else:
                print(f"✗ Price column {price_col} not found, cannot create target")
                return None, None, None # Return three None values to match expected return signature
        except Exception as e:
            print(f"✗ Failed to create target: {e}")
            return None, None, None # Return three None values

    source_features = [col for col in data.columns if col.startswith(source)]
    print(f"Initial feature count: {len(source_features)}")
    
    clean_features = []
    for col in source_features:
        if data[col].nunique() > 1 and not data[col].isna().any():
            clean_features.append(col)
            
    print(f"Valid features count: {len(clean_features)}")
    if len(clean_features) < 5:
        print(f"✗ Not enough usable features, skipping")
        return None, None, None # Return three None values
        
    X = data[clean_features]
    y = data[target_col].copy()
    
    valid_rows = ~y.isna()
    X = X[valid_rows]
    y = y[valid_rows]
    
    if len(y) == 0:
        print("✗ No valid target data")
        return None, None, None # Return three None values
        
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Up ratio: Training {y_train.mean():.2f}, Testing {y_test.mean():.2f}")
    
    selected_features = X_train.columns # Initialize with all features
    if feature_selection and len(X_train.columns) > 40:
        selected_features, importances = select_best_features(X_train, y_train, source, target)
        print(f"Selected {len(selected_features)} most important features")
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]
        
        if importances is not None:
            feature_imp_df = pd.DataFrame({
                'feature': X.columns, # Use original X columns for feature names before selection
                'importance': importances
            }).sort_values('importance', ascending=False)
            # Ensure index matches the features selected if using RandomForest importance
            # Re-align feature names for saving if necessary
            if len(feature_imp_df) == len(X.columns) and len(selected_features) < len(X.columns):
                 # Assume importances correspond to the original columns before selection
                 # Filter the importance df based on selected_features for saving
                 all_features_importance = pd.DataFrame({'feature': X.columns, 'importance': importances})
                 feature_imp_df = all_features_importance[all_features_importance['feature'].isin(selected_features)].sort_values('importance', ascending=False)
                 # Or recalculate importance based only on selected features if needed, depends on select_best_features logic
            
            # Fallback if lengths mismatch or logic is complex
            try:
                # Try creating df directly from selected features and their importances if available
                if len(selected_features) == len(importances[:len(selected_features)]):
                     feature_imp_df = pd.DataFrame({
                         'feature': selected_features,
                         'importance': importances[:len(selected_features)] # Match importance to selected features
                     }).sort_values('importance', ascending=False)
                else: # If alignment is unsure, just save top importances without names or skip
                     print("Warning: Could not align feature names with importances for saving.")

                feature_imp_df.to_csv(f'outputs/{source}_{target}_{horizon}h_feature_importance.csv', index=False)
            except Exception as e_imp:
                 print(f"Could not save feature importance: {e_imp}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model_params = optimize_model_parameters(source, target, horizon)
    
    models = {
        'RandomForest': RandomForestClassifier(**model_params['RandomForest']),
        'XGBoost': XGBClassifier(**model_params['XGBoost']),
        'LightGBM': LGBMClassifier(**model_params['LightGBM'])
    }
    
    results = []
    predictions = {}
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name} model...")
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        print(f"  ✓ Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        evaluate_by_market_condition(X_test, y_test, y_pred, target, horizon, model_name)
        
        joblib.dump(model, f'outputs/models/{source}_{target}_{horizon}h_{model_name.lower()}_model.pkl')
        joblib.dump(scaler, f'outputs/models/{source}_{target}_{horizon}h_scaler.pkl')
        joblib.dump(selected_features, f'outputs/models/{source}_{target}_{horizon}h_features.pkl')
        
        results.append({
            'source': source,
            'target': target,
            'horizon': horizon,
            'model': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'market_condition': market_condition
        })
        
        predictions[model_name] = {
            'y_pred': y_pred,
            'y_proba': y_proba,
            'y_test': y_test,
            'test_indices': X_test.index
        }
        
    try:
        top_n = 20
        best_model_name = max(results, key=lambda x: x['f1'])['model']
        best_model = models[best_model_name]
        
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            feature_names = X_train.columns # Use columns from the potentially reduced X_train
            
            feature_imp = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            feature_imp.to_csv(f'outputs/{source}_{target}_{horizon}h_{best_model_name}_feature_imp.csv', index=False)
            
            plt.figure(figsize=(12, 8))
            sns.barplot(x='importance', y='feature', data=feature_imp.head(top_n))
            plt.title(f'{source.upper()} → {target.upper()} {horizon}h: {best_model_name} Feature Importance', fontsize=14)
            plt.tight_layout()
            plt.savefig(f'outputs/figures/{source}_{target}_{horizon}h_feature_importance.png')
            plt.close()
    except Exception as e:
        print(f"Feature importance visualization failed: {e}")
        
    for model_name, pred_data in predictions.items():
        try:
            visualize_advanced_predictions(
                data, 
                pred_data['test_indices'],
                pred_data['y_test'],
                pred_data['y_pred'],
                pred_data['y_proba'],
                source,
                target,
                horizon,
                model_name,
                next((r['accuracy'] for r in results if r['model'] == model_name), None)
            )
        except Exception as e:
            print(f"Error visualizing {model_name} predictions: {e}")
            
    try:
        create_performance_curves(predictions, source, target, horizon)
    except Exception as e:
        print(f"Error creating performance curves: {e}")
        
    if source.lower() == 'sol' and target.lower() == 'eth' and horizon == 24:
        best_model_name = max(results, key=lambda x: x['accuracy'])['model']
        simulate_trading_strategy(
            data, 
            predictions[best_model_name]['test_indices'],
            predictions[best_model_name]['y_pred'],
            predictions[best_model_name]['y_proba'],
            'eth',
            horizon,
            model_name=best_model_name
        )
        
    return results, predictions, selected_features


def evaluate_by_market_condition(X_test, y_test, y_pred, target, horizon, model_name):
    if not isinstance(X_test.index, pd.DatetimeIndex):
        print("Cannot evaluate by market condition - missing time index")
        return
        
    try:
        X_test_copy = X_test.copy()
        X_test_copy['year_month'] = X_test_copy.index.to_period('M')
        X_test_copy['pred'] = y_pred
        X_test_copy['actual'] = y_test.values
        
        monthly_accuracy = X_test_copy.groupby('year_month').apply(
            lambda g: accuracy_score(g['actual'], g['pred']) if len(g) >= 24 else np.nan
        ).dropna()
        
        print(f"  Monthly accuracy range: {monthly_accuracy.min():.4f} - {monthly_accuracy.max():.4f}, Mean: {monthly_accuracy.mean():.4f}")
        
        monthly_accuracy.to_csv(f'outputs/{target}_{horizon}h_{model_name}_monthly_accuracy.csv')
    except Exception as e:
        print(f"  Monthly evaluation failed: {e}")


def simulate_trading_strategy(data, test_indices, predictions, probabilities, target, horizon, model_name):
    print(f"\nSimulating Trading Strategy - {target.upper()} {horizon}h ({model_name})")
    
    price_col = f'{target}_Close'
    if price_col not in data.columns:
        print("Cannot simulate trading - price data missing")
        return
        
    df_trade = pd.DataFrame(index=test_indices)
    df_trade['price'] = data.loc[test_indices, price_col]
    df_trade['prediction'] = predictions
    df_trade['probability'] = probabilities
    df_trade['confident_signal'] = (probabilities > 0.65) | (probabilities < 0.35)
    
    future_prices = data[price_col].shift(-horizon).loc[test_indices]
    df_trade['future_price'] = future_prices
    df_trade['actual_return'] = (future_prices - df_trade['price']) / df_trade['price'] * 100
    
    initial_balance = 10000 
    position = 0 
    balance = initial_balance 
    
    trades = []
    portfolio_values = []
    
    for idx, row in df_trade.iterrows():
        portfolio_value = balance + position * row['price']
        
        portfolio_values.append({
            'date': idx,
            'portfolio_value': portfolio_value,
            'price': row['price'],
            'position': position,
            'balance': balance
        })
        
        if row['confident_signal']:
            if row['prediction'] == 1 and position == 0:
                quantity = balance // row['price']
                if quantity > 0:
                    cost = quantity * row['price']
                    balance -= cost
                    position += quantity
                    trades.append({
                        'date': idx,
                        'action': 'BUY',
                        'price': row['price'],
                        'quantity': quantity,
                        'cost': cost,
                        'probability': row['probability']
                    })
            elif row['prediction'] == 0 and position > 0:
                revenue = position * row['price']
                balance += revenue
                trades.append({
                    'date': idx,
                    'action': 'SELL',
                    'price': row['price'],
                    'quantity': position,
                    'revenue': revenue,
                    'probability': row['probability']
                })
                position = 0
                
    if position > 0:
        last_price = df_trade['price'].iloc[-1]
        revenue = position * last_price
        balance += revenue
        trades.append({
            'date': df_trade.index[-1],
            'action': 'CLOSE',
            'price': last_price,
            'quantity': position,
            'revenue': revenue,
            'probability': 0.5
        })
        position = 0
        
    final_value = balance
    strategy_return = (final_value / initial_balance - 1) * 100
    
    buy_hold_return = (df_trade['price'].iloc[-1] / df_trade['price'].iloc[0] - 1) * 100
    
    df_trades = pd.DataFrame(trades)
    df_portfolio = pd.DataFrame(portfolio_values)
    
    if not df_trades.empty:
        df_trades.to_csv(f'outputs/trading_simulation/{target}_{horizon}h_{model_name}_trades.csv')
    df_portfolio.to_csv(f'outputs/trading_simulation/{target}_{horizon}h_{model_name}_portfolio.csv')
    
    print(f"Trading Simulation Results:")
    print(f"  Initial Balance: ${initial_balance}")
    print(f"  Final Balance: ${final_value:.2f}")
    print(f"  Strategy Return: {strategy_return:.2f}%")
    print(f"  Buy & Hold Return: {buy_hold_return:.2f}%")
    print(f"  Excess Return: {strategy_return - buy_hold_return:.2f}%")
    print(f"  Number of Trades: {len(trades)}")
    
    visualize_trading_results(df_portfolio, df_trades, target, horizon, model_name, 
                              strategy_return, buy_hold_return)

def visualize_trading_results(portfolio_df, trades_df, target, horizon, model_name, 
                              strategy_return, buy_hold_return):
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(portfolio_df['date'], portfolio_df['portfolio_value'], 
             label=f'Strategy Value (+{strategy_return:.2f}%)', linewidth=2)
    
    initial_investment = portfolio_df['portfolio_value'].iloc[0]
    buy_hold = portfolio_df['price'] / portfolio_df['price'].iloc[0] * initial_investment
    plt.plot(portfolio_df['date'], buy_hold, 
             label=f'Buy & Hold (+{buy_hold_return:.2f}%)', linestyle='--', linewidth=2)
             
    plt.title(f'{target.upper()} {horizon}h Trading Strategy Performance ({model_name})', fontsize=16)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(portfolio_df['date'], portfolio_df['price'], color='black', label=f'{target.upper()} Price')
    
    if not trades_df.empty and 'action' in trades_df.columns:
        buys = trades_df[trades_df['action'] == 'BUY']
        if not buys.empty:
            plt.scatter(buys['date'], buys['price'], marker='^', color='green', s=100, label='Buy')
            
        sells = trades_df[trades_df['action'].isin(['SELL', 'CLOSE'])]
        if not sells.empty:
            plt.scatter(sells['date'], sells['price'], marker='v', color='red', s=100, label='Sell')
            
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend()
    plt.grid(True)
    
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig(f'outputs/trading_simulation/{target}_{horizon}h_{model_name}_trading_results.png', dpi=300)
    plt.close()
    
    print(f"✓ Trading results visualization saved")

def visualize_advanced_predictions(data, test_indices, y_actual, y_pred, y_proba, source, target, horizon, model_name, accuracy):
    price_col = f'{target}_Close'
    if price_col not in data.columns:
        print(f"Price column {price_col} not found, skipping visualization")
        return
        
    test_prices = data.loc[test_indices, price_col]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    axes[0].plot(test_indices, test_prices, color='black', label=f'{target.upper()} Price')
    
    up_mask = y_pred == 1
    if np.any(up_mask):
        up_indices = test_indices[up_mask]
        up_prices = test_prices.loc[up_indices]
        up_probas = y_proba[up_mask]
        up_sizes = up_probas * 100 + 50 
        
        high_conf_up = up_probas > 0.65
        
        if np.any(high_conf_up):
            high_conf_indices = up_indices[high_conf_up.values]
            high_conf_prices = up_prices.loc[high_conf_indices]
            high_conf_sizes = up_sizes[high_conf_up]
            
            axes[0].scatter(high_conf_indices, high_conf_prices, 
                            marker='^', color='green', s=high_conf_sizes, alpha=0.9, 
                            label='High Confidence Up Signal')
                            
        low_conf_up = ~high_conf_up
        if np.any(low_conf_up):
            low_conf_indices = up_indices[low_conf_up.values]
            low_conf_prices = up_prices.loc[low_conf_indices]
            low_conf_sizes = up_sizes[low_conf_up]
            
            axes[0].scatter(low_conf_indices, low_conf_prices, 
                            marker='^', color='green', s=low_conf_sizes, alpha=0.5, 
                            label='Low Confidence Up Signal')
                            
    down_mask = y_pred == 0
    if np.any(down_mask):
        down_indices = test_indices[down_mask]
        down_prices = test_prices.loc[down_indices]
        down_probas = 1 - y_proba[down_mask]
        down_sizes = down_probas * 100 + 50
        
        high_conf_down = down_probas > 0.65
        
        if np.any(high_conf_down):
            high_conf_indices = down_indices[high_conf_down.values]
            high_conf_prices = down_prices.loc[high_conf_indices]
            high_conf_sizes = down_sizes[high_conf_down]
            
            axes[0].scatter(high_conf_indices, high_conf_prices, 
                            marker='v', color='red', s=high_conf_sizes, alpha=0.9,
                            label='High Confidence Down Signal')
                            
        low_conf_down = ~high_conf_down
        if np.any(low_conf_down):
            low_conf_indices = down_indices[low_conf_down.values]
            low_conf_prices = down_prices.loc[low_conf_indices]
            low_conf_sizes = down_sizes[low_conf_down]
            
            axes[0].scatter(low_conf_indices, low_conf_prices, 
                            marker='v', color='red', s=low_conf_sizes, alpha=0.5,
                            label='Low Confidence Down Signal')
                            
    error_mask = y_pred != y_actual
    if np.any(error_mask):
        error_indices = test_indices[error_mask]
        error_prices = test_prices.loc[error_indices]
        axes[0].scatter(error_indices, error_prices, marker='o', color='black', s=50, 
                        facecolors='none', linewidth=2, label='Incorrect Prediction')
                        
    axes[1].fill_between(test_indices, 0, y_proba, 
                         where=y_pred == 1,
                         color='green', alpha=0.4, label='Up Signal Strength')
    axes[1].fill_between(test_indices, 0, 1 - y_proba, 
                         where=y_pred == 0,
                         color='red', alpha=0.4, label='Down Signal Strength')
                         
    axes[1].axhline(y=0.5, color='black', linestyle='--', alpha=0.7)
    axes[1].axhline(y=0.65, color='green', linestyle='--', alpha=0.7, label='High Confidence Threshold')
    axes[1].axhline(y=0.35, color='red', linestyle='--', alpha=0.7)
    
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel('Signal Strength', fontsize=12)
    axes[1].legend(loc='upper left')
    axes[1].grid(True)
    
    title = f'{source.upper()} predicting {target.upper()} {horizon}h price direction ({model_name}, Accuracy: {accuracy:.4f})'
    
    if source.lower() == 'sol' and target.lower() == 'eth' and horizon == 24:
        title += " ★Best Prediction Pair★"
        
    axes[0].set_title(title, fontsize=14)
    axes[0].set_ylabel(f'{target.upper()} Price', fontsize=12)
    axes[0].legend(loc='upper left')
    axes[0].grid(True)
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=14))
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig(f'outputs/figures/{source}_{target}_{horizon}h_{model_name}_prediction.png', dpi=300)
    plt.close()
    
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_actual, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{source.upper()} → {target.upper()} {horizon}h: {model_name}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'outputs/figures/{source}_{target}_{horizon}h_{model_name}_confusion_matrix.png')
    plt.close()
    
    print(f"✓ {model_name} prediction visualization saved")

def create_performance_curves(predictions, source, target, horizon):
    plt.figure(figsize=(10, 8))
    
    for model_name, pred_data in predictions.items():
        if 'y_proba' in pred_data and 'y_test' in pred_data:
            try:
                fpr, tpr, _ = roc_curve(pred_data['y_test'], pred_data['y_proba'])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
            except Exception as e:
                print(f"Error calculating ROC curve: {e}")
                
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title(f'{source.upper()} → {target.upper()} {horizon}h: ROC Curve Comparison', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(f'outputs/figures/{source}_{target}_{horizon}h_roc_curve.png')
    plt.close()
    
    plt.figure(figsize=(10, 8))
    
    for model_name, pred_data in predictions.items():
        if 'y_proba' in pred_data and 'y_test' in pred_data:
            try:
                precision, recall, _ = precision_recall_curve(pred_data['y_test'], pred_data['y_proba'])
                pr_auc = auc(recall, precision)
                plt.plot(recall, precision, label=f'{model_name} (AUC = {pr_auc:.3f})')
            except Exception as e:
                print(f"Error calculating PR curve: {e}")
                
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'{source.upper()} → {target.upper()} {horizon}h: P-R Curve Comparison', fontsize=14)
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(f'outputs/figures/{source}_{target}_{horizon}h_pr_curve.png')
    plt.close()
    
    print("✓ Performance curves saved")

def create_adaptive_combined_signals(target, horizon, predictions_by_source, data):
    print(f"\n{'='*60}")
    print(f"Creating adaptive combined signal system for {target.upper()} {horizon}h prediction")
    print(f"{'='*60}")
    
    if not predictions_by_source:
        print("No source coin predictions available")
        return None, None # Return two None values
        
    price_col = f'{target}_Close'
    if price_col not in data.columns:
        print(f"Price column {price_col} not found, cannot create combined signals")
        return None, None # Return two None values
        
    all_indices = set()
    for source_preds in predictions_by_source.values():
        for model_preds in source_preds.values():
            all_indices.update(model_preds['test_indices'])
            
    test_indices = sorted(all_indices)
    test_prices = data.loc[test_indices, price_col]
    
    combined_df = pd.DataFrame(index=test_indices)
    combined_df[price_col] = test_prices
    
    target_col = f'{target}_future_direction_{horizon}h'
    if target_col in data.columns:
        combined_df['actual'] = data.loc[test_indices, target_col]
        
    for source, source_preds in predictions_by_source.items():
        for model_name, pred_data in source_preds.items():
            pred_indices = pred_data['test_indices']
            combined_df[f'{source}_{model_name}_pred'] = np.nan
            combined_df.loc[pred_indices, f'{source}_{model_name}_pred'] = pred_data['y_pred']
            
            combined_df[f'{source}_{model_name}_proba'] = np.nan
            combined_df.loc[pred_indices, f'{source}_{model_name}_proba'] = pred_data['y_proba']
            
    pred_cols = [col for col in combined_df.columns if col.endswith('_pred')]
    proba_cols = [col for col in combined_df.columns if col.endswith('_proba')]
    
    if pred_cols:
        combined_df['vote_signal'] = combined_df[pred_cols].mean(axis=1).round()
        
    if proba_cols:
        combined_df['weighted_proba'] = combined_df[proba_cols].mean(axis=1)
        combined_df['proba_signal'] = (combined_df['weighted_proba'] > 0.5).astype(float)
        
    if pred_cols:
        combined_df['unanimous_up'] = (combined_df[pred_cols] == 1).all(axis=1)
        combined_df['unanimous_down'] = (combined_df[pred_cols] == 0).all(axis=1)
        combined_df['unanimous_signal'] = np.nan
        combined_df.loc[combined_df['unanimous_up'], 'unanimous_signal'] = 1
        combined_df.loc[combined_df['unanimous_down'], 'unanimous_signal'] = 0
        
    if horizon == 24:
        combined_df['adaptive_signal'] = combined_df['unanimous_signal']
        print("24-hour prediction: Using unanimous signal strategy")
    else:
        combined_df['adaptive_signal'] = combined_df['proba_signal']
        print("100-hour prediction: Using weighted probability strategy")
        
    strategies = ['vote_signal', 'proba_signal', 'unanimous_signal', 'adaptive_signal']
    strategy_metrics = {}
    
    for strategy in strategies:
        if strategy in combined_df.columns:
            valid_data = combined_df.dropna(subset=[strategy, 'actual'])
            if len(valid_data) > 10: 
                try:
                    accuracy = accuracy_score(valid_data['actual'], valid_data[strategy])
                    precision = precision_score(valid_data['actual'], valid_data[strategy], zero_division=0)
                    recall = recall_score(valid_data['actual'], valid_data[strategy], zero_division=0)
                    f1 = f1_score(valid_data['actual'], valid_data[strategy], zero_division=0)
                    
                    strategy_metrics[strategy] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'coverage': len(valid_data) / len(combined_df)
                    }
                    
                    print(f"Strategy '{strategy}': Accuracy = {accuracy:.4f}, F1 = {f1:.4f}, Coverage = {len(valid_data) / len(combined_df):.2f}")
                except Exception as e:
                    print(f"Error evaluating strategy '{strategy}': {e}")
                    
    combined_df.to_csv(f'outputs/{target}_{horizon}h_combined_signals.csv')
    
    visualize_combined_signals(combined_df, target, horizon, strategy_metrics, adaptive=True)
    
    return combined_df, strategy_metrics


def visualize_combined_signals(data, target, horizon, strategy_metrics, adaptive=False):
    price_col = f'{target}_Close'
    if price_col not in data.columns or len(strategy_metrics) == 0:
        return
        
    fig, axes = plt.subplots(2, 1, figsize=(16, 12), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    axes[0].plot(data.index, data[price_col], color='black', label=f'{target.upper()} Price')
    
    strategies = {
        'vote_signal': ('Majority Vote', 'blue'),
        'proba_signal': ('Weighted Probability', 'purple'),
        'unanimous_signal': ('Unanimous Signal', 'red'),
        'adaptive_signal': ('Adaptive Strategy', 'orange')
    }
    
    ordered_strategies = list(strategies.keys())
    if adaptive and 'adaptive_signal' in ordered_strategies:
        ordered_strategies.remove('adaptive_signal')
        ordered_strategies.insert(0, 'adaptive_signal')
        
    for strategy in ordered_strategies:
        if strategy in data.columns and strategy in strategy_metrics:
            name, color = strategies[strategy]
            alpha = 0.9 if strategy == 'adaptive_signal' else 0.6
            size = 120 if strategy == 'adaptive_signal' else 80
            
            up_mask = data[strategy] == 1
            if up_mask.any():
                axes[0].scatter(data[up_mask].index, data.loc[up_mask, price_col], 
                                marker='^', color=color, s=size, alpha=alpha,
                                label=f'{name} Up Signal')
                                
            down_mask = data[strategy] == 0
            if down_mask.any():
                axes[0].scatter(data[down_mask].index, data.loc[down_mask, price_col], 
                                marker='v', color=color, s=size, alpha=alpha,
                                label=f'{name} Down Signal')
                                
    if 'weighted_proba' in data.columns:
        axes[1].fill_between(data.index, 0, data['weighted_proba'], 
                             where=data['weighted_proba'] >= 0.5,
                             color='green', alpha=0.4, label='Up Signal Strength')
        axes[1].fill_between(data.index, 0, data['weighted_proba'], 
                             where=data['weighted_proba'] < 0.5,
                             color='red', alpha=0.4, label='Down Signal Strength')
        axes[1].axhline(y=0.5, color='black', linestyle='--', alpha=0.7)
        axes[1].set_ylim(0, 1)
        axes[1].set_ylabel('Signal Strength', fontsize=12)
        axes[1].legend(loc='upper left')
        axes[1].grid(True)
        
    if adaptive and 'adaptive_signal' in strategy_metrics:
        best_strategy = 'adaptive_signal'
        best_accuracy = strategy_metrics[best_strategy]['accuracy']
        best_name = strategies[best_strategy][0]
    else:
        # Find best strategy by accuracy among available ones
        valid_strategies = {k: v for k, v in strategy_metrics.items() if k in strategies}
        if not valid_strategies:
             best_name = "N/A"
             best_accuracy = 0.0
        else:
            best_strategy = max(valid_strategies.items(), key=lambda x: x[1]['accuracy'])[0]
            best_name = strategies[best_strategy][0]
            best_accuracy = strategy_metrics[best_strategy]['accuracy']

    title = f'{target.upper()} {horizon}h Price Direction Combined Prediction\nBest Strategy: {best_name}, Accuracy: {best_accuracy:.4f}'
    
    if horizon == 24:
        title += "\n(24h prediction recommends unanimous signal)"
    else:
        title += "\n(Long-term prediction recommends weighted probability)"
        
    axes[0].set_title(title, fontsize=16)
    axes[0].set_ylabel(f'{target.upper()} Price', fontsize=12)
    axes[0].legend(loc='upper left')
    axes[0].grid(True)
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=14))
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig(f'outputs/figures/{target}_{horizon}h_combined_signals{"_adaptive" if adaptive else ""}.png', dpi=300)
    plt.close()
    
    plt.figure(figsize=(12, 6))
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'coverage']
    x = np.arange(len(metrics))
    
    num_strategies_plotted = 0
    plotted_strategies = [s for s in strategies if s in strategy_metrics]
    width = 0.8 / len(plotted_strategies) if plotted_strategies else 0.2 # Adjust bar width based on number of strategies

    for i, strategy in enumerate(plotted_strategies):
         name, color = strategies[strategy]
         values = [strategy_metrics[strategy][m] for m in metrics]
         plt.bar(x + i*width - width*(len(plotted_strategies)-1)/2, values, width, label=name, color=color) # Center bars
         num_strategies_plotted +=1
         
    plt.xlabel('Evaluation Metric')
    plt.ylabel('Score')
    plt.title(f'{target.upper()} {horizon}h Strategy Comparison')
    plt.xticks(x, metrics) # Set ticks at the center of groups
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(f'outputs/figures/{target}_{horizon}h_strategy_comparison.png')
    plt.close()
    
    print(f"✓ Combined signal visualization saved")


def main():
    print("\n" + "="*80)
    print("Cryptocurrency Cross-Prediction Model Analysis - Fixed Version")
    print("="*80 + "\n")
    
    data = load_and_clean_data('crypto_features.csv')
    
    data = create_target_variables(data)
    
    data.to_csv('processed_crypto_data.csv')
    
    horizons = [24, 100] 
    pairs = [
        ('sol', 'btc'), 
        ('eth', 'btc'), 
        ('sol', 'eth')  
    ]
    
    all_results = []
    predictions_by_target = {}
    
    prioritized_pairs = []
    for horizon in horizons:
        for source, target in pairs:
            if source.lower() == 'sol' and target.lower() == 'eth' and horizon == 24:
                prioritized_pairs.insert(0, (source, target, horizon))
            else:
                prioritized_pairs.append((source, target, horizon))
                
    for source, target, horizon in prioritized_pairs:
        print(f"\n{'='*40}")
        print(f"Prediction Horizon: {horizon} hours")
        print(f"{'='*40}")
        
        try:
            special_focus = (source.lower() == 'sol' and target.lower() == 'eth' and horizon == 24)
            if special_focus:
                print("\n🔍 Special Focus: SOL predicting ETH 24h (Best prediction pair from research)")
                
            results, predictions, _ = cross_coin_prediction(source, target, horizon, data)
            
            if results:
                all_results.extend(results)
                
                if target not in predictions_by_target:
                    predictions_by_target[target] = {}
                    
                if horizon not in predictions_by_target[target]:
                    predictions_by_target[target][horizon] = {}
                    
                predictions_by_target[target][horizon][source] = predictions
        except Exception as e:
            print(f"✗ Error predicting {source} → {target}: {e}")
            import traceback
            traceback.print_exc()
            
    combined_results = {}
    for target in predictions_by_target:
        combined_results[target] = {}
        for horizon in predictions_by_target[target]:
            try:
                _, strategy_metrics = create_adaptive_combined_signals(
                    target, horizon, predictions_by_target[target][horizon], data
                )
                if strategy_metrics: # Store metrics only if they were successfully calculated
                     combined_results[target][horizon] = strategy_metrics
            except Exception as e:
                print(f"Error creating combined signals for {target} {horizon}h: {e}")
                
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv('outputs/all_prediction_results.csv', index=False)
        
        print("\nAll model prediction results:")
        # Use try-except for groupby in case of empty results or missing columns
        try:
             print(results_df.groupby(['source', 'target', 'horizon', 'model'])['accuracy'].mean())
             best_idx = results_df['accuracy'].idxmax()
             best = results_df.loc[best_idx]
             print(f"\nBest prediction pair: {best['source'].upper()} → {best['target'].upper()} "
                   f"{best['horizon']}h ({best['model']}, Accuracy: {best['accuracy']:.4f})")
        except KeyError as e_key:
             print(f"Could not group results or find best - missing key: {e_key}")
        except Exception as e_res:
             print(f"Could not process results summary: {e_res}")

        # Report generation code removed here

    else:
        print("No successful prediction results")

if __name__ == "__main__":
    try:
        main()
        print("\n" + "="*80)
        print("Analysis complete! All results saved in 'outputs' directory")
        print("="*80)
    except Exception as e:
        import traceback
        print(f"Error during execution: {e}")
        traceback.print_exc()