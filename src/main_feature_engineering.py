"""
Main script for feature engineering in cryptocurrency price prediction.
"""
import os
import logging
from datetime import datetime

from src.feature_engineering.feature_engineer import FeatureEngineer
from src.utils.config import (
    CRYPTOCURRENCIES, QUOTE_CURRENCY,
    LOG_RETURN_LAGS, VOLATILITY_WINDOWS, RSI_PERIOD, LAGGED_RETURNS
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("feature_engineering.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to engineer features for cryptocurrency price prediction.
    """
    os.makedirs('data/features', exist_ok=True)
    
    feature_engineer = FeatureEngineer(
        processed_dir='data/processed',
        features_dir='data/features',
        log_return_lags=LOG_RETURN_LAGS,
        volatility_windows=VOLATILITY_WINDOWS,
        rsi_period=RSI_PERIOD,
        lagged_returns=LAGGED_RETURNS
    )
    
    logger.info(f"Engineering features for {CRYPTOCURRENCIES}")
    features_dict = feature_engineer.engineer_features_for_all_symbols(
        symbols=CRYPTOCURRENCIES,
        quote=QUOTE_CURRENCY
    )
    
    for symbol, df in features_dict.items():
        logger.info(f"Engineered {len(df)} rows of features for {symbol}")
        logger.info(f"Features: {list(df.columns)}")
    
    logger.info("Feature engineering completed")

if __name__ == "__main__":
    main()
