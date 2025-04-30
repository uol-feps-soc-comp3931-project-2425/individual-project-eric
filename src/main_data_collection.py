"""
Main script for collecting and preprocessing cryptocurrency data.
"""
import os
import logging
from datetime import datetime

from src.data_collection.data_collector import DataCollector
from src.preprocessing.preprocessor import DataPreprocessor
from src.utils.config import (
    CRYPTOCURRENCIES, QUOTE_CURRENCY, TIMEFRAME,
    START_DATE, END_DATE
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_collection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to collect and preprocess cryptocurrency data.
    """
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    collector = DataCollector(data_dir='data/raw')
    
    logger.info(f"Collecting data for {CRYPTOCURRENCIES} from {START_DATE} to {END_DATE}")
    data = collector.collect_all_data(
        symbols=CRYPTOCURRENCIES,
        quote=QUOTE_CURRENCY,
        start_date=START_DATE,
        end_date=END_DATE,
        timeframe=TIMEFRAME,
        use_coingecko=True  # Use CoinGecko as secondary source for validation
    )
    
    preprocessor = DataPreprocessor(
        data_dir='data/raw',
        processed_dir='data/processed'
    )
    
    logger.info(f"Preprocessing data for {CRYPTOCURRENCIES}")
    preprocessed_data = preprocessor.preprocess_all_data(
        symbols=CRYPTOCURRENCIES,
        quote=QUOTE_CURRENCY,
        timeframe=TIMEFRAME
    )
    
    logger.info("Data collection and preprocessing completed")

if __name__ == "__main__":
    main()
