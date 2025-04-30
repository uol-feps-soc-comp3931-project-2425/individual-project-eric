"""
Main script for dataset preparation in cryptocurrency price prediction.
"""
import os
import logging
import numpy as np
from datetime import datetime

from src.dataset.dataset_preparation import DatasetPreparation
from src.utils.config import (
    CRYPTOCURRENCIES, QUOTE_CURRENCY,
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, SEQUENCE_LENGTH
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dataset_preparation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to prepare datasets for cryptocurrency price prediction.
    """
    os.makedirs('data/dataset', exist_ok=True)
    
    dataset_preparation = DatasetPreparation(
        features_dir='data/features',
        dataset_dir='data/dataset',
        train_split=TRAIN_SPLIT,
        val_split=VAL_SPLIT,
        test_split=TEST_SPLIT,
        sequence_length=SEQUENCE_LENGTH
    )
    
    logger.info(f"Preparing datasets for {CRYPTOCURRENCIES}")
    datasets_dict = dataset_preparation.prepare_datasets(
        symbols=CRYPTOCURRENCIES,
        quote=QUOTE_CURRENCY,
        target_steps_ahead=[1, 5, 15, 30]
    )
    
    for horizon, datasets in datasets_dict.items():
        logger.info(f"Dataset statistics for {horizon} prediction horizon:")
        for name, data in datasets.items():
            if isinstance(data, np.ndarray):
                logger.info(f"  {name}: shape {data.shape}, mean {data.mean():.4f}, std {data.std():.4f}")
    
    logger.info("Dataset preparation completed")

if __name__ == "__main__":
    main()
