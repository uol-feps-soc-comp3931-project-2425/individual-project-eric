import os
import pandas as pd
import logging
from typing import List, Dict, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path

try:
    from .binance_api import BinanceAPI
    from .coingecko_api import CoinGeckoAPI
    from ..utils.config import (
        CRYPTOCURRENCIES, QUOTE_CURRENCY, TIMEFRAME,
        START_DATE, END_DATE, BINANCE_BASE_URL, COINGECKO_BASE_URL
    )
    CONFIG_LOADED = True
except ImportError as e:
    logging.warning(f"Could not import config from ..utils.config: {e}. Using fallback defaults.")
    try:
        from binance_api import BinanceAPI
        from coingecko_api import CoinGeckoAPI
    except ImportError:
         logging.error("Failed to import BinanceAPI or CoinGeckoAPI. Ensure they are available.")
         raise

    CRYPTOCURRENCIES = ['BTC', 'ETH']
    QUOTE_CURRENCY = 'USDT'
    TIMEFRAME = '1h'
    START_DATE = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    END_DATE = datetime.now().strftime('%Y-%m-%d')
    BINANCE_BASE_URL = 'https://api.binance.com'
    COINGECKO_BASE_URL = 'https://api.coingecko.com/api/v3'
    CONFIG_LOADED = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataCollector:
    """
    Collects cryptocurrency data from multiple sources (Binance, CoinGecko)
    and saves it to disk.

    Relies on BinanceAPI and CoinGeckoAPI classes for fetching data.
    """

    def __init__(self, data_dir: Union[str, Path] = '../../data/raw'):
        """
        Initialize the DataCollector.

        Args:
            data_dir: Directory path (string or Path object) to save the collected data.
                      Defaults to '../../data/raw' relative to where the script might be run.
                      Consider using an absolute path or configuring via environment variables
                      for better robustness.
        """
        self.data_dir = Path(data_dir).resolve()
        self.binance_api = BinanceAPI(base_url=BINANCE_BASE_URL)
        self.coingecko_api = CoinGeckoAPI(base_url=COINGECKO_BASE_URL)

        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Data directory set to: {self.data_dir}")
        except OSError as e:
            logger.error(f"Failed to create data directory {self.data_dir}: {e}")


    def collect_binance_data(self, symbol: str, quote: str = QUOTE_CURRENCY,
                           start_date: str = START_DATE, end_date: str = END_DATE,
                           timeframe: str = TIMEFRAME) -> Optional[pd.DataFrame]:
        """
        Collect OHLCV data from Binance using the BinanceAPI class.

        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC').
            quote: Quote currency (e.g., 'USDT').
            start_date: Start date in 'YYYY-MM-DD' format.
            end_date: End date in 'YYYY-MM-DD' format.
            timeframe: Timeframe for the data (e.g., '1m', '1h', '1d').

        Returns:
            DataFrame with OHLCV data and a 'symbol' column, or None if collection fails.
        """
        trading_pair = f"{symbol.upper()}{quote.upper()}"

        logger.info(f"Attempting to collect Binance data for {trading_pair} ({timeframe}) "
                    f"from {start_date} to {end_date}")

        try:
            df = self.binance_api.get_historical_klines(
                symbol=trading_pair,
                interval=timeframe,
                start_date=start_date,
                end_date=end_date
            )

            if df is None or df.empty:
                logger.warning(f"No data returned from Binance for {trading_pair} in the specified range.")
                return None

            df['symbol'] = symbol.upper()

            logger.info(f"Successfully collected {len(df)} rows of Binance data for {trading_pair}")
            return df

        except Exception as e:
            logger.error(f"An error occurred while collecting Binance data for {trading_pair}: {e}", exc_info=True)
            return None

    def collect_coingecko_data(self, symbol: str, quote: str = QUOTE_CURRENCY,
                             start_date: str = START_DATE, end_date: str = END_DATE) -> Optional[pd.DataFrame]:
        """
        Collect historical price/volume data from CoinGecko using CoinGeckoAPI.

        Note: CoinGecko's API determines granularity based on the date range.
        The underlying CoinGeckoAPI class constructs a simplified OHLC DataFrame
        where O=H=L=C=price, which may not be true OHLC depending on source granularity.
        This method does not use the 'timeframe' parameter.

        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC') - must exist in CoinGeckoAPI's map.
            quote: Quote currency (e.g., 'usdt').
            start_date: Start date in 'YYYY-MM-DD' format.
            end_date: End date in 'YYYY-MM-DD' format.

        Returns:
            DataFrame with simplified OHLCV data and a 'symbol' column, or None if collection fails.
        """
        logger.info(f"Attempting to collect CoinGecko data for {symbol.upper()}/{quote.lower()} "
                    f"from {start_date} to {end_date}")

        try:
            df = self.coingecko_api.get_historical_ohlc(
                symbol=symbol.upper(),
                vs_currency=quote.lower(),
                start_date=start_date,
                end_date=end_date
            )

            if df is None or df.empty:
                logger.warning(f"No data returned from CoinGecko for {symbol.upper()} in the specified range.")
                return None

            df['symbol'] = symbol.upper()

            logger.info(f"Successfully collected {len(df)} rows of CoinGecko data for {symbol.upper()}")
            return df

        except Exception as e:
            logger.error(f"An error occurred while collecting CoinGecko data for {symbol.upper()}: {e}", exc_info=True)
            return None

    def collect_all_data(self,
                       symbols: List[str] = CRYPTOCURRENCIES,
                       quote: str = QUOTE_CURRENCY,
                       start_date: str = START_DATE,
                       end_date: str = END_DATE,
                       timeframe: str = TIMEFRAME,
                       save_binance: bool = True,
                       save_coingecko: bool = False,
                       use_coingecko: bool = False
                       ) -> Dict[str, Dict[str, Optional[pd.DataFrame]]]:
        """
        Collects data for multiple cryptocurrencies from Binance and optionally CoinGecko.
        Saves the collected data to CSV files based on save flags.

        Args:
            symbols: List of cryptocurrency symbols (e.g., ['BTC', 'ETH']).
            quote: Quote currency (e.g., 'USDT').
            start_date: Start date in 'YYYY-MM-DD' format.
            end_date: End date in 'YYYY-MM-DD' format.
            timeframe: Timeframe for Binance data (e.g., '1m', '1h').
            save_binance: If True, save collected Binance data to CSV.
            save_coingecko: If True, save collected CoinGecko data to CSV.
            use_coingecko: If True, attempt to collect data from CoinGecko.

        Returns:
            A dictionary where keys are symbols. Each symbol maps to another dictionary
            containing 'binance' and/or 'coingecko' keys, holding the respective
            DataFrames (or None if collection failed).
            Example: {'BTC': {'binance': df_btc_binance, 'coingecko': df_btc_coingecko}}
        """
        collect_cg = use_coingecko or save_coingecko

        all_data: Dict[str, Dict[str, Optional[pd.DataFrame]]] = {}

        for symbol in symbols:
            symbol_upper = symbol.upper()
            all_data[symbol_upper] = {'binance': None, 'coingecko': None}

            logger.info(f"--- Processing symbol: {symbol_upper} ---")
            binance_df = self.collect_binance_data(
                symbol=symbol_upper,
                quote=quote,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe
            )

            if binance_df is not None:
                all_data[symbol_upper]['binance'] = binance_df

                if save_binance:
                    output_filename = f"BINANCE_{symbol_upper}{quote.upper()}_{timeframe}_{start_date}_to_{end_date}.csv"
                    output_path = self.data_dir / output_filename
                    try:
                        binance_df.to_csv(output_path, index=False)
                        logger.info(f"Saved Binance data for {symbol_upper} to {output_path}")
                    except Exception as e:
                        logger.error(f"Failed to save Binance data for {symbol_upper} to {output_path}: {e}")
            else:
                 logger.warning(f"Skipping saving for Binance {symbol_upper} due to collection failure.")

            if collect_cg:
                coingecko_df = self.collect_coingecko_data(
                    symbol=symbol_upper,
                    quote=quote,
                    start_date=start_date,
                    end_date=end_date
                )

                if coingecko_df is not None:
                    all_data[symbol_upper]['coingecko'] = coingecko_df

                    if save_coingecko:
                        output_filename_cg = f"COINGECKO_{symbol_upper}{quote.lower()}_{start_date}_to_{end_date}.csv"
                        output_path_cg = self.data_dir / output_filename_cg
                        try:
                            coingecko_df.to_csv(output_path_cg, index=False)
                            logger.info(f"Saved CoinGecko data for {symbol_upper} to {output_path_cg}")
                        except Exception as e:
                            logger.error(f"Failed to save CoinGecko data for {symbol_upper} to {output_path_cg}: {e}")
                else:
                     logger.warning(f"Skipping saving for CoinGecko {symbol_upper} due to collection failure.")

        logger.info("--- Data collection finished ---")
        return all_data

if __name__ == "__main__":
    if not CONFIG_LOADED:
        logger.warning("Running __main__ block with default fallback configurations.")

    script_dir = Path(__file__).parent
    default_data_dir = script_dir / '../../data/raw'

    collector = DataCollector(data_dir=default_data_dir)

    collected_data = collector.collect_all_data(
        symbols=CRYPTOCURRENCIES,
        quote=QUOTE_CURRENCY,
        start_date=START_DATE,
        end_date=END_DATE,
        timeframe=TIMEFRAME,
        save_binance=True,
        save_coingecko=True,
        use_coingecko=True
    )

    if 'BTC' in collected_data and collected_data['BTC'].get('binance') is not None:
        logger.info("Accessing collected BTC Binance data (first 5 rows):")
        print(collected_data['BTC']['binance'].head())

    if 'BTC' in collected_data and collected_data['BTC'].get('coingecko') is not None:
        logger.info("Accessing collected BTC CoinGecko data (first 5 rows):")
        print(collected_data['BTC']['coingecko'].head())