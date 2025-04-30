import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BinanceAPI:

    
    def __init__(self, base_url: str = 'https://api.binance.com'):
        """
        Initialize the Binance API client.
        
        Args:
            base_url: Base URL for the Binance API
        """
        self.base_url = base_url
        self.klines_endpoint = '/api/v3/klines'
        self.rate_limit_per_minute = 1200  
        self.request_count = 0
        self.last_request_time = time.time()
    
    def _handle_rate_limit(self):
        """
        Handle rate limiting with exponential backoff as mentioned in the paper.
        """
        self.request_count += 1
        
        current_time = time.time()
        if current_time - self.last_request_time >= 60:
            self.request_count = 1
            self.last_request_time = current_time
            return
        
        if self.request_count >= self.rate_limit_per_minute * 0.9: 
            backoff_time = 1 
            if self.request_count > self.rate_limit_per_minute: 
               backoff_time = min(60, 2 ** (self.request_count / self.rate_limit_per_minute)) 
            
            logger.info(f"Rate limit approaching/exceeded, backing off for {backoff_time:.2f} seconds")
            time.sleep(backoff_time)
            self.request_count = 0 
            self.last_request_time = time.time()

    
    def get_klines(self, symbol: str, interval: str = '1m', 
                  start_time: Optional[int] = None, end_time: Optional[int] = None,
                  limit: int = 1000) -> List[List]:
        self._handle_rate_limit()
        
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time 
        
        url = f"{self.base_url}{self.klines_endpoint}"
        
        retry_count = 0
        max_retries = 5 
        backoff_factor = 2 

        while retry_count < max_retries:
            try:
                response = requests.get(url, params=params, timeout=10) # Added timeout
                if response.status_code == 429: # Rate limit error
                    logger.warning("Rate limit hit (429). Applying backoff.")
                    retry_after = int(response.headers.get('Retry-After', backoff_factor ** retry_count))
                    sleep_time = min(retry_after, 60) 
                    logger.info(f"Retrying after {sleep_time} seconds...")
                    time.sleep(sleep_time)
                    retry_count += 1
                    continue 
                elif response.status_code == 418: 
                     logger.error(f"IP banned (418). Stopping request. Check your request frequency.")
                     sleep_time = 60 * 5 
                     time.sleep(sleep_time)
                     retry_count +=1 
                     continue

                response.raise_for_status() 
                return response.json()
            
            except requests.exceptions.Timeout as e:
                 logger.warning(f"Request timed out: {e}. Retrying ({retry_count+1}/{max_retries})...")
                 time.sleep(backoff_factor ** retry_count) 
                 retry_count += 1
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching klines data: {e}. Retrying ({retry_count+1}/{max_retries})...")
                sleep_time = backoff_factor ** retry_count
                logger.info(f"Backing off for {sleep_time} seconds")
                time.sleep(sleep_time)
                retry_count += 1
        
        logger.error(f"Failed to fetch klines data after {max_retries} retries.")
        return [] 

    
    def get_historical_klines(self, symbol: str, interval: str, 
                             start_date: str, end_date: str) -> pd.DataFrame:

        try:
             start_dt = datetime.strptime(start_date, '%Y-%m-%d')
             end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
             logger.error("Invalid date format. Please use YYYY-MM-DD.")
             return pd.DataFrame()

        start_ts = int(start_dt.timestamp() * 1000)
        end_ts = int(end_dt.timestamp() * 1000) 
        
        logger.info(f"Fetching data for {symbol} from {start_date} ({start_ts}) to {end_date} ({end_ts}) with interval {interval}")

        all_klines = []
        current_start = start_ts
        
        interval_map = {
            'm': 60 * 1000,
            'h': 60 * 60 * 1000,
            'd': 24 * 60 * 60 * 1000,
        }
        interval_unit = interval[-1]
        interval_value = int(interval[:-1]) if len(interval) > 1 else 1 
        
        if interval_unit not in interval_map:
             logger.error(f"Unsupported interval format: {interval}. Use formats like '1m', '5m', '1h', '1d'.")
             return pd.DataFrame()


        batch_duration_ms = 1000 * interval_value * interval_map[interval_unit]
        
        fetch_count = 0
        max_fetches = 10000 

        while current_start < end_ts and fetch_count < max_fetches:
            fetch_count += 1

            batch_end = min(current_start + batch_duration_ms - interval_map[interval_unit], end_ts) 

            logger.debug(f"Fetching batch: startTime={current_start}, endTime={batch_end}")

            klines = self.get_klines(
                symbol=symbol,
                interval=interval,
                start_time=current_start,
                end_time=batch_end, 
                limit=1000          
            )
            
            if not klines:
                if current_start >= end_ts:
                     logger.info(f"Reached end timestamp {end_ts}. No more data expected.")
                     break 
                logger.warning(f"No data returned for {symbol} from {datetime.fromtimestamp(current_start/1000)} to {datetime.fromtimestamp(batch_end/1000)}. Trying next possible start time.")
                current_start = batch_end + 1 
                time.sleep(1) 
                continue
                
            all_klines.extend(klines)
            

            last_kline_ts = klines[-1][0]
            current_start = last_kline_ts + (interval_value * interval_map[interval_unit])

            first_ts_dt = datetime.fromtimestamp(klines[0][0]/1000)
            last_ts_dt = datetime.fromtimestamp(last_kline_ts/1000)
            logger.info(f"Collected {len(klines)} candles for {symbol} from {first_ts_dt} to {last_ts_dt} (UTC). Next start: {datetime.fromtimestamp(current_start/1000)} UTC")
            
            time.sleep(0.2) 

        if fetch_count >= max_fetches:
             logger.warning("Maximum fetch count reached. Data might be incomplete.")

        if not all_klines:
            logger.warning(f"No data collected for {symbol} from {start_date} to {end_date}")
            return pd.DataFrame()
            
        logger.info(f"Total candles collected before processing: {len(all_klines)}")

        df = pd.DataFrame(all_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        

        df['timestamp'] = pd.to_numeric(df['timestamp'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True) 

        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                          'quote_asset_volume', 'number_of_trades', 
                          'taker_buy_base_asset_volume', 
                          'taker_buy_quote_asset_volume']
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce') 

        ohlcv_df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy() 
        ohlcv_df = ohlcv_df.rename(columns={'timestamp': 'datetime'})

        ohlcv_df = ohlcv_df.drop_duplicates(subset=['datetime'])
        ohlcv_df = ohlcv_df.sort_values(by='datetime').reset_index(drop=True)


        start_dt_utc = pd.to_datetime(start_date, utc=True)
        end_dt_utc = pd.to_datetime(end_date, utc=True)
        
        ohlcv_df = ohlcv_df[(ohlcv_df['datetime'] >= start_dt_utc) & (ohlcv_df['datetime'] < end_dt_utc)]

        logger.info(f"Successfully collected and processed {len(ohlcv_df)} candles for {symbol} from {start_date} to {end_date}")
        
        return ohlcv_df


binance_client = BinanceAPI()

symbol_to_fetch = 'BTCUSDT'
interval_to_fetch = '1m'  
start_date_str = '2023-12-01'
end_date_str = '2024-12-01' 

print(f"Fetching {interval_to_fetch} data for {symbol_to_fetch} from {start_date_str} to {end_date_str}...")
historical_data_df = binance_client.get_historical_klines(
    symbol=symbol_to_fetch,
    interval=interval_to_fetch,
    start_date=start_date_str,
    end_date=end_date_str
)

# 4. Check the results
if not historical_data_df.empty:
    print("Successfully fetched data:")
    print(historical_data_df.head()) # Print the first few rows
    print("...")
    print(historical_data_df.tail()) # Print the last few rows
    print(f"Total rows fetched: {len(historical_data_df)}")
    # Optional: Save the data to a CSV file
    # csv_filename = f"{symbol_to_fetch}_{interval_to_fetch}_{start_date_str}_to_{end_date_str}.csv"
    # historical_data_df.to_csv(csv_filename, index=False)
    # print(f"Data saved to {csv_filename}")
else:
    print("Failed to fetch data or no data available for the specified period.")