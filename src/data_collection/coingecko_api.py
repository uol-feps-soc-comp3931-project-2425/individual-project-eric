import time
import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CoinGeckoAPI:
    """
    Client for the CoinGecko API to collect OHLCV data.
    
    As specified in Appendix D.1.1 of the paper:
    - Secondary Source: CoinGecko API
    - Endpoint: /api/v3/coins/{id}/market_chart/range
    - Used for spot checks and cross-validation of data integrity
    """
    
    def __init__(self, base_url: str = 'https://api.coingecko.com/api/v3'):
        """
        Initialize the CoinGecko API client.
        
        Args:
            base_url: Base URL for the CoinGecko API
        """
        self.base_url = base_url
        self.coins_endpoint = '/coins'
        self.rate_limit_per_minute = 10 
        self.request_count = 0
        self.last_minute_start_time = time.time() 
        self.requests_this_minute = 0

    
    def _handle_rate_limit(self):
        """
        Handle rate limiting based on requests per minute.
        """
        current_time = time.time()
        
        if current_time - self.last_minute_start_time >= 60:
            self.requests_this_minute = 0
            self.last_minute_start_time = current_time

        if self.requests_this_minute >= self.rate_limit_per_minute * 0.9:
            wait_time = 60 - (current_time - self.last_minute_start_time)
            if wait_time > 0:
                 logger.info(f"Rate limit approaching ({self.requests_this_minute}/{self.rate_limit_per_minute}). Waiting for {wait_time:.2f} seconds to reset window.")
                 time.sleep(wait_time + 0.1) 
            self.requests_this_minute = 0
            self.last_minute_start_time = time.time()

        self.requests_this_minute += 1


    
    def get_coin_market_chart(self, coin_id: str, vs_currency: str = 'usdt',
                             from_timestamp: int = None, to_timestamp: int = None) -> Dict[str, List]:
        """
        Get market chart data for a specific coin using /market_chart/range.
        
        Args:
            coin_id: CoinGecko coin ID (e.g., 'bitcoin')
            vs_currency: Quote currency (e.g., 'usd', 'usdt')
            from_timestamp: Start time UNIX timestamp (seconds)
            to_timestamp: End time UNIX timestamp (seconds)
            
        Returns:
            Dictionary with 'prices', 'market_caps', and 'total_volumes' data.
            Timestamps within the lists are usually in milliseconds.
        """
        self._handle_rate_limit()
        
        url = f"{self.base_url}{self.coins_endpoint}/{coin_id}/market_chart/range"
        
        params = {
            'vs_currency': vs_currency.lower(), 
            'from': str(from_timestamp),       
            'to': str(to_timestamp)
        }
        
        logger.info(f"Requesting CoinGecko: {url} with params: {params}")

        retry_count = 0
        max_retries = 5
        backoff_factor = 2

        while retry_count < max_retries:
            try:
                headers = {'User-Agent': 'MyCryptoDataCollector/1.0'} 
                response = requests.get(url, params=params, headers=headers, timeout=20) 

                if response.status_code == 429:
                    logger.warning("CoinGecko Rate limit hit (429). Applying backoff.")
                    sleep_time = min(60, backoff_factor ** retry_count) 
                    logger.info(f"Retrying after {sleep_time} seconds...")
                    time.sleep(sleep_time)
                    retry_count += 1
                    continue 
                
                response.raise_for_status() 
                
                data = response.json()
                if isinstance(data, dict) and 'prices' in data and 'total_volumes' in data:
                    logger.info(f"Successfully fetched data for {coin_id}. Prices count: {len(data.get('prices', []))}, Volumes count: {len(data.get('total_volumes', []))}")
                    return data
                else:
                    logger.error(f"Unexpected response format from CoinGecko: {data}")
                    return {'prices': [], 'market_caps': [], 'total_volumes': []} 

            except requests.exceptions.Timeout as e:
                 logger.warning(f"Request timed out: {e}. Retrying ({retry_count+1}/{max_retries})...")
                 time.sleep(backoff_factor ** retry_count) 
                 retry_count += 1
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching market chart data: {e}. Retrying ({retry_count+1}/{max_retries})...")
                sleep_time = backoff_factor ** retry_count
                logger.info(f"Backing off for {sleep_time} seconds")
                time.sleep(sleep_time)
                retry_count += 1
        
        logger.error(f"Failed to fetch market chart data for {coin_id} after {max_retries} retries.")
        return {'prices': [], 'market_caps': [], 'total_volumes': []} # Return empty structure after failures

    
    def get_historical_ohlc(self, symbol: str, vs_currency: str = 'usdt',
                           start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Get historical price and volume data for a specified time range using 
        the /market_chart/range endpoint and constructs a simplified OHLC DataFrame.

        Note: CoinGecko's /market_chart/range granularity depends on the date range. 
        For ranges > 90 days, it's typically daily. Resampling to 1 minute and setting
        O=H=L=C=price is a simplification based on the original code and may not 
        represent true minutely OHLC data. For actual OHLC, consider the /coins/{id}/ohlc endpoint.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH') defined in crypto_id_map.
            vs_currency: Quote currency (default: 'usdt').
            start_date: Start date in 'YYYY-MM-DD' format (inclusive).
            end_date: End date in 'YYYY-MM-DD' format (inclusive for CoinGecko range).
            
        Returns:
            DataFrame with 'datetime', 'open', 'high', 'low', 'close', 'volume'.
            Timestamps are localized to UTC. Returns empty DataFrame on error.
        """
        self.crypto_id_map = { 
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'SOL': 'solana'
        }

        if symbol.upper() not in self.crypto_id_map:
            logger.error(f"Symbol '{symbol}' not found in CoinGecko ID mapping: {list(self.crypto_id_map.keys())}")
            return pd.DataFrame()
            
        coin_id = self.crypto_id_map[symbol.upper()]
        
        try:

            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d') 
            end_dt_inclusive = end_dt + timedelta(days=1) 

            from_timestamp = int(start_dt.timestamp())
            to_timestamp = int(end_dt_inclusive.timestamp())

            logger.info(f"Fetching CoinGecko data for {coin_id} ({symbol}) vs {vs_currency}")
            logger.info(f"Date range: {start_date} (from: {from_timestamp}) to {end_date} (to: {to_timestamp})")

        except ValueError:
             logger.error("Invalid date format. Please use YYYY-MM-DD.")
             return pd.DataFrame()

        chart_data = self.get_coin_market_chart(
            coin_id=coin_id,
            vs_currency=vs_currency,
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp
        )
        
        if not chart_data or not chart_data.get('prices') or not chart_data.get('total_volumes'):
            logger.warning(f"No data returned from CoinGecko API for {symbol} in the specified range.")
            return pd.DataFrame()
        
        try:
            prices_df = pd.DataFrame(chart_data['prices'], columns=['timestamp_ms', 'price'])
            volumes_df = pd.DataFrame(chart_data['total_volumes'], columns=['timestamp_ms', 'volume'])
            
            prices_df['datetime'] = pd.to_datetime(prices_df['timestamp_ms'], unit='ms', utc=True)
            volumes_df['datetime'] = pd.to_datetime(volumes_df['timestamp_ms'], unit='ms', utc=True)
            
            prices_df = prices_df.drop(columns=['timestamp_ms'])
            volumes_df = volumes_df.drop(columns=['timestamp_ms'])


            logger.warning("Applying resampling to 1 minute and forward fill. Note: This might create artificial data points if source granularity is lower (e.g., daily).")
            
            prices_df = prices_df.set_index('datetime').resample('1min').ffill()
            volumes_df = volumes_df.set_index('datetime').resample('1min').ffill() 

            df = pd.merge(
                prices_df, 
                volumes_df, 
                left_index=True, 
                right_index=True, 
                how='outer' 
            )

            df.ffill(inplace=True)
            df.bfill(inplace=True)

            df = df.reset_index()


            df['close'] = df['price']
            df['open'] = df['price'] 
            df['high'] = df['price'] 
            df['low'] = df['price']  
            
            df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]


            start_dt_utc = pd.to_datetime(start_date, utc=True)
            end_dt_exclusive_utc = pd.to_datetime(end_date, utc=True) + timedelta(days=1) 

            df = df[(df['datetime'] >= start_dt_utc) & (df['datetime'] < end_dt_exclusive_utc)].copy() 

            df = df.sort_values(by='datetime').reset_index(drop=True)

            logger.info(f"Successfully processed CoinGecko data for {symbol}. Rows: {len(df)}")
            return df

        except Exception as e:
            logger.error(f"Error processing CoinGecko data for {symbol}: {e}", exc_info=True)
            return pd.DataFrame()



coingecko_client = CoinGeckoAPI()

symbol_to_fetch = 'BTC'        
vs_currency_to_fetch = 'usdt'  
start_date_str = '2023-12-01'
end_date_str = '2024-12-01'   

print(f"Fetching CoinGecko data for {symbol_to_fetch}/{vs_currency_to_fetch} from {start_date_str} to {end_date_str}...")
historical_data_df_cg = coingecko_client.get_historical_ohlc(
    symbol=symbol_to_fetch,
    vs_currency=vs_currency_to_fetch,
    start_date=start_date_str,
    end_date=end_date_str
)

if not historical_data_df_cg.empty:
    print("Successfully fetched data:")
    print(historical_data_df_cg.head())
    print("...")
    print(historical_data_df_cg.tail())
    print(f"Total rows fetched: {len(historical_data_df_cg)}")
    
    min_date = historical_data_df_cg['datetime'].min()
    max_date = historical_data_df_cg['datetime'].max()
    print(f"Data ranges from {min_date} to {max_date}")

    # csv_filename_cg = f"COINGECKO_{symbol_to_fetch}_{vs_currency_to_fetch}_{start_date_str}_to_{end_date_str}.csv"
    # historical_data_df_cg.to_csv(csv_filename_cg, index=False)
    # print(f"Data saved to {csv_filename_cg}")
else:
    print("Failed to fetch data from CoinGecko or no data available.")