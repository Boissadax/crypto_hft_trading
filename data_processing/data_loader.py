"""
Data loader for cryptocurrency order book data.
Handles asynchronous data loading and preprocessing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class OrderBookDataLoader:
    """
    Loads and preprocesses cryptocurrency order book data.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the raw data directory
        """
        self.data_path = Path(data_path)
        self.data_cache = {}
        
    def load_symbol_data(self, symbol: str) -> pd.DataFrame:
        """
        Load order book data for a specific symbol.
        
        Args:
            symbol: Symbol name (e.g., 'ETH_EUR', 'XBT_EUR')
            
        Returns:
            DataFrame with order book data
        """
        if symbol in self.data_cache:
            return self.data_cache[symbol]
            
        file_path = self.data_path / f"{symbol}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        logger.info("Loading data for %s", symbol)
        
        # Load data with proper types
        df = pd.read_csv(
            file_path,
            dtype={
                'price': np.float64,
                'volume': np.float64,
                'timestamp': np.float64,
                'side': str,
                'level': np.int32
            }
        )
        
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Cache the data
        self.data_cache[symbol] = df
        
        logger.info("Loaded %d records for %s", len(df), symbol)
        return df
    
    def load_all_symbols(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Load data for all symbols.
        
        Args:
            symbols: List of symbol names
            
        Returns:
            Dictionary mapping symbol names to DataFrames
        """
        data = {}
        for symbol in symbols:
            data[symbol] = self.load_symbol_data(symbol)
        return data
    
    def get_order_book_snapshot(self, 
                               df: pd.DataFrame, 
                               timestamp: float, 
                               max_levels: int = 10) -> Dict:
        """
        Get order book snapshot at a specific timestamp.
        
        Args:
            df: Order book data DataFrame
            timestamp: Target timestamp
            max_levels: Maximum number of levels to include
            
        Returns:
            Dictionary with bid and ask levels
        """
        # Find data points closest to the timestamp
        mask = df['timestamp'] <= timestamp
        if not mask.any():
            return {'bids': [], 'asks': []}
            
        recent_data = df[mask]
        
        # Get the most recent data for each level and side
        latest_bids = (recent_data[recent_data['side'] == 'bid']
                      .groupby('level')
                      .last()
                      .sort_values('level')
                      .head(max_levels))
        
        latest_asks = (recent_data[recent_data['side'] == 'ask']
                      .groupby('level')
                      .last()
                      .sort_values('level')
                      .head(max_levels))
        
        bids = [(row['price'], row['volume']) for _, row in latest_bids.iterrows()]
        asks = [(row['price'], row['volume']) for _, row in latest_asks.iterrows()]
        
        return {
            'bids': bids,
            'asks': asks,
            'timestamp': timestamp
        }
    
    def synchronize_data(self, 
                        data_dict: Dict[str, pd.DataFrame],
                        time_window: float = 0.1) -> pd.DataFrame:
        """
        Synchronize order book data across multiple symbols
        
        Args:
            data: Dictionary of symbol -> DataFrame
            time_window: Time window for synchronization (not currently used)
        """
        """
        Synchronize data from multiple symbols.
        
        Args:
            data_dict: Dictionary of symbol DataFrames
            time_window: Time window for synchronization (seconds)
            
        Returns:
            Synchronized DataFrame with all symbols
        """
        # Get all unique timestamps
        all_timestamps = set()
        for df in data_dict.values():
            all_timestamps.update(df['timestamp'].values)
        
        timestamps = sorted(all_timestamps)
        
        synchronized_data = []
        
        for ts in timestamps:
            row = {'timestamp': ts}
            
            for symbol, df in data_dict.items():
                snapshot = self.get_order_book_snapshot(df, ts)
                
                # Add best bid/ask
                if snapshot['bids']:
                    row[f'{symbol}_best_bid'] = snapshot['bids'][0][0]
                    row[f'{symbol}_best_bid_vol'] = snapshot['bids'][0][1]
                else:
                    row[f'{symbol}_best_bid'] = np.nan
                    row[f'{symbol}_best_bid_vol'] = np.nan
                    
                if snapshot['asks']:
                    row[f'{symbol}_best_ask'] = snapshot['asks'][0][0]
                    row[f'{symbol}_best_ask_vol'] = snapshot['asks'][0][1]
                else:
                    row[f'{symbol}_best_ask'] = np.nan
                    row[f'{symbol}_best_ask_vol'] = np.nan
                
                # Add mid price
                if (f'{symbol}_best_bid' in row and 
                    f'{symbol}_best_ask' in row and
                    not pd.isna(row[f'{symbol}_best_bid']) and 
                    not pd.isna(row[f'{symbol}_best_ask'])):
                    row[f'{symbol}_mid'] = (row[f'{symbol}_best_bid'] + 
                                          row[f'{symbol}_best_ask']) / 2
                else:
                    row[f'{symbol}_mid'] = np.nan
            
            synchronized_data.append(row)
        
        df_sync = pd.DataFrame(synchronized_data)
        df_sync['datetime'] = pd.to_datetime(df_sync['timestamp'], unit='s')
        
        return df_sync.dropna()  # Remove rows with missing data

class DataPreprocessor:
    """
    Preprocesses order book data for machine learning.
    """
    
    @staticmethod
    def calculate_returns(prices: pd.Series, periods: List[int]) -> pd.DataFrame:
        """
        Calculate returns over multiple periods.
        
        Args:
            prices: Price series
            periods: List of periods for return calculation
            
        Returns:
            DataFrame with returns for each period
        """
        returns_df = pd.DataFrame()
        
        for period in periods:
            returns_df[f'return_{period}'] = prices.pct_change(period)
            
        return returns_df
    
    @staticmethod
    def calculate_volatility(prices: pd.Series, window: int) -> pd.Series:
        """
        Calculate rolling volatility.
        
        Args:
            prices: Price series
            window: Rolling window size
            
        Returns:
            Volatility series
        """
        returns = prices.pct_change()
        return returns.rolling(window=window).std()
    
    @staticmethod
    def remove_outliers(df: pd.DataFrame, columns: List[str], 
                       threshold: float = 3.0) -> pd.DataFrame:
        """
        Remove outliers using z-score method.
        
        Args:
            df: Input DataFrame
            columns: Columns to check for outliers
            threshold: Z-score threshold
            
        Returns:
            DataFrame with outliers removed
        """
        df_clean = df.copy()
        
        for col in columns:
            if col in df_clean.columns:
                z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / 
                                 df_clean[col].std())
                df_clean = df_clean[z_scores <= threshold]
        
        return df_clean
