"""
Feature Engineering Engine

Orchestrates the complete feature extraction pipeline for HFT trading data:
- Loads raw order book data from CSV files
- Maintains incremental order book state for each symbol
- Synchronizes asynchronous data using exact timestamp union
- Extracts order book and temporal features
- Returns aligned DataFrame with microsecond precision timestamps
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
from collections import defaultdict

from .synchronization import AsynchronousSync, SyncConfig
from .order_book_features import OrderBookFeatureExtractor
from .time_series_features import TimeSeriesFeatureExtractor

# Import data cache system
import sys
sys.path.append(str(Path(__file__).parent.parent))
from data_cache import ensure_processed

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Main feature engineering orchestrator for HFT trading data.
    
    Handles the complete pipeline from raw CSV data to synchronized features:
    - Order book state management per symbol
    - Asynchronous data synchronization with microsecond precision
    - Feature extraction (order book + temporal)
    - Data alignment across multiple symbols
    """
    
    def __init__(self, 
                 symbols: List[str] = None,
                 sync_config: SyncConfig = None,
                 max_levels: int = 10,
                 dataset_id: str = "DATA_0"):
        """
        Initialize feature engineering pipeline
        
        Args:
            symbols: List of trading symbols (e.g., ["BTC", "ETH"])
            sync_config: Configuration for asynchronous synchronization
            max_levels: Maximum order book levels to process
            dataset_id: Dataset identifier (e.g., "DATA_0", "DATA_1", "DATA_2")
        """
        self.symbols = symbols or ["BTC", "ETH"]
        self.sync_config = sync_config or SyncConfig()
        self.max_levels = max_levels
        self.dataset_id = dataset_id
        
        # Initialize components
        self.synchronizer = AsynchronousSync(
            config=self.sync_config,
            symbols=self.symbols
        )
        self.ob_extractor = OrderBookFeatureExtractor(max_levels=max_levels)
        self.ts_extractor = TimeSeriesFeatureExtractor()
        
        # Order book state for each symbol
        self.current_books: Dict[str, Dict[str, Dict[int, Tuple[float, float]]]] = {
            symbol: {"bid": {}, "ask": {}} for symbol in self.symbols
        }
        
        logger.info(f"FeatureEngineer initialized for symbols: {self.symbols}")
        
    def load_raw_data(self, data_dir: str = "raw_data") -> pd.DataFrame:
        """
        Load raw order book data using cached Parquet files
        
        Args:
            data_dir: Directory containing CSV files (legacy parameter, now uses dataset_id)
            
        Returns:
            DataFrame with columns: symbol, price, volume, timestamp, side, level
            
        Raises:
            FileNotFoundError: If required files are missing
            ValueError: If data structure is invalid
        """
        logger.info(f"Loading data for dataset: {self.dataset_id}")
        
        # Get cached parquet files
        try:
            parquet_paths = ensure_processed(self.dataset_id)
        except Exception as e:
            raise FileNotFoundError(f"Failed to prepare cached data for {self.dataset_id}: {e}")
        
        # Map symbols to our internal naming
        symbol_mapping = {
            "ETH": "ETH",
            "XBT": "BTC"  # Convert XBT to BTC for consistency
        }
        
        dfs = []
        
        for parquet_symbol, parquet_path in parquet_paths.items():
            internal_symbol = symbol_mapping.get(parquet_symbol, parquet_symbol)
            
            if internal_symbol not in self.symbols:
                logger.info(f"Skipping {parquet_symbol} (not in requested symbols: {self.symbols})")
                continue
                
            logger.info(f"Loading {internal_symbol} data from cached Parquet: {parquet_path}")
            
            # Load from Parquet - much faster than CSV
            df = pd.read_parquet(parquet_path)
            
            # Validate columns
            required_cols = ["price", "volume", "timestamp", "side", "level"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns in {parquet_path}: {missing_cols}")
                
            # Add symbol column with internal naming
            df["symbol"] = internal_symbol
            
            # Convert timestamp to microseconds (int)
            df["timestamp_us"] = (df["timestamp"] * 1_000_000).astype(np.int64)
            
            # Ensure proper data types
            df["price"] = df["price"].astype(np.float64)
            df["volume"] = df["volume"].astype(np.float64) 
            df["level"] = df["level"].astype(np.int8)
            
            # Validate side values
            valid_sides = {"bid", "ask"}
            invalid_sides = set(df["side"].unique()) - valid_sides
            if invalid_sides:
                raise ValueError(f"Invalid side values in {parquet_path}: {invalid_sides}")
                
            dfs.append(df)
            logger.info(f"Loaded {len(df):,} events for {internal_symbol}")
            
        if not dfs:
            raise ValueError(f"No data loaded for symbols: {self.symbols}")
            
        # Combine all data
        df_combined = pd.concat(dfs, ignore_index=True)
        
        # Sort by timestamp for chronological processing
        df_combined = df_combined.sort_values("timestamp_us")
        
        logger.info(f"Combined dataset: {len(df_combined):,} total events across {len(self.symbols)} symbols")
        
        return df_combined
        
    def _update_order_book(self, symbol: str, price: float, volume: float, 
                          side: str, level: int) -> Dict[str, Any]:
        """
        Update order book state for a symbol and return complete snapshot
        
        Args:
            symbol: Trading symbol
            price: Price level
            volume: Volume at this level  
            side: "bid" or "ask"
            level: Order book level (1 = best)
            
        Returns:
            Complete order book snapshot with features
        """
        if symbol not in self.current_books:
            self.current_books[symbol] = {"bid": {}, "ask": {}}
            
        # Update the specific level
        if volume > 0:
            self.current_books[symbol][side][level] = (price, volume)
        else:
            # Remove level if volume is 0
            self.current_books[symbol][side].pop(level, None)
            
        # Create complete order book snapshot
        order_book = {
            "bid": self.current_books[symbol]["bid"].copy(),
            "ask": self.current_books[symbol]["ask"].copy()
        }
        
        # Calculate basic features for this snapshot
        features = {"order_book": order_book}
        
        # Add basic price/volume features if data available
        if order_book["bid"] and order_book["ask"]:
            best_bid_level = min(order_book["bid"].keys())
            best_ask_level = min(order_book["ask"].keys())
            
            best_bid_price = order_book["bid"][best_bid_level][0]
            best_ask_price = order_book["ask"][best_ask_level][0]
            
            features.update({
                "best_bid": best_bid_price,
                "best_ask": best_ask_price,
                "mid_price": (best_bid_price + best_ask_price) / 2,
                "spread": best_ask_price - best_bid_price
            })
            
        return features
        
    def create_features(self, df_raw: Optional[pd.DataFrame] = None, 
                       data_dir: str = "raw_data",
                       chunk_size: int = 100000) -> pd.DataFrame:
        """
        Create complete feature set from raw order book data with memory-efficient processing
        
        Args:
            df_raw: Raw DataFrame (if None, will load from data_dir)
            data_dir: Directory for CSV files (used if df_raw is None)
            chunk_size: Number of events to process per chunk (default: 100k for memory efficiency)
            
        Returns:
            DataFrame indexed by timestamp_us with all features
            Columns prefixed by symbol (e.g., "BTC_spread", "ETH_imbalance")
        """
        # Use chunked processing to avoid RAM explosion
        if df_raw is None:
            return self._create_features_chunked(data_dir, chunk_size)
        else:
            return self._create_features_from_dataframe(df_raw)
            
    def _create_features_from_dataframe(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Create features from an existing DataFrame (legacy method)"""
        # Validate input DataFrame
        required_cols = ["symbol", "price", "volume", "timestamp_us", "side", "level"]
        missing_cols = [col for col in required_cols if col not in df_raw.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}. Available columns: {list(df_raw.columns)}")
            
        # Validate symbols
        available_symbols = set(df_raw["symbol"].unique())
        expected_symbols = set(self.symbols)
        missing_symbols = expected_symbols - available_symbols
        if missing_symbols:
            raise ValueError(f"Missing symbols in data: {missing_symbols}. Available symbols: {available_symbols}")
        
        # Data validation and cleaning
        self._validate_and_clean_data(df_raw)
        
        # Clear previous state
        self.synchronizer.clear_data()
        for symbol in self.symbols:
            self.current_books[symbol] = {"bid": {}, "ask": {}}
            
        # Process events
        events_processed = self._process_chunk_events(df_raw)
        logger.info(f"Finished processing {events_processed} events")
        
        # Finalize features
        return self._finalize_features()
    
    def _validate_and_clean_data(self, df_raw: pd.DataFrame):
        """Validate and clean input data"""
        # Validate data types and constraints
        if not df_raw["timestamp_us"].dtype in [np.int64, np.int32]:
            logger.warning("Converting timestamp_us to int64")
            df_raw["timestamp_us"] = df_raw["timestamp_us"].astype(np.int64)
            
        # Check for invalid prices/volumes
        invalid_prices = df_raw[df_raw["price"] <= 0]
        if len(invalid_prices) > 0:
            logger.warning(f"Found {len(invalid_prices)} invalid prices (<= 0), removing them")
            df_raw = df_raw[df_raw["price"] > 0]
            
        invalid_volumes = df_raw[df_raw["volume"] < 0]  
        if len(invalid_volumes) > 0:
            logger.warning(f"Found {len(invalid_volumes)} invalid volumes (< 0), removing them")
            df_raw = df_raw[df_raw["volume"] >= 0]
            
        # Validate side values
        valid_sides = {"bid", "ask"}
        invalid_sides = set(df_raw["side"].unique()) - valid_sides
        if invalid_sides:
            raise ValueError(f"Invalid side values found: {invalid_sides}. Must be one of: {valid_sides}")
            
        # Check for reasonable level values
        if df_raw["level"].min() < 1:
            raise ValueError("Order book levels must be >= 1")
        if df_raw["level"].max() > 50:
            logger.warning(f"Very deep order book levels found (max: {df_raw['level'].max()})")
            
        # Ensure chronological order
        if not df_raw["timestamp_us"].is_monotonic_increasing:
            logger.info("Sorting data by timestamp_us for chronological processing")
            df_raw = df_raw.sort_values("timestamp_us")
            
        logger.info(f"Data validation passed. Processing {len(df_raw)} events for feature extraction")

    def _finalize_features(self) -> pd.DataFrame:
        """Synchronize data and extract final features"""
        # Synchronize data across symbols
        logger.info("Starting asynchronous synchronization")
        sync_points = self.synchronizer.synchronize()
        
        if not sync_points:
            logger.warning("No synchronized points generated")
            return pd.DataFrame()
            
        logger.info(f"Generated {len(sync_points)} synchronized points")
        
        # Extract order book features
        logger.info("Extracting order book features")
        
        # Collect features per timestamp
        timestamp_features = {}
        
        for sync_point in sync_points:
            timestamp_us = sync_point.timestamp_us
            if timestamp_us not in timestamp_features:
                timestamp_features[timestamp_us] = {'timestamp_us': timestamp_us}
            
            # Extract features for each symbol at this timestamp
            for symbol in sync_point.symbols:
                if "order_book" in sync_point.symbols[symbol]:
                    order_book = sync_point.symbols[symbol]["order_book"]
                    
                    # Convert order book to snapshots format
                    snapshots = []
                    for side in ["bid", "ask"]:
                        if side in order_book:
                            for level, (price, volume) in order_book[side].items():
                                snapshots.append({
                                    "price": price,
                                    "volume": volume,
                                    "side": side,
                                    "level": level
                                })
                    
                    # Extract features for this symbol at this timestamp
                    if snapshots:
                        features = self.ob_extractor.extract_features(snapshots, symbol, timestamp_us)
                        if features:
                            # Add symbol-prefixed features to timestamp dict
                            timestamp_features[timestamp_us].update({
                                f'{symbol}_mid_price': features.mid_price,
                                f'{symbol}_spread': features.spread,
                                f'{symbol}_spread_bps': features.spread_bps,
                                f'{symbol}_bid_volume_l1': features.bid_volume_l1,
                                f'{symbol}_ask_volume_l1': features.ask_volume_l1,
                                f'{symbol}_volume_imbalance_l1': features.volume_imbalance_l1,
                                f'{symbol}_bid_volume_l5': features.bid_volume_l5,
                                f'{symbol}_ask_volume_l5': features.ask_volume_l5,
                                f'{symbol}_volume_imbalance_l5': features.volume_imbalance_l5,
                                f'{symbol}_bid_depth_l5': features.bid_depth_l5,
                                f'{symbol}_ask_depth_l5': features.ask_depth_l5,
                                f'{symbol}_depth_ratio': features.depth_ratio,
                                f'{symbol}_price_impact_100': features.price_impact_100,
                                f'{symbol}_price_impact_1000': features.price_impact_1000,
                                f'{symbol}_vwap_bid_l5': features.vwap_bid_l5,
                                f'{symbol}_vwap_ask_l5': features.vwap_ask_l5,
                                f'{symbol}_bid_slope': features.bid_slope,
                                f'{symbol}_ask_slope': features.ask_slope
                            })
        
        # Create DataFrame from features
        if timestamp_features:
            ob_features_df = pd.DataFrame(list(timestamp_features.values()))
            ob_features_df.set_index('timestamp_us', inplace=True)
        else:
            logger.warning("No order book features extracted")
            ob_features_df = pd.DataFrame()
        
        # Extract temporal features on key price/volume series
        logger.info("Extracting temporal features")
        
        # Define key fields for temporal analysis
        temporal_fields = []
        for symbol in self.symbols:
            temporal_fields.extend([
                f"{symbol}_mid_price",
                f"{symbol}_spread", 
                f"{symbol}_volume_imbalance_l1"
            ])
            
        # Filter available fields
        available_fields = [f for f in temporal_fields if f in ob_features_df.columns]
        
        if available_fields:
            ts_features_df = self.ts_extractor.extract_temporal_features(
                ob_features_df, 
                fields=available_fields
            )
            
            # Merge order book and temporal features
            final_df = ob_features_df.join(ts_features_df, how="outer")
        else:
            logger.warning("No temporal fields available")
            final_df = ob_features_df
            
        logger.info(f"Feature extraction complete. Final shape: {final_df.shape}")
        
        return final_df

    def _create_features_chunked(self, data_dir: str, chunk_size: int) -> pd.DataFrame:
        """
        Memory-efficient feature creation using chunked Parquet processing
        
        Args:
            data_dir: Directory for data files 
            chunk_size: Events per chunk
            
        Returns:
            Complete feature DataFrame
        """
        logger.info(f"Starting chunked processing with chunk_size={chunk_size:,}")
        
        # Get cached parquet files
        try:
            parquet_paths = ensure_processed(self.dataset_id)
        except Exception as e:
            raise FileNotFoundError(f"Failed to prepare cached data for {self.dataset_id}: {e}")
        
        # Symbol mapping
        symbol_mapping = {"ETH": "ETH", "XBT": "BTC"}
        
        # Clear previous state
        self.synchronizer.clear_data()
        for symbol in self.symbols:
            self.current_books[symbol] = {"bid": {}, "ask": {}}
        
        total_events = 0
        
        # Process each symbol's Parquet file in chunks
        for parquet_symbol, parquet_path in parquet_paths.items():
            internal_symbol = symbol_mapping.get(parquet_symbol, parquet_symbol)
            
            if internal_symbol not in self.symbols:
                logger.info(f"Skipping {parquet_symbol} (not in requested symbols)")
                continue
                
            logger.info(f"Processing {internal_symbol} in chunks from {parquet_path}")
            
            # Use pyarrow to read parquet in chunks without loading everything in memory
            import pyarrow.parquet as pq
            
            parquet_file = pq.ParquetFile(parquet_path)
            total_rows = parquet_file.metadata.num_rows
            logger.info(f"{internal_symbol}: {total_rows:,} total events")
            
            # Process in chunks using row groups
            for batch_idx in range(parquet_file.num_row_groups):
                # Read one row group at a time
                row_group = parquet_file.read_row_group(batch_idx)
                df_chunk = row_group.to_pandas()
                
                # Add symbol column
                df_chunk["symbol"] = internal_symbol
                df_chunk["timestamp_us"] = (df_chunk["timestamp"] * 1_000_000).astype(np.int64)
                
                # Ensure proper data types
                df_chunk["price"] = df_chunk["price"].astype(np.float64)
                df_chunk["volume"] = df_chunk["volume"].astype(np.float64)
                df_chunk["level"] = df_chunk["level"].astype(np.int8)
                
                # Sort chunk by timestamp
                df_chunk = df_chunk.sort_values("timestamp_us")
                
                # Process chunk events
                chunk_events = self._process_chunk_events(df_chunk)
                total_events += chunk_events
                
                # Clear chunk from memory
                del df_chunk
                
                logger.info(f"Processed row group {batch_idx+1}/{parquet_file.num_row_groups} ({chunk_events:,} events)")
        
        logger.info(f"Finished chunked processing: {total_events:,} total events")
        
        # Synchronize and extract features
        return self._finalize_features()
    
    def _process_chunk_events(self, df_chunk: pd.DataFrame) -> int:
        """Process events from a single chunk"""
        events_processed = 0
        
        # Process events chronologically using itertuples for performance
        for row_tuple in df_chunk.itertuples(index=False, name=None):
            # Map tuple elements (order matches df columns after processing)
            # Columns: price, volume, timestamp, side, level, symbol, timestamp_us
            price = float(row_tuple[0])   # price
            volume = float(row_tuple[1])  # volume
            timestamp = row_tuple[2]      # timestamp (original)
            side = row_tuple[3]           # side
            level = int(row_tuple[4])     # level  
            symbol = row_tuple[5]         # symbol
            timestamp_us = int(row_tuple[6])  # timestamp_us
            
            # Update order book and get complete snapshot
            features = self._update_order_book(symbol, price, volume, side, level)
            
            # Add to synchronizer
            self.synchronizer.append_event(symbol, timestamp_us, features)
            
            events_processed += 1
            
        return events_processed

def split_train_test(df: pd.DataFrame, frac: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame by time duration rather than number of rows
    
    Args:
        df: DataFrame with timestamp_us index
        frac: Fraction for training set (0.0 to 1.0)
        
    Returns:
        Tuple of (train_df, test_df)
    """
    if not 0.0 < frac < 1.0:
        raise ValueError("frac must be between 0 and 1")
        
    if len(df) == 0:
        return df.copy(), df.copy()
        
    t0 = df.index[0]
    tn = df.index[-1]
    
    # Calculate midpoint by time duration
    tmid = t0 + frac * (tn - t0)
    
    # Split by timestamp
    df_train = df[df.index < tmid].copy()
    df_test = df[df.index >= tmid].copy()
    
    logger.info(f"Split data: {len(df_train)} train samples, {len(df_test)} test samples")
    logger.info(f"Time split at: {tmid} (frac={frac})")
    
    return df_train, df_test
