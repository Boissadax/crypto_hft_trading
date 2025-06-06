"""
Asynchronous Data Synchronization

Handles synchronization of asynchronous order book data from multiple symbols:
- Union-based timestamp alignment (no fixed grid resampling)
- Forward-fill interpolation for missing data points
- Microsecond precision timestamp handling
- Complete order book state management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
from collections import defaultdict, deque
from bisect import bisect_left, bisect_right

logger = logging.getLogger(__name__)


@dataclass
class SynchronizedDataPoint:
    """Container for synchronized data at a specific timestamp"""
    timestamp_us: int
    symbols: Dict[str, Dict[str, Any]]  # symbol -> {"order_book": {...}, ...}
    interpolated_symbols: List[str]  # symbols that were forward-filled
    
    
@dataclass 
class SyncConfig:
    """Configuration for asynchronous data synchronization"""
    max_interpolation_gap_us: int = 10_000_000  # 10 seconds max forward-fill gap
    min_symbols_required: int = 1  # Minimum symbols required for sync point
    enable_cross_symbol_features: bool = False  # Disabled by default for performance


class AsynchronousSync:
    """
    Synchronizes asynchronous order book data from multiple symbols using exact timestamp union.
    
    Key features:
    - No fixed grid resampling - uses exact union of all timestamps
    - Forward-fill interpolation for missing data points
    - Microsecond precision timestamp handling
    - Complete order book state preservation
    """
    
    def __init__(self, 
                 config: SyncConfig = None,
                 symbols: List[str] = None):
        """
        Initialize asynchronous data synchronizer
        
        Args:
            config: Synchronization configuration
            symbols: List of symbols to synchronize (e.g., ["BTC", "ETH"])
        """
        self.config = config or SyncConfig()
        self.symbols = symbols or []
        
        # Data storage: symbol -> [(timestamp_us, features_dict)]
        # Features dict contains {"order_book": {...}, ...}
        self.data_buffers: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {
            symbol: [] for symbol in self.symbols
        }
        
        # Last known values for forward fill
        self.last_values: Dict[str, Optional[Dict[str, Any]]] = {
            symbol: None for symbol in self.symbols
        }
        
        # All timestamps ever seen (for union computation)
        self.all_timestamps: set = set()
        
        logger.info(f"AsynchronousSync initialized for symbols: {self.symbols}")
        
    def add_symbols(self, new_symbols: List[str]):
        """Add new symbols to track"""
        for symbol in new_symbols:
            if symbol not in self.symbols:
                self.symbols.append(symbol)
                self.data_buffers[symbol] = []
                self.last_values[symbol] = None
                logger.info(f"Added symbol: {symbol}")
                
    def append_event(self, 
                    symbol: str,
                    timestamp_us: int, 
                    features: Dict[str, Any]):
        """
        Add a new data point for a symbol
        
        Args:
            symbol: Symbol name (e.g., "BTC", "ETH")
            timestamp_us: Timestamp in microseconds (int)
            features: Dictionary containing features including "order_book"
                     Example: {"order_book": {"bid": {1: (price, vol), ...}, "ask": {...}}}
        
        Raises:
            ValueError: If timestamp_us is not an int or features missing order_book
        """
        if not isinstance(timestamp_us, int):
            raise ValueError(f"timestamp_us must be int, got {type(timestamp_us).__name__}: {timestamp_us}")
        
        if not isinstance(features, dict):
            raise ValueError(f"features must be dict, got {type(features).__name__}")
            
        if "order_book" not in features:
            raise ValueError("features dict must contain 'order_book' key")
        
        # Validate order_book structure
        order_book = features["order_book"]
        if not isinstance(order_book, dict):
            raise ValueError("order_book must be a dict")
            
        # Check for required keys
        required_keys = {"bid", "ask"}
        missing_keys = required_keys - set(order_book.keys())
        if missing_keys:
            raise ValueError(f"order_book missing required keys: {missing_keys}")
            
        # Validate bid/ask structure
        for side in ["bid", "ask"]:
            if not isinstance(order_book[side], dict):
                raise ValueError(f"order_book['{side}'] must be dict, got {type(order_book[side])}")
            
            # Check level structure if not empty
            for level, (price, volume) in order_book[side].items():
                if not isinstance(level, int) or level < 1:
                    raise ValueError(f"Invalid level {level} in {side} side, must be int >= 1")
                if not isinstance(price, (int, float)) or price <= 0:
                    raise ValueError(f"Invalid price {price} at {side} level {level}, must be positive number")
                if not isinstance(volume, (int, float)) or volume < 0:
                    raise ValueError(f"Invalid volume {volume} at {side} level {level}, must be non-negative number")
            
        if symbol not in self.symbols:
            self.add_symbols([symbol])
            
        # Store the event
        self.data_buffers[symbol].append((timestamp_us, features.copy()))
        
        # Update last known value for forward fill
        self.last_values[symbol] = features.copy()
        
        # Add to union of all timestamps
        self.all_timestamps.add(timestamp_us)
        
        # Keep data sorted by timestamp (assuming mostly chronological input)
        if len(self.data_buffers[symbol]) > 1:
            if self.data_buffers[symbol][-1][0] < self.data_buffers[symbol][-2][0]:
                # Sort if not in order
                self.data_buffers[symbol].sort(key=lambda x: x[0])
                
        logger.debug(f"Added event for {symbol} at timestamp_us={timestamp_us}")
        
    def _generate_sync_timestamps(self, 
                                 start_time_us: Optional[int] = None,
                                 end_time_us: Optional[int] = None) -> List[int]:
        """
        Generate the union of all timestamps in the specified range
        
        Args:
            start_time_us: Start timestamp (inclusive), None for earliest
            end_time_us: End timestamp (inclusive), None for latest
            
        Returns:
            Sorted list of all unique timestamps in microseconds
        """
        if not self.all_timestamps:
            return []
            
        # Filter timestamps by range
        if start_time_us is None:
            start_time_us = min(self.all_timestamps)
        if end_time_us is None:
            end_time_us = max(self.all_timestamps)
            
        filtered_timestamps = [
            ts for ts in self.all_timestamps 
            if start_time_us <= ts <= end_time_us
        ]
        
        return sorted(filtered_timestamps)
    
    def _get_value_at_timestamp(self, 
                               symbol: str, 
                               target_timestamp_us: int) -> Tuple[Optional[Dict[str, Any]], bool]:
        """
        Get the forward-filled value for a symbol at a specific timestamp
        
        Args:
            symbol: Symbol name
            target_timestamp_us: Target timestamp in microseconds
            
        Returns:
            Tuple of (features_dict, was_interpolated)
            - features_dict: The features dict at or before target timestamp, or None
            - was_interpolated: True if the value was forward-filled
        """
        if symbol not in self.data_buffers:
            return None, False
            
        buffer = self.data_buffers[symbol]
        if not buffer:
            return None, False
            
        # Find the latest value at or before target_timestamp_us
        # Binary search for efficiency
        idx = bisect_right([ts for ts, _ in buffer], target_timestamp_us) - 1
        
        if idx < 0:
            # No data available before this timestamp
            return None, False
            
        timestamp_us, features = buffer[idx]
        
        # Check if it's exactly at the timestamp (not interpolated)
        is_interpolated = timestamp_us != target_timestamp_us
        
        # Check max interpolation gap
        if is_interpolated:
            gap_us = target_timestamp_us - timestamp_us
            if gap_us > self.config.max_interpolation_gap_us:
                logger.debug(f"Gap too large for {symbol}: {gap_us/1000000:.2f}s > {self.config.max_interpolation_gap_us/1000000:.2f}s")
                return None, False
                
        return features.copy(), is_interpolated
    
    def synchronize(self, 
                   start_time_us: Optional[int] = None,
                   end_time_us: Optional[int] = None) -> List[SynchronizedDataPoint]:
        """
        Synchronize data across all symbols using exact timestamp union
        
        Args:
            start_time_us: Start timestamp (inclusive), None for earliest
            end_time_us: End timestamp (inclusive), None for latest
            
        Returns:
            List of SynchronizedDataPoint objects, sorted by timestamp
        """
        # Generate union of all timestamps
        sync_timestamps = self._generate_sync_timestamps(start_time_us, end_time_us)
        
        if not sync_timestamps:
            logger.warning("No timestamps found for synchronization")
            return []
        
        # Validate timestamps are strictly increasing
        for i in range(1, len(sync_timestamps)):
            if sync_timestamps[i] <= sync_timestamps[i-1]:
                raise ValueError(f"Timestamps not strictly increasing at index {i}: {sync_timestamps[i-1]} >= {sync_timestamps[i]}")
            
        logger.info(f"Synchronizing {len(sync_timestamps)} timestamps across {len(self.symbols)} symbols")
        
        synchronized_points = []
        
        for timestamp_us in sync_timestamps:
            symbol_data = {}
            interpolated_symbols = []
            
            # Get data for each symbol at this timestamp
            available_symbols = 0
            for symbol in self.symbols:
                features, was_interpolated = self._get_value_at_timestamp(symbol, timestamp_us)
                
                if features is not None:
                    symbol_data[symbol] = features
                    available_symbols += 1
                    
                    if was_interpolated:
                        interpolated_symbols.append(symbol)
                        
            # Only create sync point if minimum symbols requirement is met
            if available_symbols >= self.config.min_symbols_required:
                sync_point = SynchronizedDataPoint(
                    timestamp_us=timestamp_us,
                    symbols=symbol_data,
                    interpolated_symbols=interpolated_symbols
                )
                synchronized_points.append(sync_point)
                
        logger.info(f"Generated {len(synchronized_points)} synchronized data points")
        return synchronized_points
    
    def create_aligned_dataframe(self,
                                start_time_us: Optional[int] = None,
                                end_time_us: Optional[int] = None,
                                include_metadata: bool = False) -> pd.DataFrame:
        """
        Create a pandas DataFrame with aligned data across all symbols
        
        Args:
            start_time_us: Start timestamp
            end_time_us: End timestamp  
            include_metadata: Whether to include interpolation metadata columns
            
        Returns:
            DataFrame indexed by timestamp_us (int64, microseconds)
        """
        sync_points = self.synchronize(start_time_us, end_time_us)
        
        if not sync_points:
            return pd.DataFrame()
            
        # Collect all feature names across all symbols and sync points
        all_features = set()
        for point in sync_points:
            for symbol, symbol_features in point.symbols.items():
                # Flatten nested feature dicts (skip order_book for now)
                for feature_name, feature_value in symbol_features.items():
                    if feature_name != "order_book":
                        all_features.add(feature_name)
                        
        all_features = sorted(all_features)
        
        # Create column names with symbol prefixes
        columns = []
        for symbol in self.symbols:
            for feature in all_features:
                columns.append(f"{symbol}_{feature}")
                
        if include_metadata:
            for symbol in self.symbols:
                columns.append(f"{symbol}_interpolated")
                
        # Build data matrix
        data = []
        timestamps = []
        
        for point in sync_points:
            row = []
            timestamps.append(point.timestamp_us)
            
            # Fill feature values
            for symbol in self.symbols:
                for feature in all_features:
                    if (symbol in point.symbols and 
                        feature in point.symbols[symbol]):
                        value = point.symbols[symbol][feature]
                    else:
                        value = np.nan
                    row.append(value)
                    
            # Fill metadata
            if include_metadata:
                for symbol in self.symbols:
                    is_interpolated = symbol in point.interpolated_symbols
                    row.append(is_interpolated)
                    
            data.append(row)
            
        # Create DataFrame
        df = pd.DataFrame(data, columns=columns)
        df['timestamp_us'] = timestamps
        df.set_index('timestamp_us', inplace=True)
        
        # IMPORTANT: Keep timestamp_us as int64 microseconds, do NOT convert to datetime
        # This preserves microsecond precision and irregular timing
        
        return df
        
    def clear_data(self, symbol: Optional[str] = None):
        """Clear data for specific symbol or all symbols"""
        if symbol:
            if symbol in self.data_buffers:
                self.data_buffers[symbol].clear()
                self.last_values[symbol] = None
        else:
            for sym in self.symbols:
                self.data_buffers[sym].clear()
                self.last_values[sym] = None
            self.all_timestamps.clear()
            
        logger.info(f"Cleared data for {'all symbols' if not symbol else symbol}")
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get synchronization statistics"""
        stats = {
            'total_symbols': len(self.symbols),
            'total_unique_timestamps': len(self.all_timestamps),
            'symbol_stats': {}
        }
        
        for symbol in self.symbols:
            buffer = self.data_buffers[symbol]
            if buffer:
                timestamps = [ts for ts, _ in buffer]
                stats['symbol_stats'][symbol] = {
                    'total_events': len(buffer),
                    'time_span_us': max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0,
                    'first_timestamp_us': min(timestamps),
                    'last_timestamp_us': max(timestamps)
                }
            else:
                stats['symbol_stats'][symbol] = {
                    'total_events': 0,
                    'time_span_us': 0,
                    'first_timestamp_us': None,
                    'last_timestamp_us': None
                }
                
        return stats
