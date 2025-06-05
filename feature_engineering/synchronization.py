"""
Asynchronous Data Synchronization

Handles synchronization of asynchronous order book data from multiple symbols:
- Time-based alignment with configurable tolerance
- Interpolation methods for missing data points
- Efficient data resampling for analysis
- Cross-symbol feature extraction at synchronized timestamps
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
import logging
from collections import defaultdict, deque
from bisect import bisect_left, bisect_right
import heapq
from enum import Enum

logger = logging.getLogger(__name__)


class InterpolationMethod(Enum):
    """Interpolation methods for missing data"""
    FORWARD_FILL = "ffill"
    BACKWARD_FILL = "bfill"
    LINEAR = "linear"
    NEAREST = "nearest"
    ZERO = "zero"
    DROP = "drop"


@dataclass
class SynchronizedDataPoint:
    """Container for synchronized data across multiple symbols"""
    timestamp_us: int
    symbols: Dict[str, Dict[str, Any]]  # symbol -> feature dict
    interpolated_symbols: List[str]  # symbols that were interpolated
    
    
@dataclass
class SyncConfig:
    """Configuration for data synchronization"""
    tolerance_us: int = 1_000_000  # 1 second tolerance
    interpolation_method: InterpolationMethod = InterpolationMethod.FORWARD_FILL
    max_interpolation_gap_us: int = 10_000_000  # 10 seconds max interpolation
    min_symbols_required: int = 1  # Minimum symbols required for synchronized point
    resampling_frequency_us: int = 1_000_000  # 1 second resampling
    enable_cross_symbol_features: bool = True


class AsynchronousSync:
    """
    Synchronizes asynchronous order book data from multiple symbols.
    
    Provides various synchronization strategies and interpolation methods
    for creating aligned datasets suitable for lead-lag analysis.
    """
    
    def __init__(self, 
                 config: SyncConfig = None,
                 symbols: List[str] = None):
        """
        Initialize asynchronous data synchronizer
        
        Args:
            config: Synchronization configuration
            symbols: List of symbols to synchronize
        """
        self.config = config or SyncConfig()
        self.symbols = symbols or []
        
        # Data storage: symbol -> [(timestamp_us, features_dict)]
        self.data_buffers = {symbol: deque() for symbol in self.symbols}
        
        # Last known values for forward fill
        self.last_values = {symbol: None for symbol in self.symbols}
        
        # Synchronization points
        self.sync_points = []
        
        # Performance statistics
        self.sync_stats = {
            'total_sync_operations': 0,
            'interpolated_points': 0,
            'dropped_points': 0,
            'avg_sync_time_us': 0.0,
            'data_coverage': {}  # symbol -> coverage percentage
        }
        
    def add_symbols(self, new_symbols: List[str]):
        """Add new symbols to track"""
        for symbol in new_symbols:
            if symbol not in self.symbols:
                self.symbols.append(symbol)
                self.data_buffers[symbol] = deque()
                self.last_values[symbol] = None
                
    def add_data_point(self, 
                      symbol: str,
                      timestamp_us: int,
                      features: Dict[str, Any]):
        """
        Add a new data point for a symbol
        
        Args:
            symbol: Trading symbol
            timestamp_us: Timestamp in microseconds
            features: Dictionary of features/values
        """
        if symbol not in self.data_buffers:
            self.add_symbols([symbol])
            
        # Add to buffer (maintain chronological order)
        self.data_buffers[symbol].append((timestamp_us, features))
        
        # Update last known value
        self.last_values[symbol] = features
        
        # Clean old data if buffer gets too large
        self._clean_old_data(symbol, timestamp_us)
        
    def synchronize_data(self, 
                        start_time_us: Optional[int] = None,
                        end_time_us: Optional[int] = None,
                        target_timestamps: Optional[List[int]] = None) -> List[SynchronizedDataPoint]:
        """
        Synchronize data across all symbols
        
        Args:
            start_time_us: Start timestamp (if None, use earliest available)
            end_time_us: End timestamp (if None, use latest available)
            target_timestamps: Specific timestamps to synchronize (overrides start/end)
            
        Returns:
            List of synchronized data points
        """
        start_time = pd.Timestamp.now()
        
        if target_timestamps:
            sync_timestamps = sorted(target_timestamps)
        else:
            sync_timestamps = self._generate_sync_timestamps(start_time_us, end_time_us)
            
        synchronized_points = []
        
        for timestamp_us in sync_timestamps:
            sync_point = self._create_synchronized_point(timestamp_us)
            if sync_point:
                synchronized_points.append(sync_point)
                
        # Update statistics
        sync_time = (pd.Timestamp.now() - start_time).total_seconds() * 1_000_000
        self.sync_stats['total_sync_operations'] += 1
        self.sync_stats['avg_sync_time_us'] = (
            (self.sync_stats['avg_sync_time_us'] * 
             (self.sync_stats['total_sync_operations'] - 1) + sync_time) /
            self.sync_stats['total_sync_operations']
        )
        
        return synchronized_points
    
    def resample_to_frequency(self,
                            frequency_us: int,
                            start_time_us: Optional[int] = None,
                            end_time_us: Optional[int] = None) -> List[SynchronizedDataPoint]:
        """
        Resample data to a specific frequency
        
        Args:
            frequency_us: Target frequency in microseconds
            start_time_us: Start time
            end_time_us: End time
            
        Returns:
            List of resampled synchronized data points
        """
        if not start_time_us or not end_time_us:
            start_time_us, end_time_us = self._get_data_time_range()
            
        # Generate regular timestamps
        target_timestamps = list(range(start_time_us, end_time_us + 1, frequency_us))
        
        return self.synchronize_data(target_timestamps=target_timestamps)
    
    def _generate_sync_timestamps(self, 
                                 start_time_us: Optional[int],
                                 end_time_us: Optional[int]) -> List[int]:
        """Generate timestamps for synchronization"""
        if not start_time_us or not end_time_us:
            start_time_us, end_time_us = self._get_data_time_range()
            
        # Use resampling frequency to generate regular timestamps
        timestamps = list(range(
            start_time_us, 
            end_time_us + 1, 
            self.config.resampling_frequency_us
        ))
        
        return timestamps
    
    def _get_data_time_range(self) -> Tuple[int, int]:
        """Get the overall time range of available data"""
        all_timestamps = []
        
        for symbol in self.symbols:
            if self.data_buffers[symbol]:
                symbol_timestamps = [point[0] for point in self.data_buffers[symbol]]
                all_timestamps.extend(symbol_timestamps)
                
        if not all_timestamps:
            return 0, 0
            
        return min(all_timestamps), max(all_timestamps)
    
    def _create_synchronized_point(self, timestamp_us: int) -> Optional[SynchronizedDataPoint]:
        """Create a synchronized data point for a specific timestamp"""
        symbol_data = {}
        interpolated_symbols = []
        
        for symbol in self.symbols:
            data = self._get_data_at_timestamp(symbol, timestamp_us)
            
            if data is not None:
                symbol_data[symbol] = data['features']
                if data['interpolated']:
                    interpolated_symbols.append(symbol)
                    
        # Check if we have minimum required symbols
        if len(symbol_data) < self.config.min_symbols_required:
            self.sync_stats['dropped_points'] += 1
            return None
            
        # Add cross-symbol features if enabled
        if self.config.enable_cross_symbol_features and len(symbol_data) > 1:
            cross_features = self._compute_cross_symbol_features(symbol_data)
            for symbol in symbol_data:
                symbol_data[symbol].update(cross_features.get(symbol, {}))
                
        return SynchronizedDataPoint(
            timestamp_us=timestamp_us,
            symbols=symbol_data,
            interpolated_symbols=interpolated_symbols
        )
    
    def _get_data_at_timestamp(self, 
                              symbol: str, 
                              timestamp_us: int) -> Optional[Dict[str, Any]]:
        """Get data for a symbol at a specific timestamp with interpolation"""
        buffer = self.data_buffers[symbol]
        
        if not buffer:
            return None
            
        # Find exact match first
        for ts, features in buffer:
            if ts == timestamp_us:
                return {'features': features, 'interpolated': False}
                
        # Find closest points within tolerance
        closest_before = None
        closest_after = None
        
        for ts, features in buffer:
            if ts <= timestamp_us:
                if (closest_before is None or 
                    abs(timestamp_us - ts) < abs(timestamp_us - closest_before[0])):
                    closest_before = (ts, features)
            else:
                if (closest_after is None or 
                    abs(timestamp_us - ts) < abs(timestamp_us - closest_after[0])):
                    closest_after = (ts, features)
                    
        # Apply interpolation strategy
        return self._interpolate_data(
            timestamp_us, closest_before, closest_after
        )
    
    def _interpolate_data(self,
                         target_timestamp: int,
                         before_point: Optional[Tuple[int, Dict]],
                         after_point: Optional[Tuple[int, Dict]]) -> Optional[Dict[str, Any]]:
        """Interpolate data based on configuration"""
        method = self.config.interpolation_method
        tolerance = self.config.tolerance_us
        max_gap = self.config.max_interpolation_gap_us
        
        # Handle different interpolation methods
        if method == InterpolationMethod.DROP:
            return None
            
        if method == InterpolationMethod.ZERO:
            # Return zero values (need to infer structure from last known value)
            if self.last_values[symbol] is not None:
                zero_features = {k: 0.0 for k in self.last_values[symbol].keys()}
                return {'features': zero_features, 'interpolated': True}
            return None
            
        # For other methods, we need at least one point
        if not before_point and not after_point:
            return None
            
        if method == InterpolationMethod.FORWARD_FILL:
            if before_point and abs(target_timestamp - before_point[0]) <= max_gap:
                return {'features': before_point[1], 'interpolated': True}
                
        elif method == InterpolationMethod.BACKWARD_FILL:
            if after_point and abs(target_timestamp - after_point[0]) <= max_gap:
                return {'features': after_point[1], 'interpolated': True}
                
        elif method == InterpolationMethod.NEAREST:
            if before_point and after_point:
                dist_before = abs(target_timestamp - before_point[0])
                dist_after = abs(target_timestamp - after_point[0])
                
                if min(dist_before, dist_after) <= tolerance:
                    if dist_before <= dist_after:
                        return {'features': before_point[1], 'interpolated': True}
                    else:
                        return {'features': after_point[1], 'interpolated': True}
            elif before_point and abs(target_timestamp - before_point[0]) <= tolerance:
                return {'features': before_point[1], 'interpolated': True}
            elif after_point and abs(target_timestamp - after_point[0]) <= tolerance:
                return {'features': after_point[1], 'interpolated': True}
                
        elif method == InterpolationMethod.LINEAR:
            if before_point and after_point:
                dist_before = abs(target_timestamp - before_point[0])
                dist_after = abs(target_timestamp - after_point[0])
                
                if max(dist_before, dist_after) <= max_gap:
                    # Linear interpolation
                    interpolated_features = self._linear_interpolate(
                        target_timestamp, before_point, after_point
                    )
                    return {'features': interpolated_features, 'interpolated': True}
                    
        # If no interpolation method worked, try forward fill as fallback
        if before_point and abs(target_timestamp - before_point[0]) <= max_gap:
            return {'features': before_point[1], 'interpolated': True}
            
        return None
    
    def _linear_interpolate(self,
                           target_timestamp: int,
                           before_point: Tuple[int, Dict],
                           after_point: Tuple[int, Dict]) -> Dict[str, Any]:
        """Perform linear interpolation between two points"""
        t0, features0 = before_point
        t1, features1 = after_point
        
        if t1 == t0:
            return features0
            
        # Interpolation weight
        alpha = (target_timestamp - t0) / (t1 - t0)
        
        interpolated = {}
        
        # Interpolate numerical values
        for key in features0:
            if key in features1:
                val0 = features0[key]
                val1 = features1[key]
                
                # Only interpolate numerical values
                if isinstance(val0, (int, float)) and isinstance(val1, (int, float)):
                    interpolated[key] = val0 + alpha * (val1 - val0)
                else:
                    # For non-numerical, use the closest value
                    interpolated[key] = val1 if alpha > 0.5 else val0
            else:
                interpolated[key] = features0[key]
                
        return interpolated
    
    def _compute_cross_symbol_features(self, symbol_data: Dict[str, Dict]) -> Dict[str, Dict]:
        """Compute cross-symbol features like correlations and spreads"""
        cross_features = defaultdict(dict)
        
        symbols = list(symbol_data.keys())
        
        # Extract common feature names
        all_features = set()
        for features in symbol_data.values():
            all_features.update(features.keys())
            
        # Compute pairwise features
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols):
                if i >= j:  # Only compute upper triangle
                    continue
                    
                pair_key = f"{symbol1}_{symbol2}"
                
                features1 = symbol_data[symbol1]
                features2 = symbol_data[symbol2]
                
                # Price ratio
                if 'mid_price' in features1 and 'mid_price' in features2:
                    if features2['mid_price'] > 0:
                        price_ratio = features1['mid_price'] / features2['mid_price']
                        cross_features[symbol1][f'price_ratio_vs_{symbol2}'] = price_ratio
                        cross_features[symbol2][f'price_ratio_vs_{symbol1}'] = 1.0 / price_ratio
                
                # Spread ratio
                if 'spread' in features1 and 'spread' in features2:
                    if features2['spread'] > 0:
                        spread_ratio = features1['spread'] / features2['spread']
                        cross_features[symbol1][f'spread_ratio_vs_{symbol2}'] = spread_ratio
                        cross_features[symbol2][f'spread_ratio_vs_{symbol1}'] = 1.0 / spread_ratio
                
                # Volume imbalance difference
                if 'volume_imbalance_l1' in features1 and 'volume_imbalance_l1' in features2:
                    imbalance_diff = features1['volume_imbalance_l1'] - features2['volume_imbalance_l1']
                    cross_features[symbol1][f'imbalance_diff_vs_{symbol2}'] = imbalance_diff
                    cross_features[symbol2][f'imbalance_diff_vs_{symbol1}'] = -imbalance_diff
        
        return cross_features
    
    def _clean_old_data(self, symbol: str, current_timestamp: int):
        """Remove old data points to manage memory"""
        buffer = self.data_buffers[symbol]
        max_age_us = 3600 * 1_000_000  # 1 hour
        
        cutoff_time = current_timestamp - max_age_us
        
        while buffer and buffer[0][0] < cutoff_time:
            buffer.popleft()
    
    def create_aligned_dataframe(self,
                                start_time_us: Optional[int] = None,
                                end_time_us: Optional[int] = None,
                                include_metadata: bool = True) -> pd.DataFrame:
        """
        Create a pandas DataFrame with aligned data across all symbols
        
        Args:
            start_time_us: Start timestamp
            end_time_us: End timestamp
            include_metadata: Whether to include interpolation metadata
            
        Returns:
            DataFrame with multi-level columns (symbol, feature)
        """
        sync_points = self.synchronize_data(start_time_us, end_time_us)
        
        if not sync_points:
            return pd.DataFrame()
            
        # Collect all feature names across all symbols
        all_features = set()
        for point in sync_points:
            for symbol_features in point.symbols.values():
                all_features.update(symbol_features.keys())
                
        all_features = sorted(all_features)
        
        # Create multi-level column index
        columns = []
        for symbol in self.symbols:
            for feature in all_features:
                columns.append((symbol, feature))
                
        if include_metadata:
            # Add metadata columns
            for symbol in self.symbols:
                columns.append((symbol, '_interpolated'))
                
        # Create data matrix
        data = []
        timestamps = []
        
        for point in sync_points:
            row = []
            timestamps.append(point.timestamp_us)
            
            # Fill feature values
            for symbol in self.symbols:
                for feature in all_features:
                    if symbol in point.symbols and feature in point.symbols[symbol]:
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
        df = pd.DataFrame(data, columns=pd.MultiIndex.from_tuples(columns))
        df['timestamp_us'] = timestamps
        df.set_index('timestamp_us', inplace=True)
        
        # Convert timestamp index to datetime for convenience
        df.index = pd.to_datetime(df.index, unit='us')
        
        return df
    
    def compute_lead_lag_features(self,
                                 primary_symbol: str,
                                 secondary_symbol: str,
                                 lags_us: List[int],
                                 feature_name: str = 'mid_price') -> pd.DataFrame:
        """
        Compute lead-lag features between two symbols
        
        Args:
            primary_symbol: Primary symbol (potential leader)
            secondary_symbol: Secondary symbol (potential follower)
            lags_us: List of lag values in microseconds
            feature_name: Feature to analyze for lead-lag
            
        Returns:
            DataFrame with lead-lag features
        """
        # Get aligned data
        df = self.create_aligned_dataframe(include_metadata=False)
        
        if df.empty:
            return pd.DataFrame()
            
        # Extract the specific feature for both symbols
        try:
            primary_series = df[(primary_symbol, feature_name)]
            secondary_series = df[(secondary_symbol, feature_name)]
        except KeyError:
            logger.warning(f"Feature {feature_name} not found for symbols {primary_symbol} or {secondary_symbol}")
            return pd.DataFrame()
            
        # Remove NaN values
        valid_mask = primary_series.notna() & secondary_series.notna()
        primary_clean = primary_series[valid_mask]
        secondary_clean = secondary_series[valid_mask]
        
        if len(primary_clean) < 10:
            logger.warning("Insufficient data for lead-lag analysis")
            return pd.DataFrame()
            
        # Compute lead-lag features
        lead_lag_data = []
        
        for lag_us in lags_us:
            # Convert lag to number of periods (assuming regular sampling)
            sampling_period_us = self.config.resampling_frequency_us
            lag_periods = int(lag_us / sampling_period_us)
            
            if lag_periods == 0:
                continue
                
            # Shift primary series by lag
            if lag_periods > 0:
                # Positive lag: primary leads secondary
                primary_shifted = primary_clean.shift(lag_periods)
            else:
                # Negative lag: secondary leads primary
                primary_shifted = primary_clean.shift(lag_periods)
                
            # Calculate correlation
            correlation = primary_shifted.corr(secondary_clean)
            
            if not np.isnan(correlation):
                lead_lag_data.append({
                    'lag_us': lag_us,
                    'lag_periods': lag_periods,
                    'correlation': correlation,
                    'primary_symbol': primary_symbol,
                    'secondary_symbol': secondary_symbol,
                    'feature': feature_name
                })
                
        return pd.DataFrame(lead_lag_data)
    
    def get_synchronization_statistics(self) -> Dict[str, Any]:
        """Get detailed synchronization statistics"""
        stats = self.sync_stats.copy()
        
        # Compute data coverage for each symbol
        for symbol in self.symbols:
            if self.data_buffers[symbol]:
                total_points = len(self.data_buffers[symbol])
                stats['data_coverage'][symbol] = {
                    'total_points': total_points,
                    'time_span_seconds': self._get_symbol_time_span(symbol),
                    'avg_frequency_hz': self._get_symbol_frequency(symbol)
                }
            else:
                stats['data_coverage'][symbol] = {
                    'total_points': 0,
                    'time_span_seconds': 0,
                    'avg_frequency_hz': 0
                }
                
        return stats
    
    def _get_symbol_time_span(self, symbol: str) -> float:
        """Get time span of data for a symbol in seconds"""
        buffer = self.data_buffers[symbol]
        if len(buffer) < 2:
            return 0.0
            
        start_time = buffer[0][0]
        end_time = buffer[-1][0]
        
        return (end_time - start_time) / 1_000_000.0
    
    def _get_symbol_frequency(self, symbol: str) -> float:
        """Get average frequency of data for a symbol in Hz"""
        time_span = self._get_symbol_time_span(symbol)
        total_points = len(self.data_buffers[symbol])
        
        if time_span <= 0 or total_points <= 1:
            return 0.0
            
        return (total_points - 1) / time_span
    
    def clear_data(self, symbol: Optional[str] = None):
        """Clear data for specific symbol or all symbols"""
        if symbol:
            if symbol in self.data_buffers:
                self.data_buffers[symbol].clear()
                self.last_values[symbol] = None
        else:
            for symbol in self.symbols:
                self.data_buffers[symbol].clear()
                self.last_values[symbol] = None
            self.sync_points.clear()
            
        logger.info(f"Synchronization data cleared for {'all symbols' if not symbol else symbol}")
