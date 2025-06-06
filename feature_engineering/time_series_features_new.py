"""
Time Series Feature Extractor for Irregular Time Series

Extracts temporal features from irregularly-spaced time series data including:
- Returns and volatility on irregular indices  
- Moving averages using time-based windows
- Auto-correlation and temporal patterns
- Rolling statistics computed on irregular timestamps
- No fixed-frequency resampling - preserves microsecond precision
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from scipy import stats
from bisect import bisect_left, bisect_right
import warnings

logger = logging.getLogger(__name__)


class TimeSeriesFeatureExtractor:
    """
    Extracts time series features from DataFrames with irregular timestamp indices.
    
    All computations are performed on the original irregular timestamps without
    any resampling, preserving microsecond precision for HFT applications.
    """
    
    def __init__(self,
                 return_windows_sec: List[float] = None,
                 volatility_windows_sec: List[float] = None,
                 momentum_windows_sec: List[float] = None,
                 max_lookback_sec: float = 3600.0):
        """
        Initialize time series feature extractor for irregular data
        
        Args:
            return_windows_sec: Return calculation windows in seconds [1.0, 5.0, 30.0]
            volatility_windows_sec: Volatility windows in seconds [60.0, 300.0]
            momentum_windows_sec: Momentum windows in seconds [60.0, 300.0]
            max_lookback_sec: Maximum lookback for any calculation
        """
        self.return_windows_sec = return_windows_sec or [1.0, 5.0, 30.0]
        self.volatility_windows_sec = volatility_windows_sec or [60.0, 300.0]
        self.momentum_windows_sec = momentum_windows_sec or [60.0, 300.0]
        self.max_lookback_sec = max_lookback_sec
        
        # Convert seconds to microseconds for internal use
        self.return_windows_us = [int(w * 1_000_000) for w in self.return_windows_sec]
        self.volatility_windows_us = [int(w * 1_000_000) for w in self.volatility_windows_sec]
        self.momentum_windows_us = [int(w * 1_000_000) for w in self.momentum_windows_sec]
        self.max_lookback_us = int(max_lookback_sec * 1_000_000)
        
        logger.info(f"TimeSeriesFeatureExtractor initialized with irregular index support")
        
    def _find_lookback_value(self, 
                            timestamps: np.ndarray, 
                            values: np.ndarray, 
                            target_timestamp: int,
                            lookback_us: int) -> Optional[float]:
        """
        Find the last value available at or before target_timestamp - lookback_us
        
        Args:
            timestamps: Array of timestamps in microseconds (must be sorted)
            values: Array of values corresponding to timestamps
            target_timestamp: Target timestamp in microseconds
            lookback_us: Lookback window in microseconds
            
        Returns:
            Value at or before the lookback timestamp, or None if not found
        """
        target_lookback = target_timestamp - lookback_us
        
        # Binary search for the latest timestamp <= target_lookback
        idx = bisect_right(timestamps, target_lookback) - 1
        
        if idx >= 0:
            return values[idx]
        return None
        
    def _get_window_data(self, 
                        timestamps: np.ndarray,
                        values: np.ndarray,
                        target_timestamp: int, 
                        window_us: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get all data points within a time window ending at target_timestamp
        
        Args:
            timestamps: Array of timestamps in microseconds (must be sorted)
            values: Array of values
            target_timestamp: End of window timestamp
            window_us: Window size in microseconds
            
        Returns:
            Tuple of (window_timestamps, window_values)
        """
        start_time = target_timestamp - window_us
        
        # Find indices within window [start_time, target_timestamp]
        start_idx = bisect_left(timestamps, start_time)
        end_idx = bisect_right(timestamps, target_timestamp)
        
        if start_idx < end_idx:
            return timestamps[start_idx:end_idx], values[start_idx:end_idx]
        else:
            return np.array([]), np.array([])
            
    def _calculate_returns(self, 
                          df: pd.DataFrame, 
                          field: str) -> pd.DataFrame:
        """
        Calculate returns for different time windows on irregular index
        
        Args:
            df: DataFrame with timestamp_us index
            field: Field name to calculate returns for
            
        Returns:
            DataFrame with return columns added
        """
        if field not in df.columns:
            logger.warning(f"Field {field} not found, skipping returns")
            return pd.DataFrame(index=df.index)
            
        returns_df = pd.DataFrame(index=df.index)
        values = df[field].values
        timestamps = df.index.values
        
        for i, window_us in enumerate(self.return_windows_us):
            window_sec = self.return_windows_sec[i]
            col_name = f"{field}_ret_{window_sec}s"
            
            returns = []
            for j, target_ts in enumerate(timestamps):
                current_value = values[j]
                past_value = self._find_lookback_value(timestamps, values, target_ts, window_us)
                
                if past_value is not None and past_value != 0:
                    ret = (current_value - past_value) / past_value
                    returns.append(ret)
                else:
                    returns.append(np.nan)
                    
            returns_df[col_name] = returns
            
        return returns_df
        
    def _calculate_volatility(self, 
                             df: pd.DataFrame, 
                             field: str) -> pd.DataFrame:
        """
        Calculate rolling volatility on irregular index using time windows
        
        Args:
            df: DataFrame with timestamp_us index
            field: Field name to calculate volatility for
            
        Returns:
            DataFrame with volatility columns added
        """
        if field not in df.columns:
            logger.warning(f"Field {field} not found, skipping volatility")
            return pd.DataFrame(index=df.index)
            
        vol_df = pd.DataFrame(index=df.index)
        values = df[field].values
        timestamps = df.index.values
        
        for i, window_us in enumerate(self.volatility_windows_us):
            window_sec = self.volatility_windows_sec[i]
            col_name = f"{field}_vol_{window_sec}s"
            
            volatilities = []
            for target_ts in timestamps:
                window_ts, window_vals = self._get_window_data(
                    timestamps, values, target_ts, window_us
                )
                
                if len(window_vals) > 1:
                    volatility = np.std(window_vals, ddof=1)
                    volatilities.append(volatility)
                else:
                    volatilities.append(np.nan)
                    
            vol_df[col_name] = volatilities
            
        return vol_df
        
    def _calculate_momentum(self, 
                           df: pd.DataFrame, 
                           field: str) -> pd.DataFrame:
        """
        Calculate momentum indicators on irregular index
        
        Args:
            df: DataFrame with timestamp_us index
            field: Field name to calculate momentum for
            
        Returns:
            DataFrame with momentum columns added
        """
        if field not in df.columns:
            logger.warning(f"Field {field} not found, skipping momentum")
            return pd.DataFrame(index=df.index)
            
        momentum_df = pd.DataFrame(index=df.index)
        values = df[field].values
        timestamps = df.index.values
        
        for i, window_us in enumerate(self.momentum_windows_us):
            window_sec = self.momentum_windows_sec[i]
            col_name = f"{field}_momentum_{window_sec}s"
            
            momentums = []
            for target_ts in timestamps:
                window_ts, window_vals = self._get_window_data(
                    timestamps, values, target_ts, window_us
                )
                
                if len(window_vals) > 1:
                    # Simple momentum: (current - average) / average
                    current_val = window_vals[-1]
                    avg_val = np.mean(window_vals[:-1])
                    
                    if avg_val != 0:
                        momentum = (current_val - avg_val) / avg_val
                        momentums.append(momentum)
                    else:
                        momentums.append(0.0)
                else:
                    momentums.append(np.nan)
                    
            momentum_df[col_name] = momentums
            
        return momentum_df
        
    def _calculate_autocorrelation(self, 
                                  df: pd.DataFrame, 
                                  field: str,
                                  lag_windows_sec: List[float] = None) -> pd.DataFrame:
        """
        Calculate autocorrelation at various lags on irregular index
        
        Args:
            df: DataFrame with timestamp_us index
            field: Field name to calculate autocorrelation for
            lag_windows_sec: Lag windows in seconds
            
        Returns:
            DataFrame with autocorrelation columns added
        """
        lag_windows_sec = lag_windows_sec or [1.0, 5.0, 30.0]
        
        if field not in df.columns:
            logger.warning(f"Field {field} not found, skipping autocorrelation")
            return pd.DataFrame(index=df.index)
            
        autocorr_df = pd.DataFrame(index=df.index)
        values = df[field].values
        timestamps = df.index.values
        
        for lag_sec in lag_windows_sec:
            lag_us = int(lag_sec * 1_000_000)
            col_name = f"{field}_autocorr_{lag_sec}s"
            
            autocorrs = []
            for target_ts in timestamps:
                # Get current value
                current_idx = np.searchsorted(timestamps, target_ts)
                if current_idx >= len(values):
                    autocorrs.append(np.nan)
                    continue
                    
                current_val = values[current_idx]
                
                # Get lagged value
                lagged_val = self._find_lookback_value(timestamps, values, target_ts, lag_us)
                
                if lagged_val is not None:
                    # Simple autocorrelation approximation
                    # For proper autocorrelation, we'd need a window of values
                    # Here we use correlation with rolling history
                    window_size_us = max(lag_us * 2, 60_000_000)  # At least 60s window
                    window_ts, window_vals = self._get_window_data(
                        timestamps, values, target_ts, window_size_us
                    )
                    
                    if len(window_vals) > 10:
                        # Create lagged series
                        lagged_series = []
                        current_series = []
                        
                        for j in range(len(window_vals)):
                            ts = window_ts[j]
                            val = window_vals[j]
                            lag_val = self._find_lookback_value(window_ts, window_vals, ts, lag_us)
                            
                            if lag_val is not None:
                                current_series.append(val)
                                lagged_series.append(lag_val)
                                
                        if len(current_series) > 5:
                            corr = np.corrcoef(current_series, lagged_series)[0, 1]
                            autocorrs.append(corr if not np.isnan(corr) else 0.0)
                        else:
                            autocorrs.append(0.0)
                    else:
                        autocorrs.append(0.0)
                else:
                    autocorrs.append(np.nan)
                    
            autocorr_df[col_name] = autocorrs
            
        return autocorr_df
        
    def extract_temporal_features(self, 
                                 df: pd.DataFrame, 
                                 fields: List[str]) -> pd.DataFrame:
        """
        Extract all temporal features for specified fields
        
        Args:
            df: DataFrame with timestamp_us index (irregular)
            fields: List of field names to extract features for
            
        Returns:
            DataFrame with temporal feature columns, same index as input
        """
        if len(df) == 0:
            logger.warning("Empty DataFrame provided")
            return pd.DataFrame()
            
        # Verify index is timestamp_us
        if df.index.name != 'timestamp_us':
            logger.warning(f"Index name is '{df.index.name}', expected 'timestamp_us'")
            
        # Verify index is sorted
        if not df.index.is_monotonic_increasing:
            logger.warning("Index is not sorted, sorting now")
            df = df.sort_index()
            
        logger.info(f"Extracting temporal features for {len(fields)} fields on {len(df)} irregular timestamps")
        
        all_features = []
        
        for field in fields:
            if field not in df.columns:
                logger.warning(f"Field '{field}' not found in DataFrame")
                continue
                
            logger.debug(f"Processing field: {field}")
            
            # Calculate returns
            returns_df = self._calculate_returns(df, field)
            if not returns_df.empty:
                all_features.append(returns_df)
                
            # Calculate volatility  
            vol_df = self._calculate_volatility(df, field)
            if not vol_df.empty:
                all_features.append(vol_df)
                
            # Calculate momentum
            momentum_df = self._calculate_momentum(df, field)
            if not momentum_df.empty:
                all_features.append(momentum_df)
                
            # Calculate autocorrelation
            autocorr_df = self._calculate_autocorrelation(df, field)
            if not autocorr_df.empty:
                all_features.append(autocorr_df)
                
        # Combine all features
        if all_features:
            features_df = pd.concat(all_features, axis=1)
            features_df.index.name = 'timestamp_us'
            
            logger.info(f"Generated {features_df.shape[1]} temporal features")
            return features_df
        else:
            logger.warning("No temporal features generated")
            return pd.DataFrame(index=df.index)
            
    def get_feature_names(self, fields: List[str]) -> List[str]:
        """
        Get list of all feature names that would be generated
        
        Args:
            fields: List of base field names
            
        Returns:
            List of feature column names
        """
        feature_names = []
        
        for field in fields:
            # Returns
            for window_sec in self.return_windows_sec:
                feature_names.append(f"{field}_ret_{window_sec}s")
                
            # Volatility
            for window_sec in self.volatility_windows_sec:
                feature_names.append(f"{field}_vol_{window_sec}s")
                
            # Momentum
            for window_sec in self.momentum_windows_sec:
                feature_names.append(f"{field}_momentum_{window_sec}s")
                
            # Autocorrelation (default lags)
            for lag_sec in [1.0, 5.0, 30.0]:
                feature_names.append(f"{field}_autocorr_{lag_sec}s")
                
        return feature_names
