"""
Data synchronization module for handling asynchronous order book data.
Inspired by SGX methodology for high-frequency data alignment.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class DataSynchronizer:
    """
    Synchronizes asynchronous order book data from multiple cryptocurrency pairs.
    Handles microsecond-precision timestamps and different update frequencies.
    """
    
    def __init__(self, 
                 sync_method: str = 'linear_interpolation',
                 max_time_gap: float = 1.0,  # seconds
                 min_update_frequency: float = 0.001):  # 1ms minimum
        """
        Initialize data synchronizer.
        
        Args:
            sync_method: Method for synchronization ('linear_interpolation', 'forward_fill', 'backward_fill')
            max_time_gap: Maximum allowed time gap for interpolation (seconds)
            min_update_frequency: Minimum update frequency for resampling (seconds)
        """
        self.sync_method = sync_method
        self.max_time_gap = max_time_gap
        self.min_update_frequency = min_update_frequency
        self.synchronized_data = {}
        
    def synchronize_orderbooks(self, 
                             orderbook_data: Dict[str, pd.DataFrame],
                             reference_asset: str = 'ETH_EUR') -> Dict[str, pd.DataFrame]:
        """
        Synchronize multiple order book datasets to a common timeline.
        
        Args:
            orderbook_data: Dictionary of {asset_name: dataframe} with order book data
            reference_asset: Asset to use as timing reference
            
        Returns:
            Dictionary of synchronized dataframes
        """
        if reference_asset not in orderbook_data:
            raise ValueError(f"Reference asset {reference_asset} not found in data")
            
        # Get reference timeline
        reference_timeline = orderbook_data[reference_asset].index
        logger.info(f"Using {reference_asset} as reference with {len(reference_timeline)} timestamps")
        
        synchronized_data = {}
        
        for asset_name, df in orderbook_data.items():
            logger.info(f"Synchronizing {asset_name}...")
            
            if asset_name == reference_asset:
                synchronized_data[asset_name] = df.copy()
            else:
                synchronized_data[asset_name] = self._synchronize_to_timeline(
                    df, reference_timeline, asset_name
                )
                
        self.synchronized_data = synchronized_data
        return synchronized_data
    
    def _synchronize_to_timeline(self, 
                               df: pd.DataFrame, 
                               target_timeline: pd.Index,
                               asset_name: str) -> pd.DataFrame:
        """Synchronize a single dataframe to target timeline."""
        
        # Create combined timeline
        combined_timeline = df.index.union(target_timeline).sort_values()
        
        # Reindex to combined timeline
        reindexed_df = df.reindex(combined_timeline)
        
        # Apply synchronization method
        if self.sync_method == 'linear_interpolation':
            synchronized_df = self._linear_interpolation_sync(reindexed_df)
        elif self.sync_method == 'forward_fill':
            synchronized_df = self._forward_fill_sync(reindexed_df)
        elif self.sync_method == 'backward_fill':
            synchronized_df = self._backward_fill_sync(reindexed_df)
        else:
            raise ValueError(f"Unknown sync method: {self.sync_method}")
            
        # Filter to target timeline
        synchronized_df = synchronized_df.reindex(target_timeline)
        
        # Log synchronization quality
        missing_ratio = synchronized_df.isnull().sum().sum() / (len(synchronized_df) * len(synchronized_df.columns))
        logger.info(f"{asset_name} synchronization complete. Missing data ratio: {missing_ratio:.3%}")
        
        return synchronized_df
    
    def _linear_interpolation_sync(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply linear interpolation with time gap constraints."""
        
        # Calculate time differences
        time_diffs = df.index.to_series().diff()
        
        # Create mask for acceptable interpolation gaps
        gap_mask = time_diffs <= pd.Timedelta(seconds=self.max_time_gap)
        
        # Apply interpolation
        interpolated_df = df.copy()
        
        for column in df.columns:
            # Only interpolate where gaps are acceptable
            series = df[column].copy()
            
            # Mark values where gap is too large as NaN
            invalid_interp = ~gap_mask & series.isnull()
            
            # Interpolate
            series = series.interpolate(method='time', limit_area='inside')
            
            # Remove interpolated values where gap was too large
            series[invalid_interp] = np.nan
            
            interpolated_df[column] = series
            
        return interpolated_df
    
    def _forward_fill_sync(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply forward fill with time constraints."""
        
        # Calculate time differences for each non-null value
        filled_df = df.copy()
        
        for column in df.columns:
            series = df[column].copy()
            
            # Forward fill with limit
            series = series.ffill()
            
            # Create time-based mask to limit forward fill duration
            last_valid_times = series.groupby(series.notnull().cumsum()).transform('first')
            current_times = pd.Series(df.index, index=df.index)
            time_since_last = current_times - last_valid_times
            
            # Remove forward filled values that exceed time limit
            invalid_ff = time_since_last > pd.Timedelta(seconds=self.max_time_gap)
            series[invalid_ff] = np.nan
            
            filled_df[column] = series
            
        return filled_df
    
    def _backward_fill_sync(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply backward fill with time constraints."""
        return df.bfill()
    
    def create_microstructure_features(self, 
                                     synchronized_data: Dict[str, pd.DataFrame],
                                     feature_windows: List[int] = [1, 5, 10, 30]) -> pd.DataFrame:
        """
        Create microstructure features from synchronized data.
        Similar to SGX approach for market microstructure analysis.
        """
        
        all_features = []
        
        for asset_name, df in synchronized_data.items():
            asset_features = []
            
            # Basic features
            df['mid_price'] = (df['bid_price_1'] + df['ask_price_1']) / 2
            df['spread'] = df['ask_price_1'] - df['bid_price_1']
            df['spread_bps'] = (df['spread'] / df['mid_price']) * 10000
            
            # Volume imbalance
            df['volume_imbalance'] = (df['bid_quantity_1'] - df['ask_quantity_1']) / (df['bid_quantity_1'] + df['ask_quantity_1'])
            
            # Price changes
            df['price_change'] = df['mid_price'].diff()
            df['price_return'] = df['mid_price'].pct_change()
            
            # Create windowed features
            for window in feature_windows:
                # Rolling statistics
                df[f'price_vol_{window}'] = df['price_return'].rolling(window).std()
                df[f'spread_mean_{window}'] = df['spread'].rolling(window).mean()
                df[f'volume_imb_mean_{window}'] = df['volume_imbalance'].rolling(window).mean()
                
                # Price momentum
                df[f'price_momentum_{window}'] = df['mid_price'].rolling(window).apply(
                    lambda x: (x[-1] - x[0]) / x[0] if x[0] != 0 else 0
                )
                
                # Order flow features
                df[f'bid_qty_change_{window}'] = df['bid_quantity_1'].rolling(window).apply(
                    lambda x: x[-1] - x[0]
                )
                df[f'ask_qty_change_{window}'] = df['ask_quantity_1'].rolling(window).apply(
                    lambda x: x[-1] - x[0]
                )
            
            # Add asset prefix to column names
            feature_columns = [col for col in df.columns if col not in ['timestamp']]
            df_renamed = df[feature_columns].add_prefix(f'{asset_name}_')
            asset_features.append(df_renamed)
            
            all_features.extend(asset_features)
        
        # Combine all features
        if all_features:
            combined_features = pd.concat(all_features, axis=1)
            return combined_features
        else:
            return pd.DataFrame()
    
    def create_cross_asset_features(self, 
                                  eth_data: pd.DataFrame, 
                                  xbt_data: pd.DataFrame,
                                  lag_windows: List[int] = [1, 3, 5, 10, 20]) -> pd.DataFrame:
        """
        Create cross-asset features for lead-lag analysis.
        """
        
        cross_features = pd.DataFrame(index=eth_data.index)
        
        # Ensure both assets have required columns
        required_cols = ['mid_price', 'spread', 'volume_imbalance']
        for col in required_cols:
            if f'ETH_EUR_{col}' not in eth_data.columns or f'XBT_EUR_{col}' not in xbt_data.columns:
                logger.warning(f"Missing required column {col} for cross-asset features")
                continue
                
            eth_series = eth_data[f'ETH_EUR_{col}']
            xbt_series = xbt_data[f'XBT_EUR_{col}']
            
            # Price ratio
            if col == 'mid_price':
                cross_features[f'price_ratio_ETH_XBT'] = eth_series / xbt_series
                cross_features[f'price_ratio_change'] = cross_features[f'price_ratio_ETH_XBT'].pct_change()
                
            # Spread differential
            elif col == 'spread':
                cross_features[f'spread_diff_ETH_XBT'] = eth_series - xbt_series
                
            # Volume imbalance correlation
            elif col == 'volume_imbalance':
                for window in lag_windows:
                    cross_features[f'vol_imb_corr_{window}'] = eth_series.rolling(window).corr(xbt_series)
            
            # Lagged correlations
            for lag in lag_windows:
                if lag > 0:
                    # ETH leads XBT
                    cross_features[f'{col}_corr_ETH_leads_{lag}'] = eth_series.corr(xbt_series.shift(lag))
                    # XBT leads ETH  
                    cross_features[f'{col}_corr_XBT_leads_{lag}'] = eth_series.shift(lag).corr(xbt_series)
        
        return cross_features
    
    def detect_lead_lag_relationships(self, 
                                    synchronized_data: Dict[str, pd.DataFrame],
                                    max_lag: int = 50,
                                    significance_threshold: float = 0.05) -> Dict[str, Dict]:
        """
        Detect lead-lag relationships between assets using cross-correlation.
        """
        
        assets = list(synchronized_data.keys())
        relationships = {}
        
        for i, asset1 in enumerate(assets):
            for asset2 in assets[i+1:]:
                
                # Get price returns
                returns1 = synchronized_data[asset1]['mid_price'].pct_change().dropna()
                returns2 = synchronized_data[asset2]['mid_price'].pct_change().dropna()
                
                # Align data
                aligned_data = pd.concat([returns1, returns2], axis=1, join='inner')
                aligned_data.columns = [asset1, asset2]
                aligned_data = aligned_data.dropna()
                
                if len(aligned_data) > max_lag * 2:
                    # Calculate cross-correlations
                    lags = range(-max_lag, max_lag + 1)
                    cross_correlations = []
                    
                    for lag in lags:
                        if lag == 0:
                            corr = aligned_data[asset1].corr(aligned_data[asset2])
                        elif lag > 0:
                            # asset1 leads asset2
                            if len(aligned_data) > lag:
                                corr = aligned_data[asset1].iloc[:-lag].corr(aligned_data[asset2].iloc[lag:])
                            else:
                                corr = np.nan
                        else:
                            # asset2 leads asset1
                            lag_abs = abs(lag)
                            if len(aligned_data) > lag_abs:
                                corr = aligned_data[asset1].iloc[lag_abs:].corr(aligned_data[asset2].iloc[:-lag_abs])
                            else:
                                corr = np.nan
                        
                        cross_correlations.append(corr)
                    
                    # Find significant relationships
                    cross_correlations = np.array(cross_correlations)
                    max_corr_idx = np.nanargmax(np.abs(cross_correlations))
                    max_corr = cross_correlations[max_corr_idx]
                    optimal_lag = lags[max_corr_idx]
                    
                    # Store relationship
                    relationship_key = f"{asset1}_{asset2}"
                    relationships[relationship_key] = {
                        'max_correlation': max_corr,
                        'optimal_lag': optimal_lag,
                        'lead_asset': asset1 if optimal_lag > 0 else asset2,
                        'lag_asset': asset2 if optimal_lag > 0 else asset1,
                        'lag_magnitude': abs(optimal_lag),
                        'all_correlations': cross_correlations,
                        'lags': lags
                    }
                    
                    logger.info(f"Lead-lag analysis {asset1}-{asset2}: "
                              f"Max correlation {max_corr:.4f} at lag {optimal_lag}")
        
        return relationships
    
    def resample_to_frequency(self, 
                            data: pd.DataFrame, 
                            freq: str = '100ms',
                            method: str = 'last') -> pd.DataFrame:
        """
        Resample data to a specific frequency.
        
        Args:
            data: Input dataframe with datetime index
            freq: Target frequency (e.g., '100ms', '1s', '5s')
            method: Resampling method ('last', 'mean', 'first')
            
        Returns:
            Resampled dataframe
        """
        
        if method == 'last':
            resampled = data.resample(freq).last()
        elif method == 'mean':
            resampled = data.resample(freq).mean()
        elif method == 'first':
            resampled = data.resample(freq).first()
        else:
            raise ValueError(f"Unknown resampling method: {method}")
            
        # Forward fill missing values
        resampled = resampled.ffill()
        
        logger.info(f"Resampled from {len(data)} to {len(resampled)} rows at {freq} frequency")
        
        return resampled
    
    def validate_synchronization_quality(self, 
                                       synchronized_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Validate the quality of data synchronization.
        """
        
        quality_metrics = {}
        
        for asset_name, df in synchronized_data.items():
            
            # Missing data ratio
            missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
            
            # Time gaps analysis
            time_diffs = df.index.to_series().diff().dt.total_seconds()
            avg_time_gap = time_diffs.mean()
            max_time_gap = time_diffs.max()
            std_time_gap = time_diffs.std()
            
            # Data completeness by column
            completeness_by_col = (1 - df.isnull().sum() / len(df)).to_dict()
            
            quality_metrics[asset_name] = {
                'missing_data_ratio': missing_ratio,
                'avg_time_gap_seconds': avg_time_gap,
                'max_time_gap_seconds': max_time_gap,
                'std_time_gap_seconds': std_time_gap,
                'total_records': len(df),
                'completeness_by_column': completeness_by_col
            }
            
            logger.info(f"{asset_name} quality: {missing_ratio:.3%} missing, "
                       f"avg gap: {avg_time_gap:.3f}s")
        
        return quality_metrics
