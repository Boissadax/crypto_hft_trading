"""
Data Format Converter

Converts order book data between long and wide formats to enable compatibility
between the original CSV data (long format) and the async processing pipeline 
(which expects wide format).

Long format: Each row = one price/volume entry at specific side/level
Wide format: Each row = complete order book snapshot with bid_price_1, ask_price_1, etc.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

logger = logging.getLogger(__name__)

class OrderBookDataFormatter:
    """Converts order book data between different formats with progress tracking."""
    
    def __init__(self, max_levels: int = 10):
        """
        Initialize the formatter.
        
        Args:
            max_levels: Maximum number of order book levels to process
        """
        self.max_levels = max_levels
        
        # Performance tracking
        self.performance_stats = {
            'total_processing_time': 0.0,
            'snapshots_created': 0,
            'events_processed': 0,
            'optimization_hits': 0
        }
    
    def get_performance_diagnostics(self) -> Dict[str, any]:
        """
        Get detailed performance diagnostics for monitoring system efficiency.
        
        Returns:
            Dictionary with performance metrics and recommendations
        """
        return {
            'performance_stats': self.performance_stats.copy(),
            'optimization_status': {
                'tqdm_available': TQDM_AVAILABLE,
                'max_levels': self.max_levels,
                'last_optimization_ratio': getattr(self, '_last_optimization_ratio', None)
            },
            'recommendations': self._get_optimization_recommendations()
        }
    
    def _get_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on performance stats."""
        recommendations = []
        
        if not TQDM_AVAILABLE:
            recommendations.append("Install tqdm for better progress tracking: pip install tqdm")
        
        if self.performance_stats['snapshots_created'] > 10000:
            recommendations.append("Consider increasing time_window for large datasets")
        
        if self.performance_stats['optimization_hits'] > 0:
            recommendations.append("Optimizations were applied - monitor performance impact")
        
        return recommendations

    def long_to_wide(self, df: pd.DataFrame, time_window_ms: float = 100) -> pd.DataFrame:
        """
        Convert long format order book data to wide format with progress tracking.
        
        IMPORTANT: The time_window parameter is used ONLY for performance optimization,
        NOT for synchronizing timestamps! It preserves the asynchronous nature of data
        while reducing computational load by filtering snapshot density.
        
        Args:
            df: DataFrame in long format with columns:
                - datetime/timestamp: timestamp (can be Unix timestamp or datetime)
                - price: order price
                - volume: order volume  
                - side: 'bid' or 'ask'
                - level: order book level (1, 2, 3, ...)
            time_window_ms: Time window for reducing snapshot density (milliseconds)
                         - NOT for timestamp synchronization!
                         - Used only to improve performance on large datasets
                         - Set to 0 for maximum precision (all timestamps preserved)
                
        Returns:
            DataFrame in wide format with columns:
                - datetime: snapshot timestamp (original, not rounded)
                - bid_price_1, bid_price_2, ...: bid prices by level
                - ask_price_1, ask_price_2, ...: ask prices by level
                - bid_quantity_1, bid_quantity_2, ...: bid quantities by level
                - ask_quantity_1, ask_quantity_2, ...: ask quantities by level
        """
        import time
        start_time = time.time()
        
        logger.info(f"Converting {len(df)} long format records to wide format")
        logger.info(f"Time window: {time_window_ms}ms (for performance optimization, NOT synchronization)")
        
        # Progress tracking
        if not TQDM_AVAILABLE:
            logger.info("💡 Install tqdm for progress bars: pip install tqdm")
        
        # Optimize time window based on data size
        original_window = time_window_ms
        time_window_ms = self._optimize_time_window(df, time_window_ms)
        
        if time_window_ms != original_window:
            logger.info(f"⚡ Time window optimized: {original_window}ms → {time_window_ms}ms")
            self.performance_stats['optimization_hits'] += 1
        
        # Handle different timestamp column names
        timestamp_col = None
        if 'datetime' in df.columns:
            timestamp_col = 'datetime'
        elif 'timestamp' in df.columns:
            timestamp_col = 'timestamp'
        else:
            raise ValueError("No timestamp column found (expecting 'datetime' or 'timestamp')")
        
        # Convert Unix timestamps to datetime if needed
        if timestamp_col == 'timestamp':
            df = df.copy()
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Ensure we have the required columns
        required_cols = ['datetime', 'price', 'volume', 'side', 'level']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Sort by datetime and filter to max levels
        df = df.sort_values('datetime').copy()
        df = df[df['level'] <= self.max_levels]
        
        # Handle asynchronous order book data with forward-fill approach
        # CRITICAL: We preserve ALL original timestamps - no rounding or synchronization!
        # The time_window parameter only reduces snapshot density for computational efficiency
        
        all_timestamps = sorted(df['datetime'].unique())
        logger.info(f"📊 Found {len(all_timestamps)} unique timestamps (fully asynchronous)")
        
        # Strategy 1: Use time_window to reduce snapshot density (performance optimization ONLY)
        # This does NOT synchronize or round timestamps - it preserves asynchronous nature!
        if time_window_ms > 0:
            time_delta = pd.Timedelta(milliseconds=time_window_ms)
            filtered_timestamps = [all_timestamps[0]]
            
            # Progress bar for timestamp filtering
            if TQDM_AVAILABLE:
                timestamp_iter = tqdm(all_timestamps[1:], 
                                    desc="Filtering timestamps",
                                    unit="timestamp")
            else:
                timestamp_iter = all_timestamps[1:]
            
            for ts in timestamp_iter:
                if ts - filtered_timestamps[-1] >= time_delta:
                    filtered_timestamps.append(ts)
            
            reduction_pct = (1 - len(filtered_timestamps)/len(all_timestamps)) * 100
            logger.info(f"🎯 Time window filtering: {len(all_timestamps)} → {len(filtered_timestamps)} timestamps")
            logger.info(f"   Reduction: {reduction_pct:.1f}% (for performance, preserving asynchrony)")
            self._last_optimization_ratio = reduction_pct
        else:
            # If time_window = 0, keep ALL timestamps (maximum precision)
            filtered_timestamps = all_timestamps
            logger.info("🔬 Maximum precision mode: ALL timestamps preserved")
            
        # Strategy 2: Adaptive sampling for very large datasets (performance protection)
        max_snapshots = 500 if len(df) > 100000 else 1000
        if len(filtered_timestamps) > max_snapshots:
            # Take evenly distributed samples across time range
            step = max(1, len(filtered_timestamps) // max_snapshots)
            sampled_timestamps = filtered_timestamps[::step]
            # Always include the last timestamp
            if filtered_timestamps[-1] not in sampled_timestamps:
                sampled_timestamps.append(filtered_timestamps[-1])
            filtered_timestamps = sorted(sampled_timestamps)
            
            logger.info(f"🚀 Performance sampling: {len(filtered_timestamps)} snapshots selected (step={step})")
            logger.info("   Note: This is purely for computational efficiency - timestamps remain original!")
        
        logger.info(f"✅ Final: Creating {len(filtered_timestamps)} asynchronous snapshots")
        
        # OPTIMIZED VERSION: Process events incrementally with progress tracking
        df_sorted = df.sort_values('datetime').reset_index(drop=True)
        
        # Initialize data structures for forward-filling
        current_data = {}  # (side, level) -> (price, volume)
        snapshots = []
        last_processed_idx = 0
        
        # Enhanced progress tracking with tqdm
        if TQDM_AVAILABLE:
            snapshot_iterator = tqdm(enumerate(filtered_timestamps), 
                                   total=len(filtered_timestamps),
                                   desc="Creating order book snapshots",
                                   unit="snapshot",
                                   postfix={'current_level': 0, 'events_processed': 0})
        else:
            snapshot_iterator = enumerate(filtered_timestamps)
            logger.info("📈 Processing snapshots (install tqdm for detailed progress)")
        
        # Process each timestamp and create snapshots with optimized event processing
        for i, snapshot_time in snapshot_iterator:
            # Update progress for non-tqdm users
            if not TQDM_AVAILABLE and i % 100 == 0:
                progress_pct = (i / len(filtered_timestamps)) * 100
                logger.info(f"📊 Processing snapshot {i+1}/{len(filtered_timestamps)} ({progress_pct:.1f}%)")
            
            # OPTIMIZATION: Only process new events since last timestamp
            # Find events between last processed index and current timestamp
            relevant_events = df_sorted[
                (df_sorted.index >= last_processed_idx) & 
                (df_sorted['datetime'] <= snapshot_time)
            ]
            
            # Update current state with new events only
            events_this_round = 0
            for _, row in relevant_events.iterrows():
                current_data[(row['side'], row['level'])] = (row['price'], row['volume'])
                events_this_round += 1
            
            # Update performance stats
            self.performance_stats['events_processed'] += events_this_round
            
            # Update progress postfix for tqdm
            if TQDM_AVAILABLE:
                snapshot_iterator.set_postfix({
                    'events_batch': events_this_round,
                    'total_events': self.performance_stats['events_processed']
                })
            
            # Update last processed index to avoid reprocessing same events
            if not relevant_events.empty:
                last_processed_idx = relevant_events.index.max() + 1
            
            # Create snapshot
            snapshot = {'datetime': snapshot_time}
            
            # Fill in all levels for both sides
            for level in range(1, self.max_levels + 1):
                # Initialize with NaN
                snapshot[f'bid_price_{level}'] = np.nan
                snapshot[f'bid_quantity_{level}'] = np.nan
                snapshot[f'ask_price_{level}'] = np.nan
                snapshot[f'ask_quantity_{level}'] = np.nan
                
                # Fill with current data if available
                if ('bid', level) in current_data:
                    price, volume = current_data[('bid', level)]
                    snapshot[f'bid_price_{level}'] = price
                    snapshot[f'bid_quantity_{level}'] = volume
                
                if ('ask', level) in current_data:
                    price, volume = current_data[('ask', level)]
                    snapshot[f'ask_price_{level}'] = price
                    snapshot[f'ask_quantity_{level}'] = volume
            
            snapshots.append(snapshot)
        
        # Create wide format DataFrame with final performance tracking
        wide_df = pd.DataFrame(snapshots)
        wide_df = wide_df.sort_values('datetime').reset_index(drop=True)
        
        # Remove snapshots with no valid data
        price_cols = [f'bid_price_1', f'ask_price_1']
        valid_snapshots = wide_df[price_cols].notna().any(axis=1)
        wide_df = wide_df[valid_snapshots].reset_index(drop=True)
        
        # Update performance stats
        end_time = time.time()
        self.performance_stats['total_processing_time'] = end_time - start_time
        self.performance_stats['snapshots_created'] = len(wide_df)
        
        # Performance summary
        processing_speed = len(df) / (end_time - start_time) if (end_time - start_time) > 0 else 0
        
        logger.info(f"✅ Successfully created {len(wide_df)} wide format snapshots")
        logger.info(f"⚡ Performance Summary:")
        logger.info(f"   Processing time: {end_time - start_time:.2f}s")
        logger.info(f"   Processing speed: {processing_speed:,.0f} records/second")
        logger.info(f"   Events processed: {self.performance_stats['events_processed']:,}")
        logger.info(f"   Snapshots created: {self.performance_stats['snapshots_created']:,}")
        logger.info(f"   Data reduction: {(1 - len(wide_df)/len(all_timestamps))*100:.1f}% (for performance)")
        logger.info("🎯 Time window preserved asynchronous nature - no timestamp synchronization applied!")
        
        return wide_df
    
    def wide_to_long(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert wide format order book data to long format.
        
        Args:
            df: DataFrame in wide format
                
        Returns:
            DataFrame in long format
        """
        logger.info(f"Converting {len(df)} wide format snapshots to long format")
        
        long_records = []
        
        for _, row in df.iterrows():
            timestamp = row['datetime']
            
            # Extract bid data
            for level in range(1, self.max_levels + 1):
                price_col = f'bid_price_{level}'
                qty_col = f'bid_quantity_{level}'
                
                if price_col in row and qty_col in row:
                    if pd.notna(row[price_col]) and pd.notna(row[qty_col]):
                        long_records.append({
                            'datetime': timestamp,
                            'price': row[price_col],
                            'volume': row[qty_col],
                            'side': 'bid',
                            'level': level
                        })
            
            # Extract ask data
            for level in range(1, self.max_levels + 1):
                price_col = f'ask_price_{level}'
                qty_col = f'ask_quantity_{level}'
                
                if price_col in row and qty_col in row:
                    if pd.notna(row[price_col]) and pd.notna(row[qty_col]):
                        long_records.append({
                            'datetime': timestamp,
                            'price': row[price_col],
                            'volume': row[qty_col],
                            'side': 'ask',
                            'level': level
                        })
        
        long_df = pd.DataFrame(long_records)
        long_df = long_df.sort_values(['datetime', 'side', 'level']).reset_index(drop=True)
        
        logger.info(f"Created {len(long_df)} long format records")
        return long_df
    
    def validate_wide_format(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate wide format order book data.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Check required columns
        required_cols = ['datetime']
        for level in range(1, self.max_levels + 1):
            required_cols.extend([
                f'bid_price_{level}', f'ask_price_{level}',
                f'bid_quantity_{level}', f'ask_quantity_{level}'
            ])
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            validation['errors'].append(f"Missing columns: {missing_cols}")
            validation['valid'] = False
        
        if not validation['valid']:
            return validation
        
        # Check data quality
        for level in range(1, self.max_levels + 1):
            bid_price_col = f'bid_price_{level}'
            ask_price_col = f'ask_price_{level}'
            
            if bid_price_col in df.columns and ask_price_col in df.columns:
                # Check for negative spreads
                valid_data = df[[bid_price_col, ask_price_col]].dropna()
                if not valid_data.empty:
                    negative_spreads = (valid_data[ask_price_col] < valid_data[bid_price_col]).sum()
                    if negative_spreads > 0:
                        validation['warnings'].append(
                            f"Level {level}: {negative_spreads} negative spreads (ask < bid)"
                        )
        
        # Calculate statistics
        validation['stats'] = {
            'total_snapshots': len(df),
            'datetime_range': (df['datetime'].min(), df['datetime'].max()),
            'avg_bid_ask_coverage': {}
        }
        
        for level in range(1, self.max_levels + 1):
            bid_col = f'bid_price_{level}'
            ask_col = f'ask_price_{level}'
            
            if bid_col in df.columns and ask_col in df.columns:
                both_valid = df[[bid_col, ask_col]].notna().all(axis=1).sum()
                coverage = both_valid / len(df) if len(df) > 0 else 0
                validation['stats']['avg_bid_ask_coverage'][level] = coverage
        
        return validation
    
    def _optimize_time_window(self, df: pd.DataFrame, time_window_ms: float) -> float:
        """
        Optimize time window based on data density to improve performance.
        
        IMPORTANT: The time_window does NOT synchronize or round timestamps!
        It only serves as a minimum spacing filter to reduce computational load.
        
        Args:
            df: Input DataFrame
            time_window_ms: Initial time window
            
        Returns:
            Optimized time window
        """
        data_size = len(df)
        unique_timestamps = df['datetime'].nunique()
        
        # Calculate data density
        if unique_timestamps > 1:
            time_span = (df['datetime'].max() - df['datetime'].min()).total_seconds()
            avg_frequency_ms = (time_span * 1000) / unique_timestamps
            logger.info(f"Data analysis: {unique_timestamps:,} unique timestamps over {time_span:.1f}s")
            logger.info(f"Average frequency: {avg_frequency_ms:.2f}ms between timestamps")
        
        # Adaptive time window based on data size
        if data_size > 100000:  # Large dataset
            optimized_window = max(time_window_ms * 3, 500)  # At least 500ms for very large datasets
            logger.info(f"Large dataset detected ({data_size:,} records), increasing time window to {optimized_window}ms")
            logger.info("⚠️  This reduces snapshot count for performance - NO timestamp synchronization!")
        elif data_size > 50000:  # Medium dataset
            optimized_window = max(time_window_ms * 2, 200)  # At least 200ms
            logger.info(f"Medium dataset detected ({data_size:,} records), adjusting time window to {optimized_window}ms")
        else:
            optimized_window = time_window_ms
        
        return optimized_window


def demo_format_conversion():
    """Demonstrate the format conversion with sample data."""
    
    # Create sample long format data
    timestamps = pd.date_range('2024-01-01 10:00:00', periods=100, freq='100ms')
    long_data = []
    
    for i, ts in enumerate(timestamps):
        base_price = 50000 + i * 0.1  # Slowly increasing price
        
        for level in range(1, 4):  # 3 levels
            # Bid side
            bid_price = base_price - level * 0.5
            bid_qty = 1.0 + np.random.uniform(0, 2)
            long_data.append({
                'datetime': ts,
                'price': bid_price,
                'volume': bid_qty,
                'side': 'bid',
                'level': level
            })
            
            # Ask side
            ask_price = base_price + level * 0.5
            ask_qty = 1.0 + np.random.uniform(0, 2)
            long_data.append({
                'datetime': ts,
                'price': ask_price,
                'volume': ask_qty,
                'side': 'ask',
                'level': level
            })
    
    long_df = pd.DataFrame(long_data)
    print(f"Long format data shape: {long_df.shape}")
    print("\nLong format sample:")
    print(long_df.head(10))
    
    # Convert to wide format
    formatter = OrderBookDataFormatter(max_levels=3)
    wide_df = formatter.long_to_wide(long_df)
    
    print(f"\nWide format data shape: {wide_df.shape}")
    print("\nWide format sample:")
    print(wide_df.head())
    
    # Validate wide format
    validation = formatter.validate_wide_format(wide_df)
    print(f"\nValidation results:")
    print(f"Valid: {validation['valid']}")
    print(f"Errors: {validation['errors']}")
    print(f"Warnings: {validation['warnings']}")
    print(f"Coverage stats: {validation['stats']['avg_bid_ask_coverage']}")


if __name__ == "__main__":
    demo_format_conversion()
