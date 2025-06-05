"""
Multi-Period Data Loader for Crypto HFT Trading

Handles loading and organizing data from multiple time periods (DATA_0, DATA_1, DATA_2)
for temporal validation and out-of-sample testing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
from .data_loader import OrderBookDataLoader

logger = logging.getLogger(__name__)

class MultiPeriodDataLoader:
    """
    Enhanced data loader that handles multiple temporal periods for robust validation.
    
    Supports:
    - DATA_0: Training period (first day)
    - DATA_1: Validation period (second day) 
    - DATA_2: Out-of-sample testing (two days)
    """
    
    def __init__(self, base_path: str = "data/raw"):
        """
        Initialize the multi-period data loader.
        
        Args:
            base_path: Base path to raw data directory
        """
        self.base_path = Path(base_path)
        self.single_loader = OrderBookDataLoader(str(self.base_path))
        
        # Discover available periods
        self.available_periods = self._discover_periods()
        logger.info(f"Discovered periods: {self.available_periods}")
    
    def _discover_periods(self) -> List[str]:
        """Discover available data periods in the raw data directory."""
        periods = []
        for item in self.base_path.iterdir():
            if item.is_dir() and item.name.startswith('DATA_'):
                periods.append(item.name)
        return sorted(periods)
    
    def get_period_info(self, period: str, symbols: List[str] = None) -> Dict:
        """
        Get information about a specific period.
        
        Args:
            period: Period name (e.g., 'DATA_0')
            symbols: List of symbols to analyze
            
        Returns:
            Dictionary with period statistics
        """
        if symbols is None:
            symbols = ['ETH_EUR', 'XBT_EUR']
        
        period_path = self.base_path / period
        info = {
            'period': period,
            'path': str(period_path),
            'symbols': {}
        }
        
        for symbol in symbols:
            file_path = period_path / f"{symbol}.csv"
            if file_path.exists():
                # Quick metadata without loading full file
                try:
                    # Read small sample to get time range
                    sample = pd.read_csv(file_path, nrows=10000)
                    sample['datetime'] = pd.to_datetime(sample['timestamp'], unit='s')
                    
                    # Get full file size
                    total_lines = sum(1 for _ in open(file_path)) - 1  # Exclude header
                    
                    info['symbols'][symbol] = {
                        'file_size_mb': file_path.stat().st_size / 1024 / 1024,
                        'total_records': total_lines,
                        'time_start': sample['datetime'].min(),
                        'time_end': sample['datetime'].max(),
                        'sample_duration': sample['datetime'].max() - sample['datetime'].min(),
                        'columns': list(sample.columns)
                    }
                except Exception as e:
                    logger.warning(f"Error reading {symbol} in {period}: {e}")
                    info['symbols'][symbol] = {'error': str(e)}
        
        return info
    
    def get_all_periods_summary(self) -> Dict:
        """Get comprehensive summary of all available periods."""
        summary = {
            'periods': {},
            'total_records': 0,
            'total_size_mb': 0,
            'time_range': {
                'earliest': None,
                'latest': None
            }
        }
        
        for period in self.available_periods:
            period_info = self.get_period_info(period)
            summary['periods'][period] = period_info
            
            # Aggregate statistics
            for symbol_info in period_info['symbols'].values():
                if 'total_records' in symbol_info:
                    summary['total_records'] += symbol_info['total_records']
                    summary['total_size_mb'] += symbol_info['file_size_mb']
                    
                    # Track time range
                    if summary['time_range']['earliest'] is None:
                        summary['time_range']['earliest'] = symbol_info['time_start']
                        summary['time_range']['latest'] = symbol_info['time_end']
                    else:
                        if symbol_info['time_start'] < summary['time_range']['earliest']:
                            summary['time_range']['earliest'] = symbol_info['time_start']
                        if symbol_info['time_end'] > summary['time_range']['latest']:
                            summary['time_range']['latest'] = symbol_info['time_end']
        
        return summary
    
    def load_period_data(self, 
                        period: str, 
                        symbols: List[str] = None,
                        max_records: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        Load data for a specific period.
        
        Args:
            period: Period to load (e.g., 'DATA_0')
            symbols: Symbols to load
            max_records: Maximum records per symbol
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        if symbols is None:
            symbols = ['ETH_EUR', 'XBT_EUR']
        
        if period not in self.available_periods:
            raise ValueError(f"Period {period} not found. Available: {self.available_periods}")
        
        logger.info(f"Loading period {period} with symbols: {symbols}")
        
        period_data = {}
        period_path = self.base_path / period
        
        for symbol in symbols:
            file_path = period_path / f"{symbol}.csv"
            if file_path.exists():
                logger.info(f"Loading {symbol} from {file_path}")
                
                # Use the single loader to load individual files
                # Temporarily update the data path for this specific file
                original_path = self.single_loader.data_path
                self.single_loader.data_path = Path(period_path)
                df = self.single_loader.load_data(symbol, max_records)
                self.single_loader.data_path = original_path
                
                period_data[symbol] = df
                
                logger.info(f"Loaded {len(df):,} records for {symbol}")
            else:
                logger.warning(f"File not found: {file_path}")
        
        return period_data
    
    def create_temporal_splits(self, 
                              strategy: str = 'by_period') -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Create temporal train/validation/test splits using the multi-period structure.
        
        Args:
            strategy: Split strategy ('by_period' or 'chronological')
            
        Returns:
            Dictionary with train/validation/test data
        """
        if strategy == 'by_period':
            return self._split_by_period()
        elif strategy == 'chronological':
            return self._split_chronological()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _split_by_period(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Split data by period:
        - DATA_0: Training
        - DATA_1: Validation  
        - DATA_2: Test (out-of-sample)
        """
        splits = {
            'train': {},
            'validation': {},
            'test': {}
        }
        
        # Training on DATA_0
        if 'DATA_0' in self.available_periods:
            logger.info("Loading training data from DATA_0")
            splits['train'] = self.load_period_data('DATA_0')
        
        # Validation on DATA_1
        if 'DATA_1' in self.available_periods:
            logger.info("Loading validation data from DATA_1")
            splits['validation'] = self.load_period_data('DATA_1')
        
        # Test on DATA_2
        if 'DATA_2' in self.available_periods:
            logger.info("Loading test data from DATA_2")
            splits['test'] = self.load_period_data('DATA_2')
        
        return splits
    
    def _split_chronological(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Split data chronologically across all periods (70/15/15 split).
        """
        # Load all data
        all_data = {}
        for period in self.available_periods:
            period_data = self.load_period_data(period)
            for symbol, df in period_data.items():
                if symbol not in all_data:
                    all_data[symbol] = []
                all_data[symbol].append(df)
        
        # Concatenate and sort chronologically
        combined_data = {}
        for symbol, dfs in all_data.items():
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df = combined_df.sort_values('datetime').reset_index(drop=True)
            combined_data[symbol] = combined_df
        
        # Create temporal splits
        splits = {'train': {}, 'validation': {}, 'test': {}}
        
        for symbol, df in combined_data.items():
            n = len(df)
            train_end = int(0.7 * n)
            val_end = int(0.85 * n)
            
            splits['train'][symbol] = df.iloc[:train_end].copy()
            splits['validation'][symbol] = df.iloc[train_end:val_end].copy()
            splits['test'][symbol] = df.iloc[val_end:].copy()
        
        return splits
    
    def get_optimal_sample_sizes(self, target_total_records: int = 1000000) -> Dict[str, int]:
        """
        Calculate optimal sample sizes for each period to reach target total.
        
        Args:
            target_total_records: Target total number of records
            
        Returns:
            Dictionary mapping period to max_records
        """
        summary = self.get_all_periods_summary()
        
        # Calculate proportional sampling
        total_available = summary['total_records']
        sampling_ratio = min(1.0, target_total_records / total_available)
        
        optimal_sizes = {}
        for period, info in summary['periods'].items():
            period_records = sum(
                s.get('total_records', 0) 
                for s in info['symbols'].values() 
                if 'total_records' in s
            )
            optimal_sizes[period] = int(period_records * sampling_ratio)
        
        return optimal_sizes
    
    def load_balanced_sample(self, 
                           max_records_per_period: int = 100000) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Load a balanced sample from all periods.
        
        Args:
            max_records_per_period: Maximum records to load per period
            
        Returns:
            Dictionary with balanced data from all periods
        """
        balanced_data = {}
        
        for period in self.available_periods:
            logger.info(f"Loading balanced sample from {period}")
            period_data = self.load_period_data(
                period, 
                max_records=max_records_per_period
            )
            balanced_data[period] = period_data
        
        return balanced_data

    def export_period_statistics(self, output_path: str = "results/period_analysis.json"):
        """Export comprehensive period statistics to JSON."""
        import json
        
        summary = self.get_all_periods_summary()
        
        # Convert datetime objects to strings for JSON serialization
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_datetime(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetime(item) for item in obj]
            return obj
        
        summary_serializable = convert_datetime(summary)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(summary_serializable, f, indent=2)
        
        logger.info(f"Period statistics exported to {output_file}")
        return summary_serializable

if __name__ == "__main__":
    # Test the multi-period loader
    loader = MultiPeriodDataLoader()
    
    print("=== PERIOD SUMMARY ===")
    summary = loader.get_all_periods_summary()
    
    for period, info in summary['periods'].items():
        print(f"\n{period}:")
        for symbol, stats in info['symbols'].items():
            if 'total_records' in stats:
                print(f"  {symbol}: {stats['total_records']:,} records, "
                      f"{stats['file_size_mb']:.1f}MB")
                print(f"    Time: {stats['time_start']} to {stats['time_end']}")
    
    print(f"\nTotal: {summary['total_records']:,} records, {summary['total_size_mb']:.1f}MB")
