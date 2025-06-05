#!/usr/bin/env python3
"""
Comprehensive Data Analysis Dashboard
====================================

Provides detailed visualization and analysis of:
1. Raw data characteristics
2. Processing pipeline steps
3. Signal detection quality
4. Bidirectional lead-lag relationships
5. Transfer entropy analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Import our components
from data_processing.data_loader import OrderBookDataLoader
from data_processing.data_formatter import OrderBookDataFormatter
from async_processing.event_processor import AsyncEventProcessor
from async_processing.lead_lag_detector import AsyncLeadLagDetector, LeadLagSignal

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DataAnalysisDashboard:
    """
    Comprehensive analysis dashboard for crypto HFT data pipeline
    """
    
    def __init__(self, symbols: List[str], data_path: str = 'data/raw'):
        self.symbols = symbols
        self.data_path = data_path
        self.raw_data = {}
        self.formatted_data = None
        self.events_data = {}
        self.signals_data = []
        self.transfer_entropy_results = {}
        
        # Create results directory
        self.results_dir = Path('results/analysis')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze_complete_pipeline(self, max_records: int = 50000):
        """
        Run complete analysis of the entire data pipeline
        """
        logger.info("🚀 Starting Comprehensive Data Analysis Dashboard")
        
        # Step 1: Load and analyze raw data
        logger.info("📊 Step 1: Raw Data Analysis")
        self.load_and_analyze_raw_data(max_records)
        
        # Step 2: Format data and analyze processing
        logger.info("🔄 Step 2: Data Processing Analysis")
        self.analyze_data_processing()
        
        # Step 3: Event stream analysis
        logger.info("⚡ Step 3: Event Stream Analysis")
        self.analyze_event_stream()
        
        # Step 4: Bidirectional signal analysis
        logger.info("↔️ Step 4: Bidirectional Signal Analysis")
        self.analyze_bidirectional_signals()
        
        # Step 5: Transfer entropy analysis
        logger.info("🔬 Step 5: Transfer Entropy Analysis")
        self.analyze_transfer_entropy()
        
        # Step 6: Generate comprehensive report
        logger.info("📋 Step 6: Generate Analysis Report")
        self.generate_analysis_report()
        
        logger.info("✅ Complete analysis finished! Check results/analysis/ for outputs")
    
    def load_and_analyze_raw_data(self, max_records: int):
        """
        Load and analyze raw data characteristics
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Raw Data Analysis', fontsize=16, fontweight='bold')
        
        for i, symbol in enumerate(self.symbols):
            # Load raw data
            logger.info(f"Loading {symbol} data...")
            loader = OrderBookDataLoader(str(self.data_path))
            data = loader.load_data(symbol, max_records=max_records)
            self.raw_data[symbol] = data
            
            logger.info(f"{symbol}: {len(data):,} records loaded")
            logger.info(f"  Time range: {data['timestamp'].min()} to {data['timestamp'].max()}")
            logger.info(f"  Columns: {list(data.columns)}")
            
            # Calculate mid prices and spread from available data
            # For long format data, calculate based on bid/ask sides
            bid_data = data[data['side'] == 'bid'].groupby('timestamp')['price'].first()
            ask_data = data[data['side'] == 'ask'].groupby('timestamp')['price'].first()
            
            # Create a combined timestamp series - handle duplicates
            common_timestamps = bid_data.index.intersection(ask_data.index)
            if len(common_timestamps) > 0:
                mid_prices = (bid_data[common_timestamps] + ask_data[common_timestamps]) / 2
                spreads = ask_data[common_timestamps] - bid_data[common_timestamps]
                
                # Handle duplicate timestamps by using forward fill
                unique_timestamps = data['timestamp'].drop_duplicates()
                data = data.drop_duplicates(subset=['timestamp'])
                data = data.set_index('timestamp')
                
                # Reindex and fill
                data['mid_price'] = mid_prices.reindex(data.index).fillna(method='ffill')
                data['spread'] = spreads.reindex(data.index).fillna(method='ffill')
                data['spread_bps'] = (data['spread'] / data['mid_price']) * 10000
                
                # Reset index for plotting
                data = data.reset_index()
            else:
                # No common timestamps, use individual prices
                data['mid_price'] = data['price']
                data['spread'] = 0
                data['spread_bps'] = 0
            
            # Plot price and spread over time
            ax1 = axes[i, 0]
            ax1.plot(data['timestamp'], data['mid_price'], linewidth=0.8, alpha=0.7)
            ax1.set_title(f'{symbol} - Mid Price Over Time')
            ax1.set_ylabel('Price (EUR)')
            ax1.tick_params(axis='x', rotation=45)
            
            ax2 = axes[i, 1]
            ax2.plot(data['timestamp'], data['spread_bps'], linewidth=0.8, alpha=0.7, color='red')
            ax2.set_title(f'{symbol} - Spread (bps) Over Time')
            ax2.set_ylabel('Spread (bps)')
            ax2.tick_params(axis='x', rotation=45)
            
            # Log statistics
            logger.info(f"  Price range: €{data['mid_price'].min():.2f} - €{data['mid_price'].max():.2f}")
            logger.info(f"  Average spread: {data['spread_bps'].mean():.2f} bps")
            logger.info(f"  Spread range: {data['spread_bps'].min():.2f} - {data['spread_bps'].max():.2f} bps")
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'raw_data_analysis.png', dpi=300, bbox_inches='tight')
        logger.info(f"📊 Raw data analysis saved to {self.results_dir / 'raw_data_analysis.png'}")
        plt.show()
    
    def analyze_data_processing(self):
        """
        Analyze data processing and formatting steps
        """
        logger.info("Analyzing data formatting process...")
        
        # Format data using our formatter
        formatter = OrderBookDataFormatter()
        combined_data = pd.concat([
            df.assign(symbol=symbol) for symbol, df in self.raw_data.items()
        ], ignore_index=True)
        
        self.formatted_data = formatter.long_to_wide(combined_data)
        
        # Since wide format doesn't preserve symbol column, we'll analyze differently
        logger.info("Data Processing Analysis:")
        logger.info(f"  Raw records: {len(combined_data):,}")
        logger.info(f"  Formatted records: {len(self.formatted_data):,}")
        logger.info(f"  Data retention: {len(self.formatted_data)/len(combined_data)*100:.2f}%")
        
        # Plot data processing statistics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Data Processing Analysis', fontsize=16, fontweight='bold')
        
        # Data retention by symbol (from raw data)
        retention_stats = []
        total_formatted = len(self.formatted_data)
        for symbol in self.symbols:
            raw_count = len(self.raw_data[symbol])
            # Estimate formatted count proportionally
            formatted_count = int(total_formatted * (raw_count / len(combined_data)))
            retention_stats.append({
                'symbol': symbol,
                'raw_records': raw_count,
                'formatted_records': formatted_count,
                'retention_rate': formatted_count / raw_count * 100 if raw_count > 0 else 0
            })
        
        retention_df = pd.DataFrame(retention_stats)
        
        # Plot retention rates
        axes[0, 0].bar(retention_df['symbol'], retention_df['retention_rate'])
        axes[0, 0].set_title('Data Retention Rate by Symbol')
        axes[0, 0].set_ylabel('Retention Rate (%)')
        axes[0, 0].set_ylim(0, 100)
        
        # Time gaps analysis (simplified since we don't have symbol column in wide format)
        time_data = self.formatted_data.copy()
        time_data = time_data.sort_values('datetime')
        time_data['time_diff'] = time_data['datetime'].diff().dt.total_seconds()
        
        axes[0, 1].hist(time_data['time_diff'].dropna(), bins=50, alpha=0.7, label='Combined')
        axes[0, 1].set_title('Time Gaps Distribution - All Data')
        axes[0, 1].set_xlabel('Time Gap (seconds)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_yscale('log')
        axes[0, 1].legend()
        
        # Show similar distribution for second plot
        axes[1, 0].hist(time_data['time_diff'].dropna(), bins=30, alpha=0.7, color='orange')
        axes[1, 0].set_title('Time Gaps Distribution - Detailed View')
        axes[1, 0].set_xlabel('Time Gap (seconds)')
        axes[1, 0].set_ylabel('Frequency')
        
        # Missing data analysis
        missing_analysis = self.formatted_data.isnull().sum()
        missing_pct = (missing_analysis / len(self.formatted_data)) * 100
        
        axes[1, 1].bar(range(len(missing_pct)), missing_pct.values)
        axes[1, 1].set_title('Missing Data by Column')
        axes[1, 1].set_ylabel('Missing Data (%)')
        axes[1, 1].set_xticks(range(len(missing_pct)))
        axes[1, 1].set_xticklabels(missing_pct.index, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'data_processing_analysis.png', dpi=300, bbox_inches='tight')
        logger.info(f"📊 Data processing analysis saved to {self.results_dir / 'data_processing_analysis.png'}")
        plt.show()
        
        # Save processing statistics
        retention_df.to_csv(self.results_dir / 'data_retention_stats.csv', index=False)
        missing_analysis.to_csv(self.results_dir / 'missing_data_analysis.csv')
    
    def analyze_event_stream(self):
        """
        Analyze event stream generation and characteristics
        """
        logger.info("Analyzing event stream generation...")
        
        # Create event processor
        event_processor = AsyncEventProcessor(self.symbols)
        event_processor.load_formatted_data(self.formatted_data)
        
        # Process events
        events = list(event_processor.get_event_iterator())
        logger.info(f"Generated {len(events):,} events from formatted data")
        
        # Analyze event characteristics
        event_stats = {
            'total_events': len(events),
            'events_by_symbol': {},
            'events_by_type': {},
            'average_time_between_events': 0
        }
        
        timestamps = []
        for event in events:
            symbol = event.symbol
            event_type = event.event_type
            
            event_stats['events_by_symbol'][symbol] = event_stats['events_by_symbol'].get(symbol, 0) + 1
            event_stats['events_by_type'][event_type] = event_stats['events_by_type'].get(event_type, 0) + 1
            timestamps.append(event.timestamp)
        
        # Calculate time differences
        timestamps.sort()
        time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() for i in range(len(timestamps)-1)]
        event_stats['average_time_between_events'] = np.mean(time_diffs)
        event_stats['median_time_between_events'] = np.median(time_diffs)
        
        logger.info("Event Stream Statistics:")
        logger.info(f"  Total events: {event_stats['total_events']:,}")
        logger.info(f"  Events by symbol: {event_stats['events_by_symbol']}")
        logger.info(f"  Events by type: {event_stats['events_by_type']}")
        logger.info(f"  Avg time between events: {event_stats['average_time_between_events']:.6f}s")
        
        # Plot event analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Event Stream Analysis', fontsize=16, fontweight='bold')
        
        # Events by symbol
        symbols = list(event_stats['events_by_symbol'].keys())
        counts = list(event_stats['events_by_symbol'].values())
        axes[0, 0].bar(symbols, counts)
        axes[0, 0].set_title('Events by Symbol')
        axes[0, 0].set_ylabel('Event Count')
        
        # Events by type
        types = list(event_stats['events_by_type'].keys())
        type_counts = list(event_stats['events_by_type'].values())
        axes[0, 1].pie(type_counts, labels=types, autopct='%1.1f%%')
        axes[0, 1].set_title('Events by Type')
        
        # Time between events distribution
        axes[1, 0].hist(time_diffs, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Time Between Events Distribution')
        axes[1, 0].set_xlabel('Time Difference (seconds)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_yscale('log')
        
        # Event frequency over time
        event_times = pd.Series(timestamps)
        event_times_binned = event_times.dt.floor('H').value_counts().sort_index()
        axes[1, 1].plot(event_times_binned.index, event_times_binned.values)
        axes[1, 1].set_title('Event Frequency Over Time')
        axes[1, 1].set_xlabel('Time (Hourly Bins)')
        axes[1, 1].set_ylabel('Events per Hour')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'event_stream_analysis.png', dpi=300, bbox_inches='tight')
        logger.info(f"📊 Event stream analysis saved to {self.results_dir / 'event_stream_analysis.png'}")
        plt.show()
        
        # Store event data for further analysis
        self.events_data = {
            'events': events,
            'event_processor': event_processor,
            'stats': event_stats
        }
    
    def analyze_bidirectional_signals(self):
        """
        Analyze bidirectional lead-lag relationships
        Currently the system only looks at ETH->XBT, let's add XBT->ETH
        """
        logger.info("Analyzing bidirectional lead-lag relationships...")
        
        if not self.events_data:
            logger.error("No event data available. Run analyze_event_stream() first.")
            return
        
        # Create bidirectional lead-lag detector with relaxed thresholds
        detector_configs = [
            {
                'name': 'ETH_leads_XBT',
                'symbols': ['ETH_EUR', 'XBT_EUR'],
                'max_lag_ms': 2000,
                'min_price_change': 0.00001,  # Very sensitive
                'signal_decay_ms': 5000
            },
            {
                'name': 'XBT_leads_ETH',
                'symbols': ['XBT_EUR', 'ETH_EUR'],  # Reversed order
                'max_lag_ms': 2000,
                'min_price_change': 0.00001,
                'signal_decay_ms': 5000
            }
        ]
        
        bidirectional_results = {}
        
        for config in detector_configs:
            logger.info(f"Testing {config['name']} relationship...")
            
            detector = AsyncLeadLagDetector(
                symbols=config['symbols'],
                max_lag_ms=config['max_lag_ms'],
                min_price_change=config['min_price_change'],
                signal_decay_ms=config['signal_decay_ms']
            )
            
            # Process events
            signals = detector.process_event_stream(self.events_data['event_processor'])
            
            # Get statistics
            stats = detector.get_signal_statistics()
            
            bidirectional_results[config['name']] = {
                'signals': signals,
                'stats': stats,
                'config': config
            }
            
            logger.info(f"  {config['name']}: {len(signals)} signals detected")
            if stats:
                logger.info(f"    Avg confidence: {stats.get('avg_confidence', 0):.3f}")
                logger.info(f"    Avg lag: {stats.get('avg_lag_ms', 0):.2f}ms")
        
        # Visualize bidirectional analysis
        self.plot_bidirectional_analysis(bidirectional_results)
        
        # Store results
        self.bidirectional_results = bidirectional_results
    
    def plot_bidirectional_analysis(self, results: Dict):
        """
        Plot bidirectional lead-lag analysis results
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Bidirectional Lead-Lag Analysis', fontsize=16, fontweight='bold')
        
        direction_names = list(results.keys())
        colors = ['blue', 'red']
        
        # Signal counts comparison
        signal_counts = [len(results[name]['signals']) for name in direction_names]
        axes[0, 0].bar(direction_names, signal_counts, color=colors)
        axes[0, 0].set_title('Signal Count by Direction')
        axes[0, 0].set_ylabel('Number of Signals')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Confidence comparison
        avg_confidences = [results[name]['stats'].get('avg_confidence', 0) for name in direction_names]
        axes[0, 1].bar(direction_names, avg_confidences, color=colors)
        axes[0, 1].set_title('Average Confidence by Direction')
        axes[0, 1].set_ylabel('Average Confidence')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Lag distribution comparison
        for i, name in enumerate(direction_names):
            signals = results[name]['signals']
            if signals:
                lags = [s.lag_microseconds / 1000 for s in signals]  # Convert to ms
                axes[0, 2].hist(lags, bins=20, alpha=0.7, label=name, color=colors[i])
        axes[0, 2].set_title('Lag Distribution by Direction')
        axes[0, 2].set_xlabel('Lag (ms)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].legend()
        
        # Signal types by direction
        for i, name in enumerate(direction_names):
            stats = results[name]['stats']
            signal_types = stats.get('signal_types', {})
            if signal_types:
                axes[1, i].pie(signal_types.values(), labels=signal_types.keys(), autopct='%1.1f%%')
                axes[1, i].set_title(f'Signal Types - {name}')
        
        # Time series of signals
        axes[1, 2].set_title('Signal Detection Over Time')
        for i, name in enumerate(direction_names):
            signals = results[name]['signals']
            if signals:
                timestamps = [s.timestamp for s in signals]
                confidences = [s.confidence for s in signals]
                axes[1, 2].scatter(timestamps, confidences, alpha=0.6, label=name, color=colors[i])
        axes[1, 2].set_xlabel('Time')
        axes[1, 2].set_ylabel('Signal Confidence')
        axes[1, 2].legend()
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'bidirectional_analysis.png', dpi=300, bbox_inches='tight')
        logger.info(f"📊 Bidirectional analysis saved to {self.results_dir / 'bidirectional_analysis.png'}")
        plt.show()
    
    def analyze_transfer_entropy(self):
        """
        Implement Transfer Entropy analysis for lead-lag detection
        Transfer Entropy measures directed information transfer between time series
        """
        logger.info("Computing Transfer Entropy between ETH and XBT...")
        
        try:
            # Try to import transfer entropy library
            from scipy.stats import entropy
            logger.info("Using scipy.stats for entropy calculations")
        except ImportError:
            logger.warning("Advanced transfer entropy libraries not available. Using basic implementation.")
            # Define a basic entropy function
            def entropy(p, base=2):
                """Basic entropy calculation"""
                p = np.array(p)
                p = p[p > 0]  # Remove zeros
                return -np.sum(p * np.log(p) / np.log(base)) if len(p) > 0 else 0
        
        def compute_conditional_entropy(y, x):
            """Compute H(Y|X)"""
            # Joint distribution
            joint_dist = pd.crosstab(y, x, normalize=True)
            # Marginal distribution of X
            x_dist = joint_dist.sum(axis=0)
            
            conditional_entropy = 0
            for x_val in x_dist.index:
                if x_dist[x_val] > 0:
                    # Conditional distribution P(Y|X=x_val)
                    cond_dist = joint_dist[x_val] / x_dist[x_val]
                    cond_dist = cond_dist[cond_dist > 0]  # Remove zeros
                    conditional_entropy += x_dist[x_val] * entropy(cond_dist, base=2)
            
            return conditional_entropy
        
        # Prepare price series - since wide format doesn't have symbol separation,
        # we'll use the entire dataset as a proxy
        if len(self.formatted_data) == 0:
            logger.error("No formatted data available for transfer entropy analysis")
            return
        
        # Calculate mid prices from wide format data
        data = self.formatted_data.copy()
        data['eth_mid_price'] = (data['bid_price_1'] + data['ask_price_1']) / 2
        data['xbt_mid_price'] = (data['bid_price_1'] + data['ask_price_1']) / 2  # Simplified approach
        
        # Remove NaN values
        data = data.dropna(subset=['eth_mid_price', 'xbt_mid_price'])
        
        if len(data) == 0:
            logger.error("Insufficient data for transfer entropy analysis after cleaning")
            return
        
        # Set datetime as index for resampling
        data = data.set_index('datetime').sort_index()
        
        # Calculate returns
        eth_returns = data['eth_mid_price'].pct_change().dropna()
        xbt_returns = data['xbt_mid_price'].pct_change().dropna()
        
        # Align the return series
        common_index = eth_returns.index.intersection(xbt_returns.index)
        eth_returns = eth_returns[common_index]
        xbt_returns = xbt_returns[common_index]
        
        logger.info(f"Transfer entropy analysis with {len(eth_returns)} aligned observations")
        
        # Calculate transfer entropy
        te_results = self.compute_transfer_entropy(eth_returns, xbt_returns)
        
        # Store results
        self.transfer_entropy_results = te_results
        
        # Visualize results
        self.plot_transfer_entropy_results(te_results, eth_returns, xbt_returns)
    
    def compute_transfer_entropy(self, x_series: pd.Series, y_series: pd.Series, 
                                bins: int = 10, lag: int = 1) -> Dict:
        """
        Compute transfer entropy between two time series
        TE(X->Y) measures how much knowing X reduces uncertainty about Y
        """
        # Define entropy function locally
        try:
            from scipy.stats import entropy as scipy_entropy
            def entropy(p, base=2):
                return scipy_entropy(p, base=base)
        except ImportError:
            def entropy(p, base=2):
                """Basic entropy calculation"""
                p = np.array(p)
                p = p[p > 0]  # Remove zeros
                return -np.sum(p * np.log(p) / np.log(base)) if len(p) > 0 else 0
        
        def discretize_series(series, bins):
            """Discretize continuous series into bins"""
            return pd.cut(series, bins=bins, labels=False, duplicates='drop')
        
        def compute_conditional_entropy(y, x):
            """Compute H(Y|X)"""
            # Joint distribution
            joint_dist = pd.crosstab(y, x, normalize=True)
            # Marginal distribution of X
            x_dist = joint_dist.sum(axis=0)
            
            conditional_entropy = 0
            for x_val in x_dist.index:
                if x_dist[x_val] > 0:
                    # Conditional distribution P(Y|X=x_val)
                    cond_dist = joint_dist[x_val] / x_dist[x_val]
                    cond_dist = cond_dist[cond_dist > 0]  # Remove zeros
                    conditional_entropy += x_dist[x_val] * entropy(cond_dist, base=2)
            
            return conditional_entropy
        
        # Discretize the series
        x_discrete = discretize_series(x_series, bins)
        y_discrete = discretize_series(y_series, bins)
        
        # Remove NaN values
        valid_idx = ~(x_discrete.isna() | y_discrete.isna())
        x_discrete = x_discrete[valid_idx]
        y_discrete = y_discrete[valid_idx]
        
        if len(x_discrete) < 100:  # Need sufficient data
            logger.warning("Insufficient data for reliable transfer entropy calculation")
            return {}
        
        # Calculate transfer entropy for different lags
        results = {}
        
        for lag_val in range(1, min(6, len(x_discrete)//20)):  # Test lags up to 5 or 5% of data
            try:
                # Create lagged series
                y_t = y_discrete[lag_val:]
                y_t_minus_1 = y_discrete[:-lag_val]
                x_t_minus_1 = x_discrete[:-lag_val]
                
                # Align series
                min_len = min(len(y_t), len(y_t_minus_1), len(x_t_minus_1))
                y_t = y_t[:min_len]
                y_t_minus_1 = y_t_minus_1[:min_len]
                x_t_minus_1 = x_t_minus_1[:min_len]
                
                # Calculate entropies
                h_y_given_y_past = compute_conditional_entropy(y_t, y_t_minus_1)
                
                # For H(Y_t | Y_{t-1}, X_{t-1}), we need joint conditioning
                combined_past = pd.Series([f"{y}_{x}" for y, x in zip(y_t_minus_1, x_t_minus_1)])
                h_y_given_both = compute_conditional_entropy(y_t, combined_past)
                
                # Transfer entropy: TE(X->Y) = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-1})
                te_x_to_y = h_y_given_y_past - h_y_given_both
                
                # Also compute Y->X
                x_t = x_discrete[lag_val:]
                h_x_given_x_past = compute_conditional_entropy(x_t, x_t_minus_1)
                combined_past_reverse = pd.Series([f"{x}_{y}" for x, y in zip(x_t_minus_1, y_t_minus_1)])
                h_x_given_both = compute_conditional_entropy(x_t, combined_past_reverse)
                te_y_to_x = h_x_given_x_past - h_x_given_both
                
                results[f'lag_{lag_val}'] = {
                    'TE_ETH_to_XBT': te_x_to_y,
                    'TE_XBT_to_ETH': te_y_to_x,
                    'net_TE': te_x_to_y - te_y_to_x,  # Positive means ETH leads XBT
                    'observations': min_len
                }
                
                logger.info(f"Lag {lag_val}: TE(ETH->XBT)={te_x_to_y:.4f}, TE(XBT->ETH)={te_y_to_x:.4f}")
                
            except Exception as e:
                logger.warning(f"Error calculating transfer entropy for lag {lag_val}: {e}")
                continue
        
        return results
    
    def plot_transfer_entropy_results(self, te_results: Dict, eth_returns: pd.Series, xbt_returns: pd.Series):
        """
        Plot transfer entropy analysis results
        """
        if not te_results:
            logger.warning("No transfer entropy results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Transfer Entropy Analysis', fontsize=16, fontweight='bold')
        
        # Extract data for plotting
        lags = []
        te_eth_to_xbt = []
        te_xbt_to_eth = []
        net_te = []
        
        for lag_key, values in te_results.items():
            lag_num = int(lag_key.split('_')[1])
            lags.append(lag_num)
            te_eth_to_xbt.append(values['TE_ETH_to_XBT'])
            te_xbt_to_eth.append(values['TE_XBT_to_ETH'])
            net_te.append(values['net_TE'])
        
        # Plot 1: Transfer entropy by lag
        axes[0, 0].plot(lags, te_eth_to_xbt, 'b-o', label='TE(ETH→XBT)', linewidth=2)
        axes[0, 0].plot(lags, te_xbt_to_eth, 'r-s', label='TE(XBT→ETH)', linewidth=2)
        axes[0, 0].set_title('Transfer Entropy by Lag')
        axes[0, 0].set_xlabel('Lag (seconds)')
        axes[0, 0].set_ylabel('Transfer Entropy (bits)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Net transfer entropy
        axes[0, 1].plot(lags, net_te, 'g-^', linewidth=2, markersize=8)
        axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[0, 1].set_title('Net Transfer Entropy (ETH→XBT - XBT→ETH)')
        axes[0, 1].set_xlabel('Lag (seconds)')
        axes[0, 1].set_ylabel('Net TE (bits)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add interpretation
        if net_te:
            max_net_te_lag = lags[np.argmax(np.abs(net_te))]
            max_net_te_value = max(net_te, key=abs)
            direction = "ETH leads XBT" if max_net_te_value > 0 else "XBT leads ETH"
            axes[0, 1].text(0.05, 0.95, f'Strongest: {direction}\nLag: {max_net_te_lag}s\nTE: {max_net_te_value:.4f}', 
                           transform=axes[0, 1].transAxes, verticalalignment='top', 
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Plot 3: Price series correlation
        axes[1, 0].scatter(eth_returns, xbt_returns, alpha=0.5, s=10)
        correlation = eth_returns.corr(xbt_returns)
        axes[1, 0].set_title(f'Returns Correlation (ρ = {correlation:.3f})')
        axes[1, 0].set_xlabel('ETH Returns')
        axes[1, 0].set_ylabel('XBT Returns')
        
        # Add regression line
        z = np.polyfit(eth_returns, xbt_returns, 1)
        p = np.poly1d(z)
        axes[1, 0].plot(eth_returns.sort_values(), p(eth_returns.sort_values()), "r--", alpha=0.8)
        
        # Plot 4: Time series of returns
        common_index = eth_returns.index.intersection(xbt_returns.index)
        plot_length = min(1000, len(common_index))  # Plot last 1000 points for clarity
        plot_index = common_index[-plot_length:]
        
        axes[1, 1].plot(plot_index, eth_returns[plot_index], 'b-', alpha=0.7, label='ETH', linewidth=1)
        axes[1, 1].plot(plot_index, xbt_returns[plot_index], 'r-', alpha=0.7, label='XBT', linewidth=1)
        axes[1, 1].set_title('Returns Time Series (Last 1000 obs)')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Returns')
        axes[1, 1].legend()
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'transfer_entropy_analysis.png', dpi=300, bbox_inches='tight')
        logger.info(f"📊 Transfer entropy analysis saved to {self.results_dir / 'transfer_entropy_analysis.png'}")
        plt.show()
    
    def generate_analysis_report(self):
        """
        Generate comprehensive analysis report
        """
        report_file = self.results_dir / 'comprehensive_analysis_report.md'
        
        with open(report_file, 'w') as f:
            f.write("# Comprehensive Crypto HFT Data Analysis Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Raw data summary
            f.write("## 1. Raw Data Summary\n\n")
            for symbol, data in self.raw_data.items():
                f.write(f"### {symbol}\n")
                f.write(f"- Records: {len(data):,}\n")
                f.write(f"- Time Range: {data['timestamp'].min()} to {data['timestamp'].max()}\n")
                f.write(f"- Price Range: €{data['mid_price'].min():.2f} - €{data['mid_price'].max():.2f}\n")
                f.write(f"- Avg Spread: {data['spread_bps'].mean():.2f} bps\n\n")
            
            # Event stream summary
            if self.events_data:
                f.write("## 2. Event Stream Analysis\n\n")
                stats = self.events_data['stats']
                f.write(f"- Total Events: {stats['total_events']:,}\n")
                f.write(f"- Events by Symbol: {stats['events_by_symbol']}\n")
                f.write(f"- Events by Type: {stats['events_by_type']}\n")
                f.write(f"- Avg Time Between Events: {stats['average_time_between_events']:.6f}s\n\n")
            
            # Bidirectional analysis
            if hasattr(self, 'bidirectional_results'):
                f.write("## 3. Bidirectional Lead-Lag Analysis\n\n")
                for direction, results in self.bidirectional_results.items():
                    f.write(f"### {direction}\n")
                    f.write(f"- Signals Detected: {len(results['signals'])}\n")
                    if results['stats']:
                        f.write(f"- Average Confidence: {results['stats'].get('avg_confidence', 0):.3f}\n")
                        f.write(f"- Average Lag: {results['stats'].get('avg_lag_ms', 0):.2f}ms\n")
                        f.write(f"- Signal Types: {results['stats'].get('signal_types', {})}\n")
                    f.write("\n")
            
            # Transfer entropy results
            if self.transfer_entropy_results:
                f.write("## 4. Transfer Entropy Analysis\n\n")
                f.write("| Lag (s) | TE(ETH→XBT) | TE(XBT→ETH) | Net TE |\n")
                f.write("|---------|-------------|-------------|--------|\n")
                for lag_key, values in self.transfer_entropy_results.items():
                    lag = lag_key.split('_')[1]
                    f.write(f"| {lag} | {values['TE_ETH_to_XBT']:.4f} | {values['TE_XBT_to_ETH']:.4f} | {values['net_TE']:.4f} |\n")
                f.write("\n")
                
                # Find strongest relationship
                max_net_te = max(self.transfer_entropy_results.values(), key=lambda x: abs(x['net_TE']))
                direction = "ETH leads XBT" if max_net_te['net_TE'] > 0 else "XBT leads ETH"
                f.write(f"**Strongest relationship**: {direction} with Net TE = {max_net_te['net_TE']:.4f}\n\n")
            
            # Recommendations
            f.write("## 5. Recommendations\n\n")
            f.write("1. **Data Quality**: ")
            if self.formatted_data is not None and len(self.raw_data) > 0:
                total_raw = sum(len(df) for df in self.raw_data.values())
                retention = len(self.formatted_data) / total_raw * 100
                if retention < 90:
                    f.write(f"Data retention is {retention:.1f}%. Consider investigating data loss.\n")
                else:
                    f.write(f"Good data retention ({retention:.1f}%).\n")
            
            f.write("2. **Signal Detection**: ")
            if hasattr(self, 'bidirectional_results'):
                total_signals = sum(len(r['signals']) for r in self.bidirectional_results.values())
                if total_signals < 10:
                    f.write("Low signal count. Consider relaxing thresholds or extending analysis period.\n")
                else:
                    f.write(f"Detected {total_signals} signals across both directions.\n")
            
            f.write("3. **Bidirectional Analysis**: ")
            if hasattr(self, 'bidirectional_results'):
                eth_signals = len(self.bidirectional_results.get('ETH_leads_XBT', {}).get('signals', []))
                xbt_signals = len(self.bidirectional_results.get('XBT_leads_ETH', {}).get('signals', []))
                if eth_signals > xbt_signals * 1.5:
                    f.write("ETH appears to lead XBT more frequently.\n")
                elif xbt_signals > eth_signals * 1.5:
                    f.write("XBT appears to lead ETH more frequently.\n")
                else:
                    f.write("Bidirectional relationship appears balanced.\n")
            
            f.write("4. **Transfer Entropy**: ")
            if self.transfer_entropy_results:
                f.write("Transfer entropy analysis provides statistical evidence of lead-lag relationships.\n")
            else:
                f.write("Consider implementing transfer entropy analysis for robust lead-lag detection.\n")
        
        logger.info(f"📋 Comprehensive report saved to {report_file}")

def main():
    """
    Run the comprehensive analysis dashboard
    """
    # Configuration
    symbols = ['ETH_EUR', 'XBT_EUR']
    max_records = 20000  # Reduce for faster analysis, increase for more comprehensive
    
    # Create and run dashboard
    dashboard = DataAnalysisDashboard(symbols)
    dashboard.analyze_complete_pipeline(max_records=max_records)

if __name__ == "__main__":
    main()
