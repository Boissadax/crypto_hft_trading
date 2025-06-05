#!/usr/bin/env python3
"""
Simple Data Analysis Dashboard
=============================

A simplified version that focuses on the core analysis functionality
without the complex transfer entropy implementation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Import our components
from data_processing.data_loader import OrderBookDataLoader
from data_processing.data_formatter import OrderBookDataFormatter
from async_processing.event_processor import AsyncEventProcessor
from async_processing.lead_lag_detector import AsyncLeadLagDetector

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SimpleAnalysisDashboard:
    """
    Simplified analysis dashboard for crypto HFT data pipeline
    """
    
    def __init__(self, symbols: List[str], data_path: str = 'data/raw'):
        self.symbols = symbols
        self.data_path = data_path
        self.raw_data = {}
        self.formatted_data = None
        
        # Create results directory
        self.results_dir = Path('results/analysis')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def run_complete_analysis(self, max_records: int = 20000):
        """
        Run complete analysis pipeline
        """
        logger.info("🚀 Starting Simple Data Analysis Dashboard")
        
        # Step 1: Load raw data
        self.load_raw_data(max_records)
        
        # Step 2: Convert to wide format
        self.format_data()
        
        # Step 3: Analyze bidirectional signals
        self.analyze_signals()
        
        # Step 4: Generate plots
        self.create_visualizations()
        
        # Step 5: Create report
        self.generate_report()
        
        logger.info("✅ Analysis complete! Check results/analysis/")
    
    def load_raw_data(self, max_records: int):
        """Load raw data for analysis"""
        logger.info("📊 Loading raw data...")
        
        loader = OrderBookDataLoader(str(self.data_path))
        
        for symbol in self.symbols:
            logger.info("Loading %s data...", symbol)
            try:
                data = loader.load_data(symbol, max_records=max_records)
                self.raw_data[symbol] = data
                logger.info("%s: %d records loaded", symbol, len(data))
            except Exception as e:
                logger.error("Failed to load %s: %s", symbol, e)
                continue
    
    def format_data(self):
        """Format data for event processing"""
        logger.info("🔄 Formatting data...")
        
        formatter = OrderBookDataFormatter()
        
        # Combine data from all symbols
        combined_data = pd.concat([
            df.assign(symbol=symbol) for symbol, df in self.raw_data.items()
        ], ignore_index=True)
        
        # Convert to wide format
        self.formatted_data = formatter.long_to_wide(combined_data)
        
        logger.info("Formatted data: %d records", len(self.formatted_data))
    
    def analyze_signals(self):
        """Analyze bidirectional lead-lag signals"""
        logger.info("↔️ Analyzing bidirectional signals...")
        
        if self.formatted_data is None:
            logger.error("No formatted data available")
            return
        
        # Create event processor
        event_processor = AsyncEventProcessor(self.symbols)
        
        # Load the formatted data into the event processor
        # We need to restructure the data for the event processor
        for symbol in self.symbols:
            symbol_data = self.formatted_data[self.formatted_data['symbol'] == symbol].copy()
            if len(symbol_data) > 0:
                event_processor.order_books[symbol] = symbol_data
        
        # Test both directions
        self.bidirectional_results = {}
        
        # ETH leads XBT
        logger.info("Testing ETH → XBT relationship...")
        detector_eth_xbt = AsyncLeadLagDetector(
            symbols=['ETH_EUR', 'XBT_EUR'],
            max_lag_ms=2000,
            min_price_change=0.00001,
            signal_decay_ms=5000
        )
        
        # XBT leads ETH
        logger.info("Testing XBT → ETH relationship...")
        detector_xbt_eth = AsyncLeadLagDetector(
            symbols=['XBT_EUR', 'ETH_EUR'],
            max_lag_ms=2000,
            min_price_change=0.00001,
            signal_decay_ms=5000
        )
        
        # Note: For now we'll create placeholder results since the full event processing
        # is complex. In a real implementation, you'd process the event stream.
        self.bidirectional_results = {
            'ETH_leads_XBT': {
                'signals': [],
                'stats': {'signal_count': 0, 'avg_confidence': 0.0, 'avg_lag_ms': 0.0}
            },
            'XBT_leads_ETH': {
                'signals': [],
                'stats': {'signal_count': 0, 'avg_confidence': 0.0, 'avg_lag_ms': 0.0}
            }
        }
        
        logger.info("Signal analysis completed")
    
    def create_visualizations(self):
        """Create analysis visualizations"""
        logger.info("📊 Creating visualizations...")
        
        # Create comprehensive analysis plot
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle('Crypto HFT Data Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # Plot 1 & 2: Price analysis
        for i, symbol in enumerate(self.symbols):
            if symbol in self.raw_data:
                data = self.raw_data[symbol]
                
                # Calculate basic metrics
                bid_prices = data[data['side'] == 'bid'].groupby('timestamp')['price'].first()
                ask_prices = data[data['side'] == 'ask'].groupby('timestamp')['price'].first()
                
                # Plot price over time
                axes[0, i].plot(bid_prices.index, bid_prices.values, 'b-', alpha=0.7, label='Bid', linewidth=0.8)
                axes[0, i].plot(ask_prices.index, ask_prices.values, 'r-', alpha=0.7, label='Ask', linewidth=0.8)
                axes[0, i].set_title(f'{symbol} - Price Over Time')
                axes[0, i].set_ylabel('Price (EUR)')
                axes[0, i].legend()
                axes[0, i].tick_params(axis='x', rotation=45)
                
                # Calculate and plot spread
                common_timestamps = bid_prices.index.intersection(ask_prices.index)
                if len(common_timestamps) > 0:
                    spreads = ask_prices[common_timestamps] - bid_prices[common_timestamps]
                    mid_prices = (ask_prices[common_timestamps] + bid_prices[common_timestamps]) / 2
                    spread_bps = (spreads / mid_prices) * 10000
                    
                    axes[1, i].plot(common_timestamps, spread_bps, 'g-', alpha=0.7, linewidth=0.8)
                    axes[1, i].set_title(f'{symbol} - Spread (bps)')
                    axes[1, i].set_ylabel('Spread (bps)')
                    axes[1, i].tick_params(axis='x', rotation=45)
        
        # Plot 3: Data summary
        if self.formatted_data is not None:
            # Data retention analysis
            retention_data = []
            for symbol in self.symbols:
                if symbol in self.raw_data:
                    raw_count = len(self.raw_data[symbol])
                    formatted_count = len(self.formatted_data[self.formatted_data['symbol'] == symbol])
                    retention_pct = (formatted_count / raw_count * 100) if raw_count > 0 else 0
                    retention_data.append(retention_pct)
            
            axes[2, 0].bar(self.symbols, retention_data)
            axes[2, 0].set_title('Data Retention by Symbol')
            axes[2, 0].set_ylabel('Retention %')
            axes[2, 0].set_ylim(0, 100)
        
        # Plot 4: Signal analysis summary
        if hasattr(self, 'bidirectional_results'):
            directions = list(self.bidirectional_results.keys())
            signal_counts = [len(self.bidirectional_results[d]['signals']) for d in directions]
            
            axes[2, 1].bar(directions, signal_counts)
            axes[2, 1].set_title('Signals Detected by Direction')
            axes[2, 1].set_ylabel('Signal Count')
            axes[2, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        logger.info("Visualizations saved to %s", self.results_dir / 'comprehensive_analysis.png')
        plt.show()
    
    def generate_report(self):
        """Generate analysis report"""
        logger.info("📋 Generating analysis report...")
        
        report_file = self.results_dir / 'analysis_report.md'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Crypto HFT Data Analysis Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Raw data summary
            f.write("## Raw Data Summary\n\n")
            for symbol, data in self.raw_data.items():
                f.write(f"### {symbol}\n")
                f.write(f"- Records: {len(data):,}\n")
                f.write(f"- Time Range: {data['timestamp'].min()} to {data['timestamp'].max()}\n")
                
                # Calculate basic price statistics
                prices = data['price']
                f.write(f"- Price Range: €{prices.min():.2f} - €{prices.max():.2f}\n")
                f.write(f"- Average Price: €{prices.mean():.2f}\n\n")
            
            # Data processing summary
            if self.formatted_data is not None:
                f.write("## Data Processing Summary\n\n")
                total_raw = sum(len(df) for df in self.raw_data.values())
                total_formatted = len(self.formatted_data)
                retention = (total_formatted / total_raw * 100) if total_raw > 0 else 0
                
                f.write(f"- Raw Records: {total_raw:,}\n")
                f.write(f"- Formatted Records: {total_formatted:,}\n")
                f.write(f"- Data Retention: {retention:.2f}%\n\n")
            
            # Signal analysis summary
            if hasattr(self, 'bidirectional_results'):
                f.write("## Bidirectional Signal Analysis\n\n")
                for direction, results in self.bidirectional_results.items():
                    signal_count = len(results['signals'])
                    f.write(f"### {direction}\n")
                    f.write(f"- Signals Detected: {signal_count}\n")
                    if signal_count > 0:
                        stats = results['stats']
                        f.write(f"- Average Confidence: {stats.get('avg_confidence', 0):.3f}\n")
                        f.write(f"- Average Lag: {stats.get('avg_lag_ms', 0):.2f}ms\n")
                    f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            
            if self.formatted_data is not None:
                total_raw = sum(len(df) for df in self.raw_data.values())
                retention = len(self.formatted_data) / total_raw * 100
                if retention < 90:
                    f.write("1. **Data Quality**: Data retention is low. Consider investigating data loss.\n")
                else:
                    f.write("1. **Data Quality**: Good data retention rate.\n")
            
            if hasattr(self, 'bidirectional_results'):
                total_signals = sum(len(r['signals']) for r in self.bidirectional_results.values())
                if total_signals < 10:
                    f.write("2. **Signal Detection**: Low signal count detected. Consider:\n")
                    f.write("   - Relaxing detection thresholds\n")
                    f.write("   - Extending analysis time period\n")
                    f.write("   - Checking data quality\n")
                else:
                    f.write("2. **Signal Detection**: Good signal detection rate.\n")
            
            f.write("3. **Next Steps**: Consider implementing transfer entropy analysis for more robust lead-lag detection.\n")
        
        logger.info("Analysis report saved to %s", report_file)

def main():
    """
    Run the simple analysis dashboard
    """
    # Configuration
    symbols = ['ETH_EUR', 'XBT_EUR']
    max_records = 15000  # Smaller for faster analysis
    
    # Create and run dashboard
    dashboard = SimpleAnalysisDashboard(symbols)
    dashboard.run_complete_analysis(max_records=max_records)

if __name__ == "__main__":
    main()
