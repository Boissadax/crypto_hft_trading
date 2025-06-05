#!/usr/bin/env python3
"""
Text-based Analysis Dashboard
===========================

Focuses on text output to avoid display issues
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# Import our components
from data_processing.data_loader import OrderBookDataLoader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def text_analysis():
    """Run text-based analysis"""
    
    print("=" * 60)
    print("🚀 CRYPTO HFT DATA ANALYSIS DASHBOARD")
    print("=" * 60)
    
    # Configuration
    symbols = ['ETH_EUR', 'XBT_EUR']
    max_records = 5000
    
    # Create results directory
    results_dir = Path('results/analysis')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load data
    print("\n📊 STEP 1: LOADING RAW DATA")
    print("-" * 40)
    
    loader = OrderBookDataLoader('data/raw')
    raw_data = {}
    
    for symbol in symbols:
        print(f"Loading {symbol}...")
        try:
            data = loader.load_data(symbol, max_records=max_records)
            raw_data[symbol] = data
            
            print(f"✅ {symbol}: {len(data):,} records loaded")
            print(f"   Time range: {data['datetime'].min()} to {data['datetime'].max()}")
            print(f"   Price range: €{data['price'].min():.2f} - €{data['price'].max():.2f}")
            print(f"   Levels: {sorted(data['level'].unique())}")
            print(f"   Sides: {sorted(data['side'].unique())}")
            print()
            
        except Exception as e:
            print(f"❌ Failed to load {symbol}: {e}")
    
    # Step 2: Analyze data characteristics
    print("\n📈 STEP 2: DATA CHARACTERISTICS ANALYSIS")
    print("-" * 40)
    
    analysis_results = {}
    
    for symbol, data in raw_data.items():
        print(f"\n🔍 Analyzing {symbol}:")
        
        # Basic statistics
        bid_data = data[data['side'] == 'bid']
        ask_data = data[data['side'] == 'ask']
        
        print(f"   📊 Record Distribution:")
        print(f"      - Bid records: {len(bid_data):,} ({len(bid_data)/len(data)*100:.1f}%)")
        print(f"      - Ask records: {len(ask_data):,} ({len(ask_data)/len(data)*100:.1f}%)")
        
        # Price analysis
        print(f"   💰 Price Analysis:")
        print(f"      - Overall range: €{data['price'].min():.2f} - €{data['price'].max():.2f}")
        print(f"      - Average price: €{data['price'].mean():.2f}")
        print(f"      - Price std: €{data['price'].std():.2f}")
        
        if len(bid_data) > 0 and len(ask_data) > 0:
            # Calculate spreads where we have both bid and ask at same timestamp
            grouped_bids = bid_data.groupby('timestamp')['price'].max()  # Best bid
            grouped_asks = ask_data.groupby('timestamp')['price'].min()  # Best ask
            
            common_times = grouped_bids.index.intersection(grouped_asks.index)
            if len(common_times) > 0:
                spreads = grouped_asks[common_times] - grouped_bids[common_times]
                mid_prices = (grouped_asks[common_times] + grouped_bids[common_times]) / 2
                spread_bps = (spreads / mid_prices) * 10000
                
                print(f"   📏 Spread Analysis ({len(common_times)} observations):")
                print(f"      - Average spread: {spreads.mean():.4f} EUR ({spread_bps.mean():.2f} bps)")
                print(f"      - Spread range: {spreads.min():.4f} - {spreads.max():.4f} EUR")
                print(f"      - Spread std: {spread_bps.std():.2f} bps")
                
                analysis_results[symbol] = {
                    'records': len(data),
                    'bid_records': len(bid_data),
                    'ask_records': len(ask_data),
                    'price_range': (data['price'].min(), data['price'].max()),
                    'avg_spread_bps': spread_bps.mean(),
                    'spread_observations': len(common_times)
                }
        
        # Volume analysis
        print(f"   📦 Volume Analysis:")
        print(f"      - Volume range: {data['volume'].min():.2f} - {data['volume'].max():.2f}")
        print(f"      - Average volume: {data['volume'].mean():.2f}")
        print(f"      - Total volume: {data['volume'].sum():.2f}")
        
        # Level analysis
        level_dist = data['level'].value_counts().sort_index()
        print(f"   🎯 Level Distribution:")
        for level in sorted(level_dist.index)[:5]:  # Show first 5 levels
            count = level_dist[level]
            print(f"      - Level {level}: {count:,} records ({count/len(data)*100:.1f}%)")
    
    # Step 3: Bidirectional analysis preparation
    print("\n↔️ STEP 3: BIDIRECTIONAL RELATIONSHIP PREPARATION")
    print("-" * 40)
    
    if len(raw_data) >= 2:
        symbols_list = list(raw_data.keys())
        symbol1, symbol2 = symbols_list[0], symbols_list[1]
        
        print(f"🔄 Analyzing relationship between {symbol1} and {symbol2}:")
        
        data1, data2 = raw_data[symbol1], raw_data[symbol2]
        
        # Find overlapping time periods
        time1_range = (data1['timestamp'].min(), data1['timestamp'].max())
        time2_range = (data2['timestamp'].min(), data2['timestamp'].max())
        
        overlap_start = max(time1_range[0], time2_range[0])
        overlap_end = min(time1_range[1], time2_range[1])
        overlap_duration = overlap_end - overlap_start
        
        print(f"   ⏱️ Time Overlap Analysis:")
        print(f"      - {symbol1} time range: {pd.Timestamp(time1_range[0], unit='s')} to {pd.Timestamp(time1_range[1], unit='s')}")
        print(f"      - {symbol2} time range: {pd.Timestamp(time2_range[0], unit='s')} to {pd.Timestamp(time2_range[1], unit='s')}")
        print(f"      - Overlap period: {overlap_duration:.0f} seconds ({overlap_duration/3600:.1f} hours)")
        
        # Calculate data density in overlap period
        overlap_data1 = data1[(data1['timestamp'] >= overlap_start) & (data1['timestamp'] <= overlap_end)]
        overlap_data2 = data2[(data2['timestamp'] >= overlap_start) & (data2['timestamp'] <= overlap_end)]
        
        print(f"   📊 Data Density in Overlap:")
        print(f"      - {symbol1}: {len(overlap_data1):,} records ({len(overlap_data1)/overlap_duration:.2f} records/sec)")
        print(f"      - {symbol2}: {len(overlap_data2):,} records ({len(overlap_data2)/overlap_duration:.2f} records/sec)")
        
        # Basic correlation analysis (simplified)
        if len(overlap_data1) > 0 and len(overlap_data2) > 0:
            # Get price series for both symbols
            prices1 = overlap_data1.groupby('timestamp')['price'].mean()
            prices2 = overlap_data2.groupby('timestamp')['price'].mean()
            
            # Find common timestamps
            common_times = prices1.index.intersection(prices2.index)
            
            if len(common_times) > 10:
                corr = prices1[common_times].corr(prices2[common_times])
                print(f"   🔗 Basic Price Correlation: {corr:.4f}")
                print(f"      - Based on {len(common_times)} common timestamps")
            else:
                print(f"   ⚠️ Insufficient common timestamps ({len(common_times)}) for correlation analysis")
    
    # Step 4: Generate comprehensive report
    print("\n📋 STEP 4: GENERATING COMPREHENSIVE REPORT")
    print("-" * 40)
    
    report_file = results_dir / 'comprehensive_analysis_report.md'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Comprehensive Crypto HFT Data Analysis Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Executive Summary\n\n")
        total_records = sum(len(data) for data in raw_data.values())
        f.write(f"- **Total Records Analyzed**: {total_records:,}\n")
        f.write(f"- **Symbols Analyzed**: {', '.join(raw_data.keys())}\n")
        f.write(f"- **Analysis Period**: {max_records:,} records per symbol\n\n")
        
        f.write("## Detailed Analysis by Symbol\n\n")
        
        for symbol, data in raw_data.items():
            f.write(f"### {symbol}\n\n")
            f.write(f"- **Records**: {len(data):,}\n")
            f.write(f"- **Time Range**: {data['datetime'].min()} to {data['datetime'].max()}\n")
            f.write(f"- **Price Range**: €{data['price'].min():.2f} - €{data['price'].max():.2f}\n")
            f.write(f"- **Average Price**: €{data['price'].mean():.2f}\n")
            f.write(f"- **Volume Range**: {data['volume'].min():.2f} - {data['volume'].max():.2f}\n")
            f.write(f"- **Total Volume**: {data['volume'].sum():.2f}\n")
            
            bid_count = len(data[data['side'] == 'bid'])
            ask_count = len(data[data['side'] == 'ask'])
            f.write(f"- **Bid Records**: {bid_count:,} ({bid_count/len(data)*100:.1f}%)\n")
            f.write(f"- **Ask Records**: {ask_count:,} ({ask_count/len(data)*100:.1f}%)\n")
            
            if symbol in analysis_results:
                result = analysis_results[symbol]
                f.write(f"- **Average Spread**: {result['avg_spread_bps']:.2f} bps\n")
                f.write(f"- **Spread Observations**: {result['spread_observations']:,}\n")
            
            f.write("\n")
        
        f.write("## Recommendations\n\n")
        f.write("1. **Data Quality**: ")
        
        # Assess data quality
        min_records = min(len(data) for data in raw_data.values())
        if min_records < 1000:
            f.write("Consider increasing the sample size for more robust analysis.\n")
        else:
            f.write("Good data coverage for analysis.\n")
        
        f.write("2. **Signal Detection**: For bidirectional lead-lag analysis, consider:\n")
        f.write("   - Implementing event-based processing\n")
        f.write("   - Using adaptive thresholds based on volatility\n")
        f.write("   - Analyzing both price and volume signals\n")
        
        f.write("3. **Next Steps**:\n")
        f.write("   - Implement transfer entropy analysis\n")
        f.write("   - Add real-time signal detection\n")
        f.write("   - Optimize thresholds based on historical performance\n")
    
    print(f"✅ Comprehensive report saved to: {report_file}")
    
    # Step 5: Summary
    print("\n🎉 ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"📁 Results saved in: {results_dir}")
    print(f"📊 Total records analyzed: {sum(len(data) for data in raw_data.values()):,}")
    print(f"📈 Symbols analyzed: {', '.join(raw_data.keys())}")
    
    if analysis_results:
        print("\n💡 KEY INSIGHTS:")
        for symbol, result in analysis_results.items():
            print(f"   • {symbol}: {result['spread_observations']:,} price observations, avg spread {result['avg_spread_bps']:.2f} bps")
    
    print("\n🔧 RECOMMENDED NEXT STEPS:")
    print("   1. Run bidirectional signal detection")
    print("   2. Implement transfer entropy analysis")
    print("   3. Optimize detection thresholds")
    print("   4. Add real-time monitoring")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    text_analysis()
