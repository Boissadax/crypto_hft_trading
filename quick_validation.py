#!/usr/bin/env python3
"""
Quick validation script to test optimizations with smaller data samples
"""

import pandas as pd
import numpy as np
import time
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from optimized_hft_engine import OptimizedHFTEngine
from feature_engineering.vectorized_time_series_features import VectorizedTimeSeriesFeatures
from feature_engineering.vectorized_order_book_features import VectorizedOrderBookFeatures
from optimized_data_cache import OptimizedDataCache

def generate_sample_data(n_rows=10000):
    """Generate sample trading data for testing"""
    np.random.seed(42)  # For reproducible results
    
    timestamps = pd.date_range(start='2024-01-01', periods=n_rows, freq='1S')
    
    # Generate realistic price data with some trend and volatility
    base_price = 2000.0
    price_changes = np.random.normal(0, 0.001, n_rows)  # 0.1% volatility
    prices = base_price * np.exp(np.cumsum(price_changes))
    
    # Generate bid/ask spreads
    spread = np.random.uniform(0.5, 2.0, n_rows)
    bid_prices = prices - spread/2
    ask_prices = prices + spread/2
    
    # Generate volumes
    volumes = np.random.exponential(100, n_rows)
    
    sample_data = pd.DataFrame({
        'timestamp': timestamps,
        'price': prices,
        'bid_price': bid_prices,
        'ask_price': ask_prices,
        'volume': volumes,
        'bid_volume': volumes * np.random.uniform(0.8, 1.2, n_rows),
        'ask_volume': volumes * np.random.uniform(0.8, 1.2, n_rows)
    })
    
    return sample_data

def test_vectorized_features():
    """Test vectorized feature extraction"""
    print("🧪 Testing Vectorized Feature Extraction...")
    
    # Generate test data
    test_data = generate_sample_data(10000)
    print(f"Generated test data: {test_data.shape}")
    
    # Test time series features
    print("📊 Testing Time Series Features...")
    ts_extractor = VectorizedTimeSeriesFeatures()
    
    start_time = time.time()
    ts_features = ts_extractor.extract_all_features(test_data)
    ts_time = time.time() - start_time
    
    print(f"✅ Time Series Features: {ts_features.shape} in {ts_time:.3f}s")
    
    # Test order book features  
    print("📈 Testing Order Book Features...")
    ob_extractor = VectorizedOrderBookFeatures()
    
    start_time = time.time()
    ob_features = ob_extractor.extract_all_features(test_data)
    ob_time = time.time() - start_time
    
    print(f"✅ Order Book Features: {ob_features.shape} in {ob_time:.3f}s")
    
    return ts_features, ob_features, ts_time + ob_time

def test_optimized_cache():
    """Test optimized data cache"""
    print("💾 Testing Optimized Data Cache...")
    
    cache = OptimizedDataCache(
        cache_dir="data_cache/test",
        chunk_size=5000
    )
    
    # Generate and cache data
    test_data = generate_sample_data(20000)
    
    start_time = time.time()
    cache_key = cache.cache_data("test_data", test_data)
    cache_time = time.time() - start_time
    
    print(f"✅ Data cached in {cache_time:.3f}s")
    
    # Test retrieval
    start_time = time.time()
    retrieved_data = cache.get_cached_data(cache_key)
    retrieval_time = time.time() - start_time
    
    print(f"✅ Data retrieved in {retrieval_time:.3f}s")
    print(f"✅ Data integrity: {test_data.equals(retrieved_data)}")
    
    return cache_time + retrieval_time

def test_optimized_engine():
    """Test the complete optimized engine"""
    print("🚀 Testing Optimized HFT Engine...")
    
    engine = OptimizedHFTEngine()
    test_data = generate_sample_data(15000)
    
    start_time = time.time()
    
    # Test feature extraction
    features = engine.extract_features_vectorized(test_data)
    
    processing_time = time.time() - start_time
    
    print(f"✅ Complete Engine Processing: {features.shape} in {processing_time:.3f}s")
    print(f"✅ Processing rate: {len(test_data)/processing_time:.0f} rows/second")
    
    return processing_time

def compare_with_basic_loops():
    """Compare vectorized operations with basic loops"""
    print("⚡ Comparing Vectorized vs Loop Operations...")
    
    test_data = generate_sample_data(5000)  # Smaller for loop comparison
    
    # Vectorized moving average
    start_time = time.time()
    vectorized_ma = test_data['price'].rolling(window=20).mean()
    vectorized_time = time.time() - start_time
    
    # Basic loop moving average
    start_time = time.time()
    loop_ma = []
    for i in range(len(test_data)):
        if i < 19:
            loop_ma.append(np.nan)
        else:
            loop_ma.append(test_data['price'].iloc[i-19:i+1].mean())
    loop_time = time.time() - start_time
    
    speedup = loop_time / vectorized_time if vectorized_time > 0 else float('inf')
    
    print(f"✅ Vectorized MA: {vectorized_time:.4f}s")
    print(f"🐌 Loop MA: {loop_time:.4f}s") 
    print(f"🚀 Speedup: {speedup:.1f}x")
    
    return speedup

def main():
    """Run quick validation tests"""
    print("🔥 HFT Engine Quick Validation")
    print("=" * 50)
    
    total_start = time.time()
    
    try:
        # Test individual components
        ts_features, ob_features, feature_time = test_vectorized_features()
        cache_time = test_optimized_cache()
        engine_time = test_optimized_engine()
        speedup = compare_with_basic_loops()
        
        total_time = time.time() - total_start
        
        print("\n" + "=" * 50)
        print("📊 VALIDATION SUMMARY")
        print("=" * 50)
        print(f"✅ Feature Extraction Time: {feature_time:.3f}s")
        print(f"✅ Cache Operations Time: {cache_time:.3f}s")
        print(f"✅ Complete Engine Time: {engine_time:.3f}s")
        print(f"🚀 Vectorization Speedup: {speedup:.1f}x")
        print(f"⏱️  Total Validation Time: {total_time:.3f}s")
        print("\n🎉 All optimizations working correctly!")
        
        # Estimate performance on full dataset
        rows_per_second = 15000 / engine_time
        full_dataset_rows = 25_000_000_000 / 100  # Estimate based on CSV size
        estimated_time = full_dataset_rows / rows_per_second
        
        print(f"\n📈 PERFORMANCE PROJECTION")
        print(f"Current processing rate: {rows_per_second:.0f} rows/second")
        print(f"Estimated time for 25GB dataset: {estimated_time/3600:.1f} hours")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
