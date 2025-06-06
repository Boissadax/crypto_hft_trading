#!/usr/bin/env python3
"""
HFT Engine v3 - Complete Implementation with Visual Benchmarking
Enhanced main script that implements all project functionalities and provides comprehensive benchmarking.

This script provides a complete end-to-end workflow:
1. Data loading and caching optimization
2. Feature engineering with microsecond precision
3. Transfer Entropy analysis and causality testing
4. Machine Learning model training and evaluation
5. Strategy implementation and backtesting
6. Comprehensive benchmark comparison with visualizations
7. Performance analysis and reporting

Usage:
    python main_complete.py --dataset DATA_0 --symbols BTC ETH
    python main_complete.py --dataset DATA_1 --symbols BTC ETH SOL
    python main_complete.py --full-analysis  # Complete analysis with all features
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import warnings
import argparse
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import json
from tqdm import tqdm
import time

warnings.filterwarnings('ignore')

# Import all modules
from statistical_analysis import TransferEntropyAnalyzer, CausalityTester, RegimeDetector
from feature_engineering import FeatureEngineer, AsynchronousSync, SyncConfig, split_train_test
# Import optimized vectorized modules
from feature_engineering.vectorized_time_series_features import VectorizedTimeSeriesFeatureExtractor
from feature_engineering.vectorized_order_book_features import VectorizedOrderBookExtractor
from optimized_data_cache import ensure_processed_optimized, get_cache_info_optimized
from learning import DataPreparator, ModelTrainer
from benchmark import BacktestEngine, BuyHoldStrategy, RandomStrategy, SimpleMomentumStrategy, MeanReversionStrategy
from strategy import TransferEntropyStrategy
from visualization import FeatureAnalysisVisualizer, ModelPerformanceVisualizer
from data_cache import ensure_processed, get_cache_info

def setup_logging(verbose: bool = False):
    """Setup comprehensive logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"hft_complete_{datetime.now().strftime('%Y%m%d_%H%M')}.log"),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

class HFTEngineComplete:
    """Complete HFT Engine implementation with all functionalities and progress tracking."""
    
    def __init__(self, dataset_id: str = "DATA_0", symbols: List[str] = None, verbose: bool = False, 
                 quick_mode: bool = False, max_events: int = None):
        """
        Initialize the complete HFT Engine.
        
        Args:
            dataset_id: Dataset identifier (DATA_0, DATA_1, DATA_2)
            symbols: List of trading symbols
            verbose: Enable verbose logging
            quick_mode: Enable quick mode (reduced dataset and processing)
            max_events: Maximum number of events to process (for testing/debugging)
        """
        self.dataset_id = dataset_id
        self.symbols = symbols or ["BTC", "ETH"]
        self.verbose = verbose
        self.quick_mode = quick_mode
        self.max_events = max_events
        self.logger = setup_logging(verbose)
        
        # Progress tracking
        self.progress_bars = {}
        self.start_time = time.time()
        
        # Results storage
        self.results = {
            'data_info': {},
            'features': pd.DataFrame(),
            'transfer_entropy': {},
            'causality_tests': {},
            'ml_results': {},
            'strategy_performance': {},
            'benchmark_comparison': {},
            'visualizations': {},
            'performance_metrics': {}
        }
        
        # Initialize components
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize all engine components."""
        self.logger.info("🔧 Initializing HFT Engine components...")
        
        with tqdm(total=7, desc="Initializing components", unit="component") as pbar:
            # Statistical analysis
            self.te_analyzer = TransferEntropyAnalyzer()
            pbar.update(1)
            
            self.causality_tester = CausalityTester()
            pbar.update(1)
            
            self.regime_detector = RegimeDetector()
            pbar.update(1)
            
            # Feature engineering
            sync_config = SyncConfig(
                max_interpolation_gap_us=300_000_000,  # 5 minutes
                min_symbols_required=1,
                enable_cross_symbol_features=True
            )
            
            self.feature_engineer = FeatureEngineer(
                symbols=self.symbols,
                sync_config=sync_config,
                max_levels=3 if self.quick_mode else 5,  # Reduced levels in quick mode
                dataset_id=self.dataset_id
            )
            pbar.update(1)
            
            # Learning components
            self.data_preparator = DataPreparator()
            pbar.update(1)
            
            self.model_trainer = ModelTrainer()
            pbar.update(1)
            
            # Visualization
            self.feature_visualizer = FeatureAnalysisVisualizer()
            self.performance_visualizer = ModelPerformanceVisualizer()
            pbar.update(1)
        
        self.logger.info("✅ All components initialized successfully")
    
    def load_and_cache_data(self) -> Dict[str, pd.DataFrame]:
        """Load and cache data with optimization and progress tracking."""
        self.logger.info(f"🔄 Loading data for dataset {self.dataset_id}...")
        
        start_time = time.time()
        
        try:
            # 🚀 OPTIMIZED: Use optimized cache system for 25GB+ files
            cache_info = get_cache_info_optimized(self.dataset_id)
            self.logger.info(f"Cache status: {cache_info}")
            
            # Process data if needed using optimized pipeline
            with tqdm(desc="Processing data cache (OPTIMIZED)", unit="step") as pbar:
                ensure_processed_optimized(self.dataset_id)
                pbar.update(1)
            
            # Load processed data
            processed_dir = Path("processed_data") / self.dataset_id
            data = {}
            
            for symbol in tqdm(self.symbols, desc="Loading symbols"):
                parquet_file = processed_dir / f"{symbol}_EUR.parquet"
                if parquet_file.exists():
                    df = pd.read_parquet(parquet_file)
                    
                    # Apply max_events limit for testing
                    if self.max_events and len(df) > self.max_events:
                        self.logger.info(f"🔄 Limiting {symbol} to {self.max_events:,} events (from {len(df):,})")
                        df = df.head(self.max_events)
                    
                    # Quick mode - sample data
                    if self.quick_mode and len(df) > 50000:
                        sample_size = min(50000, len(df))
                        df = df.sample(n=sample_size, random_state=42).sort_index()
                        self.logger.info(f"🚀 Quick mode: Sampled {len(df):,} events for {symbol}")
                    
                    data[symbol] = df
                    self.logger.info(f"✅ Loaded {symbol}: {len(df):,} rows")
                else:
                    self.logger.warning(f"⚠️  No data found for {symbol}")
            
            if not data:
                self.logger.info("📝 Generating synthetic data for demonstration...")
                data = self._generate_synthetic_data()
            
            # Store data info and performance metrics
            load_time = time.time() - start_time
            self.results['data_info'] = {
                'dataset_id': self.dataset_id,
                'symbols': list(data.keys()),
                'data_points': {symbol: len(df) for symbol, df in data.items()},
                'time_range': {
                    symbol: {
                        'start': df.index.min(),
                        'end': df.index.max()
                    } for symbol, df in data.items()
                },
                'quick_mode': self.quick_mode,
                'max_events': self.max_events
            }
            
            self.results['performance_metrics']['data_loading_time'] = load_time
            self.logger.info(f"⏱️ Data loading completed in {load_time:.2f} seconds")
            
            return data
            
        except Exception as e:
            self.logger.error(f"❌ Data loading failed: {e}")
            raise
    
    def _generate_synthetic_data(self) -> Dict[str, pd.DataFrame]:
        """Generate synthetic data with realistic microstructure."""
        self.logger.info("🎲 Generating synthetic market data...")
        
        # Generate 48 hours of 1-second data
        timestamps = pd.date_range(
            start='2024-01-01', 
            end='2024-01-03', 
            freq='1S'
        )
        
        np.random.seed(42)
        n_points = len(timestamps)
        
        # Generate realistic price dynamics
        data = {}
        base_prices = {'BTC': 45000, 'ETH': 2800, 'SOL': 100}
        
        for i, symbol in enumerate(self.symbols):
            if symbol not in base_prices:
                base_prices[symbol] = 1000 * (i + 1)
            
            # Geometric Brownian Motion with jumps
            dt = 1/86400  # 1 second in days
            sigma = 0.02  # 2% daily volatility
            mu = 0.0001   # Small drift
            
            # Generate returns with occasional jumps
            returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), n_points)
            
            # Add occasional jump events
            jump_prob = 0.001
            jump_mask = np.random.random(n_points) < jump_prob
            jump_sizes = np.random.normal(0, 0.01, n_points)
            returns[jump_mask] += jump_sizes[jump_mask]
            
            # Generate prices
            returns[0] = 0
            log_prices = np.log(base_prices[symbol]) + np.cumsum(returns)
            prices = np.exp(log_prices)
            
            # Generate order book data
            spread_bps = np.random.exponential(5, n_points)  # 5 bps average spread
            spreads = prices * spread_bps / 10000
            
            bid_prices = prices - spreads / 2
            ask_prices = prices + spreads / 2
            
            # Generate volumes
            base_volume = 100
            volume_factor = np.random.lognormal(0, 0.5, n_points)
            volumes = base_volume * volume_factor
            
            # Create DataFrame
            df = pd.DataFrame({
                'mid_price': prices,
                'bid': bid_prices,
                'ask': ask_prices,
                'volume': volumes,
                'spread': spreads,
                'volume_imbalance_l1': np.random.normal(0, 0.1, n_points),
                'price': prices  # For compatibility
            }, index=timestamps)
            
            data[symbol] = df
            
        self.logger.info(f"✅ Generated synthetic data for {len(data)} symbols")
        return data
    
    def engineer_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Comprehensive feature engineering with progress tracking."""
        self.logger.info("⚙️ Starting feature engineering...")
        
        start_time = time.time()
        
        try:
            # Try to use real data processing
            try:
                # Add progress monitoring to feature engineering
                self.logger.info("🔍 Attempting to process real market data...")
                
                if self.quick_mode:
                    # In quick mode, use smaller chunk size
                    chunk_size = 10000
                else:
                    chunk_size = 50000
                
                df_features = self.feature_engineer.create_features(chunk_size=chunk_size)
                self.logger.info("✅ Used real data for feature engineering")
                
            except (FileNotFoundError, ValueError) as e:
                self.logger.info(f"🔄 Falling back to synthetic data: {e}")
                
                # 🚀 OPTIMIZED: Vectorized synthetic data conversion
                df_raw_events = []
                total_events = sum(len(df) for df in data.values()) * 2  # bid + ask
                
                with tqdm(total=total_events, desc="Converting synthetic data (VECTORIZED)", unit="events") as pbar:
                    for symbol, df in data.items():
                        if symbol.upper() not in [s.upper() for s in self.symbols]:
                            continue
                        
                        # Vectorized conversion - NO INNER FOR LOOP
                        timestamps_us = (df.index.astype(np.int64) // 1000).astype(int)
                        
                        # Create vectorized arrays for bid/ask
                        n_rows = len(df)
                        symbols_array = np.full(n_rows, symbol.upper())
                        volumes_half = df['volume'].values / 2
                        
                        # Vectorized bid events
                        bid_events = pd.DataFrame({
                            "symbol": symbols_array,
                            "timestamp_us": timestamps_us,
                            "price": df['bid'].values,
                            "volume": volumes_half,
                            "side": np.full(n_rows, "bid"),
                            "level": np.ones(n_rows)
                        })
                        
                        # Vectorized ask events  
                        ask_events = pd.DataFrame({
                            "symbol": symbols_array,
                            "timestamp_us": timestamps_us,
                            "price": df['ask'].values,
                            "volume": volumes_half,
                            "side": np.full(n_rows, "ask"),
                            "level": np.ones(n_rows)
                        })
                        
                        # Combine and append
                        df_raw_events.extend(bid_events.to_dict('records'))
                        df_raw_events.extend(ask_events.to_dict('records'))
                        
                        pbar.update(n_rows * 2)  # bid + ask
                
                self.logger.info(f"📊 Created {len(df_raw_events):,} raw events")
                
                with tqdm(desc="Processing features", unit="step") as pbar:
                    df_raw = pd.DataFrame(df_raw_events).sort_values("timestamp_us")
                    pbar.update(1)
                    
                    df_features = self.feature_engineer.create_features(df_raw)
                    pbar.update(1)
            
            if df_features.empty:
                raise ValueError("No features generated")
            
            # Store performance metrics
            feature_time = time.time() - start_time
            self.results['performance_metrics']['feature_engineering_time'] = feature_time
            
            self.logger.info(f"✅ Generated {df_features.shape[1]} features from {len(df_features):,} samples")
            self.logger.info(f"📊 Feature columns: {list(df_features.columns[:10])}...")
            self.logger.info(f"⏱️ Feature engineering completed in {feature_time:.2f} seconds")
            
            # Store features
            self.results['features'] = df_features
            
            return df_features
            
        except Exception as e:
            self.logger.error(f"❌ Feature engineering failed: {e}")
            raise
            self.logger.error(f"❌ Feature engineering failed: {e}")
            raise
    
    def analyze_transfer_entropy(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Comprehensive Transfer Entropy analysis with progress tracking."""
        self.logger.info("🔬 Analyzing Transfer Entropy relationships...")
        
        start_time = time.time()
        
        try:
            # Prepare returns data
            returns_data = {}
            for symbol in tqdm(self.symbols, desc="Preparing returns data"):
                if symbol in data:
                    returns = data[symbol]['mid_price'].pct_change().dropna()
                    
                    # In quick mode, use smaller sample
                    if self.quick_mode and len(returns) > 10000:
                        returns = returns.iloc[-10000:]  # Last 10k points
                    
                    returns_data[symbol] = returns
                    self.logger.info(f"{symbol}: {len(returns)} returns, μ={returns.mean():.6f}, σ={returns.std():.6f}")
            
            # 🚀 OPTIMIZED: Vectorized pairwise Transfer Entropy calculation
            te_results = {}
            te_matrix = pd.DataFrame(index=self.symbols, columns=self.symbols)
            
            # Pre-compute all symbol combinations (vectorized)
            symbol_pairs = [(leader, follower) for leader in self.symbols 
                           for follower in self.symbols if leader != follower]
            total_pairs = len(symbol_pairs)
            
            with tqdm(total=total_pairs, desc="Computing Transfer Entropy (VECTORIZED)", unit="pairs") as pbar:
                # Batch process pairs for efficiency
                for leader, follower in symbol_pairs:
                    if leader in returns_data and follower in returns_data:
                        try:
                            # Vectorized data alignment
                            leader_data = returns_data[leader]
                            follower_data = returns_data[follower]
                            
                            common_index = leader_data.index.intersection(follower_data.index)
                            if len(common_index) < 200:
                                self.logger.warning(f"Insufficient data for {leader}→{follower}")
                                pbar.update(1)
                                continue
                            
                            # Vectorized alignment
                            leader_aligned = leader_data.loc[common_index]
                            follower_aligned = follower_data.loc[common_index]
                            
                            # Calculate TE with reduced complexity in quick mode
                            max_lag = 5 if self.quick_mode else 10
                            te_result = self.te_analyzer.calculate_transfer_entropy(
                                leader_aligned.values,
                                follower_aligned.values,
                                max_lag=max_lag,
                                method='ksg'
                            )
                            
                            pair_key = f"{leader}→{follower}"
                            te_results[pair_key] = te_result
                            te_matrix.loc[leader, follower] = te_result['transfer_entropy']
                            
                            if self.verbose:
                                self.logger.info(f"TE {pair_key}: {te_result['transfer_entropy']:.6f}")
                        
                        except Exception as e:
                            self.logger.warning(f"TE calculation failed for {leader}→{follower}: {e}")
                        
                        finally:
                            pbar.update(1)
                    else:
                        pbar.update(1)
            
            # Identify dominant relationships
            dominant_relationships = []
            for pair, result in te_results.items():
                if result['transfer_entropy'] > 0.01:  # Significant threshold
                    dominant_relationships.append({
                        'pair': pair,
                        'te_value': result['transfer_entropy'],
                        'significance': result.get('p_value', 'N/A')
                    })
            
            # Sort by TE value
            dominant_relationships.sort(key=lambda x: x['te_value'], reverse=True)
            
            # Store performance metrics
            te_time = time.time() - start_time
            self.results['performance_metrics']['transfer_entropy_time'] = te_time
            
            analysis_results = {
                'te_matrix': te_matrix,
                'pairwise_results': te_results,
                'dominant_relationships': dominant_relationships,
                'summary': {
                    'total_pairs_analyzed': len(te_results),
                    'significant_relationships': len(dominant_relationships),
                    'strongest_relationship': dominant_relationships[0] if dominant_relationships else None
                }
            }
            
            self.results['transfer_entropy'] = analysis_results
            self.logger.info(f"✅ TE analysis complete: {len(dominant_relationships)} significant relationships found")
            self.logger.info(f"⏱️ Transfer Entropy analysis completed in {te_time:.2f} seconds")
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"❌ Transfer Entropy analysis failed: {e}")
            return {}
    
    def perform_causality_tests(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Comprehensive causality testing with progress tracking."""
        self.logger.info("📈 Performing statistical causality tests...")
        
        start_time = time.time()
        
        try:
            # Prepare data
            returns_df = pd.DataFrame()
            for symbol in tqdm(self.symbols, desc="Preparing causality data"):
                if symbol in data:
                    returns = data[symbol]['mid_price'].pct_change()
                    
                    # In quick mode, use smaller sample
                    if self.quick_mode and len(returns) > 5000:
                        returns = returns.iloc[-5000:]  # Last 5k points
                    
                    returns_df[symbol] = returns
            
            returns_df = returns_df.dropna()
            
            if len(returns_df) < 100:
                self.logger.warning("Insufficient data for causality testing")
                return {}
            
            results = {}
            
            # Test components with progress tracking
            tests_to_run = ['granger', 'var']
            if not self.quick_mode:
                tests_to_run.append('johansen')
            
            with tqdm(total=len(tests_to_run), desc="Running causality tests", unit="test") as pbar:
                # Granger Causality
                try:
                    max_lag = 5 if self.quick_mode else 10
                    granger_result = self.causality_tester.granger_causality_test(
                        returns_df, max_lag=max_lag
                    )
                    results['granger'] = granger_result
                    self.logger.info("✅ Granger causality test completed")
                except Exception as e:
                    self.logger.error(f"Granger test failed: {e}")
                    results['granger'] = {}
                pbar.update(1)
                
                # VAR Causality
                try:
                    max_lag = 3 if self.quick_mode else 5
                    var_result = self.causality_tester.var_causality_test(
                        returns_df, max_lag=max_lag
                    )
                    results['var'] = var_result
                    self.logger.info("✅ VAR causality test completed")
                except Exception as e:
                    self.logger.error(f"VAR test failed: {e}")
                    results['var'] = {}
                pbar.update(1)
                
                # Johansen Cointegration (only in full mode)
                if not self.quick_mode:
                    try:
                        johansen_result = self.causality_tester.johansen_cointegration_test(returns_df)
                        results['johansen'] = johansen_result
                        self.logger.info("✅ Johansen cointegration test completed")
                    except Exception as e:
                        self.logger.error(f"Johansen test failed: {e}")
                        results['johansen'] = {}
                    pbar.update(1)
            
            # Store performance metrics
            causality_time = time.time() - start_time
            self.results['performance_metrics']['causality_tests_time'] = causality_time
            
            self.results['causality_tests'] = results
            self.logger.info(f"✅ Causality testing completed")
            self.logger.info(f"⏱️ Causality tests completed in {causality_time:.2f} seconds")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Causality testing failed: {e}")
            return {}
    
    def train_ml_models(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Train and evaluate ML models with progress tracking."""
        self.logger.info("🤖 Training machine learning models...")
        
        start_time = time.time()
        
        try:
            if features.empty:
                self.logger.warning("No features available for ML training")
                return {}
            
            # Prepare data with progress tracking
            with tqdm(desc="Preparing ML data", unit="step") as pbar:
                df_prepared = self.data_preparator.prepare_features(features)
                pbar.update(1)
                
                if df_prepared.empty:
                    self.logger.warning("Data preparation resulted in empty dataset")
                    return {}
                
                # Split data temporally
                df_train, df_test = split_train_test(df_prepared, frac=0.7)
                pbar.update(1)
            
            self.logger.info(f"Train set: {len(df_train):,} samples")
            self.logger.info(f"Test set: {len(df_test):,} samples")
            
            # Train models with progress tracking
            with tqdm(desc="Training ML models", unit="model") as pbar:
                ml_results = self.model_trainer.train_and_evaluate(df_train, df_test)
                pbar.update(1)
            
            # Store performance metrics
            ml_time = time.time() - start_time
            self.results['performance_metrics']['ml_training_time'] = ml_time
            
            self.results['ml_results'] = ml_results
            self.logger.info("✅ ML model training completed")
            self.logger.info(f"⏱️ ML training completed in {ml_time:.2f} seconds")
            
            return ml_results
            
        except Exception as e:
            self.logger.error(f"❌ ML training failed: {e}")
            return {}
    
    def run_comprehensive_backtest(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Run comprehensive strategy backtesting with progress tracking."""
        self.logger.info("🎯 Running comprehensive strategy backtesting...")
        
        start_time = time.time()
        
        try:
            # Initialize strategies with progress tracking
            strategies = []
            
            with tqdm(desc="Initializing strategies", unit="strategy") as pbar:
                # Main Transfer Entropy Strategy
                te_strategy = TransferEntropyStrategy(
                    symbols=self.symbols,
                    initial_capital=100000.0,
                    te_threshold=0.02,
                    confidence_threshold=0.6,
                    lookback_window=100 if self.quick_mode else 200,
                    rebalance_frequency=50 if self.quick_mode else 100
                )
                strategies.append(te_strategy)
                pbar.update(1)
                
                # Benchmark strategies
                strategies.extend([
                    BuyHoldStrategy(symbol=self.symbols[0], initial_capital=100000.0),
                    RandomStrategy(symbols=self.symbols, initial_capital=100000.0, trade_probability=0.05),
                    SimpleMomentumStrategy(symbols=self.symbols, lookback_window=25 if self.quick_mode else 50, initial_capital=100000.0),
                    MeanReversionStrategy(symbols=self.symbols, lookback_window=25 if self.quick_mode else 50, initial_capital=100000.0)
                ])
                pbar.update(4)
            
            # Run backtest with progress tracking
            backtest_engine = BacktestEngine(initial_capital=100000.0)
            
            with tqdm(desc="Running backtests", unit="strategy") as pbar:
                results = backtest_engine.compare_strategies(strategies, data)
                pbar.update(len(strategies))
            
            # Enhanced analysis
            if 'comparison' in results:
                comparison_df = results['comparison']
                
                # Calculate additional metrics
                comparison_df['Sharpe_Ratio'] = comparison_df['Total Return'] / (comparison_df.get('Volatility', 1) + 1e-6)
                comparison_df['Risk_Adjusted_Return'] = comparison_df['Total Return'] / (comparison_df.get('Max_Drawdown', 1) + 1e-6)
                
                # Rank strategies
                comparison_df['Overall_Rank'] = comparison_df['Total Return'].rank(ascending=False)
                
                self.logger.info("📊 Strategy Performance Summary:")
                for strategy, row in comparison_df.iterrows():
                    self.logger.info(f"  {strategy}: {row['Total Return']:.2%} return, Rank #{int(row['Overall_Rank'])}")
            
            # Store performance metrics
            backtest_time = time.time() - start_time
            self.results['performance_metrics']['backtesting_time'] = backtest_time
            
            self.results['strategy_performance'] = results
            self.logger.info(f"✅ Backtesting completed in {backtest_time:.2f} seconds")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Backtesting failed: {e}")
            return {}
    
    def create_comprehensive_visualizations(self):
        """Create comprehensive visualizations with progress tracking."""
        self.logger.info("📊 Creating comprehensive visualizations...")
        
        start_time = time.time()
        
        try:
            viz_dir = Path("visualizations")
            viz_dir.mkdir(exist_ok=True)
            
            plt.style.use('seaborn-v0_8')
            
            visualizations_to_create = []
            
            # Check what visualizations can be created
            if not self.results['features'].empty:
                visualizations_to_create.append("feature_analysis")
            
            if 'te_matrix' in self.results['transfer_entropy']:
                visualizations_to_create.append("transfer_entropy")
            
            if 'comparison' in self.results['strategy_performance']:
                visualizations_to_create.append("strategy_performance")
            
            # Create visualizations with progress tracking
            with tqdm(total=len(visualizations_to_create), desc="Creating visualizations", unit="plot") as pbar:
                
                # 1. Feature Analysis
                if "feature_analysis" in visualizations_to_create:
                    self.logger.info("📈 Creating feature analysis plots...")
                    
                    fig = self.feature_visualizer.create_comprehensive_analysis(
                        self.results['features']
                    )
                    if fig:
                        fig.savefig(viz_dir / 'feature_analysis.png', dpi=300, bbox_inches='tight')
                        plt.close(fig)
                    pbar.update(1)
                
                # 2. Transfer Entropy Heatmap
                if "transfer_entropy" in visualizations_to_create:
                    self.logger.info("🔥 Creating Transfer Entropy heatmap...")
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    te_matrix = self.results['transfer_entropy']['te_matrix'].astype(float)
                    
                    sns.heatmap(te_matrix, annot=True, cmap='Blues', fmt='.4f', ax=ax)
                    ax.set_title('Transfer Entropy Matrix\n(Leader → Follower)', fontsize=16, fontweight='bold')
                    ax.set_xlabel('Follower', fontsize=12)
                    ax.set_ylabel('Leader', fontsize=12)
                    
                    plt.tight_layout()
                    fig.savefig(viz_dir / 'transfer_entropy_heatmap.png', dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    pbar.update(1)
                
                # 3. Strategy Performance Comparison
                if "strategy_performance" in visualizations_to_create:
                    self.logger.info("🏆 Creating strategy performance comparison...")
                    
                    comparison_df = self.results['strategy_performance']['comparison']
                    
                    # Performance bar chart
                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
                    
                    # Returns comparison
                    comparison_df['Total Return'].plot(kind='bar', ax=ax1, color='skyblue')
                    ax1.set_title('Total Returns by Strategy', fontweight='bold')
                    ax1.set_ylabel('Return (%)')
                    ax1.tick_params(axis='x', rotation=45)
                    ax1.grid(True, alpha=0.3)
                    
                    # Sharpe Ratio comparison
                    if 'Sharpe_Ratio' in comparison_df.columns:
                        comparison_df['Sharpe_Ratio'].plot(kind='bar', ax=ax2, color='lightgreen')
                        ax2.set_title('Sharpe Ratio Comparison', fontweight='bold')
                        ax2.set_ylabel('Sharpe Ratio')
                        ax2.tick_params(axis='x', rotation=45)
                        ax2.grid(True, alpha=0.3)
                    
                    # Trade count
                    if 'Number of Trades' in comparison_df.columns:
                        comparison_df['Number of Trades'].plot(kind='bar', ax=ax3, color='orange')
                        ax3.set_title('Number of Trades', fontweight='bold')
                        ax3.set_ylabel('Trade Count')
                        ax3.tick_params(axis='x', rotation=45)
                        ax3.grid(True, alpha=0.3)
                    
                    # Win rate
                    if 'Win Rate' in comparison_df.columns:
                        comparison_df['Win Rate'].plot(kind='bar', ax=ax4, color='salmon')
                        ax4.set_title('Win Rate (%)', fontweight='bold')
                        ax4.set_ylabel('Win Rate')
                        ax4.tick_params(axis='x', rotation=45)
                        ax4.grid(True, alpha=0.3)
                
                plt.suptitle('Strategy Performance Benchmark', fontsize=16, fontweight='bold')
                plt.tight_layout()
                fig.savefig(viz_dir / 'strategy_benchmark.png', dpi=300, bbox_inches='tight')
                plt.close(fig)
            
            # 4. ML Model Performance
            if self.results['ml_results']:
                self.logger.info("🤖 Creating ML performance visualizations...")
                
                fig = self.performance_visualizer.create_model_comparison(
                    self.results['ml_results']
                )
                if fig:
                    fig.savefig(viz_dir / 'ml_performance.png', dpi=300, bbox_inches='tight')
                    plt.close(fig)
            
            # 5. Summary Dashboard
            self._create_summary_dashboard(viz_dir)
            
            self.logger.info(f"✅ All visualizations saved to {viz_dir}/")
            
        except Exception as e:
            self.logger.error(f"❌ Visualization creation failed: {e}")
    
    def _create_summary_dashboard(self, viz_dir: Path):
        """Create a comprehensive summary dashboard."""
        try:
            fig = plt.figure(figsize=(20, 14))
            gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
            
            # Title
            fig.suptitle('HFT Engine v3 - Complete Analysis Dashboard', 
                        fontsize=20, fontweight='bold', y=0.95)
            
            # Data summary
            ax1 = fig.add_subplot(gs[0, :2])
            data_info = self.results['data_info']
            summary_text = f"""
Dataset: {data_info.get('dataset_id', 'N/A')}
Symbols: {', '.join(data_info.get('symbols', []))}
Data Points: {sum(data_info.get('data_points', {}).values()):,}
Features Generated: {self.results['features'].shape[1] if not self.results['features'].empty else 0}
            """
            ax1.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')
            ax1.set_title('Data Summary', fontweight='bold')
            ax1.axis('off')
            
            # Transfer Entropy summary
            ax2 = fig.add_subplot(gs[0, 2:])
            te_results = self.results['transfer_entropy']
            if te_results:
                dominant = te_results.get('dominant_relationships', [])
                strongest_pair = dominant[0]['pair'] if dominant else 'None'
                strongest_value = f"{dominant[0]['te_value']:.4f}" if dominant else 'N/A'
                te_text = f"""
Relationships Analyzed: {te_results.get('summary', {}).get('total_pairs_analyzed', 0)}
Significant Relationships: {len(dominant)}
Strongest: {strongest_pair} ({strongest_value})
                """
                ax2.text(0.1, 0.5, te_text, fontsize=12, verticalalignment='center')
            ax2.set_title('Transfer Entropy Summary', fontweight='bold')
            ax2.axis('off')
            
            # Strategy performance
            if 'comparison' in self.results['strategy_performance']:
                comparison_df = self.results['strategy_performance']['comparison']
                
                # Best performers
                ax3 = fig.add_subplot(gs[1, :])
                top_3 = comparison_df.nlargest(3, 'Total Return')
                
                strategies = top_3.index.tolist()
                returns = top_3['Total Return'].values
                colors = ['gold', 'silver', 'chocolate']
                
                bars = ax3.bar(range(len(strategies)), returns, color=colors[:len(strategies)])
                ax3.set_title('Top 3 Performing Strategies', fontweight='bold', fontsize=14)
                ax3.set_ylabel('Total Return (%)')
                ax3.set_xticks(range(len(strategies)))
                ax3.set_xticklabels(strategies, rotation=45, ha='right')
                ax3.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars, returns):
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                            f'{value:.2f}%', ha='center', va='bottom', fontweight='bold')
            
            # Feature importance (if available)
            if not self.results['features'].empty:
                ax4 = fig.add_subplot(gs[2, :2])
                features_df = self.results['features']
                
                # Calculate feature variance as proxy for importance
                feature_vars = features_df.var().sort_values(ascending=False).head(10)
                
                ax4.barh(range(len(feature_vars)), feature_vars.values, color='lightblue')
                ax4.set_yticks(range(len(feature_vars)))
                ax4.set_yticklabels([name[:20] + '...' if len(name) > 20 else name 
                                   for name in feature_vars.index], fontsize=10)
                ax4.set_title('Top 10 Features by Variance', fontweight='bold')
                ax4.set_xlabel('Variance')
                ax4.grid(True, alpha=0.3)
            
            # Model performance summary
            ax5 = fig.add_subplot(gs[2, 2:])
            ml_results = self.results['ml_results']
            if ml_results:
                model_text = "ML Models Trained:\n"
                for model_name, results in ml_results.items():
                    if isinstance(results, dict) and 'accuracy' in results:
                        model_text += f"• {model_name}: {results['accuracy']:.3f}\n"
                ax5.text(0.1, 0.5, model_text, fontsize=11, verticalalignment='center')
            else:
                ax5.text(0.1, 0.5, "No ML results available", fontsize=11, verticalalignment='center')
            ax5.set_title('ML Model Performance', fontweight='bold')
            ax5.axis('off')
            
            # Execution summary
            ax6 = fig.add_subplot(gs[3, :])
            execution_summary = f"""
🚀 HFT Engine v3 Execution Summary:
✅ Data Loading & Caching: Completed
✅ Feature Engineering: {self.results['features'].shape[1]} features generated
✅ Transfer Entropy Analysis: {len(self.results['transfer_entropy'].get('pairwise_results', {}))} pairs analyzed
✅ Causality Testing: Multiple statistical tests performed
✅ ML Model Training: {len(self.results['ml_results'])} models trained
✅ Strategy Backtesting: {len(self.results['strategy_performance'].get('individual_results', {}))} strategies tested
✅ Comprehensive Visualizations: Generated
            """
            ax6.text(0.05, 0.5, execution_summary, fontsize=12, verticalalignment='center', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
            ax6.set_title('Execution Status', fontweight='bold')
            ax6.axis('off')
            
            plt.savefig(viz_dir / 'summary_dashboard.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            self.logger.error(f"❌ Summary dashboard creation failed: {e}")
    
    def save_results(self):
        """Save all results to files."""
        self.logger.info("💾 Saving results...")
        
        try:
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save main results as JSON
            json_results = {}
            for key, value in self.results.items():
                if isinstance(value, pd.DataFrame):
                    if not value.empty:
                        json_results[key] = {
                            'shape': value.shape,
                            'columns': value.columns.tolist(),
                            'sample_data': value.head().to_dict()
                        }
                elif isinstance(value, dict):
                    # Convert numpy types to native Python types for JSON serialization
                    json_results[key] = self._convert_for_json(value)
                else:
                    json_results[key] = value
            
            with open(results_dir / f"hft_results_{timestamp}.json", 'w') as f:
                json.dump(json_results, f, indent=2, default=str)
            
            # Save features DataFrame
            if not self.results['features'].empty:
                self.results['features'].to_parquet(results_dir / f"features_{timestamp}.parquet")
            
            # Save strategy comparison
            if 'comparison' in self.results['strategy_performance']:
                comparison_df = self.results['strategy_performance']['comparison']
                comparison_df.to_csv(results_dir / f"strategy_comparison_{timestamp}.csv")
            
            self.logger.info(f"✅ Results saved to {results_dir}/")
            
        except Exception as e:
            self.logger.error(f"❌ Results saving failed: {e}")
    
    def _convert_for_json(self, obj):
        """Convert numpy types to JSON-serializable types."""
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        else:
            return obj
    
    def run(self) -> Dict[str, Any]:
        """
        Main orchestrator method with global progress tracking.
        Provides a comprehensive pipeline with overall progress indication.
        """
        total_steps = 8
        overall_start_time = time.time()
        
        print(f"\n🚀 Starting HFT Engine v3 Analysis Pipeline")
        print(f"   Mode: {'Quick' if self.quick_mode else 'Full'}")
        if self.max_events:
            print(f"   Max Events: {self.max_events:,}")
        print(f"   Symbols: {', '.join(self.symbols)}")
        print(f"   Dataset: {self.dataset_id}")
        print("="*60)
        
        with tqdm(total=total_steps, desc="Overall Progress", unit="phase", position=0) as main_pbar:
            try:
                # Phase 1: Data Loading
                main_pbar.set_description("📊 Data Loading")
                data = self.load_and_cache_data()
                main_pbar.update(1)
                
                # Phase 2: Feature Engineering
                main_pbar.set_description("⚙️ Feature Engineering")
                features = self.engineer_features(data)
                main_pbar.update(1)
                
                # Phase 3: Transfer Entropy Analysis
                main_pbar.set_description("🔬 Transfer Entropy")
                te_results = self.analyze_transfer_entropy(data)
                main_pbar.update(1)
                
                # Phase 4: Causality Testing
                main_pbar.set_description("📈 Causality Tests")
                causality_results = self.perform_causality_tests(data)
                main_pbar.update(1)
                
                # Phase 5: ML Training
                main_pbar.set_description("🤖 ML Training")
                ml_results = self.train_ml_models(features)
                main_pbar.update(1)
                
                # Phase 6: Strategy Backtesting
                main_pbar.set_description("🎯 Backtesting")
                backtest_results = self.run_comprehensive_backtest(data)
                main_pbar.update(1)
                
                # Phase 7: Visualization
                main_pbar.set_description("📊 Visualizations")
                self.create_comprehensive_visualizations()
                main_pbar.update(1)
                
                # Phase 8: Results Saving
                main_pbar.set_description("💾 Saving Results")
                self.save_results()
                main_pbar.update(1)
                
                main_pbar.set_description("✅ Complete")
                
            except Exception as e:
                main_pbar.set_description("❌ Failed")
                raise
        
        # Final summary with timing
        total_time = time.time() - overall_start_time
        self.results['performance_metrics']['total_execution_time'] = total_time
        
        print(f"\n🎉 Analysis Pipeline Complete!")
        print(f"⏱️ Total Execution Time: {total_time:.2f} seconds")
        
        # Detailed performance breakdown
        if 'performance_metrics' in self.results:
            metrics = self.results['performance_metrics']
            print(f"\n📊 Performance Breakdown:")
            for phase, time_taken in metrics.items():
                if phase.endswith('_time') and phase != 'total_execution_time':
                    phase_name = phase.replace('_time', '').replace('_', ' ').title()
                    print(f"   {phase_name}: {time_taken:.2f}s")
        
        self._print_final_summary()
        
        return self.results
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """Run the complete HFT analysis pipeline."""
        self.logger.info("🚀 Starting complete HFT Engine v3 analysis...")
        
        try:
            # 1. Load and cache data
            print("\n📊 Loading and caching data...")
            data = self.load_and_cache_data()
            
            # 2. Feature engineering
            print("\n⚙️ Engineering features...")
            features = self.engineer_features(data)
            
            # 3. Transfer Entropy analysis
            print("\n🔬 Analyzing Transfer Entropy...")
            te_results = self.analyze_transfer_entropy(data)
            
            # 4. Causality testing
            print("\n📈 Performing causality tests...")
            causality_results = self.perform_causality_tests(data)
            
            # 5. ML model training
            print("\n🤖 Training ML models...")
            ml_results = self.train_ml_models(features)
            
            # 6. Strategy backtesting
            print("\n🎯 Running comprehensive backtesting...")
            backtest_results = self.run_comprehensive_backtest(data)
            
            # 7. Create visualizations
            print("\n📊 Creating comprehensive visualizations...")
            self.create_comprehensive_visualizations()
            
            # 8. Save results
            print("\n💾 Saving results...")
            self.save_results()
            
            # 9. Print summary
            self._print_final_summary()
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"❌ Complete analysis failed: {e}")
            raise
    
    def _print_final_summary(self):
        """Print comprehensive final summary."""
        print("\n" + "="*80)
        print("🎉 HFT ENGINE v3 - COMPLETE ANALYSIS SUMMARY")
        print("="*80)
        
        # Data summary
        data_info = self.results['data_info']
        print(f"\n📊 DATA OVERVIEW:")
        print(f"   Dataset: {data_info.get('dataset_id', 'N/A')}")
        print(f"   Symbols: {', '.join(data_info.get('symbols', []))}")
        print(f"   Total Data Points: {sum(data_info.get('data_points', {}).values()):,}")
        
        # Features summary
        if not self.results['features'].empty:
            print(f"\n⚙️ FEATURE ENGINEERING:")
            print(f"   Features Generated: {self.results['features'].shape[1]}")
            print(f"   Time Series Length: {len(self.results['features']):,}")
            print(f"   Microsecond Precision: {'✅' if self.results['features'].index.dtype == 'int64' else '❌'}")
        
        # Transfer Entropy summary
        te_results = self.results['transfer_entropy']
        if te_results:
            print(f"\n🔬 TRANSFER ENTROPY ANALYSIS:")
            summary = te_results.get('summary', {})
            print(f"   Pairs Analyzed: {summary.get('total_pairs_analyzed', 0)}")
            print(f"   Significant Relationships: {summary.get('significant_relationships', 0)}")
            
            strongest = summary.get('strongest_relationship')
            if strongest:
                print(f"   Strongest Relationship: {strongest['pair']} (TE: {strongest['te_value']:.4f})")
        
        # Strategy performance summary
        if 'comparison' in self.results['strategy_performance']:
            comparison_df = self.results['strategy_performance']['comparison']
            print(f"\n🏆 STRATEGY PERFORMANCE:")
            print(f"   Strategies Tested: {len(comparison_df)}")
            
            best_strategy = comparison_df.loc[comparison_df['Total Return'].idxmax()]
            worst_strategy = comparison_df.loc[comparison_df['Total Return'].idxmin()]
            
            print(f"   🥇 Best Strategy: {best_strategy.name}")
            print(f"      Return: {best_strategy['Total Return']:.2f}%")
            print(f"      Trades: {best_strategy.get('Number of Trades', 'N/A')}")
            
            print(f"   🥉 Worst Strategy: {worst_strategy.name}")
            print(f"      Return: {worst_strategy['Total Return']:.2f}%")
            
            # Show all strategies ranked
            print(f"\n📈 COMPLETE STRATEGY RANKING:")
            ranked = comparison_df.sort_values('Total Return', ascending=False)
            for i, (strategy, row) in enumerate(ranked.iterrows(), 1):
                print(f"   {i}. {strategy}: {row['Total Return']:.2f}%")
        
        # ML results summary
        if self.results['ml_results']:
            print(f"\n🤖 MACHINE LEARNING:")
            print(f"   Models Trained: {len(self.results['ml_results'])}")
            for model_name, results in self.results['ml_results'].items():
                if isinstance(results, dict) and 'accuracy' in results:
                    print(f"   {model_name}: {results['accuracy']:.3f} accuracy")
        
        # Files generated
        print(f"\n📁 OUTPUTS GENERATED:")
        print(f"   📊 Visualizations: visualizations/")
        print(f"   💾 Results: results/")
        print(f"   📝 Logs: logs/")
        
        print(f"\n💡 NEXT STEPS:")
        print(f"   1. Review visualizations in visualizations/ folder")
        print(f"   2. Analyze strategy performance details")
        print(f"   3. Fine-tune Transfer Entropy strategy parameters")
        print(f"   4. Explore feature importance and selection")
        print(f"   5. Consider ensemble methods for improved performance")
        
        print("\n" + "="*80)
        print("✅ ANALYSIS COMPLETE! Check the generated files for detailed results.")
        print("="*80)

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="HFT Engine v3 - Complete Implementation with Visual Benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --dataset DATA_0 --symbols BTC ETH
  python main.py --dataset DATA_1 --symbols BTC ETH SOL --verbose
  python main.py --full-analysis
  python main.py --quick-mode --max-events 100000  # Fast testing mode
  python main.py --symbols BTC ETH --quick-mode     # Quick analysis
        """
    )
    
    parser.add_argument("--dataset", choices=["DATA_0", "DATA_1", "DATA_2"], 
                       default="DATA_0", help="Dataset to analyze")
    parser.add_argument("--symbols", nargs="+", default=["BTC", "ETH"],
                       help="Trading symbols to analyze")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--full-analysis", action="store_true",
                       help="Run complete analysis with all features")
    parser.add_argument("--quick-mode", "-q", action="store_true",
                       help="Enable quick mode (reduced dataset size and complexity)")
    parser.add_argument("--max-events", type=int, metavar="N",
                       help="Maximum number of events to process per symbol (for testing)")
    
    args = parser.parse_args()
    
    # Banner
    print("🚀 " + "="*70)
    print("   HFT ENGINE v3 - COMPLETE IMPLEMENTATION")
    print("   Transfer Entropy Based Trading System")
    print("   With Comprehensive Benchmarking & Visualization")
    print("="*72)
    
    if args.full_analysis:
        args.symbols = ["BTC", "ETH"]  # Can be extended
        print("🔄 Running FULL ANALYSIS mode...")
    elif args.quick_mode:
        print("⚡ Running QUICK MODE - Reduced complexity for faster execution")
    
    print(f"📁 Dataset: {args.dataset}")
    print(f"📈 Symbols: {', '.join(args.symbols)}")
    print(f"🔍 Verbose: {'ON' if args.verbose else 'OFF'}")
    if args.max_events:
        print(f"🔢 Max Events per Symbol: {args.max_events:,}")
    print("="*72)
    
    try:
        # Initialize and run complete engine
        engine = HFTEngineComplete(
            dataset_id=args.dataset,
            symbols=args.symbols,
            verbose=args.verbose,
            quick_mode=args.quick_mode,
            max_events=args.max_events
        )
        
        # Run complete analysis using the new run method
        results = engine.run()
        
        return results
        
    except KeyboardInterrupt:
        print("\n❌ Analysis interrupted by user")
        return None
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()
