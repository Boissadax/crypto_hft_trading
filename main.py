"""
HFT Engine v3 - Transfer Entropy Based Trading System
Main execution script for the refactored HFT trading system.

This script demonstrates the complete workflow:
1. Data loading and preprocessing
2. Transfer Entropy analysis
3. Strategy backtesting
4. Performance evaluation
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import refactored modules
from statistical_analysis import TransferEntropyAnalyzer, CausalityTester, RegimeDetector
from feature_engineering import FeatureEngineer, AsynchronousSync
from learning import DataPreparator, ModelTrainer
from benchmark import BacktestEngine, BuyHoldStrategy, RandomStrategy
from strategy import TransferEntropyStrategy

def setup_logging():
    """Setup logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"hft_engine_{datetime.now().strftime('%Y%m%d')}.log"),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def load_sample_data():
    """
    Load sample cryptocurrency data.
    In production, this would connect to real data sources.
    """
    logger = logging.getLogger(__name__)
    
    # Try to load existing data
    data_dir = Path("raw_data")
    data_files = {
        'BTC': data_dir / "XBT_EUR.csv",
        'ETH': data_dir / "ETH_EUR.csv"
    }
    
    data = {}
    
    for symbol, file_path in data_files.items():
        try:
            if file_path.exists():
                df = pd.read_csv(file_path)
                logger.info(f"Loaded {symbol} data: {len(df)} rows")
                
                # Basic preprocessing
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                
                # Calculate mid price
                if 'price' in df.columns:
                    df['mid_price'] = df['price']
                elif 'bid' in df.columns and 'ask' in df.columns:
                    df['mid_price'] = (df['bid'] + df['ask']) / 2
                else:
                    logger.warning(f"Could not calculate mid_price for {symbol}")
                    continue
                
                data[symbol] = df
            else:
                logger.warning(f"Data file not found: {file_path}")
                
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
    
    if not data:
        logger.info("No data files found, generating synthetic data")
        data = generate_synthetic_data()
    
    return data

def generate_synthetic_data():
    """Generate synthetic cryptocurrency data for demonstration."""
    logger = logging.getLogger(__name__)
    logger.info("Generating synthetic cryptocurrency data")
    
    # Generate 1 week of 1-minute data
    timestamps = pd.date_range(
        start='2024-01-01', 
        end='2024-01-08', 
        freq='1min'
    )
    
    np.random.seed(42)
    
    # Generate correlated price series
    n_points = len(timestamps)
    
    # BTC as leader
    btc_returns = np.random.normal(0, 0.002, n_points)
    btc_returns[0] = 0
    btc_prices = 50000 * np.exp(np.cumsum(btc_returns))
    
    # ETH follows BTC with some lag and noise
    eth_returns = np.zeros(n_points)
    eth_returns[1:] = 0.7 * btc_returns[:-1] + np.random.normal(0, 0.0015, n_points-1)
    eth_prices = 3000 * np.exp(np.cumsum(eth_returns))
    
    # Create DataFrames
    data = {}
    
    for symbol, prices in [('BTC', btc_prices), ('ETH', eth_prices)]:
        df = pd.DataFrame({
            'mid_price': prices,
            'volume': np.random.exponential(100, n_points),
            'spread': np.random.exponential(0.01, n_points) * prices / 10000,
        }, index=timestamps)
        
        # Add some order book features
        df['bid'] = df['mid_price'] - df['spread'] / 2
        df['ask'] = df['mid_price'] + df['spread'] / 2
        df['volume_imbalance_l1'] = np.random.normal(0, 0.1, n_points)
        
        data[symbol] = df
    
    return data

def demonstrate_transfer_entropy_analysis(data):
    """Demonstrate Transfer Entropy analysis."""
    logger = logging.getLogger(__name__)
    logger.info("=== Transfer Entropy Analysis ===")
    
    # Initialize analyzer
    te_analyzer = TransferEntropyAnalyzer()
    
    # Get price series
    symbols = list(data.keys())
    if len(symbols) < 2:
        logger.warning("Need at least 2 symbols for TE analysis")
        return {}
    
    # Calculate returns
    returns_data = {}
    for symbol in symbols:
        returns = data[symbol]['mid_price'].pct_change().dropna()
        returns_data[symbol] = returns
        logger.info(f"{symbol} returns: mean={returns.mean():.6f}, std={returns.std():.6f}")
    
    # Calculate Transfer Entropy between all pairs
    te_results = {}
    
    for i, leader in enumerate(symbols):
        for j, follower in enumerate(symbols):
            if i != j:
                try:
                    # Align data
                    leader_data = returns_data[leader]
                    follower_data = returns_data[follower]
                    
                    # Find common index
                    common_index = leader_data.index.intersection(follower_data.index)
                    if len(common_index) < 100:
                        logger.warning(f"Insufficient overlapping data for {leader}->{follower}")
                        continue
                    
                    leader_aligned = leader_data.loc[common_index]
                    follower_aligned = follower_data.loc[common_index]
                    
                    # Calculate TE
                    te_result = te_analyzer.calculate_transfer_entropy(
                        leader_aligned.values,
                        follower_aligned.values,
                        max_lag=5
                    )
                    
                    te_results[f"{leader}->{follower}"] = te_result
                    
                    logger.info(f"TE {leader}->{follower}: {te_result['transfer_entropy']:.6f} "
                              f"(p-value: {te_result.get('p_value', 'N/A')})")
                    
                except Exception as e:
                    logger.error(f"Error calculating TE for {leader}->{follower}: {e}")
    
    return te_results

def demonstrate_causality_testing(data):
    """Demonstrate statistical causality testing."""
    logger = logging.getLogger(__name__)
    logger.info("=== Causality Testing ===")
    
    causality_tester = CausalityTester()
    
    # Prepare data for causality tests
    symbols = list(data.keys())
    if len(symbols) < 2:
        return {}
    
    # Create aligned DataFrame of returns
    returns_df = pd.DataFrame()
    for symbol in symbols:
        returns_df[symbol] = data[symbol]['mid_price'].pct_change()
    
    returns_df = returns_df.dropna()
    
    if len(returns_df) < 50:
        logger.warning("Insufficient data for causality testing")
        return {}
    
    # Granger causality test
    try:
        granger_result = causality_tester.granger_causality_test(
            returns_df, max_lag=5
        )
        logger.info(f"Granger causality results: {granger_result['summary']}")
    except Exception as e:
        logger.error(f"Granger causality test failed: {e}")
        granger_result = {}
    
    # VAR causality test
    try:
        var_result = causality_tester.var_causality_test(
            returns_df, max_lag=5
        )
        logger.info(f"VAR causality test completed")
    except Exception as e:
        logger.error(f"VAR causality test failed: {e}")
        var_result = {}
    
    return {
        'granger': granger_result,
        'var': var_result
    }

def demonstrate_feature_engineering(data):
    """Demonstrate feature engineering capabilities."""
    logger = logging.getLogger(__name__)
    logger.info("=== Feature Engineering ===")
    
    feature_engineer = FeatureEngineer()
    
    # Test with first symbol
    symbol = list(data.keys())[0]
    df = data[symbol].copy()
    
    logger.info(f"Original data shape for {symbol}: {df.shape}")
    
    # Create features
    try:
        features_df = feature_engineer.create_features(df)
        logger.info(f"Features created: {features_df.shape}")
        logger.info(f"Feature columns: {list(features_df.columns)}")
        
        # Show some statistics
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        logger.info(f"Numeric features: {len(numeric_cols)}")
        
        return features_df
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        return pd.DataFrame()

def demonstrate_strategy_backtesting(data):
    """Demonstrate strategy backtesting."""
    logger = logging.getLogger(__name__)
    logger.info("=== Strategy Backtesting ===")
    
    symbols = list(data.keys())
    
    # Initialize strategies
    te_strategy = TransferEntropyStrategy(
        symbols=symbols,
        initial_capital=100000.0,
        te_threshold=0.05,
        confidence_threshold=0.6,
        lookback_window=100,
        rebalance_frequency=50
    )
    
    # Baseline strategies
    buy_hold = BuyHoldStrategy(symbol=symbols[0], initial_capital=100000.0)
    random_strategy = RandomStrategy(symbols=symbols, initial_capital=100000.0)
    
    # Initialize backtest engine
    backtest_engine = BacktestEngine(initial_capital=100000.0)
    
    try:
        # Run comparison
        results = backtest_engine.compare_strategies(
            strategies=[te_strategy, buy_hold, random_strategy],
            data=data
        )
        
        # Display results
        comparison_df = results['comparison']
        logger.info("Strategy Comparison Results:")
        logger.info(f"\n{comparison_df}")
        
        # Detailed results for TE strategy
        te_results = results['individual_results']['Transfer_Entropy_Strategy']
        logger.info(f"\nTransfer Entropy Strategy Details:")
        logger.info(f"Final Value: ${te_results['final_value']:,.2f}")
        logger.info(f"Total Return: {te_results['total_return']:.2%}")
        logger.info(f"Number of Trades: {te_results['num_trades']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Backtesting failed: {e}")
        return {}

def main():
    """Main execution function."""
    print("🚀 HFT Engine v3 - Transfer Entropy Based Trading System")
    print("=" * 60)
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting HFT Engine v3")
    
    try:
        # Load data
        print("\n📊 Loading Data...")
        data = load_sample_data()
        
        if not data:
            logger.error("No data available for analysis")
            return
        
        logger.info(f"Loaded data for symbols: {list(data.keys())}")
        
        # Transfer Entropy Analysis
        print("\n🔬 Transfer Entropy Analysis...")
        te_results = demonstrate_transfer_entropy_analysis(data)
        
        # Causality Testing
        print("\n📈 Statistical Causality Testing...")
        causality_results = demonstrate_causality_testing(data)
        
        # Feature Engineering
        print("\n⚙️ Feature Engineering...")
        features = demonstrate_feature_engineering(data)
        
        # Strategy Backtesting
        print("\n🎯 Strategy Backtesting...")
        backtest_results = demonstrate_strategy_backtesting(data)
        
        # Summary
        print("\n" + "=" * 60)
        print("✅ Analysis Complete!")
        
        if te_results:
            print(f"🔬 Transfer Entropy relationships found: {len(te_results)}")
        
        if not features.empty:
            print(f"⚙️ Features engineered: {features.shape[1]} features")
        
        if backtest_results:
            print(f"🎯 Strategy backtesting completed")
            
            # Best performing strategy
            if 'comparison' in backtest_results:
                comparison_df = backtest_results['comparison']
                best_strategy = comparison_df.loc[comparison_df['Total Return'].idxmax()]
                print(f"🏆 Best Strategy: {best_strategy.name} ({best_strategy['Total Return']:.2%} return)")
        
        print("\n💡 Next Steps:")
        print("   1. Explore detailed results in the notebooks/")
        print("   2. Adjust strategy parameters for optimization")
        print("   3. Add more sophisticated ML models")
        print("   4. Implement real-time data feeds")
        
        logger.info("HFT Engine v3 execution completed successfully")
        
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        raise

if __name__ == "__main__":
    main()
