"""
Main execution pipeline for crypto HFT trading strategy.
Orchestrates data loading, feature extraction, model training, and signal generation.
"""

import pandas as pd
import numpy as np
import yaml
import logging
from datetime import datetime
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from data_processing.data_loader import OrderBookDataLoader, DataPreprocessor
from data_processing.feature_extractor import OrderBookFeatureExtractor
from models.ml_models import MLModelManager
from models.model_selection import RollingModelSelection
from strategy.signal_generator import CrossAssetSignalGenerator
from strategy.risk_manager import RiskManager
from utils.metrics import PerformanceMetrics
from utils.visualization import TradingVisualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CryptoTradingPipeline:
    """
    Main pipeline for cryptocurrency high-frequency trading strategy.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the trading pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize components
        self.data_loader = OrderBookDataLoader(self.config['data']['raw_data_path'])
        self.feature_extractor = OrderBookFeatureExtractor(
            max_levels=self.config['features']['depth_levels']
        )
        self.model_manager = MLModelManager(self.config)
        self.signal_generator = CrossAssetSignalGenerator(
            self.config,
            signal_threshold=self.config['strategy']['signal_threshold'],
            confidence_threshold=self.config['strategy']['confidence_threshold']
        )
        self.risk_manager = RiskManager(self.config)
        self.performance_metrics = PerformanceMetrics()
        self.visualizer = TradingVisualizer()
        
        # Data storage
        self.raw_data = {}
        self.synchronized_data = None
        self.features_data = None
        self.targets_data = None
        self.predictions = {}
        
    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def load_data(self) -> None:
        """Load raw order book data for all symbols."""
        logger.info("Loading raw order book data")
        
        symbols = self.config['data']['symbols']
        self.raw_data = self.data_loader.load_all_symbols(symbols)
        
        for symbol, df in self.raw_data.items():
            logger.info(f"Loaded {len(df)} records for {symbol}")
            logger.info(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    def synchronize_data(self) -> None:
        """Synchronize data across all symbols."""
        logger.info("Synchronizing data across symbols")
        
        self.synchronized_data = self.data_loader.synchronize_data(
            self.raw_data,
            time_window=0.1  # 100ms synchronization window
        )
        
        logger.info(f"Synchronized data shape: {self.synchronized_data.shape}")
        logger.info(f"Synchronized date range: {self.synchronized_data['datetime'].min()} to {self.synchronized_data['datetime'].max()}")
    
    def extract_features(self) -> None:
        """Extract features from synchronized data."""
        logger.info("Extracting features from order book data")
        
        symbols = self.config['data']['symbols']
        if len(symbols) != 2:
            raise ValueError("Exactly 2 symbols required for cross-asset analysis")
        
        symbol1, symbol2 = symbols[0], symbols[1]
        
        # Extract features
        self.features_data = self.feature_extractor.extract_all_features(
            self.synchronized_data,
            symbol1=symbol1,
            symbol2=symbol2,
            window_size=100  # 10 seconds at 100ms frequency
        )
        
        logger.info(f"Extracted features shape: {self.features_data.shape}")
        logger.info(f"Number of features: {len(self.features_data.columns) - 1}")  # -1 for timestamp
        
        # Create target variables for both symbols
        horizons = self.config['models']['prediction_horizons']
        
        targets_list = []
        for symbol in symbols:
            targets = self.feature_extractor.create_target_variables(
                self.synchronized_data,
                symbol=symbol,
                horizons=horizons
            )
            targets_list.append(targets)
        
        # Merge all targets
        self.targets_data = targets_list[0]
        for targets in targets_list[1:]:
            self.targets_data = self.targets_data.merge(targets, on='timestamp', how='inner')
        
        logger.info(f"Created targets shape: {self.targets_data.shape}")
    
    def train_models(self, target_symbol: str = None) -> None:
        """Train machine learning models."""
        if target_symbol is None:
            target_symbol = self.config['data']['symbols'][0]  # Default to first symbol
        
        logger.info(f"Training models for {target_symbol}")
        
        # Prepare data for training
        merged_data = self.features_data.merge(self.targets_data, on='timestamp', how='inner')
        merged_data = merged_data.dropna()
        
        # Select target variable (direction prediction)
        target_col = f'target_{target_symbol}_1.0s_direction'  # 1 second horizon
        if target_col not in merged_data.columns:
            raise ValueError(f"Target column {target_col} not found")
        
        # Prepare features and targets
        feature_cols = [col for col in merged_data.columns 
                       if col not in ['timestamp'] and not col.startswith('target_')]
        
        X = merged_data[feature_cols]
        y = merged_data[target_col]
        
        # Split data (time series split)
        split_idx = int(len(merged_data) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")
        
        # Train all models
        results = self.model_manager.train_all_models(
            X_train, y_train, X_test, y_test, model_type='classification'
        )
        
        logger.info("Model training results:")
        logger.info(f"\n{results[['model', 'accuracy', 'precision', 'recall', 'f1_score']].to_string()}")
        
        return results
    
    def run_rolling_selection(self, target_symbol: str = None) -> pd.DataFrame:
        """Run rolling model selection."""
        if target_symbol is None:
            target_symbol = self.config['data']['symbols'][0]
        
        logger.info(f"Running rolling model selection for {target_symbol}")
        
        # Initialize rolling selection
        rolling_selection = RollingModelSelection(
            self.config,
            window_size_seconds=1800,  # 30 minutes
            prediction_horizon_seconds=1.0,  # 1 second
            update_frequency_seconds=300  # 5 minutes
        )
        
        # Prepare rolling data
        rolling_data = rolling_selection.prepare_rolling_data(
            self.features_data,
            self.targets_data,
            target_symbol
        )
        
        # Run rolling selection
        results = rolling_selection.run_rolling_selection(rolling_data)
        
        # Get performance summary
        summary = rolling_selection.get_model_performance_summary()
        logger.info("Rolling model performance summary:")
        logger.info(f"\n{summary.to_string()}")
        
        return results
    
    def generate_trading_signals(self, target_symbol: str = None) -> list:
        """Generate real-time trading signals."""
        if target_symbol is None:
            target_symbol = self.config['data']['symbols'][0]
        
        logger.info(f"Generating trading signals for {target_symbol}")
        
        # Use the best performing model for signal generation
        best_model = 'RandomForest'  # This should come from model selection results
        
        signals = []
        
        # Simulate real-time signal generation
        merged_data = self.features_data.merge(self.targets_data, on='timestamp', how='inner')
        merged_data = merged_data.dropna()
        
        # Take last 100 samples for simulation
        simulation_data = merged_data.tail(100)
        
        for idx, row in simulation_data.iterrows():
            # Prepare features
            feature_cols = [col for col in row.index 
                           if col not in ['timestamp'] and not col.startswith('target_')]
            features = row[feature_cols].values.reshape(1, -1)
            
            # Generate prediction
            try:
                prediction = self.model_manager.predict(best_model, pd.DataFrame([row[feature_cols]]))
                probabilities = self.model_manager.predict_proba(best_model, pd.DataFrame([row[feature_cols]]))
                
                # Create market data dict
                market_data = {
                    f'{target_symbol}_best_bid': row.get(f'{target_symbol}_best_bid', np.nan),
                    f'{target_symbol}_best_ask': row.get(f'{target_symbol}_best_ask', np.nan),
                    f'{target_symbol}_mid': row.get(f'{target_symbol}_mid', np.nan)
                }
                
                # Generate signal
                signal = self.signal_generator.generate_cross_asset_signal(
                    primary_predictions={best_model: prediction[0]},
                    primary_probabilities={best_model: probabilities[0]},
                    market_data=market_data,
                    timestamp=row['timestamp'],
                    target_symbol=target_symbol
                )
                
                signals.append(signal)
                
            except Exception as e:
                logger.error(f"Error generating signal: {e}")
                continue
        
        logger.info(f"Generated {len(signals)} trading signals")
        
        # Calculate signal statistics
        signal_stats = self.signal_generator.get_signal_statistics()
        logger.info(f"Signal statistics: {signal_stats}")
        
        return signals
    
    def backtest_strategy(self, signals: list) -> dict:
        """Backtest the trading strategy."""
        logger.info("Running strategy backtest")
        
        # Initialize risk manager
        self.risk_manager = RiskManager(self.config, initial_capital=100000.0)
        
        # Simulate trading
        for signal in signals:
            if signal['signal_type'] != 'HOLD':
                symbol = signal['symbol']
                current_price = signal.get('market_conditions', {}).get(f'{symbol}_mid', 100.0)
                
                # Calculate position size
                position_size = self.risk_manager.calculate_position_size(
                    signal_strength=signal['strength'],
                    current_price=current_price,
                    volatility=0.02  # 2% daily volatility assumption
                )
                
                if signal['direction'] == -1:
                    position_size = -position_size
                
                # Check if we should trade
                if self.risk_manager.check_risk_limits(symbol, position_size, current_price)['approved']:
                    # Open position
                    self.risk_manager.open_position(
                        symbol=symbol,
                        size=position_size,
                        entry_price=current_price,
                        signal_info=signal,
                        timestamp=signal['timestamp']
                    )
        
        # Get portfolio summary
        portfolio_summary = self.risk_manager.get_portfolio_summary()
        
        logger.info("Backtest results:")
        logger.info(f"Total return: {portfolio_summary['total_return']:.2%}")
        logger.info(f"Number of trades: {portfolio_summary['num_trades']}")
        logger.info(f"Win rate: {portfolio_summary['win_rate']:.2%}")
        logger.info(f"Max drawdown: {portfolio_summary['max_drawdown']:.2%}")
        
        return portfolio_summary
    
    def run_complete_pipeline(self) -> dict:
        """Run the complete trading pipeline."""
        logger.info("Starting complete crypto trading pipeline")
        
        try:
            # Step 1: Load and synchronize data
            self.load_data()
            self.synchronize_data()
            
            # Step 2: Extract features
            self.extract_features()
            
            # Step 3: Train models
            model_results = self.train_models()
            
            # Step 4: Run rolling model selection
            rolling_results = self.run_rolling_selection()
            
            # Step 5: Generate trading signals
            signals = self.generate_trading_signals()
            
            # Step 6: Backtest strategy
            backtest_results = self.backtest_strategy(signals)
            
            # Step 7: Generate visualizations
            self.generate_visualizations(signals, backtest_results)
            
            logger.info("Pipeline completed successfully")
            
            return {
                'model_results': model_results,
                'rolling_results': rolling_results,
                'signals': signals,
                'backtest_results': backtest_results
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def generate_visualizations(self, signals: list, backtest_results: dict) -> None:
        """Generate visualization plots."""
        logger.info("Generating visualizations")
        
        # Plot price data
        if self.synchronized_data is not None:
            self.visualizer.plot_price_series(
                self.synchronized_data,
                self.config['data']['symbols']
            )
        
        # Plot signals
        if signals:
            self.visualizer.plot_trading_signals(signals)
        
        # Plot performance
        self.visualizer.plot_performance_metrics(backtest_results)
        
        logger.info("Visualizations saved")

if __name__ == "__main__":
    # Run the complete pipeline
    pipeline = CryptoTradingPipeline()
    results = pipeline.run_complete_pipeline()
    
    print("Pipeline execution completed!")
    print(f"Final results: {results['backtest_results']}")
