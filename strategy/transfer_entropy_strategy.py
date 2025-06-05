"""
Transfer Entropy Lead-Lag Strategy

Main trading strategy using Transfer Entropy for lead-lag relationships:
- Identifies lead-lag pairs using Transfer Entropy analysis
- Generates trading signals based on causality strength
- Incorporates regime detection and statistical validation
- Uses machine learning models for signal refinement
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from ..data_synchronization import DataSynchronizer
from ..feature_engineering import FeatureEngineer
from ..statistical_analysis import TransferEntropyAnalyzer, CausalityTester, RegimeDetector
from ..learning import DataPreparator, ModelTrainer, TransferEntropyModel
from ..benchmark import BaseBenchmarkStrategy, TradeSignal, SignalType

@dataclass
class LeadLagPair:
    """Container for lead-lag relationship information."""
    leader: str
    follower: str
    te_value: float
    te_pvalue: float
    confidence: float
    lag: int
    strength: str  # 'strong', 'moderate', 'weak'
    regime: str
    last_updated: int

@dataclass
class TradingSignal:
    """Enhanced trading signal with causality information."""
    timestamp: int
    symbol: str
    signal: SignalType
    confidence: float
    price: float
    causality_strength: float
    lead_lag_info: Dict[str, Any]
    ml_prediction: Optional[float] = None
    regime: Optional[str] = None
    metadata: Dict[str, Any] = None

class TransferEntropyStrategy(BaseBenchmarkStrategy):
    """
    Main Transfer Entropy-based trading strategy.
    
    Uses Transfer Entropy to identify lead-lag relationships between
    cryptocurrency pairs and generates trading signals based on
    causality patterns.
    """
    
    def __init__(self,
                 symbols: List[str],
                 initial_capital: float = 100000.0,
                 transaction_cost: float = 0.001,
                 te_threshold: float = 0.1,
                 confidence_threshold: float = 0.6,
                 lookback_window: int = 100,
                 rebalance_frequency: int = 50,
                 max_positions: int = 3,
                 position_size: float = 0.3,
                 use_ml: bool = True,
                 use_regime_detection: bool = True):
        """
        Initialize Transfer Entropy strategy.
        
        Args:
            symbols: List of cryptocurrency symbols to trade
            initial_capital: Initial capital amount
            transaction_cost: Transaction cost as fraction
            te_threshold: Minimum Transfer Entropy value for signal
            confidence_threshold: Minimum confidence for trades
            lookback_window: Window for TE calculation
            rebalance_frequency: How often to recalculate TE relationships
            max_positions: Maximum number of concurrent positions
            position_size: Position size as fraction of capital
            use_ml: Whether to use ML models for signal refinement
            use_regime_detection: Whether to use regime detection
        """
        super().__init__('Transfer_Entropy_Strategy', initial_capital, transaction_cost)
        
        # Strategy parameters
        self.symbols = symbols
        self.te_threshold = te_threshold
        self.confidence_threshold = confidence_threshold
        self.lookback_window = lookback_window
        self.rebalance_frequency = rebalance_frequency
        self.max_positions = max_positions
        self.position_size = position_size
        self.use_ml = use_ml
        self.use_regime_detection = use_regime_detection
        
        # Initialize components
        self.synchronizer = DataSynchronizer()
        self.feature_engineer = FeatureEngineer()
        self.te_analyzer = TransferEntropyAnalyzer()
        self.causality_tester = CausalityTester()
        self.regime_detector = RegimeDetector() if use_regime_detection else None
        self.data_preparator = DataPreparator() if use_ml else None
        self.model_trainer = ModelTrainer() if use_ml else None
        
        # State variables
        self.price_history: Dict[str, List[float]] = {symbol: [] for symbol in symbols}
        self.lead_lag_pairs: List[LeadLagPair] = []
        self.current_regime = 'normal'
        self.last_rebalance = 0
        self.step_count = 0
        self.ml_models = {}
        
        # Performance tracking
        self.signal_history = []
        self.performance_metrics = {}
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    def initialize(self, start_timestamp: int):
        """Initialize strategy at start of backtesting."""
        self.logger.info("Initializing Transfer Entropy Strategy")
        self.last_rebalance = start_timestamp
        self.step_count = 0
    
    def generate_signals(self, 
                        timestamp: int,
                        market_data: Dict[str, Any]) -> List[TradeSignal]:
        """Generate trading signals based on Transfer Entropy analysis."""
        self.step_count += 1
        
        # Update price history
        self._update_price_history(market_data)
        
        # Check if we need to rebalance (recalculate TE relationships)
        if (self.step_count % self.rebalance_frequency == 0 or 
            timestamp - self.last_rebalance > self.rebalance_frequency):
            self._rebalance_relationships(timestamp)
            self.last_rebalance = timestamp
        
        # Generate signals based on current lead-lag relationships
        signals = self._generate_te_signals(timestamp, market_data)
        
        # Filter signals based on confidence and position limits
        filtered_signals = self._filter_signals(signals)
        
        # Store signals for analysis
        self.signal_history.extend(filtered_signals)
        
        return filtered_signals
    
    def _update_price_history(self, market_data: Dict[str, Any]):
        """Update price history for all symbols."""
        for symbol in self.symbols:
            if symbol in market_data:
                price = market_data[symbol].get('mid_price', 0)
                if price > 0:
                    self.price_history[symbol].append(price)
                    
                    # Keep only lookback window
                    if len(self.price_history[symbol]) > self.lookback_window * 2:
                        self.price_history[symbol] = self.price_history[symbol][-self.lookback_window * 2:]
    
    def _rebalance_relationships(self, timestamp: int):
        """Recalculate Transfer Entropy relationships between symbols."""
        self.logger.info(f"Rebalancing TE relationships at timestamp {timestamp}")
        
        # Check if we have enough data
        min_data_length = min(len(hist) for hist in self.price_history.values() if hist)
        if min_data_length < self.lookback_window:
            self.logger.warning(f"Insufficient data for TE calculation: {min_data_length} < {self.lookback_window}")
            return
        
        # Convert price history to synchronized DataFrame
        price_df = pd.DataFrame({
            symbol: hist[-self.lookback_window:] 
            for symbol, hist in self.price_history.items() 
            if len(hist) >= self.lookback_window
        })
        
        if price_df.empty:
            return
        
        # Calculate returns
        returns_df = price_df.pct_change().dropna()
        
        # Detect current regime if enabled
        if self.use_regime_detection and self.regime_detector:
            try:
                regime_results = self.regime_detector.detect_hmm_regimes(
                    returns_df.mean(axis=1), n_regimes=3
                )
                self.current_regime = regime_results['current_regime']
            except Exception as e:
                self.logger.warning(f"Regime detection failed: {e}")
                self.current_regime = 'normal'
        
        # Calculate Transfer Entropy for all pairs
        new_pairs = []
        
        for i, leader in enumerate(self.symbols):
            for j, follower in enumerate(self.symbols):
                if i != j and leader in returns_df.columns and follower in returns_df.columns:
                    try:
                        # Calculate Transfer Entropy
                        te_result = self.te_analyzer.calculate_transfer_entropy(
                            returns_df[leader].values,
                            returns_df[follower].values,
                            max_lag=5
                        )
                        
                        # Validate with statistical tests
                        causality_result = self.causality_tester.granger_causality_test(
                            returns_df[[leader, follower]],
                            max_lag=5
                        )
                        
                        # Determine relationship strength
                        te_value = te_result['transfer_entropy']
                        te_pvalue = te_result.get('p_value', 1.0)
                        
                        if te_value > self.te_threshold and te_pvalue < 0.05:
                            # Determine strength based on TE value
                            if te_value > 0.3:
                                strength = 'strong'
                                confidence = min(0.9, te_value * 2)
                            elif te_value > 0.15:
                                strength = 'moderate'
                                confidence = min(0.7, te_value * 3)
                            else:
                                strength = 'weak'
                                confidence = min(0.5, te_value * 5)
                            
                            pair = LeadLagPair(
                                leader=leader,
                                follower=follower,
                                te_value=te_value,
                                te_pvalue=te_pvalue,
                                confidence=confidence,
                                lag=te_result.get('optimal_lag', 1),
                                strength=strength,
                                regime=self.current_regime,
                                last_updated=timestamp
                            )
                            new_pairs.append(pair)
                    
                    except Exception as e:
                        self.logger.warning(f"Failed to calculate TE for {leader}->{follower}: {e}")
        
        # Update lead-lag pairs
        self.lead_lag_pairs = new_pairs
        
        # Train ML models if enabled
        if self.use_ml and self.data_preparator and self.model_trainer:
            self._train_ml_models(returns_df)
        
        self.logger.info(f"Found {len(self.lead_lag_pairs)} significant lead-lag relationships")
    
    def _train_ml_models(self, returns_df: pd.DataFrame):
        """Train ML models for signal refinement."""
        try:
            # Prepare features and targets
            features_df = self.feature_engineer.create_features(returns_df)
            
            # Create targets for each symbol
            targets = {}
            for symbol in self.symbols:
                if symbol in returns_df.columns:
                    targets[symbol] = self.data_preparator.create_target_variables(
                        returns_df[symbol], 
                        prediction_horizon=1,
                        target_type='direction'
                    )
            
            # Train models for each symbol
            for symbol in self.symbols:
                if symbol in targets and not targets[symbol].empty:
                    try:
                        # Prepare data
                        X, y = self.data_preparator.prepare_features_targets(
                            features_df, targets[symbol]
                        )
                        
                        if len(X) > 50:  # Minimum data requirement
                            # Train model
                            model = self.model_trainer.train_model(
                                X, y, 
                                model_type='random_forest',
                                use_ensemble=True
                            )
                            self.ml_models[symbol] = model
                            
                    except Exception as e:
                        self.logger.warning(f"Failed to train ML model for {symbol}: {e}")
        
        except Exception as e:
            self.logger.warning(f"ML model training failed: {e}")
    
    def _generate_te_signals(self, 
                           timestamp: int,
                           market_data: Dict[str, Any]) -> List[TradingSignal]:
        """Generate trading signals based on lead-lag relationships."""
        signals = []
        
        for pair in self.lead_lag_pairs:
            leader = pair.leader
            follower = pair.follower
            
            # Check if we have current data for both symbols
            if (leader not in market_data or follower not in market_data or
                leader not in self.price_history or follower not in self.price_history):
                continue
            
            # Get recent price movements
            leader_prices = self.price_history[leader][-10:]  # Last 10 prices
            follower_prices = self.price_history[follower][-10:]
            
            if len(leader_prices) < 5 or len(follower_prices) < 5:
                continue
            
            # Calculate recent returns
            leader_returns = np.diff(leader_prices) / leader_prices[:-1]
            follower_returns = np.diff(follower_prices) / follower_prices[:-1]
            
            # Generate signal based on leader movement
            if len(leader_returns) >= pair.lag:
                # Look at leader's movement with appropriate lag
                lagged_leader_return = leader_returns[-(pair.lag + 1)]
                current_leader_return = leader_returns[-1]
                
                # Generate signal for follower based on leader's movement
                signal_strength = abs(lagged_leader_return) * pair.confidence
                
                if signal_strength > 0.01:  # Minimum signal strength
                    # Determine signal direction
                    if lagged_leader_return > 0:
                        signal_type = SignalType.BUY
                    else:
                        signal_type = SignalType.SELL
                    
                    # Adjust confidence based on various factors
                    confidence = pair.confidence * signal_strength
                    
                    # Regime adjustment
                    if pair.regime != self.current_regime:
                        confidence *= 0.7  # Reduce confidence for regime mismatch
                    
                    # ML model adjustment if available
                    ml_prediction = None
                    if self.use_ml and follower in self.ml_models:
                        try:
                            # Prepare features for prediction
                            recent_data = pd.DataFrame({
                                leader: leader_prices[-20:],
                                follower: follower_prices[-20:]
                            }).pct_change().dropna()
                            
                            if len(recent_data) >= 10:
                                features = self.feature_engineer.create_features(recent_data)
                                if not features.empty:
                                    ml_prediction = self.ml_models[follower].predict(
                                        features.iloc[-1:].values
                                    )[0]
                                    
                                    # Adjust confidence based on ML prediction
                                    if (signal_type == SignalType.BUY and ml_prediction > 0.5) or \
                                       (signal_type == SignalType.SELL and ml_prediction < 0.5):
                                        confidence *= 1.2  # Boost confidence
                                    else:
                                        confidence *= 0.8  # Reduce confidence
                        
                        except Exception as e:
                            self.logger.debug(f"ML prediction failed for {follower}: {e}")
                    
                    # Create enhanced signal
                    if confidence >= self.confidence_threshold:
                        signal = TradingSignal(
                            timestamp=timestamp,
                            symbol=follower,
                            signal=signal_type,
                            confidence=min(1.0, confidence),
                            price=market_data[follower].get('mid_price', 0),
                            causality_strength=pair.te_value,
                            lead_lag_info={
                                'leader': leader,
                                'lag': pair.lag,
                                'te_value': pair.te_value,
                                'te_pvalue': pair.te_pvalue,
                                'strength': pair.strength
                            },
                            ml_prediction=ml_prediction,
                            regime=self.current_regime,
                            metadata={
                                'strategy': 'transfer_entropy',
                                'leader_return': lagged_leader_return,
                                'signal_strength': signal_strength,
                                'pair_age': timestamp - pair.last_updated
                            }
                        )
                        signals.append(signal)
        
        return signals
    
    def _filter_signals(self, signals: List[TradingSignal]) -> List[TradeSignal]:
        """Filter and convert signals to standard format."""
        if not signals:
            return []
        
        # Sort by confidence
        signals.sort(key=lambda s: s.confidence, reverse=True)
        
        # Filter by position limits
        current_positions = len(self.positions)
        available_positions = self.max_positions - current_positions
        
        # Convert to standard TradeSignal format
        filtered_signals = []
        symbols_traded = set()
        
        for signal in signals:
            # Skip if we already have a signal for this symbol
            if signal.symbol in symbols_traded:
                continue
            
            # Skip if we would exceed position limits
            if (signal.signal != SignalType.HOLD and 
                signal.symbol not in self.positions and 
                len(filtered_signals) >= available_positions):
                continue
            
            # Convert to standard format
            trade_signal = TradeSignal(
                timestamp=signal.timestamp,
                symbol=signal.symbol,
                signal=signal.signal,
                confidence=signal.confidence,
                price=signal.price,
                metadata={
                    'causality_strength': signal.causality_strength,
                    'lead_lag_info': signal.lead_lag_info,
                    'ml_prediction': signal.ml_prediction,
                    'regime': signal.regime,
                    **signal.metadata
                }
            )
            
            filtered_signals.append(trade_signal)
            symbols_traded.add(signal.symbol)
        
        return filtered_signals
    
    def update(self, timestamp: int, market_data: Dict[str, Any], portfolio_state: Any):
        """Update strategy state after each timestep."""
        # Update position values
        self.update_positions(market_data, timestamp)
        
        # Update performance metrics periodically
        if self.step_count % 100 == 0:
            self._update_performance_metrics()
    
    def _update_performance_metrics(self):
        """Update strategy performance metrics."""
        if not self.signal_history:
            return
        
        # Analyze signal quality
        signals_df = pd.DataFrame([
            {
                'timestamp': s.timestamp,
                'symbol': s.symbol,
                'signal': s.signal.value,
                'confidence': s.confidence,
                'causality_strength': s.metadata.get('causality_strength', 0),
                'regime': s.metadata.get('regime', 'unknown')
            }
            for s in self.signal_history[-1000:]  # Last 1000 signals
        ])
        
        # Calculate metrics
        self.performance_metrics.update({
            'total_signals': len(signals_df),
            'avg_confidence': signals_df['confidence'].mean(),
            'avg_causality_strength': signals_df['causality_strength'].mean(),
            'signal_distribution': signals_df['signal'].value_counts().to_dict(),
            'regime_distribution': signals_df['regime'].value_counts().to_dict(),
            'active_pairs': len(self.lead_lag_pairs),
            'current_regime': self.current_regime
        })
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get detailed strategy information."""
        return {
            'strategy_name': self.name,
            'symbols': self.symbols,
            'te_threshold': self.te_threshold,
            'confidence_threshold': self.confidence_threshold,
            'lookback_window': self.lookback_window,
            'rebalance_frequency': self.rebalance_frequency,
            'active_pairs': len(self.lead_lag_pairs),
            'current_regime': self.current_regime,
            'use_ml': self.use_ml,
            'use_regime_detection': self.use_regime_detection,
            'performance_metrics': self.performance_metrics,
            'lead_lag_pairs': [
                {
                    'leader': pair.leader,
                    'follower': pair.follower,
                    'te_value': pair.te_value,
                    'confidence': pair.confidence,
                    'strength': pair.strength,
                    'lag': pair.lag
                }
                for pair in self.lead_lag_pairs
            ]
        }
    
    def get_detailed_performance(self) -> Dict[str, Any]:
        """Get detailed performance analysis."""
        base_performance = self.get_performance_summary()
        
        # Add strategy-specific metrics
        strategy_metrics = {
            'causality_analysis': {
                'active_relationships': len(self.lead_lag_pairs),
                'relationship_strength': {
                    'strong': len([p for p in self.lead_lag_pairs if p.strength == 'strong']),
                    'moderate': len([p for p in self.lead_lag_pairs if p.strength == 'moderate']),
                    'weak': len([p for p in self.lead_lag_pairs if p.strength == 'weak'])
                },
                'avg_te_value': np.mean([p.te_value for p in self.lead_lag_pairs]) if self.lead_lag_pairs else 0,
                'avg_confidence': np.mean([p.confidence for p in self.lead_lag_pairs]) if self.lead_lag_pairs else 0
            },
            'signal_quality': {
                'total_signals': len(self.signal_history),
                'avg_signal_confidence': np.mean([s.confidence for s in self.signal_history]) if self.signal_history else 0,
                'regime_consistency': self.current_regime
            },
            'ml_performance': {
                'models_trained': len(self.ml_models),
                'symbols_with_ml': list(self.ml_models.keys())
            } if self.use_ml else {},
            'regime_detection': {
                'current_regime': self.current_regime,
                'regime_enabled': self.use_regime_detection
            }
        }
        
        return {
            **base_performance,
            'strategy_specific': strategy_metrics
        }
