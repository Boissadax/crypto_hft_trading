"""
Trading signal generation and strategy implementation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TradingSignalGenerator:
    """
    Generates trading signals based on model predictions and market conditions.
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 signal_threshold: float = 0.6,
                 confidence_threshold: float = 0.7):
        """
        Initialize the signal generator.
        
        Args:
            config: Configuration dictionary
            signal_threshold: Minimum probability threshold for signal generation
            confidence_threshold: Minimum confidence threshold for signal
        """
        self.config = config
        self.signal_threshold = signal_threshold
        self.confidence_threshold = confidence_threshold
        
        # Signal history
        self.signal_history = []
        self.last_signal_time = {}
        
    def generate_signal(self,
                       predictions: Dict[str, float],
                       probabilities: Dict[str, np.ndarray],
                       market_data: Dict[str, float],
                       timestamp: float) -> Dict[str, Any]:
        """
        Generate trading signal based on model predictions.
        
        Args:
            predictions: Dictionary of model predictions
            probabilities: Dictionary of prediction probabilities
            market_data: Current market data
            timestamp: Current timestamp
            
        Returns:
            Dictionary with signal information
        """
        signal = {
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp),
            'signal_type': 'HOLD',
            'confidence': 0.0,
            'symbol': None,
            'direction': None,
            'strength': 0.0,
            'models_agreement': 0.0,
            'market_conditions': {}
        }
        
        # Analyze model consensus
        model_votes = []
        model_confidences = []
        
        for model_name, pred in predictions.items():
            if model_name in probabilities:
                prob = probabilities[model_name]
                if len(prob) > 1:  # Binary classification
                    confidence = max(prob)
                    model_confidences.append(confidence)
                    
                    # Vote based on prediction and confidence
                    if confidence >= self.confidence_threshold:
                        if pred == 1 and prob[1] >= self.signal_threshold:
                            model_votes.append(1)  # Buy signal
                        elif pred == 0 and prob[0] >= self.signal_threshold:
                            model_votes.append(-1)  # Sell signal
                        else:
                            model_votes.append(0)  # Hold
                    else:
                        model_votes.append(0)  # Hold due to low confidence
        
        if not model_votes:
            return signal
        
        # Calculate consensus
        buy_votes = sum(1 for v in model_votes if v == 1)
        sell_votes = sum(1 for v in model_votes if v == -1)
        total_votes = len(model_votes)
        
        models_agreement = max(buy_votes, sell_votes) / total_votes
        avg_confidence = np.mean(model_confidences) if model_confidences else 0.0
        
        # Generate signal based on consensus
        if buy_votes > sell_votes and models_agreement >= 0.6:
            signal_type = 'BUY'
            direction = 1
            strength = buy_votes / total_votes
        elif sell_votes > buy_votes and models_agreement >= 0.6:
            signal_type = 'SELL'
            direction = -1
            strength = sell_votes / total_votes
        else:
            signal_type = 'HOLD'
            direction = 0
            strength = 0.0
        
        # Update signal
        signal.update({
            'signal_type': signal_type,
            'confidence': avg_confidence,
            'direction': direction,
            'strength': strength,
            'models_agreement': models_agreement,
            'model_votes': model_votes,
            'market_conditions': self._analyze_market_conditions(market_data)
        })
        
        # Store signal history
        self.signal_history.append(signal.copy())
        
        return signal
    
    def _analyze_market_conditions(self, market_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze current market conditions.
        
        Args:
            market_data: Current market data
            
        Returns:
            Dictionary with market condition analysis
        """
        conditions = {
            'volatility_regime': 'normal',
            'liquidity_status': 'normal',
            'spread_status': 'normal'
        }
        
        # Analyze spreads
        for symbol in ['ETH_EUR', 'XBT_EUR']:
            bid_key = f'{symbol}_best_bid'
            ask_key = f'{symbol}_best_ask'
            
            if bid_key in market_data and ask_key in market_data:
                bid = market_data[bid_key]
                ask = market_data[ask_key]
                
                if not pd.isna(bid) and not pd.isna(ask):
                    spread = ask - bid
                    mid_price = (bid + ask) / 2
                    rel_spread = spread / mid_price if mid_price > 0 else np.nan
                    
                    conditions[f'{symbol}_spread'] = spread
                    conditions[f'{symbol}_rel_spread'] = rel_spread
                    
                    # Classify spread status
                    if not pd.isna(rel_spread):
                        if rel_spread > 0.002:  # 0.2%
                            conditions['spread_status'] = 'wide'
                        elif rel_spread < 0.0005:  # 0.05%
                            conditions['spread_status'] = 'tight'
        
        return conditions
    
    def should_trade(self, 
                    signal: Dict[str, Any],
                    current_positions: Dict[str, float],
                    risk_limits: Dict[str, float]) -> bool:
        """
        Determine if a trade should be executed based on signal and risk management.
        
        Args:
            signal: Generated trading signal
            current_positions: Current positions by symbol
            risk_limits: Risk limits and constraints
            
        Returns:
            Boolean indicating whether to trade
        """
        # Don't trade on HOLD signals
        if signal['signal_type'] == 'HOLD':
            return False
        
        # Check minimum confidence
        if signal['confidence'] < self.confidence_threshold:
            return False
        
        # Check minimum model agreement
        if signal['models_agreement'] < 0.6:
            return False
        
        # Check time since last signal (avoid overtrading)
        min_interval = self.config.get('execution', {}).get('min_trade_interval_seconds', 1.0)
        current_time = signal['timestamp']
        
        if signal['symbol'] in self.last_signal_time:
            time_since_last = current_time - self.last_signal_time[signal['symbol']]
            if time_since_last < min_interval:
                return False
        
        # Check position limits
        symbol = signal.get('symbol')
        if symbol and symbol in current_positions:
            current_pos = current_positions[symbol]
            max_pos = risk_limits.get('max_position_size', 0.1)
            
            if signal['direction'] == 1 and current_pos >= max_pos:
                return False  # Already at maximum long position
            elif signal['direction'] == -1 and current_pos <= -max_pos:
                return False  # Already at maximum short position
        
        # Check market conditions
        market_conditions = signal.get('market_conditions', {})
        spread_status = market_conditions.get('spread_status', 'normal')
        
        if spread_status == 'wide':
            # Avoid trading in wide spread conditions
            return False
        
        # Update last signal time
        if symbol:
            self.last_signal_time[symbol] = current_time
        
        return True
    
    def get_signal_statistics(self, 
                            lookback_hours: int = 24) -> Dict[str, Any]:
        """
        Get statistics on recent signals.
        
        Args:
            lookback_hours: Number of hours to look back
            
        Returns:
            Dictionary with signal statistics
        """
        if not self.signal_history:
            return {}
        
        # Filter recent signals
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
        recent_signals = [
            s for s in self.signal_history 
            if s['datetime'] >= cutoff_time
        ]
        
        if not recent_signals:
            return {}
        
        # Calculate statistics
        total_signals = len(recent_signals)
        buy_signals = sum(1 for s in recent_signals if s['signal_type'] == 'BUY')
        sell_signals = sum(1 for s in recent_signals if s['signal_type'] == 'SELL')
        hold_signals = sum(1 for s in recent_signals if s['signal_type'] == 'HOLD')
        
        avg_confidence = np.mean([s['confidence'] for s in recent_signals])
        avg_agreement = np.mean([s['models_agreement'] for s in recent_signals])
        avg_strength = np.mean([s['strength'] for s in recent_signals])
        
        return {
            'total_signals': total_signals,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'hold_signals': hold_signals,
            'buy_ratio': buy_signals / total_signals,
            'sell_ratio': sell_signals / total_signals,
            'hold_ratio': hold_signals / total_signals,
            'avg_confidence': avg_confidence,
            'avg_agreement': avg_agreement,
            'avg_strength': avg_strength
        }

class CrossAssetSignalGenerator(TradingSignalGenerator):
    """
    Generates signals based on cross-asset relationships and lead-lag effects.
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 signal_threshold: float = 0.6,
                 confidence_threshold: float = 0.7,
                 max_lag_seconds: float = 5.0):
        """
        Initialize the cross-asset signal generator.
        
        Args:
            config: Configuration dictionary
            signal_threshold: Minimum probability threshold for signal generation
            confidence_threshold: Minimum confidence threshold for signal
            max_lag_seconds: Maximum lag to consider for lead-lag relationships
        """
        super().__init__(config, signal_threshold, confidence_threshold)
        self.max_lag_seconds = max_lag_seconds
        self.price_history = {'ETH_EUR': [], 'XBT_EUR': []}
        
    def update_price_history(self, 
                           market_data: Dict[str, float],
                           timestamp: float):
        """
        Update price history for lead-lag analysis.
        
        Args:
            market_data: Current market data
            timestamp: Current timestamp
        """
        for symbol in ['ETH_EUR', 'XBT_EUR']:
            mid_key = f'{symbol}_mid'
            if mid_key in market_data and not pd.isna(market_data[mid_key]):
                self.price_history[symbol].append({
                    'timestamp': timestamp,
                    'price': market_data[mid_key]
                })
                
                # Keep only recent history (max_lag_seconds * 2)
                cutoff_time = timestamp - self.max_lag_seconds * 2
                self.price_history[symbol] = [
                    p for p in self.price_history[symbol] 
                    if p['timestamp'] >= cutoff_time
                ]
    
    def detect_lead_lag_signal(self, 
                              target_symbol: str,
                              lead_symbol: str) -> Dict[str, Any]:
        """
        Detect lead-lag signals between two symbols.
        
        Args:
            target_symbol: Symbol to generate signal for
            lead_symbol: Symbol that may lead the target
            
        Returns:
            Dictionary with lead-lag signal information
        """
        signal = {
            'has_signal': False,
            'direction': 0,
            'strength': 0.0,
            'lag_seconds': 0.0,
            'correlation': 0.0
        }
        
        # Check if we have enough history
        target_history = self.price_history.get(target_symbol, [])
        lead_history = self.price_history.get(lead_symbol, [])
        
        if len(target_history) < 10 or len(lead_history) < 10:
            return signal
        
        # Convert to DataFrames
        target_df = pd.DataFrame(target_history)
        lead_df = pd.DataFrame(lead_history)
        
        # Calculate returns
        target_df['return'] = target_df['price'].pct_change()
        lead_df['return'] = lead_df['price'].pct_change()
        
        # Test different lags
        best_correlation = 0.0
        best_lag = 0.0
        
        lag_range = np.arange(0.1, self.max_lag_seconds, 0.1)  # Test lags in 0.1s increments
        
        for lag in lag_range:
            # Shift lead returns by lag
            lead_shifted = lead_df.copy()
            lead_shifted['timestamp'] = lead_shifted['timestamp'] + lag
            
            # Merge on timestamp (approximate matching)
            merged = pd.merge_asof(
                target_df.sort_values('timestamp'),
                lead_shifted[['timestamp', 'return']].sort_values('timestamp'),
                on='timestamp',
                suffixes=('_target', '_lead'),
                tolerance=0.05  # 50ms tolerance
            )
            
            # Calculate correlation
            valid_data = merged.dropna()
            if len(valid_data) >= 5:
                correlation = valid_data['return_target'].corr(valid_data['return_lead'])
                
                if abs(correlation) > abs(best_correlation):
                    best_correlation = correlation
                    best_lag = lag
        
        # Generate signal if correlation is significant
        min_correlation = self.config.get('strategy', {}).get('min_correlation', 0.3)
        
        if abs(best_correlation) >= min_correlation:
            signal.update({
                'has_signal': True,
                'direction': 1 if best_correlation > 0 else -1,
                'strength': abs(best_correlation),
                'lag_seconds': best_lag,
                'correlation': best_correlation
            })
        
        return signal
    
    def generate_cross_asset_signal(self,
                                  primary_predictions: Dict[str, float],
                                  primary_probabilities: Dict[str, np.ndarray],
                                  market_data: Dict[str, float],
                                  timestamp: float,
                                  target_symbol: str) -> Dict[str, Any]:
        """
        Generate cross-asset trading signal.
        
        Args:
            primary_predictions: Predictions from primary models
            primary_probabilities: Prediction probabilities
            market_data: Current market data
            timestamp: Current timestamp
            target_symbol: Symbol to generate signal for
            
        Returns:
            Dictionary with combined signal information
        """
        # Update price history
        self.update_price_history(market_data, timestamp)
        
        # Generate primary signal
        primary_signal = self.generate_signal(
            primary_predictions, 
            primary_probabilities, 
            market_data, 
            timestamp
        )
        
        # Detect lead-lag signals
        other_symbol = 'XBT_EUR' if target_symbol == 'ETH_EUR' else 'ETH_EUR'
        lead_lag_signal = self.detect_lead_lag_signal(target_symbol, other_symbol)
        
        # Combine signals
        combined_signal = primary_signal.copy()
        combined_signal['symbol'] = target_symbol
        combined_signal['lead_lag_info'] = lead_lag_signal
        
        # Adjust signal strength based on lead-lag
        if lead_lag_signal['has_signal']:
            # Boost signal if lead-lag agrees with primary signal
            if (primary_signal['direction'] * lead_lag_signal['direction'] > 0):
                combined_signal['strength'] = min(1.0, 
                    primary_signal['strength'] + lead_lag_signal['strength'] * 0.3)
                combined_signal['confidence'] = min(1.0,
                    primary_signal['confidence'] + 0.1)
            # Reduce signal if lead-lag disagrees
            elif primary_signal['direction'] != 0:
                combined_signal['strength'] *= 0.7
                combined_signal['confidence'] *= 0.9
        
        return combined_signal
