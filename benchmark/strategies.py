"""
Benchmark Strategies

Implements baseline strategies for performance comparison:
- Buy & Hold: Simple passive strategy
- Random Strategy: Random trading decisions
- Simple Momentum: Basic momentum-based trading
- Mean Reversion: Basic mean reversion strategy
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import random
from enum import Enum

class SignalType(Enum):
    """Trading signal types."""
    BUY = 1
    SELL = -1
    HOLD = 0

@dataclass
class TradeSignal:
    """Container for trading signals."""
    timestamp: int
    symbol: str
    signal: SignalType
    confidence: float
    price: float
    metadata: Dict[str, Any] = None

@dataclass
class Position:
    """Trading position information."""
    symbol: str
    quantity: float
    entry_price: float
    entry_time: int
    current_price: float
    unrealized_pnl: float
    
class BaseBenchmarkStrategy(ABC):
    """Base class for benchmark strategies."""
    
    def __init__(self, 
                 name: str,
                 initial_capital: float = 100000.0,
                 transaction_cost: float = 0.001):
        """
        Initialize benchmark strategy.
        
        Args:
            name: Strategy name
            initial_capital: Initial capital amount
            transaction_cost: Transaction cost as fraction of trade value
        """
        self.name = name
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.current_capital = initial_capital
        self.positions = {}  # symbol -> Position
        self.trade_history = []
        self.performance_metrics = {}
        
    @abstractmethod
    def generate_signal(self, 
                       timestamp: int,
                       market_data: Dict[str, Any]) -> Optional[TradeSignal]:
        """Generate trading signal based on market data."""
        pass
    
    def generate_signals(self, 
                        timestamp: int,
                        market_data: Dict[str, Any]) -> List[TradeSignal]:
        """Generate list of trading signals (wrapper for compatibility)."""
        signal = self.generate_signal(timestamp, market_data)
        return [signal] if signal is not None else []
    
    def initialize(self, start_timestamp: int):
        """Initialize strategy at start of backtesting."""
        pass
    
    def update(self, timestamp: int, market_data: Dict[str, Any], portfolio_state: Any):
        """Update strategy state after each timestep."""
        self.update_positions(market_data, timestamp)
    
    def execute_signal(self, signal: TradeSignal, current_prices: Dict[str, float]):
        """Execute a trading signal."""
        symbol = signal.symbol
        current_price = current_prices.get(symbol, signal.price)
        
        if signal.signal == SignalType.BUY:
            self._execute_buy(signal, current_price)
        elif signal.signal == SignalType.SELL:
            self._execute_sell(signal, current_price)
        # HOLD requires no action
    
    def _execute_buy(self, signal: TradeSignal, current_price: float):
        """Execute buy order."""
        symbol = signal.symbol
        
        # Calculate position size (use all available capital)
        available_capital = self.current_capital * 0.95  # Keep 5% as buffer
        transaction_cost_amount = available_capital * self.transaction_cost
        net_capital = available_capital - transaction_cost_amount
        
        if net_capital > 0 and current_price > 0:
            quantity = net_capital / current_price
            
            # Close any existing short position
            if symbol in self.positions and self.positions[symbol].quantity < 0:
                self._close_position(symbol, current_price, signal.timestamp)
            
            # Open new long position
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=current_price,
                entry_time=signal.timestamp,
                current_price=current_price,
                unrealized_pnl=0.0
            )
            
            self.current_capital -= available_capital
            
            self.trade_history.append({
                'timestamp': signal.timestamp,
                'symbol': symbol,
                'action': 'BUY',
                'quantity': quantity,
                'price': current_price,
                'capital_after': self.current_capital,
                'transaction_cost': transaction_cost_amount
            })
    
    def _execute_sell(self, signal: TradeSignal, current_price: float):
        """Execute sell order."""
        symbol = signal.symbol
        
        # Close any existing long position
        if symbol in self.positions and self.positions[symbol].quantity > 0:
            self._close_position(symbol, current_price, signal.timestamp)
    
    def _close_position(self, symbol: str, current_price: float, timestamp: int):
        """Close existing position."""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # Calculate PnL
        if position.quantity > 0:  # Long position
            pnl = position.quantity * (current_price - position.entry_price)
        else:  # Short position  
            pnl = abs(position.quantity) * (position.entry_price - current_price)
        
        # Apply transaction cost
        trade_value = abs(position.quantity) * current_price
        transaction_cost_amount = trade_value * self.transaction_cost
        net_pnl = pnl - transaction_cost_amount
        
        # Update capital
        self.current_capital += trade_value + net_pnl
        
        self.trade_history.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'action': 'CLOSE',
            'quantity': -position.quantity,
            'price': current_price,
            'pnl': net_pnl,
            'capital_after': self.current_capital,
            'transaction_cost': transaction_cost_amount,
            'holding_period': timestamp - position.entry_time
        })
        
        # Remove position
        del self.positions[symbol]
    
    def update_positions(self, current_prices: Dict[str, float], timestamp: int):
        """Update position values with current prices."""
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position.current_price = current_prices[symbol]
                
                if position.quantity > 0:  # Long position
                    position.unrealized_pnl = position.quantity * (position.current_price - position.entry_price)
                else:  # Short position
                    position.unrealized_pnl = abs(position.quantity) * (position.entry_price - position.current_price)
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value."""
        total_value = self.current_capital
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position_value = abs(position.quantity) * current_prices[symbol]
                total_value += position_value + position.unrealized_pnl
        
        return total_value
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.trade_history:
            return {'error': 'No trades executed'}
        
        trades_df = pd.DataFrame(self.trade_history)
        
        # Basic metrics
        total_trades = len(trades_df)
        profitable_trades = len(trades_df[trades_df.get('pnl', 0) > 0])
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # PnL metrics
        total_pnl = trades_df['pnl'].sum() if 'pnl' in trades_df.columns else 0
        total_costs = trades_df['transaction_cost'].sum() if 'transaction_cost' in trades_df.columns else 0
        
        # Final capital
        final_capital = self.current_capital
        total_return = (final_capital - self.initial_capital) / self.initial_capital
        
        return {
            'strategy_name': self.name,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_costs': total_costs,
            'final_capital': final_capital,
            'total_return': total_return,
            'initial_capital': self.initial_capital
        }

class BuyHoldStrategy(BaseBenchmarkStrategy):
    """
    Buy and Hold benchmark strategy.
    
    Simply buys at the beginning and holds until the end.
    """
    
    def __init__(self, 
                 symbol: str = 'BTC',
                 initial_capital: float = 100000.0,
                 transaction_cost: float = 0.001):
        super().__init__('Buy_and_Hold', initial_capital, transaction_cost)
        self.target_symbol = symbol
        self.has_bought = False
        
    def generate_signal(self, 
                       timestamp: int,
                       market_data: Dict[str, Any]) -> Optional[TradeSignal]:
        """Generate buy signal only once at the beginning."""
        if not self.has_bought and self.target_symbol in market_data:
            self.has_bought = True
            return TradeSignal(
                timestamp=timestamp,
                symbol=self.target_symbol,
                signal=SignalType.BUY,
                confidence=1.0,
                price=market_data[self.target_symbol].get('mid_price', 0),
                metadata={'strategy': 'buy_hold'}
            )
        
        return TradeSignal(
            timestamp=timestamp,
            symbol=self.target_symbol,
            signal=SignalType.HOLD,
            confidence=1.0,
            price=market_data.get(self.target_symbol, {}).get('mid_price', 0),
            metadata={'strategy': 'buy_hold'}
        )

class RandomStrategy(BaseBenchmarkStrategy):
    """
    Random trading strategy for baseline comparison.
    
    Makes random buy/sell/hold decisions.
    """
    
    def __init__(self,
                 symbols: List[str] = None,
                 trade_probability: float = 0.1,
                 initial_capital: float = 100000.0,
                 transaction_cost: float = 0.001,
                 random_seed: int = 42):
        super().__init__('Random_Strategy', initial_capital, transaction_cost)
        self.symbols = symbols or ['BTC', 'ETH']
        self.trade_probability = trade_probability
        random.seed(random_seed)
        np.random.seed(random_seed)
        
    def generate_signal(self, 
                       timestamp: int,
                       market_data: Dict[str, Any]) -> Optional[TradeSignal]:
        """Generate random trading signals."""
        
        # Decide whether to trade
        if random.random() > self.trade_probability:
            return None
        
        # Select random symbol
        available_symbols = [s for s in self.symbols if s in market_data]
        if not available_symbols:
            return None
        
        symbol = random.choice(available_symbols)
        
        # Random signal type
        signal_type = random.choice([SignalType.BUY, SignalType.SELL, SignalType.HOLD])
        
        # Random confidence
        confidence = random.uniform(0.1, 1.0)
        
        return TradeSignal(
            timestamp=timestamp,
            symbol=symbol,
            signal=signal_type,
            confidence=confidence,
            price=market_data[symbol].get('mid_price', 0),
            metadata={'strategy': 'random'}
        )

class SimpleMomentumStrategy(BaseBenchmarkStrategy):
    """
    Simple momentum strategy for comparison.
    
    Buys when price is above moving average, sells when below.
    """
    
    def __init__(self,
                 symbols: List[str] = None,
                 lookback_window: int = 20,
                 initial_capital: float = 100000.0,
                 transaction_cost: float = 0.001):
        super().__init__('Simple_Momentum', initial_capital, transaction_cost)
        self.symbols = symbols or ['BTC', 'ETH']
        self.lookback_window = lookback_window
        self.price_history = {symbol: [] for symbol in self.symbols}
        
    def generate_signal(self, 
                       timestamp: int,
                       market_data: Dict[str, Any]) -> Optional[TradeSignal]:
        """Generate momentum-based signals."""
        
        signals = []
        
        for symbol in self.symbols:
            if symbol in market_data:
                current_price = market_data[symbol].get('mid_price', 0)
                
                if current_price > 0:
                    # Update price history
                    self.price_history[symbol].append(current_price)
                    
                    # Keep only lookback window
                    if len(self.price_history[symbol]) > self.lookback_window:
                        self.price_history[symbol].pop(0)
                    
                    # Generate signal if we have enough history
                    if len(self.price_history[symbol]) >= self.lookback_window:
                        moving_avg = np.mean(self.price_history[symbol])
                        
                        if current_price > moving_avg * 1.01:  # 1% above MA
                            signal_type = SignalType.BUY
                            confidence = min(1.0, (current_price / moving_avg - 1) * 10)
                        elif current_price < moving_avg * 0.99:  # 1% below MA
                            signal_type = SignalType.SELL
                            confidence = min(1.0, (1 - current_price / moving_avg) * 10)
                        else:
                            signal_type = SignalType.HOLD
                            confidence = 0.5
                        
                        signals.append(TradeSignal(
                            timestamp=timestamp,
                            symbol=symbol,
                            signal=signal_type,
                            confidence=confidence,
                            price=current_price,
                            metadata={
                                'strategy': 'momentum',
                                'moving_avg': moving_avg,
                                'price_ratio': current_price / moving_avg
                            }
                        ))
        
        # Return highest confidence signal
        if signals:
            return max(signals, key=lambda s: s.confidence)
        
        return None

class MeanReversionStrategy(BaseBenchmarkStrategy):
    """
    Simple mean reversion strategy.
    
    Buys when price is significantly below moving average,
    sells when significantly above.
    """
    
    def __init__(self,
                 symbols: List[str] = None,
                 lookback_window: int = 20,
                 std_threshold: float = 2.0,
                 initial_capital: float = 100000.0,
                 transaction_cost: float = 0.001):
        super().__init__('Mean_Reversion', initial_capital, transaction_cost)
        self.symbols = symbols or ['BTC', 'ETH']
        self.lookback_window = lookback_window
        self.std_threshold = std_threshold
        self.price_history = {symbol: [] for symbol in self.symbols}
        
    def generate_signal(self, 
                       timestamp: int,
                       market_data: Dict[str, Any]) -> Optional[TradeSignal]:
        """Generate mean reversion signals."""
        
        signals = []
        
        for symbol in self.symbols:
            if symbol in market_data:
                current_price = market_data[symbol].get('mid_price', 0)
                
                if current_price > 0:
                    # Update price history
                    self.price_history[symbol].append(current_price)
                    
                    # Keep only lookback window
                    if len(self.price_history[symbol]) > self.lookback_window:
                        self.price_history[symbol].pop(0)
                    
                    # Generate signal if we have enough history
                    if len(self.price_history[symbol]) >= self.lookback_window:
                        prices = np.array(self.price_history[symbol])
                        moving_avg = np.mean(prices)
                        moving_std = np.std(prices)
                        
                        if moving_std > 0:
                            z_score = (current_price - moving_avg) / moving_std
                            
                            if z_score < -self.std_threshold:  # Price significantly below average
                                signal_type = SignalType.BUY
                                confidence = min(1.0, abs(z_score) / self.std_threshold)
                            elif z_score > self.std_threshold:  # Price significantly above average
                                signal_type = SignalType.SELL
                                confidence = min(1.0, abs(z_score) / self.std_threshold)
                            else:
                                signal_type = SignalType.HOLD
                                confidence = 0.1
                            
                            signals.append(TradeSignal(
                                timestamp=timestamp,
                                symbol=symbol,
                                signal=signal_type,
                                confidence=confidence,
                                price=current_price,
                                metadata={
                                    'strategy': 'mean_reversion',
                                    'z_score': z_score,
                                    'moving_avg': moving_avg,
                                    'moving_std': moving_std
                                }
                            ))
        
        # Return highest confidence signal
        if signals:
            return max(signals, key=lambda s: s.confidence)
        
        return None
