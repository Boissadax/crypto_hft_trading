"""
Backtesting Engine

Comprehensive backtesting framework for strategy evaluation:
- Event-driven backtesting architecture
- Portfolio management and position tracking
- Transaction cost modeling
- Multi-asset strategy support
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from .strategies import BaseBenchmarkStrategy, TradeSignal, SignalType, Position
from .metrics import PerformanceAnalyzer, PerformanceMetrics

class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"

class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class Order:
    """Trading order representation."""
    order_id: str
    timestamp: int
    symbol: str
    side: SignalType
    quantity: float
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    fill_price: Optional[float] = None
    fill_time: Optional[int] = None
    commission: float = 0.0

@dataclass
class Trade:
    """Executed trade representation."""
    trade_id: str
    timestamp: int
    symbol: str
    side: SignalType
    quantity: float
    price: float
    commission: float
    order_id: str

@dataclass
class PortfolioState:
    """Portfolio state snapshot."""
    timestamp: int
    cash: float
    positions: Dict[str, Position]
    portfolio_value: float
    unrealized_pnl: float
    realized_pnl: float

class TransactionCostModel:
    """Transaction cost modeling."""
    
    def __init__(self,
                 fixed_cost: float = 0.0,
                 percentage_cost: float = 0.001,
                 price_impact: float = 0.0,
                 min_cost: float = 0.0):
        """
        Initialize transaction cost model.
        
        Args:
            fixed_cost: Fixed cost per trade
            percentage_cost: Percentage of trade value
            price_impact: Price impact coefficient
            min_cost: Minimum cost per trade
        """
        self.fixed_cost = fixed_cost
        self.percentage_cost = percentage_cost
        self.price_impact = price_impact
        self.min_cost = min_cost
    
    def calculate_cost(self, quantity: float, price: float, side: SignalType) -> float:
        """
        Calculate transaction cost for a trade.
        
        Args:
            quantity: Trade quantity
            price: Trade price
            side: Trade side (buy/sell)
            
        Returns:
            Total transaction cost
        """
        trade_value = abs(quantity) * price
        
        # Fixed cost
        cost = self.fixed_cost
        
        # Percentage cost
        cost += trade_value * self.percentage_cost
        
        # Price impact (simplified model)
        if self.price_impact > 0:
            impact = trade_value * self.price_impact
            cost += impact
        
        # Apply minimum cost
        cost = max(cost, self.min_cost)
        
        return cost

class Portfolio:
    """Portfolio management for backtesting."""
    
    def __init__(self,
                 initial_cash: float = 100000.0,
                 transaction_cost_model: Optional[TransactionCostModel] = None):
        """
        Initialize portfolio.
        
        Args:
            initial_cash: Initial cash amount
            transaction_cost_model: Transaction cost model
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.transaction_cost_model = transaction_cost_model or TransactionCostModel()
        
        # Tracking
        self.trades: List[Trade] = []
        self.portfolio_history: List[PortfolioState] = []
        self.realized_pnl = 0.0
        
    def execute_order(self, order: Order, current_prices: Dict[str, float]) -> bool:
        """
        Execute an order.
        
        Args:
            order: Order to execute
            current_prices: Current market prices
            
        Returns:
            True if order was executed successfully
        """
        if order.symbol not in current_prices:
            order.status = OrderStatus.REJECTED
            return False
        
        current_price = current_prices[order.symbol]
        
        # Determine execution price based on order type
        if order.order_type == OrderType.MARKET:
            execution_price = current_price
        elif order.order_type == OrderType.LIMIT:
            if order.side == SignalType.BUY and current_price <= order.price:
                execution_price = order.price
            elif order.side == SignalType.SELL and current_price >= order.price:
                execution_price = order.price
            else:
                return False  # Order not filled
        else:
            execution_price = current_price  # Simplified for stop orders
        
        # Calculate transaction cost
        commission = self.transaction_cost_model.calculate_cost(
            order.quantity, execution_price, order.side
        )
        
        # Check if we have enough cash (for buy orders)
        if order.side == SignalType.BUY:
            required_cash = order.quantity * execution_price + commission
            if required_cash > self.cash:
                order.status = OrderStatus.REJECTED
                return False
        
        # Execute the trade
        self._execute_trade(order, execution_price, commission)
        
        order.status = OrderStatus.FILLED
        order.fill_price = execution_price
        order.fill_time = order.timestamp
        order.commission = commission
        
        return True
    
    def _execute_trade(self, order: Order, price: float, commission: float):
        """Execute a trade and update portfolio."""
        symbol = order.symbol
        quantity = order.quantity if order.side == SignalType.BUY else -order.quantity
        
        # Create trade record
        trade = Trade(
            trade_id=f"trade_{len(self.trades)}",
            timestamp=order.timestamp,
            symbol=symbol,
            side=order.side,
            quantity=abs(quantity),
            price=price,
            commission=commission,
            order_id=order.order_id
        )
        self.trades.append(trade)
        
        # Update cash
        cash_flow = -quantity * price - commission
        self.cash += cash_flow
        
        # Update positions
        if symbol in self.positions:
            position = self.positions[symbol]
            
            # Calculate realized PnL if reducing position
            if (position.quantity > 0 and quantity < 0) or (position.quantity < 0 and quantity > 0):
                # Partial or full close
                close_quantity = min(abs(quantity), abs(position.quantity))
                if position.quantity > 0:  # Long position
                    realized_pnl = close_quantity * (price - position.entry_price) - commission
                else:  # Short position
                    realized_pnl = close_quantity * (position.entry_price - price) - commission
                
                self.realized_pnl += realized_pnl
            
            # Update position
            new_quantity = position.quantity + quantity
            
            if abs(new_quantity) < 1e-8:  # Position closed
                del self.positions[symbol]
            else:
                # Update average entry price for additions
                if (position.quantity > 0 and quantity > 0) or (position.quantity < 0 and quantity < 0):
                    total_cost = position.quantity * position.entry_price + quantity * price
                    position.entry_price = total_cost / new_quantity
                
                position.quantity = new_quantity
        else:
            # New position
            if abs(quantity) > 1e-8:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=price,
                    entry_time=order.timestamp,
                    current_price=price,
                    unrealized_pnl=0.0
                )
    
    def update_positions(self, current_prices: Dict[str, float], timestamp: int):
        """Update position values with current prices."""
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position.current_price = current_prices[symbol]
                
                # Calculate unrealized PnL
                if position.quantity > 0:  # Long position
                    position.unrealized_pnl = position.quantity * (
                        position.current_price - position.entry_price
                    )
                else:  # Short position
                    position.unrealized_pnl = position.quantity * (
                        position.entry_price - position.current_price
                    )
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value."""
        portfolio_value = self.cash
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                portfolio_value += position.quantity * current_prices[symbol]
        
        return portfolio_value
    
    def get_unrealized_pnl(self) -> float:
        """Get total unrealized PnL."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    def snapshot(self, timestamp: int, current_prices: Dict[str, float]) -> PortfolioState:
        """Take a portfolio snapshot."""
        self.update_positions(current_prices, timestamp)
        
        return PortfolioState(
            timestamp=timestamp,
            cash=self.cash,
            positions=self.positions.copy(),
            portfolio_value=self.get_portfolio_value(current_prices),
            unrealized_pnl=self.get_unrealized_pnl(),
            realized_pnl=self.realized_pnl
        )

class BacktestEngine:
    """Main backtesting engine."""
    
    def __init__(self,
                 initial_capital: float = 100000.0,
                 transaction_cost_model: Optional[TransactionCostModel] = None,
                 logging_level: int = logging.INFO):
        """
        Initialize backtest engine.
        
        Args:
            initial_capital: Initial capital
            transaction_cost_model: Transaction cost model
            logging_level: Logging level
        """
        self.initial_capital = initial_capital
        self.transaction_cost_model = transaction_cost_model or TransactionCostModel()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging_level)
        
        # Results storage
        self.results: Dict[str, Any] = {}
    
    def run_backtest(self,
                    strategy: BaseBenchmarkStrategy,
                    data: Dict[str, pd.DataFrame],
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Run backtest for a strategy.
        
        Args:
            strategy: Trading strategy to test
            data: Dictionary of price data by symbol
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            Dictionary with backtest results
        """
        self.logger.info(f"Starting backtest for strategy: {strategy.name}")
        
        # Initialize portfolio
        portfolio = Portfolio(
            initial_cash=self.initial_capital,
            transaction_cost_model=self.transaction_cost_model
        )
        
        # Prepare data
        aligned_data = self._align_data(data, start_date, end_date)
        if aligned_data.empty:
            raise ValueError("No data available for backtesting")
        
        # Initialize strategy
        strategy.initialize(aligned_data.index[0])
        
        # Track performance
        portfolio_values = []
        signals_generated = []
        orders_executed = []
        
        # Main backtesting loop
        for timestamp in aligned_data.index:
            current_data = aligned_data.loc[timestamp].to_dict()
            
            # Generate signals
            signals = strategy.generate_signals(timestamp, current_data)
            signals_generated.extend(signals)
            
            # Convert signals to orders
            orders = self._signals_to_orders(signals, timestamp)
            
            # Execute orders
            for order in orders:
                if portfolio.execute_order(order, current_data):
                    orders_executed.append(order)
            
            # Update portfolio
            portfolio_snapshot = portfolio.snapshot(timestamp, current_data)
            portfolio.portfolio_history.append(portfolio_snapshot)
            portfolio_values.append(portfolio_snapshot.portfolio_value)
            
            # Update strategy
            strategy.update(timestamp, current_data, portfolio_snapshot)
        
        # Calculate results
        results = self._calculate_results(
            portfolio, aligned_data, portfolio_values, strategy.name
        )
        
        self.logger.info(f"Backtest completed for {strategy.name}")
        return results
    
    def compare_strategies(self,
                          strategies: List[BaseBenchmarkStrategy],
                          data: Dict[str, pd.DataFrame],
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Compare multiple strategies.
        
        Args:
            strategies: List of strategies to compare
            data: Price data
            start_date: Start date
            end_date: End date
            
        Returns:
            Comparison results
        """
        results = {}
        
        for strategy in strategies:
            results[strategy.name] = self.run_backtest(
                strategy, data, start_date, end_date
            )
        
        # Create comparison summary
        comparison = self._create_comparison_summary(results)
        
        return {
            'individual_results': results,
            'comparison': comparison
        }
    
    def _align_data(self, data: Dict[str, pd.DataFrame], 
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None) -> pd.DataFrame:
        """Align and filter data."""
        if not data:
            return pd.DataFrame()
        
        # Get common timestamps
        common_index = None
        for symbol, df in data.items():
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)
        
        if common_index is None or len(common_index) == 0:
            return pd.DataFrame()
        
        # Create aligned DataFrame
        aligned_data = pd.DataFrame(index=common_index)
        
        for symbol, df in data.items():
            # Use close price as default
            price_col = 'close' if 'close' in df.columns else df.columns[0]
            aligned_data[symbol] = df.loc[common_index, price_col]
        
        # Filter by date range
        if start_date:
            aligned_data = aligned_data[aligned_data.index >= start_date]
        if end_date:
            aligned_data = aligned_data[aligned_data.index <= end_date]
        
        return aligned_data.dropna()
    
    def _signals_to_orders(self, signals: List[TradeSignal], timestamp: int) -> List[Order]:
        """Convert trading signals to orders."""
        orders = []
        
        for i, signal in enumerate(signals):
            if signal.signal == SignalType.HOLD:
                continue
            
            order = Order(
                order_id=f"order_{timestamp}_{i}",
                timestamp=timestamp,
                symbol=signal.symbol,
                side=signal.signal,
                quantity=1000.0,  # Fixed quantity for simplicity
                order_type=OrderType.MARKET
            )
            orders.append(order)
        
        return orders
    
    def _calculate_results(self,
                          portfolio: Portfolio,
                          data: pd.DataFrame,
                          portfolio_values: List[float],
                          strategy_name: str) -> Dict[str, Any]:
        """Calculate comprehensive backtest results."""
        # Create returns series
        portfolio_series = pd.Series(portfolio_values, index=data.index)
        returns = portfolio_series.pct_change().dropna()
        
        # Calculate performance metrics
        analyzer = PerformanceAnalyzer()
        metrics = analyzer.calculate_metrics(
            returns=returns,
            trades=pd.DataFrame([
                {
                    'timestamp': trade.timestamp,
                    'symbol': trade.symbol,
                    'side': trade.side.value,
                    'quantity': trade.quantity,
                    'price': trade.price,
                    'commission': trade.commission,
                    'pnl': 0  # Simplified - would need position tracking for accurate PnL
                }
                for trade in portfolio.trades
            ]) if portfolio.trades else None
        )
        
        return {
            'strategy_name': strategy_name,
            'portfolio_values': portfolio_series,
            'returns': returns,
            'metrics': metrics,
            'trades': portfolio.trades,
            'portfolio_history': portfolio.portfolio_history,
            'final_value': portfolio_values[-1] if portfolio_values else portfolio.initial_cash,
            'total_return': (portfolio_values[-1] / portfolio.initial_cash - 1) if portfolio_values else 0,
            'num_trades': len(portfolio.trades)
        }
    
    def _create_comparison_summary(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Create comparison summary of strategies."""
        comparison_data = []
        
        for strategy_name, result in results.items():
            metrics = result['metrics']
            
            comparison_data.append({
                'Strategy': strategy_name,
                'Total Return': result['total_return'],
                'Annualized Return': metrics.annualized_return,
                'Volatility': metrics.annualized_volatility,
                'Sharpe Ratio': metrics.sharpe_ratio,
                'Max Drawdown': metrics.max_drawdown,
                'Calmar Ratio': metrics.calmar_ratio,
                'Win Rate': metrics.win_rate,
                'Num Trades': result['num_trades']
            })
        
        return pd.DataFrame(comparison_data).set_index('Strategy')
