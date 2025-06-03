"""
Portfolio management module for crypto HFT trading strategy.
Handles position management, order execution, and portfolio optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    PARTIAL = "partial"

@dataclass
class Order:
    """Represents a trading order."""
    timestamp: datetime
    asset: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: Optional[float]
    order_type: OrderType
    order_id: str
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    commission: float = 0.0

@dataclass
class Position:
    """Represents a position in an asset."""
    asset: str
    quantity: float
    avg_entry_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    last_update: datetime = None

class PortfolioManager:
    """
    Manages portfolio positions, orders, and risk for crypto HFT strategy.
    """
    
    def __init__(self,
                 initial_capital: float = 100000.0,
                 max_position_size: float = 0.1,  # 10% of portfolio per asset
                 commission_rate: float = 0.001,  # 0.1% commission
                 slippage_model: str = 'linear',
                 max_leverage: float = 1.0):
        """
        Initialize portfolio manager.
        
        Args:
            initial_capital: Starting capital
            max_position_size: Maximum position size as fraction of portfolio
            commission_rate: Trading commission rate
            slippage_model: Model for market impact ('linear', 'sqrt', 'none')
            max_leverage: Maximum leverage allowed
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_size = max_position_size
        self.commission_rate = commission_rate
        self.slippage_model = slippage_model
        self.max_leverage = max_leverage
        
        # Portfolio state
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.cash = initial_capital
        self.portfolio_value = initial_capital
        
        # Performance tracking
        self.portfolio_history = []
        self.trade_history = []
        self.pnl_history = []
        
        # Risk metrics
        self.max_drawdown = 0.0
        self.peak_portfolio_value = initial_capital
        
    def update_portfolio_value(self, current_prices: Dict[str, float]):
        """Update portfolio value based on current market prices."""
        
        position_value = 0.0
        total_unrealized_pnl = 0.0
        
        for asset, position in self.positions.items():
            if asset in current_prices:
                current_price = current_prices[asset]
                position_market_value = position.quantity * current_price
                position_value += position_market_value
                
                # Update unrealized PnL
                position.unrealized_pnl = (current_price - position.avg_entry_price) * position.quantity
                total_unrealized_pnl += position.unrealized_pnl
                position.last_update = datetime.now()
        
        self.portfolio_value = self.cash + position_value
        
        # Update peak and drawdown
        if self.portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = self.portfolio_value
        
        current_drawdown = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Record portfolio state
        portfolio_state = {
            'timestamp': datetime.now(),
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'position_value': position_value,
            'unrealized_pnl': total_unrealized_pnl,
            'drawdown': current_drawdown
        }
        self.portfolio_history.append(portfolio_state)
        
        return self.portfolio_value
    
    def place_order(self,
                   asset: str,
                   side: str,
                   quantity: float,
                   price: Optional[float] = None,
                   order_type: OrderType = OrderType.MARKET,
                   order_id: Optional[str] = None) -> str:
        """
        Place a trading order.
        
        Args:
            asset: Asset symbol
            side: 'buy' or 'sell'
            quantity: Order quantity
            price: Limit price (for limit orders)
            order_type: Type of order
            order_id: Custom order ID
            
        Returns:
            Order ID
        """
        
        if order_id is None:
            order_id = f"{asset}_{side}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Validate order
        if not self._validate_order(asset, side, quantity, price):
            logger.warning("Order validation failed: %s %s %s", asset, side, quantity)
            return None
        
        order = Order(
            timestamp=datetime.now(),
            asset=asset,
            side=side,
            quantity=quantity,
            price=price,
            order_type=order_type,
            order_id=order_id
        )
        
        self.orders[order_id] = order
        logger.info(f"Order placed: {order_id} - {asset} {side} {quantity}")
        
        return order_id
    
    def execute_order(self,
                     order_id: str,
                     execution_price: float,
                     execution_quantity: Optional[float] = None,
                     timestamp: Optional[datetime] = None) -> bool:
        """
        Execute a pending order.
        
        Args:
            order_id: Order to execute
            execution_price: Price at which order was executed
            execution_quantity: Quantity executed (default: full order)
            timestamp: Execution timestamp
            
        Returns:
            True if execution successful
        """
        
        if order_id not in self.orders:
            logger.error(f"Order {order_id} not found")
            return False
        
        order = self.orders[order_id]
        
        if order.status != OrderStatus.PENDING:
            logger.warning(f"Order {order_id} is not pending (status: {order.status})")
            return False
        
        if execution_quantity is None:
            execution_quantity = order.quantity
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Calculate slippage
        if order.price is not None:
            slippage = self._calculate_slippage(order, execution_price, execution_quantity)
            execution_price += slippage
        
        # Calculate commission
        commission = execution_quantity * execution_price * self.commission_rate
        
        # Update order
        order.filled_quantity += execution_quantity
        order.filled_price = execution_price
        order.commission += commission
        
        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIAL
        
        # Update position
        self._update_position(order, execution_price, execution_quantity, commission, timestamp)
        
        # Record trade
        trade_record = {
            'timestamp': timestamp,
            'order_id': order_id,
            'asset': order.asset,
            'side': order.side,
            'quantity': execution_quantity,
            'price': execution_price,
            'commission': commission,
            'slippage': slippage if order.price else 0.0
        }
        self.trade_history.append(trade_record)
        
        logger.info(f"Order executed: {order_id} - {execution_quantity} @ {execution_price}")
        
        return True
    
    def _validate_order(self, asset: str, side: str, quantity: float, price: Optional[float]) -> bool:
        """Validate order parameters."""
        
        if quantity <= 0:
            return False
        
        # Check position size limits
        if side == 'buy':
            estimated_value = quantity * (price or 1000)  # Conservative estimate
            max_position_value = self.portfolio_value * self.max_position_size
            
            if estimated_value > max_position_value:
                logger.warning(f"Order exceeds position size limit: {estimated_value} > {max_position_value}")
                return False
        
        # Check available cash for buy orders
        if side == 'buy':
            required_cash = quantity * (price or 1000) * (1 + self.commission_rate)
            if required_cash > self.cash:
                logger.warning(f"Insufficient cash: {required_cash} > {self.cash}")
                return False
        
        # Check available position for sell orders
        if side == 'sell':
            current_position = self.positions.get(asset, Position(asset, 0.0, 0.0))
            if quantity > current_position.quantity:
                logger.warning(f"Insufficient position: {quantity} > {current_position.quantity}")
                return False
        
        return True
    
    def _calculate_slippage(self, order: Order, execution_price: float, quantity: float) -> float:
        """Calculate market impact slippage."""
        
        if self.slippage_model == 'none':
            return 0.0
        
        # Simple slippage models
        base_slippage = 0.001  # 0.1% base slippage
        
        if self.slippage_model == 'linear':
            # Linear in quantity
            slippage_factor = quantity / 1000  # Normalize by 1000 units
            slippage = base_slippage * slippage_factor * execution_price
        elif self.slippage_model == 'sqrt':
            # Square root law
            slippage_factor = np.sqrt(quantity / 1000)
            slippage = base_slippage * slippage_factor * execution_price
        else:
            slippage = 0.0
        
        # Apply direction (negative for sells)
        if order.side == 'sell':
            slippage = -slippage
        
        return slippage
    
    def _update_position(self,
                        order: Order,
                        execution_price: float,
                        execution_quantity: float,
                        commission: float,
                        timestamp: datetime):
        """Update position based on order execution."""
        
        asset = order.asset
        
        if asset not in self.positions:
            self.positions[asset] = Position(asset, 0.0, 0.0)
        
        position = self.positions[asset]
        
        if order.side == 'buy':
            # Update position for buy
            total_quantity = position.quantity + execution_quantity
            if total_quantity > 0:
                # Calculate new average entry price
                total_cost = (position.quantity * position.avg_entry_price +
                             execution_quantity * execution_price + commission)
                position.avg_entry_price = total_cost / total_quantity
                position.quantity = total_quantity
            
            # Update cash
            self.cash -= execution_quantity * execution_price + commission
            
        elif order.side == 'sell':
            # Calculate realized PnL
            realized_pnl = (execution_price - position.avg_entry_price) * execution_quantity - commission
            position.realized_pnl += realized_pnl
            position.quantity -= execution_quantity
            
            # Update cash
            self.cash += execution_quantity * execution_price - commission
            
            # Remove position if fully closed
            if position.quantity <= 0:
                del self.positions[asset]
        
        # Record PnL
        pnl_record = {
            'timestamp': timestamp,
            'asset': asset,
            'realized_pnl': realized_pnl if order.side == 'sell' else 0.0,
            'commission': commission
        }
        self.pnl_history.append(pnl_record)
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        if order.status == OrderStatus.PENDING:
            order.status = OrderStatus.CANCELLED
            logger.info(f"Order cancelled: {order_id}")
            return True
        
        return False
    
    def get_position(self, asset: str) -> Optional[Position]:
        """Get current position for an asset."""
        return self.positions.get(asset)
    
    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary."""
        
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
        
        return {
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_realized_pnl': total_realized_pnl,
            'max_drawdown': self.max_drawdown,
            'return_pct': (self.portfolio_value - self.initial_capital) / self.initial_capital * 100,
            'num_positions': len(self.positions),
            'num_trades': len(self.trade_history)
        }
    
    def get_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics."""
        
        if len(self.portfolio_history) < 2:
            return {}
        
        # Convert to DataFrame for analysis
        portfolio_df = pd.DataFrame(self.portfolio_history)
        portfolio_df.set_index('timestamp', inplace=True)
        
        # Calculate returns
        portfolio_returns = portfolio_df['portfolio_value'].pct_change().dropna()
        
        # Performance metrics
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1  # Assume 252 trading days
        volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized volatility
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown analysis
        portfolio_values = portfolio_df['portfolio_value']
        running_max = portfolio_values.cummax()
        drawdowns = (portfolio_values - running_max) / running_max
        max_drawdown = drawdowns.min()
        
        # Win rate and trade analysis
        if self.trade_history:
            trades_df = pd.DataFrame(self.trade_history)
            # Calculate per-trade PnL (simplified)
            trade_pnls = []
            for trade in self.trade_history:
                # This is a simplified calculation
                if trade['side'] == 'sell':
                    # Assume we're closing a position
                    trade_pnls.append(trade['quantity'] * trade['price'] - trade['commission'])
            
            if trade_pnls:
                win_rate = len([pnl for pnl in trade_pnls if pnl > 0]) / len(trade_pnls)
                avg_win = np.mean([pnl for pnl in trade_pnls if pnl > 0]) if any(pnl > 0 for pnl in trade_pnls) else 0
                avg_loss = np.mean([pnl for pnl in trade_pnls if pnl < 0]) if any(pnl < 0 for pnl in trade_pnls) else 0
                profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            else:
                win_rate = 0
                avg_win = 0
                avg_loss = 0
                profit_factor = 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_trades': len(self.trade_history)
        }
    
    def rebalance_portfolio(self,
                          target_weights: Dict[str, float],
                          current_prices: Dict[str, float],
                          min_trade_size: float = 0.01) -> List[str]:
        """
        Rebalance portfolio to target weights.
        
        Args:
            target_weights: Dictionary of {asset: weight}
            current_prices: Current market prices
            min_trade_size: Minimum trade size to execute
            
        Returns:
            List of order IDs placed
        """
        
        order_ids = []
        
        # Calculate current weights
        self.update_portfolio_value(current_prices)
        current_weights = {}
        
        for asset, position in self.positions.items():
            if asset in current_prices:
                position_value = position.quantity * current_prices[asset]
                current_weights[asset] = position_value / self.portfolio_value
        
        # Calculate required trades
        for asset, target_weight in target_weights.items():
            if asset not in current_prices:
                logger.warning(f"No price data for {asset}, skipping rebalance")
                continue
            
            current_weight = current_weights.get(asset, 0.0)
            weight_diff = target_weight - current_weight
            
            if abs(weight_diff) > 0.01:  # 1% threshold
                target_value = target_weight * self.portfolio_value
                current_value = current_weight * self.portfolio_value
                trade_value = target_value - current_value
                
                trade_quantity = abs(trade_value) / current_prices[asset]
                
                if trade_quantity >= min_trade_size:
                    side = 'buy' if trade_value > 0 else 'sell'
                    
                    order_id = self.place_order(
                        asset=asset,
                        side=side,
                        quantity=trade_quantity,
                        order_type=OrderType.MARKET
                    )
                    
                    if order_id:
                        order_ids.append(order_id)
        
        logger.info(f"Rebalancing complete: {len(order_ids)} orders placed")
        return order_ids
    
    def export_portfolio_data(self) -> Dict[str, pd.DataFrame]:
        """Export portfolio data for analysis."""
        
        portfolio_df = pd.DataFrame(self.portfolio_history)
        trades_df = pd.DataFrame(self.trade_history)
        pnl_df = pd.DataFrame(self.pnl_history)
        
        positions_data = []
        for asset, position in self.positions.items():
            positions_data.append({
                'asset': asset,
                'quantity': position.quantity,
                'avg_entry_price': position.avg_entry_price,
                'unrealized_pnl': position.unrealized_pnl,
                'realized_pnl': position.realized_pnl,
                'last_update': position.last_update
            })
        positions_df = pd.DataFrame(positions_data)
        
        return {
            'portfolio_history': portfolio_df,
            'trades': trades_df,
            'pnl_history': pnl_df,
            'current_positions': positions_df
        }
