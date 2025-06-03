"""
Risk management for cryptocurrency trading strategy.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RiskManager:
    """
    Manages risk for cryptocurrency trading strategy.
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 initial_capital: float = 100000.0):
        """
        Initialize the risk manager.
        
        Args:
            config: Configuration dictionary
            initial_capital: Initial trading capital
        """
        self.config = config
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Risk parameters
        self.max_position_size = config.get('strategy', {}).get('max_position_size', 0.1)
        self.stop_loss = config.get('strategy', {}).get('stop_loss', 0.002)
        self.take_profit = config.get('strategy', {}).get('take_profit', 0.005)
        
        # Position tracking
        self.positions = {}  # symbol -> position info
        self.trade_history = []
        self.daily_pnl = []
        
        # Risk metrics
        self.max_drawdown = 0.0
        self.peak_capital = initial_capital
        
    def calculate_position_size(self, 
                              signal_strength: float,
                              current_price: float,
                              volatility: float) -> float:
        """
        Calculate appropriate position size based on signal strength and risk.
        
        Args:
            signal_strength: Strength of trading signal (0-1)
            current_price: Current price of the asset
            volatility: Current volatility estimate
            
        Returns:
            Position size as fraction of capital
        """
        # Base position size from signal strength
        base_size = self.max_position_size * signal_strength
        
        # Adjust for volatility (reduce size in high volatility)
        if volatility > 0:
            vol_adjustment = min(1.0, 0.02 / volatility)  # Target 2% daily volatility
            base_size *= vol_adjustment
        
        # Ensure position size doesn't exceed limits
        max_size = min(base_size, self.max_position_size)
        
        # Convert to dollar amount
        dollar_amount = max_size * self.current_capital
        
        # Convert to number of units
        position_size = dollar_amount / current_price if current_price > 0 else 0.0
        
        return position_size
    
    def check_risk_limits(self, 
                         symbol: str,
                         proposed_position: float,
                         current_price: float) -> Dict[str, Any]:
        """
        Check if proposed position violates risk limits.
        
        Args:
            symbol: Trading symbol
            proposed_position: Proposed position size
            current_price: Current price
            
        Returns:
            Dictionary with risk check results
        """
        risk_check = {
            'approved': True,
            'warnings': [],
            'violations': [],
            'adjusted_position': proposed_position
        }
        
        # Check maximum position size
        position_value = abs(proposed_position * current_price)
        max_position_value = self.max_position_size * self.current_capital
        
        if position_value > max_position_value:
            risk_check['violations'].append(f"Position size exceeds maximum: {position_value:.2f} > {max_position_value:.2f}")
            risk_check['adjusted_position'] = np.sign(proposed_position) * max_position_value / current_price
            risk_check['approved'] = False
        
        # Check total portfolio exposure
        total_exposure = position_value
        for pos_symbol, pos_info in self.positions.items():
            if pos_symbol != symbol:
                total_exposure += abs(pos_info['size'] * pos_info['current_price'])
        
        max_total_exposure = 0.8 * self.current_capital  # Maximum 80% exposure
        if total_exposure > max_total_exposure:
            risk_check['violations'].append(f"Total exposure exceeds limit: {total_exposure:.2f} > {max_total_exposure:.2f}")
            risk_check['approved'] = False
        
        # Check drawdown limits
        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        max_allowed_drawdown = 0.20  # 20% maximum drawdown
        
        if current_drawdown > max_allowed_drawdown:
            risk_check['violations'].append(f"Maximum drawdown exceeded: {current_drawdown:.2%} > {max_allowed_drawdown:.2%}")
            risk_check['approved'] = False
        
        # Warning for large positions
        if position_value > 0.05 * self.current_capital:  # 5% of capital
            risk_check['warnings'].append(f"Large position size: {position_value:.2f} ({position_value/self.current_capital:.1%} of capital)")
        
        return risk_check
    
    def open_position(self, 
                     symbol: str,
                     size: float,
                     entry_price: float,
                     signal_info: Dict[str, Any],
                     timestamp: float) -> bool:
        """
        Open a new position.
        
        Args:
            symbol: Trading symbol
            size: Position size (positive for long, negative for short)
            entry_price: Entry price
            signal_info: Information about the trading signal
            timestamp: Timestamp of position opening
            
        Returns:
            Boolean indicating if position was opened successfully
        """
        # Check risk limits
        risk_check = self.check_risk_limits(symbol, size, entry_price)
        
        if not risk_check['approved']:
            logger.warning(f"Position opening rejected for {symbol}: {risk_check['violations']}")
            return False
        
        # Calculate stop loss and take profit levels
        if size > 0:  # Long position
            stop_loss_price = entry_price * (1 - self.stop_loss)
            take_profit_price = entry_price * (1 + self.take_profit)
        else:  # Short position
            stop_loss_price = entry_price * (1 + self.stop_loss)
            take_profit_price = entry_price * (1 - self.take_profit)
        
        # Create position
        position = {
            'symbol': symbol,
            'size': risk_check['adjusted_position'],
            'entry_price': entry_price,
            'current_price': entry_price,
            'stop_loss_price': stop_loss_price,
            'take_profit_price': take_profit_price,
            'entry_timestamp': timestamp,
            'entry_datetime': datetime.fromtimestamp(timestamp),
            'signal_info': signal_info,
            'unrealized_pnl': 0.0,
            'max_favorable_excursion': 0.0,
            'max_adverse_excursion': 0.0
        }
        
        # Store position
        self.positions[symbol] = position
        
        # Update capital (account for transaction costs)
        transaction_cost = abs(risk_check['adjusted_position'] * entry_price) * 0.001  # 0.1% transaction cost
        self.current_capital -= transaction_cost
        
        logger.info(f"Opened position: {symbol} size={risk_check['adjusted_position']:.4f} at {entry_price:.2f}")
        
        return True
    
    def update_position(self, 
                       symbol: str,
                       current_price: float,
                       timestamp: float) -> Dict[str, Any]:
        """
        Update position with current market price.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            timestamp: Current timestamp
            
        Returns:
            Dictionary with position update information
        """
        if symbol not in self.positions:
            return {'action': 'none', 'reason': 'no_position'}
        
        position = self.positions[symbol]
        old_price = position['current_price']
        position['current_price'] = current_price
        
        # Calculate unrealized PnL
        if position['size'] > 0:  # Long position
            position['unrealized_pnl'] = position['size'] * (current_price - position['entry_price'])
        else:  # Short position
            position['unrealized_pnl'] = position['size'] * (position['entry_price'] - current_price)
        
        # Update max favorable/adverse excursion
        if position['size'] > 0:  # Long position
            if current_price > position['entry_price']:
                position['max_favorable_excursion'] = max(
                    position['max_favorable_excursion'],
                    current_price - position['entry_price']
                )
            else:
                position['max_adverse_excursion'] = min(
                    position['max_adverse_excursion'],
                    current_price - position['entry_price']
                )
        else:  # Short position
            if current_price < position['entry_price']:
                position['max_favorable_excursion'] = max(
                    position['max_favorable_excursion'],
                    position['entry_price'] - current_price
                )
            else:
                position['max_adverse_excursion'] = min(
                    position['max_adverse_excursion'],
                    position['entry_price'] - current_price
                )
        
        # Check for stop loss or take profit
        action = 'hold'
        reason = ''
        
        if position['size'] > 0:  # Long position
            if current_price <= position['stop_loss_price']:
                action = 'close'
                reason = 'stop_loss'
            elif current_price >= position['take_profit_price']:
                action = 'close'
                reason = 'take_profit'
        else:  # Short position
            if current_price >= position['stop_loss_price']:
                action = 'close'
                reason = 'stop_loss'
            elif current_price <= position['take_profit_price']:
                action = 'close'
                reason = 'take_profit'
        
        # Check for time-based exit (optional)
        position_duration = timestamp - position['entry_timestamp']
        max_position_duration = 3600  # 1 hour maximum
        
        if position_duration > max_position_duration:
            action = 'close'
            reason = 'time_limit'
        
        return {
            'action': action,
            'reason': reason,
            'unrealized_pnl': position['unrealized_pnl'],
            'duration': position_duration
        }
    
    def close_position(self, 
                      symbol: str,
                      exit_price: float,
                      timestamp: float,
                      reason: str = 'manual') -> Dict[str, Any]:
        """
        Close an existing position.
        
        Args:
            symbol: Trading symbol
            exit_price: Exit price
            timestamp: Timestamp of position closing
            reason: Reason for closing
            
        Returns:
            Dictionary with trade information
        """
        if symbol not in self.positions:
            return {'success': False, 'reason': 'no_position'}
        
        position = self.positions[symbol]
        
        # Calculate realized PnL
        if position['size'] > 0:  # Long position
            realized_pnl = position['size'] * (exit_price - position['entry_price'])
        else:  # Short position
            realized_pnl = position['size'] * (position['entry_price'] - exit_price)
        
        # Account for transaction costs
        transaction_cost = abs(position['size'] * exit_price) * 0.001  # 0.1% transaction cost
        net_pnl = realized_pnl - transaction_cost
        
        # Update capital
        self.current_capital += net_pnl
        
        # Update peak capital and drawdown
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Create trade record
        trade = {
            'symbol': symbol,
            'entry_timestamp': position['entry_timestamp'],
            'exit_timestamp': timestamp,
            'entry_datetime': position['entry_datetime'],
            'exit_datetime': datetime.fromtimestamp(timestamp),
            'duration': timestamp - position['entry_timestamp'],
            'size': position['size'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'realized_pnl': realized_pnl,
            'net_pnl': net_pnl,
            'transaction_cost': transaction_cost,
            'return_pct': realized_pnl / abs(position['size'] * position['entry_price']),
            'max_favorable_excursion': position['max_favorable_excursion'],
            'max_adverse_excursion': position['max_adverse_excursion'],
            'exit_reason': reason,
            'signal_info': position['signal_info']
        }
        
        # Store trade
        self.trade_history.append(trade)
        
        # Remove position
        del self.positions[symbol]
        
        logger.info(f"Closed position: {symbol} PnL={net_pnl:.2f} ({trade['return_pct']:.2%}) Reason={reason}")
        
        return {
            'success': True,
            'trade': trade,
            'new_capital': self.current_capital
        }
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get current portfolio summary.
        
        Returns:
            Dictionary with portfolio metrics
        """
        total_unrealized_pnl = sum(pos['unrealized_pnl'] for pos in self.positions.values())
        total_position_value = sum(
            abs(pos['size'] * pos['current_price']) 
            for pos in self.positions.values()
        )
        
        # Calculate portfolio metrics
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        
        if self.trade_history:
            wins = [t for t in self.trade_history if t['net_pnl'] > 0]
            losses = [t for t in self.trade_history if t['net_pnl'] <= 0]
            
            win_rate = len(wins) / len(self.trade_history) if self.trade_history else 0.0
            avg_win = np.mean([t['net_pnl'] for t in wins]) if wins else 0.0
            avg_loss = np.mean([t['net_pnl'] for t in losses]) if losses else 0.0
            profit_factor = abs(sum(t['net_pnl'] for t in wins) / sum(t['net_pnl'] for t in losses)) if losses else np.inf
        else:
            win_rate = 0.0
            avg_win = 0.0
            avg_loss = 0.0
            profit_factor = 0.0
        
        return {
            'current_capital': self.current_capital,
            'initial_capital': self.initial_capital,
            'total_return': total_return,
            'unrealized_pnl': total_unrealized_pnl,
            'total_position_value': total_position_value,
            'exposure': total_position_value / self.current_capital if self.current_capital > 0 else 0.0,
            'max_drawdown': self.max_drawdown,
            'num_positions': len(self.positions),
            'num_trades': len(self.trade_history),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'positions': {symbol: {
                'size': pos['size'],
                'entry_price': pos['entry_price'],
                'current_price': pos['current_price'],
                'unrealized_pnl': pos['unrealized_pnl']
            } for symbol, pos in self.positions.items()}
        }
    
    def calculate_var(self, confidence_level: float = 0.95, 
                     lookback_days: int = 30) -> float:
        """
        Calculate Value at Risk (VaR) based on historical returns.
        
        Args:
            confidence_level: Confidence level for VaR calculation
            lookback_days: Number of days to look back for calculation
            
        Returns:
            VaR value
        """
        if len(self.daily_pnl) < 10:
            return 0.0
        
        # Get recent daily returns
        recent_pnl = self.daily_pnl[-lookback_days:] if len(self.daily_pnl) >= lookback_days else self.daily_pnl
        returns = [pnl / self.initial_capital for pnl in recent_pnl]
        
        # Calculate VaR as percentile
        var = np.percentile(returns, (1 - confidence_level) * 100)
        
        return var * self.current_capital
