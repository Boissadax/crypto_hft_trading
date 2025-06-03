"""
Performance metrics calculation for trading strategy evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """
    Calculates various performance metrics for trading strategies.
    """
    
    def __init__(self):
        """Initialize the performance metrics calculator."""
        pass
    
    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """
        Calculate returns from price series.
        
        Args:
            prices: Price series
            
        Returns:
            Returns series
        """
        return prices.pct_change().dropna()
    
    def calculate_sharpe_ratio(self, 
                             returns: pd.Series, 
                             risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Returns series
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Sharpe ratio
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        # Convert to daily risk-free rate
        daily_rf_rate = risk_free_rate / 252
        
        excess_returns = returns - daily_rf_rate
        return excess_returns.mean() / returns.std() * np.sqrt(252)
    
    def calculate_sortino_ratio(self, 
                              returns: pd.Series,
                              risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sortino ratio.
        
        Args:
            returns: Returns series
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Sortino ratio
        """
        if len(returns) == 0:
            return 0.0
        
        daily_rf_rate = risk_free_rate / 252
        excess_returns = returns - daily_rf_rate
        
        # Calculate downside deviation
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return np.inf
        
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
        
        if downside_deviation == 0:
            return np.inf
        
        return excess_returns.mean() / downside_deviation * np.sqrt(252)
    
    def calculate_max_drawdown(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate maximum drawdown.
        
        Args:
            returns: Returns series
            
        Returns:
            Dictionary with max drawdown metrics
        """
        if len(returns) == 0:
            return {'max_drawdown': 0.0, 'max_drawdown_duration': 0}
        
        # Calculate cumulative returns
        cumulative = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = cumulative.expanding().max()
        
        # Calculate drawdown
        drawdown = (cumulative - running_max) / running_max
        
        max_drawdown = drawdown.min()
        
        # Calculate drawdown duration
        drawdown_start = None
        max_duration = 0
        current_duration = 0
        
        for i, dd in enumerate(drawdown):
            if dd < 0:
                if drawdown_start is None:
                    drawdown_start = i
                current_duration = i - drawdown_start + 1
            else:
                if current_duration > max_duration:
                    max_duration = current_duration
                drawdown_start = None
                current_duration = 0
        
        return {
            'max_drawdown': abs(max_drawdown),
            'max_drawdown_duration': max_duration,
            'drawdown_series': drawdown
        }
    
    def calculate_var(self, 
                     returns: pd.Series, 
                     confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: Returns series
            confidence_level: Confidence level for VaR
            
        Returns:
            VaR value
        """
        if len(returns) == 0:
            return 0.0
        
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    def calculate_cvar(self, 
                      returns: pd.Series, 
                      confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR).
        
        Args:
            returns: Returns series
            confidence_level: Confidence level for CVaR
            
        Returns:
            CVaR value
        """
        if len(returns) == 0:
            return 0.0
        
        var = self.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    def calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """
        Calculate Calmar ratio.
        
        Args:
            returns: Returns series
            
        Returns:
            Calmar ratio
        """
        if len(returns) == 0:
            return 0.0
        
        annual_return = (1 + returns.mean()) ** 252 - 1
        max_dd = self.calculate_max_drawdown(returns)['max_drawdown']
        
        if max_dd == 0:
            return np.inf
        
        return annual_return / max_dd
    
    def calculate_hit_ratio(self, predictions: List[int], 
                          actual: List[int]) -> float:
        """
        Calculate hit ratio (accuracy).
        
        Args:
            predictions: Predicted values
            actual: Actual values
            
        Returns:
            Hit ratio
        """
        if len(predictions) != len(actual) or len(predictions) == 0:
            return 0.0
        
        correct = sum(p == a for p, a in zip(predictions, actual))
        return correct / len(predictions)
    
    def calculate_information_ratio(self, 
                                  portfolio_returns: pd.Series,
                                  benchmark_returns: pd.Series) -> float:
        """
        Calculate information ratio.
        
        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Information ratio
        """
        if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) == 0:
            return 0.0
        
        active_returns = portfolio_returns - benchmark_returns
        tracking_error = active_returns.std()
        
        if tracking_error == 0:
            return 0.0
        
        return active_returns.mean() / tracking_error * np.sqrt(252)
    
    def calculate_omega_ratio(self, 
                            returns: pd.Series, 
                            threshold: float = 0.0) -> float:
        """
        Calculate Omega ratio.
        
        Args:
            returns: Returns series
            threshold: Threshold return
            
        Returns:
            Omega ratio
        """
        if len(returns) == 0:
            return 0.0
        
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        
        if losses.sum() == 0:
            return np.inf
        
        return gains.sum() / losses.sum()
    
    def calculate_trade_metrics(self, trades: List[Dict]) -> Dict[str, float]:
        """
        Calculate trading-specific metrics.
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Dictionary with trade metrics
        """
        if not trades:
            return {}
        
        # Extract trade returns
        trade_returns = [trade['return_pct'] for trade in trades if 'return_pct' in trade]
        trade_pnl = [trade['net_pnl'] for trade in trades if 'net_pnl' in trade]
        
        if not trade_returns:
            return {}
        
        # Win/Loss analysis
        wins = [r for r in trade_returns if r > 0]
        losses = [r for r in trade_returns if r <= 0]
        
        # Calculate metrics
        metrics = {
            'total_trades': len(trades),
            'win_rate': len(wins) / len(trade_returns) if trade_returns else 0.0,
            'avg_win': np.mean(wins) if wins else 0.0,
            'avg_loss': np.mean(losses) if losses else 0.0,
            'largest_win': max(wins) if wins else 0.0,
            'largest_loss': min(losses) if losses else 0.0,
            'avg_trade_return': np.mean(trade_returns),
            'total_pnl': sum(trade_pnl) if trade_pnl else 0.0,
            'profit_factor': abs(sum(wins) / sum(losses)) if losses else np.inf,
            'expectancy': np.mean(trade_returns) if trade_returns else 0.0
        }
        
        # Trade duration analysis
        durations = [trade['duration'] for trade in trades if 'duration' in trade]
        if durations:
            metrics.update({
                'avg_trade_duration': np.mean(durations),
                'max_trade_duration': max(durations),
                'min_trade_duration': min(durations)
            })
        
        return metrics
    
    def generate_performance_report(self, 
                                  portfolio_returns: pd.Series,
                                  trades: List[Dict] = None,
                                  benchmark_returns: pd.Series = None) -> Dict[str, float]:
        """
        Generate comprehensive performance report.
        
        Args:
            portfolio_returns: Portfolio returns series
            trades: List of trade dictionaries (optional)
            benchmark_returns: Benchmark returns (optional)
            
        Returns:
            Dictionary with all performance metrics
        """
        report = {}
        
        # Return metrics
        if len(portfolio_returns) > 0:
            cumulative_return = (1 + portfolio_returns).prod() - 1
            annual_return = (1 + portfolio_returns.mean()) ** 252 - 1
            annual_volatility = portfolio_returns.std() * np.sqrt(252)
            
            report.update({
                'cumulative_return': cumulative_return,
                'annual_return': annual_return,
                'annual_volatility': annual_volatility,
                'sharpe_ratio': self.calculate_sharpe_ratio(portfolio_returns),
                'sortino_ratio': self.calculate_sortino_ratio(portfolio_returns),
                'calmar_ratio': self.calculate_calmar_ratio(portfolio_returns),
                'var_95': self.calculate_var(portfolio_returns, 0.95),
                'cvar_95': self.calculate_cvar(portfolio_returns, 0.95),
                'omega_ratio': self.calculate_omega_ratio(portfolio_returns)
            })
            
            # Drawdown metrics
            dd_metrics = self.calculate_max_drawdown(portfolio_returns)
            report.update({
                'max_drawdown': dd_metrics['max_drawdown'],
                'max_drawdown_duration': dd_metrics['max_drawdown_duration']
            })
        
        # Trade metrics
        if trades:
            trade_metrics = self.calculate_trade_metrics(trades)
            report.update(trade_metrics)
        
        # Benchmark comparison
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            report['information_ratio'] = self.calculate_information_ratio(
                portfolio_returns, benchmark_returns
            )
            
            # Tracking error
            if len(portfolio_returns) == len(benchmark_returns):
                tracking_error = (portfolio_returns - benchmark_returns).std() * np.sqrt(252)
                report['tracking_error'] = tracking_error
        
        return report
    
    def calculate_correlation_metrics(self, 
                                    price_series1: pd.Series,
                                    price_series2: pd.Series) -> Dict[str, float]:
        """
        Calculate correlation metrics between two price series.
        
        Args:
            price_series1: First price series
            price_series2: Second price series
            
        Returns:
            Dictionary with correlation metrics
        """
        if len(price_series1) != len(price_series2) or len(price_series1) < 2:
            return {}
        
        # Calculate returns
        returns1 = self.calculate_returns(price_series1)
        returns2 = self.calculate_returns(price_series2)
        
        # Align returns
        min_len = min(len(returns1), len(returns2))
        returns1 = returns1.tail(min_len)
        returns2 = returns2.tail(min_len)
        
        if len(returns1) < 2:
            return {}
        
        # Calculate correlations
        correlation = returns1.corr(returns2)
        
        # Rolling correlation
        rolling_corr = returns1.rolling(window=20).corr(returns2)
        
        return {
            'correlation': correlation,
            'avg_rolling_correlation': rolling_corr.mean(),
            'correlation_volatility': rolling_corr.std(),
            'min_correlation': rolling_corr.min(),
            'max_correlation': rolling_corr.max()
        }
