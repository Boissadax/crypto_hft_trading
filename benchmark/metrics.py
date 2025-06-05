"""
Performance Metrics

Comprehensive performance evaluation metrics for trading strategies:
- Risk-adjusted returns (Sharpe, Sortino, Calmar)
- Maximum drawdown and risk metrics
- Trading performance metrics
- Statistical significance tests
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    
    # Return metrics
    total_return: float
    annualized_return: float
    cumulative_return: float
    
    # Risk metrics
    volatility: float
    annualized_volatility: float
    max_drawdown: float
    max_drawdown_duration: int
    
    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    
    # Trading metrics
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    
    # Statistical metrics
    alpha: float
    beta: float
    r_squared: float
    
    # Additional metrics
    var_95: float
    cvar_95: float
    skewness: float
    kurtosis: float

class PerformanceAnalyzer:
    """Comprehensive performance analysis for trading strategies."""
    
    def __init__(self, 
                 risk_free_rate: float = 0.02,
                 trading_days: int = 252):
        """
        Initialize performance analyzer.
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe calculation
            trading_days: Number of trading days per year
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days
    
    def calculate_metrics(self,
                         returns: pd.Series,
                         benchmark_returns: Optional[pd.Series] = None,
                         trades: Optional[pd.DataFrame] = None) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            returns: Strategy returns series
            benchmark_returns: Benchmark returns for relative metrics
            trades: DataFrame with trade details
            
        Returns:
            PerformanceMetrics object with all calculated metrics
        """
        # Basic return metrics
        total_return = (1 + returns).prod() - 1
        periods_per_year = self.trading_days / len(returns) * len(returns)
        annualized_return = (1 + total_return) ** (self.trading_days / len(returns)) - 1
        cumulative_return = (1 + returns).cumprod() - 1
        
        # Risk metrics
        volatility = returns.std()
        annualized_volatility = volatility * np.sqrt(self.trading_days)
        
        # Drawdown calculations
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Drawdown duration
        drawdown_periods = self._calculate_drawdown_duration(drawdown)
        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        
        # Risk-adjusted metrics
        sharpe_ratio = self._calculate_sharpe(returns, annualized_return, annualized_volatility)
        sortino_ratio = self._calculate_sortino(returns, annualized_return)
        calmar_ratio = self._calculate_calmar(annualized_return, max_drawdown)
        
        # Information ratio (vs benchmark)
        information_ratio = 0.0
        alpha, beta, r_squared = 0.0, 1.0, 0.0
        
        if benchmark_returns is not None:
            information_ratio = self._calculate_information_ratio(returns, benchmark_returns)
            alpha, beta, r_squared = self._calculate_alpha_beta(returns, benchmark_returns)
        
        # Trading metrics
        trading_metrics = self._calculate_trading_metrics(trades) if trades is not None else {
            'total_trades': 0, 'win_rate': 0.0, 'profit_factor': 0.0,
            'avg_win': 0.0, 'avg_loss': 0.0
        }
        
        # Risk metrics
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            cumulative_return=cumulative_return.iloc[-1],
            volatility=volatility,
            annualized_volatility=annualized_volatility,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            information_ratio=information_ratio,
            alpha=alpha,
            beta=beta,
            r_squared=r_squared,
            var_95=var_95,
            cvar_95=cvar_95,
            skewness=skewness,
            kurtosis=kurtosis,
            **trading_metrics
        )
    
    def _calculate_sharpe(self, returns: pd.Series, ann_return: float, ann_vol: float) -> float:
        """Calculate Sharpe ratio."""
        if ann_vol == 0:
            return 0.0
        return (ann_return - self.risk_free_rate) / ann_vol
    
    def _calculate_sortino(self, returns: pd.Series, ann_return: float) -> float:
        """Calculate Sortino ratio."""
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return np.inf
        
        downside_deviation = downside_returns.std() * np.sqrt(self.trading_days)
        if downside_deviation == 0:
            return np.inf
        
        return (ann_return - self.risk_free_rate) / downside_deviation
    
    def _calculate_calmar(self, ann_return: float, max_dd: float) -> float:
        """Calculate Calmar ratio."""
        if max_dd == 0:
            return np.inf
        return ann_return / abs(max_dd)
    
    def _calculate_information_ratio(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate Information ratio vs benchmark."""
        active_returns = returns - benchmark_returns
        tracking_error = active_returns.std() * np.sqrt(self.trading_days)
        
        if tracking_error == 0:
            return 0.0
        
        return active_returns.mean() * self.trading_days / tracking_error
    
    def _calculate_alpha_beta(self, returns: pd.Series, benchmark_returns: pd.Series) -> Tuple[float, float, float]:
        """Calculate alpha, beta, and R-squared vs benchmark."""
        try:
            # Align returns
            aligned_data = pd.concat([returns, benchmark_returns], axis=1).dropna()
            if len(aligned_data) < 2:
                return 0.0, 1.0, 0.0
            
            strategy_returns = aligned_data.iloc[:, 0]
            bench_returns = aligned_data.iloc[:, 1]
            
            # Linear regression
            slope, intercept, r_value, _, _ = stats.linregress(bench_returns, strategy_returns)
            
            beta = slope
            alpha = intercept * self.trading_days  # Annualized alpha
            r_squared = r_value ** 2
            
            return alpha, beta, r_squared
            
        except Exception:
            return 0.0, 1.0, 0.0
    
    def _calculate_drawdown_duration(self, drawdown: pd.Series) -> List[int]:
        """Calculate duration of drawdown periods."""
        is_dd = drawdown < 0
        dd_periods = []
        current_period = 0
        
        for dd in is_dd:
            if dd:
                current_period += 1
            else:
                if current_period > 0:
                    dd_periods.append(current_period)
                current_period = 0
        
        # Add final period if still in drawdown
        if current_period > 0:
            dd_periods.append(current_period)
        
        return dd_periods
    
    def _calculate_trading_metrics(self, trades: pd.DataFrame) -> Dict[str, float]:
        """Calculate trading-specific metrics."""
        if trades is None or len(trades) == 0:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0
            }
        
        # Assume trades DataFrame has 'pnl' column
        if 'pnl' not in trades.columns:
            return {
                'total_trades': len(trades),
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0
            }
        
        total_trades = len(trades)
        winning_trades = trades[trades['pnl'] > 0]
        losing_trades = trades[trades['pnl'] < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
        
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0.0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0.0
        
        gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0.0
        gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0.0
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }

class RiskAnalyzer:
    """Advanced risk analysis for trading strategies."""
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize risk analyzer.
        
        Args:
            confidence_level: Confidence level for risk calculations
        """
        self.confidence_level = confidence_level
    
    def calculate_var_cvar(self, returns: pd.Series, 
                          confidence_level: Optional[float] = None) -> Tuple[float, float]:
        """
        Calculate Value at Risk and Conditional Value at Risk.
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level (default uses instance level)
            
        Returns:
            Tuple of (VaR, CVaR)
        """
        if confidence_level is None:
            confidence_level = self.confidence_level
        
        alpha = 1 - confidence_level
        
        # Historical VaR
        var = np.percentile(returns, alpha * 100)
        
        # CVaR (Expected Shortfall)
        cvar = returns[returns <= var].mean()
        
        return var, cvar
    
    def calculate_maximum_drawdown_details(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Calculate detailed maximum drawdown information.
        
        Args:
            returns: Series of returns
            
        Returns:
            Dictionary with drawdown details
        """
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        
        # Find maximum drawdown
        max_dd_idx = drawdown.idxmin()
        max_dd_value = drawdown.min()
        
        # Find peak before drawdown
        peak_idx = running_max.loc[:max_dd_idx].idxmax()
        peak_value = running_max.loc[peak_idx]
        
        # Find recovery point
        recovery_idx = None
        recovery_value = None
        
        post_dd = cum_returns.loc[max_dd_idx:]
        recovery_candidates = post_dd[post_dd >= peak_value]
        
        if len(recovery_candidates) > 0:
            recovery_idx = recovery_candidates.index[0]
            recovery_value = recovery_candidates.iloc[0]
        
        return {
            'max_drawdown': max_dd_value,
            'peak_date': peak_idx,
            'trough_date': max_dd_idx,
            'recovery_date': recovery_idx,
            'peak_value': peak_value,
            'trough_value': cum_returns.loc[max_dd_idx],
            'recovery_value': recovery_value,
            'drawdown_duration': (max_dd_idx - peak_idx) if peak_idx and max_dd_idx else 0,
            'recovery_duration': (recovery_idx - max_dd_idx) if recovery_idx and max_dd_idx else None
        }
    
    def rolling_risk_metrics(self, 
                           returns: pd.Series,
                           window: int = 252) -> pd.DataFrame:
        """
        Calculate rolling risk metrics.
        
        Args:
            returns: Series of returns
            window: Rolling window size
            
        Returns:
            DataFrame with rolling risk metrics
        """
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = (returns.rolling(window).mean() * 252) / rolling_vol
        
        # Rolling VaR and CVaR
        rolling_var = returns.rolling(window).quantile(0.05)
        rolling_cvar = returns.rolling(window).apply(
            lambda x: x[x <= x.quantile(0.05)].mean()
        )
        
        # Rolling maximum drawdown
        def rolling_max_dd(window_returns):
            if len(window_returns) < 2:
                return 0
            cum_ret = (1 + window_returns).cumprod()
            running_max = cum_ret.expanding().max()
            dd = (cum_ret - running_max) / running_max
            return dd.min()
        
        rolling_max_dd = returns.rolling(window).apply(rolling_max_dd)
        
        return pd.DataFrame({
            'rolling_volatility': rolling_vol,
            'rolling_sharpe': rolling_sharpe,
            'rolling_var': rolling_var,
            'rolling_cvar': rolling_cvar,
            'rolling_max_drawdown': rolling_max_dd
        })

class StatisticalTester:
    """Statistical significance testing for strategy performance."""
    
    def __init__(self):
        """Initialize statistical tester."""
        pass
    
    def t_test_returns(self, 
                      strategy_returns: pd.Series,
                      benchmark_returns: pd.Series) -> Dict[str, Any]:
        """
        Perform t-test comparing strategy vs benchmark returns.
        
        Args:
            strategy_returns: Strategy returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Dictionary with test results
        """
        # Align returns
        aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
        if len(aligned) < 2:
            return {'t_statistic': 0, 'p_value': 1.0, 'significant': False}
        
        strat_ret = aligned.iloc[:, 0]
        bench_ret = aligned.iloc[:, 1]
        
        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(strat_ret, bench_ret)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'strategy_mean': strat_ret.mean(),
            'benchmark_mean': bench_ret.mean(),
            'difference': strat_ret.mean() - bench_ret.mean()
        }
    
    def jarque_bera_test(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Test for normality using Jarque-Bera test.
        
        Args:
            returns: Series of returns
            
        Returns:
            Dictionary with test results
        """
        jb_stat, p_value = stats.jarque_bera(returns.dropna())
        
        return {
            'jb_statistic': jb_stat,
            'p_value': p_value,
            'is_normal': p_value > 0.05,
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns)
        }
    
    def ljung_box_test(self, returns: pd.Series, lags: int = 10) -> Dict[str, Any]:
        """
        Test for autocorrelation using Ljung-Box test.
        
        Args:
            returns: Series of returns
            lags: Number of lags to test
            
        Returns:
            Dictionary with test results
        """
        from statsmodels.stats.diagnostic import acorr_ljungbox
        
        try:
            result = acorr_ljungbox(returns.dropna(), lags=lags, return_df=True)
            
            return {
                'lb_statistic': result['lb_stat'].iloc[-1],
                'p_value': result['lb_pvalue'].iloc[-1],
                'no_autocorrelation': result['lb_pvalue'].iloc[-1] > 0.05,
                'lags_tested': lags
            }
        except Exception as e:
            return {
                'lb_statistic': 0,
                'p_value': 1.0,
                'no_autocorrelation': True,
                'lags_tested': lags,
                'error': str(e)
            }
