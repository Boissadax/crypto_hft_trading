"""
Visualization utilities for crypto HFT trading strategy analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union
import warnings

warnings.filterwarnings('ignore')

class TradingVisualizer:
    """Comprehensive visualization tools for trading analysis."""
    
    def __init__(self, style='seaborn', figsize=(12, 8)):
        try:
            plt.style.use(style)
        except (OSError, FileNotFoundError) as e:
            warnings.warn(f"Matplotlib style '{style}' not found. Using default style. ({e})")
        self.figsize = figsize
        sns.set_palette("husl")
    
    def plot_orderbook_features(self, features_df: pd.DataFrame, 
                              asset: str, 
                              feature_names: List[str],
                              title: str = None) -> plt.Figure:
        """Plot order book features over time."""
        fig, axes = plt.subplots(len(feature_names), 1, figsize=(15, 3*len(feature_names)))
        if len(feature_names) == 1:
            axes = [axes]
            
        for i, feature in enumerate(feature_names):
            if feature in features_df.columns:
                axes[i].plot(features_df.index, features_df[feature], 
                           label=f'{asset} {feature}', linewidth=0.8)
                axes[i].set_ylabel(feature)
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        if title:
            fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        return fig
    
    def plot_price_and_volume(self, data: pd.DataFrame, 
                            asset: str,
                            price_col: str = 'mid_price',
                            volume_col: str = 'volume') -> plt.Figure:
        """Plot price and volume with dual y-axis."""
        fig, ax1 = plt.subplots(figsize=self.figsize)
        
        # Price
        color = 'tab:blue'
        ax1.set_xlabel('Time')
        ax1.set_ylabel(f'{asset} Price', color=color)
        ax1.plot(data.index, data[price_col], color=color, linewidth=1)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)
        
        # Volume
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Volume', color=color)
        ax2.bar(data.index, data[volume_col], color=color, alpha=0.3, width=0.8)
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title(f'{asset} Price and Volume')
        plt.tight_layout()
        return fig
    
    def plot_correlation_matrix(self, features_df: pd.DataFrame,
                               title: str = "Feature Correlation Matrix") -> plt.Figure:
        """Plot correlation matrix heatmap."""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Calculate correlation matrix
        corr_matrix = features_df.corr()
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, fmt='.2f', cbar_kws={"shrink": .5})
        
        plt.title(title, fontsize=16)
        plt.tight_layout()
        return fig
    
    def plot_lead_lag_analysis(self, eth_data: pd.DataFrame, 
                              xbt_data: pd.DataFrame,
                              max_lag: int = 100) -> plt.Figure:
        """Plot lead-lag cross-correlation analysis."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Calculate cross-correlation
        eth_returns = eth_data['mid_price'].pct_change().dropna()
        xbt_returns = xbt_data['mid_price'].pct_change().dropna()
        
        # Align data by timestamp
        aligned_data = pd.concat([eth_returns, xbt_returns], axis=1, join='inner')
        aligned_data.columns = ['ETH', 'XBT']
        aligned_data = aligned_data.dropna()
        
        if len(aligned_data) > max_lag * 2:
            # Cross-correlation
            lags = range(-max_lag, max_lag + 1)
            cross_corr = []
            
            for lag in lags:
                if lag == 0:
                    corr = aligned_data['ETH'].corr(aligned_data['XBT'])
                elif lag > 0:
                    corr = aligned_data['ETH'].iloc[lag:].corr(aligned_data['XBT'].iloc[:-lag])
                else:
                    corr = aligned_data['ETH'].iloc[:lag].corr(aligned_data['XBT'].iloc[-lag:])
                cross_corr.append(corr)
            
            # Plot cross-correlation
            ax1.plot(lags, cross_corr, 'b-', linewidth=2)
            ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax1.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            ax1.set_xlabel('Lag (ETH leads <- -> XBT leads)')
            ax1.set_ylabel('Cross-correlation')
            ax1.set_title('ETH-XBT Cross-correlation Analysis')
            ax1.grid(True, alpha=0.3)
            
            # Find max correlation and lag
            max_corr_idx = np.argmax(np.abs(cross_corr))
            max_lag_val = lags[max_corr_idx]
            max_corr_val = cross_corr[max_corr_idx]
            ax1.plot(max_lag_val, max_corr_val, 'ro', markersize=8)
            ax1.text(max_lag_val, max_corr_val + 0.1, 
                    f'Max: {max_corr_val:.3f} at lag {max_lag_val}')
        
        # Plot price movements
        ax2.plot(aligned_data.index, aligned_data['ETH'].cumsum(), 
                label='ETH Cumulative Returns', alpha=0.8)
        ax2.plot(aligned_data.index, aligned_data['XBT'].cumsum(), 
                label='XBT Cumulative Returns', alpha=0.8)
        ax2.set_ylabel('Cumulative Returns')
        ax2.set_title('Price Movement Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_signal_performance(self, signals: pd.DataFrame,
                               returns: pd.DataFrame,
                               asset: str) -> plt.Figure:
        """Plot trading signals and their performance."""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot price and signals
        ax1.plot(returns.index, returns['price'], 'b-', alpha=0.7, label='Price')
        
        # Buy signals
        buy_signals = signals[signals['signal'] > 0]
        if not buy_signals.empty:
            ax1.scatter(buy_signals.index, 
                       returns.loc[buy_signals.index, 'price'],
                       color='green', marker='^', s=50, label='Buy Signal')
        
        # Sell signals
        sell_signals = signals[signals['signal'] < 0]
        if not sell_signals.empty:
            ax1.scatter(sell_signals.index,
                       returns.loc[sell_signals.index, 'price'],
                       color='red', marker='v', s=50, label='Sell Signal')
        
        ax1.set_ylabel(f'{asset} Price')
        ax1.set_title('Trading Signals')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot signal strength
        ax2.plot(signals.index, signals['signal_strength'], 'purple', linewidth=1)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_ylabel('Signal Strength')
        ax2.set_title('Signal Strength Over Time')
        ax2.grid(True, alpha=0.3)
        
        # Plot cumulative returns
        if 'strategy_returns' in returns.columns:
            ax3.plot(returns.index, returns['strategy_returns'].cumsum(), 
                    'green', label='Strategy Returns', linewidth=2)
        ax3.plot(returns.index, returns['market_returns'].cumsum(), 
                'blue', label='Market Returns', linewidth=2)
        ax3.set_ylabel('Cumulative Returns')
        ax3.set_xlabel('Time')
        ax3.set_title('Strategy vs Market Performance')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_risk_metrics(self, portfolio_data: pd.DataFrame) -> plt.Figure:
        """Plot risk management metrics."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Portfolio value
        ax1.plot(portfolio_data.index, portfolio_data['portfolio_value'], 'b-', linewidth=2)
        ax1.set_ylabel('Portfolio Value')
        ax1.set_title('Portfolio Value Over Time')
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        running_max = portfolio_data['portfolio_value'].cummax()
        drawdown = (portfolio_data['portfolio_value'] - running_max) / running_max * 100
        ax2.fill_between(portfolio_data.index, drawdown, 0, color='red', alpha=0.3)
        ax2.plot(portfolio_data.index, drawdown, 'r-', linewidth=1)
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_title('Portfolio Drawdown')
        ax2.grid(True, alpha=0.3)
        
        # Position size
        ax3.plot(portfolio_data.index, portfolio_data['position_size'], 'g-', linewidth=1)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.set_ylabel('Position Size')
        ax3.set_title('Position Size Over Time')
        ax3.grid(True, alpha=0.3)
        
        # Rolling volatility
        if 'returns' in portfolio_data.columns:
            rolling_vol = portfolio_data['returns'].rolling(window=100).std() * np.sqrt(252)
            ax4.plot(portfolio_data.index, rolling_vol, 'orange', linewidth=1)
            ax4.set_ylabel('Annualized Volatility')
            ax4.set_title('Rolling Volatility (100-period)')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_model_performance(self, predictions: pd.DataFrame,
                              actual: pd.DataFrame,
                              model_name: str = "Model") -> plt.Figure:
        """Plot model prediction performance."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Predictions vs Actual
        ax1.scatter(actual, predictions, alpha=0.5)
        ax1.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--')
        ax1.set_xlabel('Actual')
        ax1.set_ylabel('Predicted')
        ax1.set_title(f'{model_name}: Predictions vs Actual')
        ax1.grid(True, alpha=0.3)
        
        # Time series comparison
        sample_idx = min(1000, len(predictions))
        ax2.plot(range(sample_idx), actual[:sample_idx], 'b-', label='Actual', alpha=0.7)
        ax2.plot(range(sample_idx), predictions[:sample_idx], 'r-', label='Predicted', alpha=0.7)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Value')
        ax2.set_title('Time Series Comparison (First 1000 points)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Prediction errors
        errors = predictions - actual
        ax3.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        ax3.axvline(x=errors.mean(), color='red', linestyle='--', 
                   label=f'Mean: {errors.mean():.4f}')
        ax3.set_xlabel('Prediction Error')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Prediction Error Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Error over time
        ax4.plot(range(len(errors)), errors, 'purple', alpha=0.7, linewidth=0.5)
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Prediction Error')
        ax4.set_title('Prediction Error Over Time')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, feature_importance: Dict[str, float],
                               title: str = "Feature Importance") -> plt.Figure:
        """Plot feature importance from ML models."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        features = list(feature_importance.keys())
        importance = list(feature_importance.values())
        
        # Sort by importance
        sorted_idx = np.argsort(importance)
        sorted_features = [features[i] for i in sorted_idx]
        sorted_importance = [importance[i] for i in sorted_idx]
        
        # Create horizontal bar chart
        ax.barh(range(len(sorted_features)), sorted_importance, color='skyblue', alpha=0.8)
        ax.set_yticks(range(len(sorted_features)))
        ax.set_yticklabels(sorted_features)
        ax.set_xlabel('Importance')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

class LivePlotter:
    """Real-time plotting utilities for live trading."""
    
    def __init__(self, max_points: int = 1000):
        plt.ion()  # Interactive mode
        self.max_points = max_points
        self.data_buffer = {}
        
    def update_plot(self, data: Dict[str, float], plot_name: str):
        """Update real-time plot with new data point."""
        if plot_name not in self.data_buffer:
            self.data_buffer[plot_name] = []
            
        self.data_buffer[plot_name].append(data)
        
        # Keep only recent points
        if len(self.data_buffer[plot_name]) > self.max_points:
            self.data_buffer[plot_name] = self.data_buffer[plot_name][-self.max_points:]
            
        # Update plot (implementation depends on specific requirements)
        self._refresh_plot(plot_name)
    
    def _refresh_plot(self, plot_name: str):
        """Refresh the specified plot."""
        # Implementation for real-time plot updates
        pass

# Utility functions
def save_plot(fig: plt.Figure, filename: str, dpi: int = 300):
    """Save plot to file."""
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    
def close_all_plots():
    """Close all matplotlib figures."""
    plt.close('all')
