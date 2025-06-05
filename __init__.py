"""
HFT Engine V3 - Optimized High-Frequency Trading Engine
======================================================

Professional-grade high-frequency trading engine optimized for:
- Sub-second lead-lag cross-crypto signals
- Asynchronous order book processing at scale
- Advanced performance analytics and backtesting
"""

__version__ = "3.0.0"
__author__ = "HFT Analytics Team"

# Core optimized components
from core.optimized_engine import OptimizedTradingEngine, OptimizedEngineConfig
from core.streaming_backtest_runner import StreamingBacktestRunner, BacktestConfig
from data.streaming_data_handler import OptimizedDataHandler, OrderBookSnapshot
from strategies.optimized_strategies import (
    OptimizedLeadLagStrategy, 
    OptimizedRandomStrategy, 
    OptimizedBuyHoldStrategy
)

__all__ = [
    # Core engine
    'OptimizedTradingEngine',
    'OptimizedEngineConfig',
    'StreamingBacktestRunner', 
    'BacktestConfig',
    
    # Data handling
    'OptimizedDataHandler',
    'OrderBookSnapshot',
    
    # Strategies
    'OptimizedLeadLagStrategy',
    'OptimizedRandomStrategy', 
    'OptimizedBuyHoldStrategy'
]
