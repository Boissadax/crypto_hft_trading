"""
Benchmark Framework

Implements comprehensive benchmarking for lead-lag strategies:
- Buy & Hold benchmark
- Random strategy benchmark  
- Simple momentum strategies
- Performance comparison metrics
- Statistical significance testing
"""

from .backtesting import BacktestEngine
from .strategies import BaseBenchmarkStrategy, BuyHoldStrategy, RandomStrategy, SimpleMomentumStrategy, MeanReversionStrategy
from .metrics import PerformanceAnalyzer, PerformanceMetrics

__all__ = [
    'BacktestEngine',
    'BaseBenchmarkStrategy',
    'BuyHoldStrategy', 
    'RandomStrategy',
    'SimpleMomentumStrategy',
    'MeanReversionStrategy',
    'PerformanceAnalyzer',
    'PerformanceMetrics'
]
