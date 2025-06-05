"""
Time Series Feature Extractor

Extracts temporal features from time series of order book data including:
- Price returns and volatility measures
- Moving averages and momentum indicators
- Auto-correlation and temporal patterns
- Rolling statistics and trend indicators
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
from numba import njit
from scipy import stats
from collections import deque
import warnings

logger = logging.getLogger(__name__)


@dataclass
class TimeSeriesFeatures:
    """Container for time series features at a specific timestamp"""
    timestamp_us: int
    symbol: str
    
    # Returns and volatility
    return_1s: float      # 1-second return
    return_5s: float      # 5-second return
    return_30s: float     # 30-second return
    volatility_1min: float   # 1-minute rolling volatility
    volatility_5min: float   # 5-minute rolling volatility
    
    # Moving averages
    sma_1min: float       # 1-minute simple moving average
    sma_5min: float       # 5-minute simple moving average
    ema_1min: float       # 1-minute exponential moving average
    ema_5min: float       # 5-minute exponential moving average
    
    # Momentum indicators
    rsi_1min: float       # 1-minute RSI
    momentum_1min: float  # 1-minute momentum
    price_velocity: float # Price velocity (change rate)
    price_acceleration: float # Price acceleration
    
    # Statistical measures
    skewness_1min: float  # 1-minute rolling skewness
    kurtosis_1min: float  # 1-minute rolling kurtosis
    autocorr_lag1: float  # Lag-1 autocorrelation
    hurst_exponent: float # Hurst exponent (trending vs mean-reverting)
    
    # Trend indicators
    trend_strength: float # Trend strength indicator
    trend_direction: int  # Trend direction (-1, 0, 1)
    regime_indicator: float # Market regime indicator
    
    # Volume-price relationships
    volume_price_correlation: float # Rolling volume-price correlation
    vwap_deviation: float # Deviation from VWAP
    
    # High-frequency patterns
    microstructure_noise: float # Microstructure noise estimate
    bid_ask_bounce: float # Bid-ask bounce indicator


class TimeSeriesFeatureExtractor:
    """
    Extracts comprehensive time series features from order book price data.
    
    Maintains rolling windows of historical data and computes various
    temporal indicators optimized for high-frequency trading analysis.
    """
    
    def __init__(self,
                 window_sizes: Dict[str, int] = None,
                 max_history_seconds: int = 3600,
                 sampling_frequency_us: int = 1_000_000,  # 1 second
                 enable_hurst: bool = True):
        """
        Initialize time series feature extractor
        
        Args:
            window_sizes: Dictionary of window sizes for different features
            max_history_seconds: Maximum seconds of history to maintain
            sampling_frequency_us: Sampling frequency in microseconds
            enable_hurst: Whether to compute Hurst exponent (computationally expensive)
        """
        # Default window sizes (in number of observations)
        self.window_sizes = window_sizes or {
            'volatility_short': 60,    # 1 minute at 1Hz
            'volatility_long': 300,    # 5 minutes at 1Hz
            'sma_short': 60,
            'sma_long': 300,
            'ema_short': 60,
            'ema_long': 300,
            'rsi': 60,
            'momentum': 60,
            'statistics': 60,
            'correlation': 300,
            'trend': 300
        }
        
        self.max_history_seconds = max_history_seconds
        self.sampling_frequency_us = sampling_frequency_us
        self.enable_hurst = enable_hurst
        
        # Rolling data storage (symbol -> deque of (timestamp, price, volume))
        self.price_history = {}
        self.volume_history = {}
        self.feature_history = {}
        
        # Pre-computed intermediate values for efficiency
        self.ema_cache = {}  # EMA state cache
        self.rsi_cache = {}  # RSI calculation cache
        
        # Statistics tracking
        self.extraction_stats = {
            'total_extractions': 0,
            'avg_extraction_time_us': 0.0,
            'history_size_bytes': 0
        }
        
    def extract_features(self,
                        timestamp_us: int,
                        symbol: str,
                        mid_price: float,
                        volume: float,
                        additional_data: Dict = None) -> Optional[TimeSeriesFeatures]:
        """
        Extract time series features for a given timestamp and price
        
        Args:
            timestamp_us: Timestamp in microseconds
            symbol: Trading symbol
            mid_price: Mid price at this timestamp
            volume: Volume at this timestamp
            additional_data: Additional data (spread, imbalance, etc.)
            
        Returns:
            TimeSeriesFeatures object or None if insufficient history
        """
        start_time = pd.Timestamp.now()
        
        # Initialize history for new symbol
        if symbol not in self.price_history:
            self._initialize_symbol_history(symbol)
            
        # Add new data point
        self._add_data_point(symbol, timestamp_us, mid_price, volume)
        
        # Clean old data
        self._clean_old_data(symbol, timestamp_us)
        
        # Check if we have sufficient history
        if len(self.price_history[symbol]) < self.window_sizes['volatility_short']:
            return None
            
        # Extract all features
        features = self._compute_all_time_series_features(
            symbol, timestamp_us, additional_data or {}
        )
        
        # Update statistics
        extraction_time = (pd.Timestamp.now() - start_time).total_seconds() * 1_000_000
        self.extraction_stats['total_extractions'] += 1
        self.extraction_stats['avg_extraction_time_us'] = (
            (self.extraction_stats['avg_extraction_time_us'] * 
             (self.extraction_stats['total_extractions'] - 1) + extraction_time) /
            self.extraction_stats['total_extractions']
        )
        
        return features
    
    def _initialize_symbol_history(self, symbol: str):
        """Initialize history storage for a new symbol"""
        max_points = int(self.max_history_seconds * 1_000_000 / self.sampling_frequency_us)
        
        self.price_history[symbol] = deque(maxlen=max_points)
        self.volume_history[symbol] = deque(maxlen=max_points)
        self.feature_history[symbol] = deque(maxlen=max_points)
        
        # Initialize EMA cache
        self.ema_cache[symbol] = {
            'short': None,
            'long': None
        }
        
        # Initialize RSI cache
        self.rsi_cache[symbol] = {
            'gains': deque(maxlen=self.window_sizes['rsi']),
            'losses': deque(maxlen=self.window_sizes['rsi']),
            'avg_gain': 0.0,
            'avg_loss': 0.0
        }
    
    def _add_data_point(self, symbol: str, timestamp_us: int, price: float, volume: float):
        """Add new data point to history"""
        self.price_history[symbol].append((timestamp_us, price))
        self.volume_history[symbol].append((timestamp_us, volume))
    
    def _clean_old_data(self, symbol: str, current_timestamp_us: int):
        """Remove data points older than max_history_seconds"""
        cutoff_time = current_timestamp_us - (self.max_history_seconds * 1_000_000)
        
        # Clean price history
        while (self.price_history[symbol] and 
               self.price_history[symbol][0][0] < cutoff_time):
            self.price_history[symbol].popleft()
            
        # Clean volume history
        while (self.volume_history[symbol] and 
               self.volume_history[symbol][0][0] < cutoff_time):
            self.volume_history[symbol].popleft()
    
    def _compute_all_time_series_features(self,
                                        symbol: str,
                                        timestamp_us: int,
                                        additional_data: Dict) -> TimeSeriesFeatures:
        """Compute all time series features"""
        
        # Get price and volume arrays
        prices = np.array([p[1] for p in self.price_history[symbol]], dtype=np.float64)
        timestamps = np.array([p[0] for p in self.price_history[symbol]], dtype=np.int64)
        volumes = np.array([v[1] for v in self.volume_history[symbol]], dtype=np.float64)
        
        current_price = prices[-1]
        
        # Compute returns
        returns = self._compute_returns(prices, timestamps, timestamp_us)
        
        # Compute volatility measures
        volatility_1min = self._compute_volatility(prices, self.window_sizes['volatility_short'])
        volatility_5min = self._compute_volatility(prices, self.window_sizes['volatility_long'])
        
        # Compute moving averages
        sma_1min = self._compute_sma(prices, self.window_sizes['sma_short'])
        sma_5min = self._compute_sma(prices, self.window_sizes['sma_long'])
        
        ema_1min = self._compute_ema(symbol, current_price, 'short')
        ema_5min = self._compute_ema(symbol, current_price, 'long')
        
        # Compute momentum indicators
        rsi_1min = self._compute_rsi(symbol, prices)
        momentum_1min = self._compute_momentum(prices, self.window_sizes['momentum'])
        
        price_velocity, price_acceleration = self._compute_velocity_acceleration(
            prices, timestamps
        )
        
        # Compute statistical measures
        stats_window = min(len(prices), self.window_sizes['statistics'])
        recent_prices = prices[-stats_window:]
        
        skewness_1min = float(stats.skew(recent_prices)) if len(recent_prices) > 3 else 0.0
        kurtosis_1min = float(stats.kurtosis(recent_prices)) if len(recent_prices) > 3 else 0.0
        
        autocorr_lag1 = self._compute_autocorrelation(recent_prices, lag=1)
        
        # Hurst exponent (computationally expensive, optional)
        hurst_exponent = 0.5  # Default value
        if self.enable_hurst and len(prices) >= 50:
            hurst_exponent = self._compute_hurst_exponent(recent_prices)
        
        # Trend indicators
        trend_strength, trend_direction = self._compute_trend_indicators(prices)
        regime_indicator = self._compute_regime_indicator(prices, volatility_1min)
        
        # Volume-price relationships
        corr_window = min(len(prices), self.window_sizes['correlation'])
        recent_volumes = volumes[-corr_window:]
        recent_prices_for_corr = prices[-corr_window:]
        
        volume_price_correlation = np.corrcoef(
            recent_prices_for_corr, recent_volumes
        )[0, 1] if len(recent_prices_for_corr) > 1 else 0.0
        
        # Handle NaN correlation
        if np.isnan(volume_price_correlation):
            volume_price_correlation = 0.0
        
        # VWAP deviation
        vwap = np.sum(recent_prices_for_corr * recent_volumes) / np.sum(recent_volumes) if np.sum(recent_volumes) > 0 else current_price
        vwap_deviation = (current_price - vwap) / vwap if vwap > 0 else 0.0
        
        # High-frequency patterns
        microstructure_noise = self._estimate_microstructure_noise(prices)
        bid_ask_bounce = additional_data.get('bid_ask_bounce', 0.0)
        
        return TimeSeriesFeatures(
            timestamp_us=timestamp_us,
            symbol=symbol,
            return_1s=returns.get('1s', 0.0),
            return_5s=returns.get('5s', 0.0),
            return_30s=returns.get('30s', 0.0),
            volatility_1min=volatility_1min,
            volatility_5min=volatility_5min,
            sma_1min=sma_1min,
            sma_5min=sma_5min,
            ema_1min=ema_1min,
            ema_5min=ema_5min,
            rsi_1min=rsi_1min,
            momentum_1min=momentum_1min,
            price_velocity=price_velocity,
            price_acceleration=price_acceleration,
            skewness_1min=skewness_1min,
            kurtosis_1min=kurtosis_1min,
            autocorr_lag1=autocorr_lag1,
            hurst_exponent=hurst_exponent,
            trend_strength=trend_strength,
            trend_direction=trend_direction,
            regime_indicator=regime_indicator,
            volume_price_correlation=volume_price_correlation,
            vwap_deviation=vwap_deviation,
            microstructure_noise=microstructure_noise,
            bid_ask_bounce=bid_ask_bounce
        )
    
    def _compute_returns(self, prices: np.ndarray, timestamps: np.ndarray, current_timestamp: int) -> Dict[str, float]:
        """Compute returns at different time horizons"""
        returns = {}
        
        # Define time horizons in microseconds
        horizons = {
            '1s': 1_000_000,
            '5s': 5_000_000,
            '30s': 30_000_000
        }
        
        current_price = prices[-1]
        
        for horizon_name, horizon_us in horizons.items():
            target_time = current_timestamp - horizon_us
            
            # Find the closest timestamp
            time_diffs = np.abs(timestamps - target_time)
            closest_idx = np.argmin(time_diffs)
            
            # Only use if within reasonable tolerance (10% of horizon)
            if time_diffs[closest_idx] <= horizon_us * 0.1:
                past_price = prices[closest_idx]
                returns[horizon_name] = (current_price - past_price) / past_price if past_price > 0 else 0.0
            else:
                returns[horizon_name] = 0.0
        
        return returns
    
    @staticmethod
    @njit
    def _compute_volatility(prices: np.ndarray, window: int) -> float:
        """Compute rolling volatility using returns"""
        if len(prices) < 2:
            return 0.0
            
        window = min(window, len(prices) - 1)
        recent_prices = prices[-(window + 1):]
        
        # Compute log returns
        returns = np.log(recent_prices[1:] / recent_prices[:-1])
        
        # Handle any invalid returns
        valid_returns = returns[np.isfinite(returns)]
        
        if len(valid_returns) < 2:
            return 0.0
            
        return float(np.std(valid_returns))
    
    @staticmethod
    @njit
    def _compute_sma(prices: np.ndarray, window: int) -> float:
        """Compute simple moving average"""
        if len(prices) == 0:
            return 0.0
            
        window = min(window, len(prices))
        return float(np.mean(prices[-window:]))
    
    def _compute_ema(self, symbol: str, current_price: float, period_type: str) -> float:
        """Compute exponential moving average with state caching"""
        alpha_map = {
            'short': 2.0 / (self.window_sizes['ema_short'] + 1),
            'long': 2.0 / (self.window_sizes['ema_long'] + 1)
        }
        
        alpha = alpha_map[period_type]
        
        if self.ema_cache[symbol][period_type] is None:
            # Initialize with current price
            self.ema_cache[symbol][period_type] = current_price
        else:
            # Update EMA
            prev_ema = self.ema_cache[symbol][period_type]
            self.ema_cache[symbol][period_type] = alpha * current_price + (1 - alpha) * prev_ema
        
        return self.ema_cache[symbol][period_type]
    
    def _compute_rsi(self, symbol: str, prices: np.ndarray) -> float:
        """Compute RSI with incremental calculation"""
        if len(prices) < 2:
            return 50.0  # Neutral RSI
            
        # Current price change
        current_change = prices[-1] - prices[-2]
        
        rsi_cache = self.rsi_cache[symbol]
        
        # Add gain/loss to cache
        if current_change > 0:
            rsi_cache['gains'].append(current_change)
            rsi_cache['losses'].append(0.0)
        elif current_change < 0:
            rsi_cache['gains'].append(0.0)
            rsi_cache['losses'].append(-current_change)
        else:
            rsi_cache['gains'].append(0.0)
            rsi_cache['losses'].append(0.0)
        
        # Calculate RSI
        if len(rsi_cache['gains']) < self.window_sizes['rsi']:
            return 50.0  # Not enough data
            
        avg_gain = np.mean(rsi_cache['gains'])
        avg_loss = np.mean(rsi_cache['losses'])
        
        if avg_loss == 0:
            return 100.0
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    @staticmethod
    @njit
    def _compute_momentum(prices: np.ndarray, window: int) -> float:
        """Compute price momentum"""
        if len(prices) < window + 1:
            return 0.0
            
        current_price = prices[-1]
        past_price = prices[-(window + 1)]
        
        return (current_price - past_price) / past_price if past_price > 0 else 0.0
    
    def _compute_velocity_acceleration(self, prices: np.ndarray, timestamps: np.ndarray) -> Tuple[float, float]:
        """Compute price velocity and acceleration"""
        if len(prices) < 3:
            return 0.0, 0.0
            
        # Use last 3 points for velocity/acceleration
        recent_prices = prices[-3:]
        recent_times = timestamps[-3:]
        
        # Convert timestamps to seconds
        times_sec = recent_times / 1_000_000.0
        
        # Compute velocity (price change per second)
        if len(recent_prices) >= 2:
            dt = times_sec[-1] - times_sec[-2]
            if dt > 0:
                velocity = (recent_prices[-1] - recent_prices[-2]) / dt
            else:
                velocity = 0.0
        else:
            velocity = 0.0
            
        # Compute acceleration
        if len(recent_prices) >= 3:
            dt1 = times_sec[-2] - times_sec[-3]
            dt2 = times_sec[-1] - times_sec[-2]
            
            if dt1 > 0 and dt2 > 0:
                velocity1 = (recent_prices[-2] - recent_prices[-3]) / dt1
                velocity2 = (recent_prices[-1] - recent_prices[-2]) / dt2
                acceleration = (velocity2 - velocity1) / ((dt1 + dt2) / 2)
            else:
                acceleration = 0.0
        else:
            acceleration = 0.0
            
        return float(velocity), float(acceleration)
    
    @staticmethod
    def _compute_autocorrelation(prices: np.ndarray, lag: int = 1) -> float:
        """Compute autocorrelation at specified lag"""
        if len(prices) <= lag:
            return 0.0
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            correlation = np.corrcoef(prices[:-lag], prices[lag:])[0, 1]
            
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    def _compute_hurst_exponent(self, prices: np.ndarray) -> float:
        """Compute Hurst exponent using R/S analysis"""
        if len(prices) < 10:
            return 0.5
            
        try:
            # Compute log returns
            log_returns = np.diff(np.log(prices))
            
            # Remove any invalid values
            log_returns = log_returns[np.isfinite(log_returns)]
            
            if len(log_returns) < 10:
                return 0.5
                
            # R/S analysis
            n = len(log_returns)
            mean_return = np.mean(log_returns)
            
            # Cumulative deviations
            cumulative_deviations = np.cumsum(log_returns - mean_return)
            
            # Range
            R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
            
            # Standard deviation
            S = np.std(log_returns)
            
            if S == 0:
                return 0.5
                
            # R/S ratio
            rs_ratio = R / S
            
            # Hurst exponent
            hurst = np.log(rs_ratio) / np.log(n)
            
            # Bound between 0 and 1
            return float(np.clip(hurst, 0.0, 1.0))
            
        except Exception:
            return 0.5
    
    def _compute_trend_indicators(self, prices: np.ndarray) -> Tuple[float, int]:
        """Compute trend strength and direction"""
        if len(prices) < self.window_sizes['trend']:
            return 0.0, 0
            
        window = min(len(prices), self.window_sizes['trend'])
        recent_prices = prices[-window:]
        
        # Linear regression to determine trend
        x = np.arange(len(recent_prices))
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent_prices)
            
            # Trend strength is R-squared
            trend_strength = float(r_value ** 2)
            
            # Trend direction
            if slope > 0 and p_value < 0.05:
                trend_direction = 1  # Uptrend
            elif slope < 0 and p_value < 0.05:
                trend_direction = -1  # Downtrend
            else:
                trend_direction = 0  # No clear trend
                
        except Exception:
            trend_strength = 0.0
            trend_direction = 0
        
        return trend_strength, trend_direction
    
    def _compute_regime_indicator(self, prices: np.ndarray, volatility: float) -> float:
        """Compute market regime indicator (trending vs mean-reverting)"""
        if len(prices) < 20:
            return 0.0
            
        # Use ratio of volatility to absolute price change
        window = min(20, len(prices))
        recent_prices = prices[-window:]
        
        price_range = np.max(recent_prices) - np.min(recent_prices)
        avg_price = np.mean(recent_prices)
        
        if avg_price == 0 or volatility == 0:
            return 0.0
            
        # Normalized price range
        normalized_range = price_range / avg_price
        
        # Regime indicator: high values suggest trending, low values suggest mean-reverting
        regime = normalized_range / volatility if volatility > 0 else 0.0
        
        return float(regime)
    
    def _estimate_microstructure_noise(self, prices: np.ndarray) -> float:
        """Estimate microstructure noise using realized variance decomposition"""
        if len(prices) < 10:
            return 0.0
            
        # Use last 10 prices for noise estimation
        recent_prices = prices[-10:]
        
        # Compute first differences
        first_diff = np.diff(recent_prices)
        
        if len(first_diff) < 2:
            return 0.0
            
        # Noise estimate using variance of first differences
        noise_variance = np.var(first_diff) / 2  # Theoretical relationship
        
        return float(np.sqrt(noise_variance))
    
    def get_feature_statistics(self) -> Dict[str, Any]:
        """Get extraction statistics"""
        stats = self.extraction_stats.copy()
        
        # Calculate memory usage
        total_points = sum(len(hist) for hist in self.price_history.values())
        stats['total_data_points'] = total_points
        stats['symbols_tracked'] = len(self.price_history)
        stats['estimated_memory_mb'] = total_points * 24 / (1024 * 1024)  # 24 bytes per data point
        
        return stats
    
    def clear_history(self, symbol: Optional[str] = None):
        """Clear history for specific symbol or all symbols"""
        if symbol:
            if symbol in self.price_history:
                self.price_history[symbol].clear()
                self.volume_history[symbol].clear()
                self.feature_history[symbol].clear()
                # Reset caches
                self.ema_cache[symbol] = {'short': None, 'long': None}
                self.rsi_cache[symbol] = {
                    'gains': deque(maxlen=self.window_sizes['rsi']),
                    'losses': deque(maxlen=self.window_sizes['rsi']),
                    'avg_gain': 0.0,
                    'avg_loss': 0.0
                }
        else:
            # Clear all
            self.price_history.clear()
            self.volume_history.clear()
            self.feature_history.clear()
            self.ema_cache.clear()
            self.rsi_cache.clear()
            
        logger.info(f"Time series history cleared for {'all symbols' if not symbol else symbol}")
