"""
Feature extraction for cryptocurrency order book data.
Extracts features for machine learning models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class OrderBookFeatureExtractor:
    """
    Extracts features from order book data for machine learning.
    """
    
    def __init__(self, max_levels: int = 10):
        """
        Initialize the feature extractor.
        
        Args:
            max_levels: Maximum number of order book levels to consider
        """
        self.max_levels = max_levels
        self.scaler = StandardScaler()
        
    def extract_spread_features(self, 
                              best_bid: float, 
                              best_ask: float) -> Dict[str, float]:
        """
        Extract bid-ask spread features.
        
        Args:
            best_bid: Best bid price
            best_ask: Best ask price
            
        Returns:
            Dictionary of spread features
        """
        if pd.isna(best_bid) or pd.isna(best_ask):
            return {
                'spread_absolute': np.nan,
                'spread_relative': np.nan,
                'mid_price': np.nan
            }
        
        spread_abs = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2
        spread_rel = spread_abs / mid_price if mid_price > 0 else np.nan
        
        return {
            'spread_absolute': spread_abs,
            'spread_relative': spread_rel,
            'mid_price': mid_price
        }
    
    def extract_imbalance_features(self, 
                                 bid_volumes: List[float],
                                 ask_volumes: List[float]) -> Dict[str, float]:
        """
        Extract order book imbalance features.
        
        Args:
            bid_volumes: List of bid volumes
            ask_volumes: List of ask volumes
            
        Returns:
            Dictionary of imbalance features
        """
        if not bid_volumes or not ask_volumes:
            return {
                'volume_imbalance': np.nan,
                'depth_imbalance': np.nan,
                'total_bid_volume': np.nan,
                'total_ask_volume': np.nan
            }
        
        total_bid_vol = sum(bid_volumes)
        total_ask_vol = sum(ask_volumes)
        
        # Volume imbalance
        vol_imbalance = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol) \
                       if (total_bid_vol + total_ask_vol) > 0 else np.nan
        
        # Depth imbalance (number of levels)
        depth_imbalance = (len(bid_volumes) - len(ask_volumes)) / \
                         (len(bid_volumes) + len(ask_volumes)) \
                         if (len(bid_volumes) + len(ask_volumes)) > 0 else np.nan
        
        return {
            'volume_imbalance': vol_imbalance,
            'depth_imbalance': depth_imbalance,
            'total_bid_volume': total_bid_vol,
            'total_ask_volume': total_ask_vol
        }
    
    def extract_weighted_price_features(self, 
                                      bids: List[Tuple[float, float]],
                                      asks: List[Tuple[float, float]]) -> Dict[str, float]:
        """
        Extract volume-weighted and depth-weighted price features.
        
        Args:
            bids: List of (price, volume) tuples for bids
            asks: List of (price, volume) tuples for asks
            
        Returns:
            Dictionary of weighted price features
        """
        features = {}
        
        # Volume-weighted bid price
        if bids:
            _, bid_volumes = zip(*bids)
            total_bid_volume = sum(bid_volumes)
            if total_bid_volume > 0:
                vwap_bid = sum(p * v for p, v in bids) / total_bid_volume
                features['vwap_bid'] = vwap_bid
            else:
                features['vwap_bid'] = np.nan
        else:
            features['vwap_bid'] = np.nan
            
        # Volume-weighted ask price
        if asks:
            _, ask_volumes = zip(*asks)
            total_ask_volume = sum(ask_volumes)
            if total_ask_volume > 0:
                vwap_ask = sum(p * v for p, v in asks) / total_ask_volume
                features['vwap_ask'] = vwap_ask
            else:
                features['vwap_ask'] = np.nan
        else:
            features['vwap_ask'] = np.nan
            
        # Depth-weighted prices (weighted by inverse of level)
        if bids:
            weights = [1.0 / (i + 1) for i in range(len(bids))]
            total_weight = sum(weights)
            dwap_bid = sum(bids[i][0] * weights[i] for i in range(len(bids))) / total_weight
            features['dwap_bid'] = dwap_bid
        else:
            features['dwap_bid'] = np.nan
            
        if asks:
            weights = [1.0 / (i + 1) for i in range(len(asks))]
            total_weight = sum(weights)
            dwap_ask = sum(asks[i][0] * weights[i] for i in range(len(asks))) / total_weight
            features['dwap_ask'] = dwap_ask
        else:
            features['dwap_ask'] = np.nan
            
        return features
    
    def extract_momentum_features(self, 
                                prices: pd.Series, 
                                windows: List[int]) -> Dict[str, float]:
        """
        Extract price momentum features.
        
        Args:
            prices: Price series
            windows: List of window sizes for momentum calculation
            
        Returns:
            Dictionary of momentum features
        """
        features = {}
        
        for window in windows:
            if len(prices) >= window:
                # Simple momentum (price change)
                momentum = prices.iloc[-1] - prices.iloc[-window]
                features[f'momentum_{window}'] = momentum
                
                # Relative momentum (percentage change)
                if prices.iloc[-window] != 0:
                    rel_momentum = momentum / prices.iloc[-window]
                    features[f'rel_momentum_{window}'] = rel_momentum
                else:
                    features[f'rel_momentum_{window}'] = np.nan
                    
                # Volatility
                if window > 1:
                    returns = prices.pct_change().dropna()
                    if len(returns) >= window - 1:
                        volatility = returns.tail(window - 1).std()
                        features[f'volatility_{window}'] = volatility
                    else:
                        features[f'volatility_{window}'] = np.nan
                else:
                    features[f'volatility_{window}'] = np.nan
            else:
                features[f'momentum_{window}'] = np.nan
                features[f'rel_momentum_{window}'] = np.nan
                features[f'volatility_{window}'] = np.nan
                
        return features
    
    def extract_cross_correlation_features(self, 
                                         price1: pd.Series,
                                         price2: pd.Series,
                                         max_lag: int = 50) -> Dict[str, float]:
        """
        Extract cross-correlation features between two price series.
        
        Args:
            price1: First price series
            price2: Second price series
            max_lag: Maximum lag to consider
            
        Returns:
            Dictionary of cross-correlation features
        """
        features = {}
        
        if len(price1) < 10 or len(price2) < 10:
            return {
                'max_correlation': np.nan,
                'best_lag': np.nan,
                'correlation_at_0': np.nan
            }
        
        # Align series
        min_len = min(len(price1), len(price2))
        p1 = price1.tail(min_len).values
        p2 = price2.tail(min_len).values
        
        # Calculate returns
        r1 = np.diff(p1) / p1[:-1]
        r2 = np.diff(p2) / p2[:-1]
        
        # Remove NaN values
        valid_mask = ~(np.isnan(r1) | np.isnan(r2))
        r1 = r1[valid_mask]
        r2 = r2[valid_mask]
        
        if len(r1) < 5:
            return {
                'max_correlation': np.nan,
                'best_lag': np.nan,
                'correlation_at_0': np.nan
            }
        
        # Cross-correlation
        correlations = []
        lags = range(-max_lag, max_lag + 1)
        
        for lag in lags:
            if lag == 0:
                if len(r1) == len(r2):
                    corr, _ = pearsonr(r1, r2)
                    correlations.append(corr)
                else:
                    correlations.append(np.nan)
            elif lag > 0:
                if len(r1) > lag and len(r2) > lag:
                    corr, _ = pearsonr(r1[:-lag], r2[lag:])
                    correlations.append(corr)
                else:
                    correlations.append(np.nan)
            else:  # lag < 0
                lag_abs = abs(lag)
                if len(r1) > lag_abs and len(r2) > lag_abs:
                    corr, _ = pearsonr(r1[lag_abs:], r2[:-lag_abs])
                    correlations.append(corr)
                else:
                    correlations.append(np.nan)
        
        # Find maximum correlation and corresponding lag
        valid_correlations = [c for c in correlations if not np.isnan(c)]
        if valid_correlations:
            max_corr_idx = np.nanargmax(np.abs(correlations))
            features['max_correlation'] = correlations[max_corr_idx]
            features['best_lag'] = lags[max_corr_idx]
            
            # Correlation at lag 0
            zero_lag_idx = len(lags) // 2
            features['correlation_at_0'] = correlations[zero_lag_idx]
        else:
            features['max_correlation'] = np.nan
            features['best_lag'] = np.nan
            features['correlation_at_0'] = np.nan
            
        return features
    
    def extract_all_features(self, 
                           df: pd.DataFrame,
                           symbol1: str,
                           symbol2: str,
                           window_size: int = 100) -> pd.DataFrame:
        """
        Extract all features from synchronized order book data.
        
        Args:
            df: Synchronized DataFrame with order book data
            symbol1: First symbol (e.g., 'ETH_EUR')
            symbol2: Second symbol (e.g., 'XBT_EUR')
            window_size: Rolling window size for feature calculation
            
        Returns:
            DataFrame with extracted features
        """
        feature_list = []
        
        for i in range(window_size, len(df)):
            features = {'timestamp': df.iloc[i]['timestamp']}
            
            # Current values
            current_row = df.iloc[i]
            
            # Extract features for each symbol
            for symbol in [symbol1, symbol2]:
                # Spread features
                best_bid = current_row[f'{symbol}_best_bid']
                best_ask = current_row[f'{symbol}_best_ask']
                spread_features = self.extract_spread_features(best_bid, best_ask)
                
                for key, value in spread_features.items():
                    features[f'{symbol}_{key}'] = value
                
                # Imbalance features (simplified - using best bid/ask volumes)
                bid_vol = current_row[f'{symbol}_best_bid_vol']
                ask_vol = current_row[f'{symbol}_best_ask_vol']
                
                if not pd.isna(bid_vol) and not pd.isna(ask_vol):
                    imbalance_features = self.extract_imbalance_features([bid_vol], [ask_vol])
                    for key, value in imbalance_features.items():
                        features[f'{symbol}_{key}'] = value
                
                # Momentum features
                price_series = df[f'{symbol}_mid'].iloc[i-window_size:i]
                momentum_features = self.extract_momentum_features(
                    price_series, [5, 10, 20, 50]
                )
                for key, value in momentum_features.items():
                    features[f'{symbol}_{key}'] = value
            
            # Cross-correlation features
            price1_series = df[f'{symbol1}_mid'].iloc[i-window_size:i]
            price2_series = df[f'{symbol2}_mid'].iloc[i-window_size:i]
            
            cross_corr_features = self.extract_cross_correlation_features(
                price1_series, price2_series, max_lag=10
            )
            for key, value in cross_corr_features.items():
                features[f'cross_{key}'] = value
            
            feature_list.append(features)
        
        features_df = pd.DataFrame(feature_list)
        return features_df.fillna(method='ffill').fillna(0)  # Forward fill then zero fill
    
    def create_target_variables(self, 
                              df: pd.DataFrame,
                              symbol: str,
                              horizons: List[float]) -> pd.DataFrame:
        """
        Create target variables for prediction.
        
        Args:
            df: DataFrame with price data
            symbol: Symbol to create targets for
            horizons: List of prediction horizons in seconds
            
        Returns:
            DataFrame with target variables
        """
        targets_df = pd.DataFrame()
        targets_df['timestamp'] = df['timestamp']
        
        price_col = f'{symbol}_mid'
        if price_col not in df.columns:
            logger.warning("Price column %s not found", price_col)
            return targets_df
        
        for horizon in horizons:
            # Convert horizon to number of rows (assuming 100ms frequency)
            horizon_rows = int(horizon * 10)  # 10 rows per second at 100ms frequency
            
            if horizon_rows >= len(df):
                targets_df[f'target_{symbol}_{horizon}s'] = np.nan
                targets_df[f'target_{symbol}_{horizon}s_direction'] = np.nan
                continue
            
            # Calculate future returns
            future_prices = df[price_col].shift(-horizon_rows)
            current_prices = df[price_col]
            
            returns = (future_prices - current_prices) / current_prices
            
            # Binary direction (1 for up, 0 for down)
            direction = (returns > 0).astype(int)
            
            targets_df[f'target_{symbol}_{horizon}s'] = returns
            targets_df[f'target_{symbol}_{horizon}s_direction'] = direction
        
        return targets_df
