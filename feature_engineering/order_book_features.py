"""
Order Book Feature Extractor

Extracts comprehensive features from order book snapshots including:
- Price and volume imbalances
- Spread metrics and bid-ask dynamics
- Volume flow indicators
- Order book depth and liquidity measures
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from numba import njit
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class OrderBookFeatures:
    """Container for order book features at a specific timestamp"""
    timestamp_us: int
    symbol: str
    
    # Basic spread metrics
    mid_price: float
    spread: float
    spread_bps: float  # basis points
    
    # Volume imbalances
    bid_volume_l1: float
    ask_volume_l1: float
    volume_imbalance_l1: float  # (bid - ask) / (bid + ask)
    
    # Multi-level metrics (levels 1-5)
    bid_volume_l5: float
    ask_volume_l5: float
    volume_imbalance_l5: float
    
    # Depth metrics
    bid_depth_l5: float  # Total volume in first 5 bid levels
    ask_depth_l5: float  # Total volume in first 5 ask levels
    depth_ratio: float   # bid_depth / ask_depth
    
    # Price impact estimates
    price_impact_100: float  # Price impact for 100 unit trade
    price_impact_1000: float # Price impact for 1000 unit trade
    
    # Weighted prices
    vwap_bid_l5: float  # Volume weighted average bid price (5 levels)
    vwap_ask_l5: float  # Volume weighted average ask price (5 levels)
    
    # Order book slope (price elasticity)
    bid_slope: float  # Average price difference between consecutive bid levels
    ask_slope: float  # Average price difference between consecutive ask levels


class OrderBookFeatureExtractor:
    """
    Extracts comprehensive features from order book snapshots.
    
    Features include imbalances, spreads, depth metrics, and price impact estimates.
    Optimized for real-time processing with Numba acceleration where possible.
    """
    
    def __init__(self, 
                 max_levels: int = 10,
                 volume_impact_sizes: List[float] = [100.0, 1000.0],
                 enable_caching: bool = True):
        """
        Initialize feature extractor
        
        Args:
            max_levels: Maximum number of order book levels to process
            volume_impact_sizes: Trade sizes for price impact calculation
            enable_caching: Whether to cache computed features
        """
        self.max_levels = max_levels
        self.volume_impact_sizes = volume_impact_sizes
        self.enable_caching = enable_caching
        
        # Feature cache for performance
        self._feature_cache = defaultdict(dict) if enable_caching else None
        
        # Statistics tracking
        self.extraction_stats = {
            'total_extractions': 0,
            'cache_hits': 0,
            'invalid_books': 0,
            'avg_extraction_time_us': 0.0
        }
        
    def extract_features(self, 
                        snapshots: List[Dict],
                        symbol: str,
                        timestamp_us: int) -> Optional[OrderBookFeatures]:
        """
        Extract features from order book snapshots at a specific timestamp
        
        Args:
            snapshots: List of order book snapshots (price, volume, side, level)
            symbol: Trading symbol
            timestamp_us: Timestamp in microseconds
            
        Returns:
            OrderBookFeatures object or None if invalid order book
        """
        start_time = pd.Timestamp.now()
        
        # Check cache first
        if self.enable_caching:
            cache_key = f"{symbol}_{timestamp_us}"
            if cache_key in self._feature_cache:
                self.extraction_stats['cache_hits'] += 1
                return self._feature_cache[cache_key]
        
        # Organize snapshots by side and level
        order_book = self._organize_order_book(snapshots)
        
        if not self._is_valid_order_book(order_book):
            self.extraction_stats['invalid_books'] += 1
            return None
            
        # Extract all features
        features = self._compute_all_features(order_book, symbol, timestamp_us)
        
        # Cache if enabled
        if self.enable_caching and features:
            self._feature_cache[cache_key] = features
            
        # Update statistics
        extraction_time = (pd.Timestamp.now() - start_time).total_seconds() * 1_000_000
        self.extraction_stats['total_extractions'] += 1
        self.extraction_stats['avg_extraction_time_us'] = (
            (self.extraction_stats['avg_extraction_time_us'] * 
             (self.extraction_stats['total_extractions'] - 1) + extraction_time) /
            self.extraction_stats['total_extractions']
        )
        
        return features
    
    def _organize_order_book(self, snapshots: List[Dict]) -> Dict[str, Dict[int, Dict]]:
        """Organize snapshots into structured order book by side and level"""
        order_book = {'bid': {}, 'ask': {}}
        
        for snapshot in snapshots:
            side = snapshot['side']
            level = snapshot['level']
            
            order_book[side][level] = {
                'price': snapshot['price'],
                'volume': snapshot['volume']
            }
                
        return order_book
    
    def _is_valid_order_book(self, order_book: Dict) -> bool:
        """Check if order book has minimum required data"""
        # Need at least level 1 on both sides
        return (1 in order_book['bid'] and 1 in order_book['ask'] and
                order_book['bid'][1]['volume'] > 0 and
                order_book['ask'][1]['volume'] > 0 and
                order_book['bid'][1]['price'] < order_book['ask'][1]['price'])
    
    def _compute_all_features(self, 
                            order_book: Dict,
                            symbol: str,
                            timestamp_us: int) -> OrderBookFeatures:
        """Compute all order book features"""
        
        # Basic level 1 metrics
        bid_l1 = order_book['bid'][1]
        ask_l1 = order_book['ask'][1]
        
        mid_price = (bid_l1['price'] + ask_l1['price']) / 2
        spread = ask_l1['price'] - bid_l1['price']
        spread_bps = (spread / mid_price) * 10000 if mid_price > 0 else 0
        
        # Volume imbalances
        volume_imbalance_l1 = self._compute_volume_imbalance(
            bid_l1['volume'], ask_l1['volume']
        )
        
        # Multi-level metrics
        bid_volume_l5, bid_depth_l5, vwap_bid = self._compute_side_metrics(
            order_book['bid'], max_levels=5
        )
        ask_volume_l5, ask_depth_l5, vwap_ask = self._compute_side_metrics(
            order_book['ask'], max_levels=5
        )
        
        volume_imbalance_l5 = self._compute_volume_imbalance(
            bid_volume_l5, ask_volume_l5
        )
        
        depth_ratio = bid_depth_l5 / ask_depth_l5 if ask_depth_l5 > 0 else np.inf
        
        # Price impact estimates
        price_impact_100 = self._estimate_price_impact(order_book, 100.0)
        price_impact_1000 = self._estimate_price_impact(order_book, 1000.0)
        
        # Order book slopes
        bid_slope = self._compute_price_slope(order_book['bid'])
        ask_slope = self._compute_price_slope(order_book['ask'])
        
        return OrderBookFeatures(
            timestamp_us=timestamp_us,
            symbol=symbol,
            mid_price=mid_price,
            spread=spread,
            spread_bps=spread_bps,
            bid_volume_l1=bid_l1['volume'],
            ask_volume_l1=ask_l1['volume'],
            volume_imbalance_l1=volume_imbalance_l1,
            bid_volume_l5=bid_volume_l5,
            ask_volume_l5=ask_volume_l5,
            volume_imbalance_l5=volume_imbalance_l5,
            bid_depth_l5=bid_depth_l5,
            ask_depth_l5=ask_depth_l5,
            depth_ratio=depth_ratio,
            price_impact_100=price_impact_100,
            price_impact_1000=price_impact_1000,
            vwap_bid_l5=vwap_bid,
            vwap_ask_l5=vwap_ask,
            bid_slope=bid_slope,
            ask_slope=ask_slope
        )
    
    @staticmethod
    @njit
    def _compute_volume_imbalance(bid_volume: float, ask_volume: float) -> float:
        """Compute volume imbalance: (bid - ask) / (bid + ask)"""
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0.0
        return (bid_volume - ask_volume) / total_volume
    
    def _compute_side_metrics(self, 
                            side_data: Dict[int, Dict],
                            max_levels: int = 5) -> Tuple[float, float, float]:
        """
        Compute aggregated metrics for one side of the order book
        
        Returns:
            (level_1_volume, total_depth, vwap)
        """
        if not side_data or 1 not in side_data:
            return 0.0, 0.0, 0.0
            
        level_1_volume = side_data[1]['volume']
        total_volume = 0.0
        weighted_price_sum = 0.0
        
        for level in range(1, min(max_levels + 1, max(side_data.keys()) + 1)):
            if level in side_data:
                volume = side_data[level]['volume']
                price = side_data[level]['price']
                
                total_volume += volume
                weighted_price_sum += price * volume
        
        vwap = weighted_price_sum / total_volume if total_volume > 0 else 0.0
        
        return level_1_volume, total_volume, vwap
    
    def _estimate_price_impact(self, order_book: Dict, trade_size: float) -> float:
        """
        Estimate price impact for a market order of given size
        
        Simulates walking through the order book until trade_size is filled
        Returns relative price impact as percentage
        """
        if trade_size <= 0:
            return 0.0
            
        # For simplicity, estimate impact on ask side (buy order)
        ask_side = order_book['ask']
        if not ask_side or 1 not in ask_side:
            return np.inf
            
        remaining_size = trade_size
        total_cost = 0.0
        
        for level in sorted(ask_side.keys()):
            if remaining_size <= 0:
                break
                
            level_data = ask_side[level]
            available_volume = level_data['volume']
            price = level_data['price']
            
            volume_to_take = min(remaining_size, available_volume)
            total_cost += volume_to_take * price
            remaining_size -= volume_to_take
        
        if remaining_size > 0:
            # Could not fill the entire order
            return np.inf
            
        average_price = total_cost / trade_size
        reference_price = ask_side[1]['price']  # Level 1 ask price
        
        return ((average_price - reference_price) / reference_price) * 100
    
    def _compute_price_slope(self, side_data: Dict[int, Dict]) -> float:
        """
        Compute average price difference between consecutive levels
        
        Measures the price elasticity of the order book
        """
        if len(side_data) < 2:
            return 0.0
            
        price_diffs = []
        sorted_levels = sorted(side_data.keys())
        
        for i in range(len(sorted_levels) - 1):
            level1 = sorted_levels[i]
            level2 = sorted_levels[i + 1]
            
            price1 = side_data[level1]['price']
            price2 = side_data[level2]['price']
            
            price_diffs.append(abs(price2 - price1))
        
        return np.mean(price_diffs) if price_diffs else 0.0
    
    def extract_batch_features(self, 
                             order_book_data: List[Tuple],
                             chunk_size: int = 1000) -> pd.DataFrame:
        """
        Extract features for a batch of order book snapshots
        
        Args:
            order_book_data: List of (symbol, timestamp_us, snapshots) tuples
            chunk_size: Size of processing chunks for memory efficiency
            
        Returns:
            DataFrame with extracted features
        """
        features_list = []
        
        for i in range(0, len(order_book_data), chunk_size):
            chunk = order_book_data[i:i + chunk_size]
            
            for symbol, timestamp_us, snapshots in chunk:
                features = self.extract_features(snapshots, symbol, timestamp_us)
                if features:
                    # Convert to dictionary for DataFrame creation
                    feature_dict = {
                        'timestamp_us': features.timestamp_us,
                        'symbol': features.symbol,
                        'mid_price': features.mid_price,
                        'spread': features.spread,
                        'spread_bps': features.spread_bps,
                        'bid_volume_l1': features.bid_volume_l1,
                        'ask_volume_l1': features.ask_volume_l1,
                        'volume_imbalance_l1': features.volume_imbalance_l1,
                        'bid_volume_l5': features.bid_volume_l5,
                        'ask_volume_l5': features.ask_volume_l5,
                        'volume_imbalance_l5': features.volume_imbalance_l5,
                        'bid_depth_l5': features.bid_depth_l5,
                        'ask_depth_l5': features.ask_depth_l5,
                        'depth_ratio': features.depth_ratio,
                        'price_impact_100': features.price_impact_100,
                        'price_impact_1000': features.price_impact_1000,
                        'vwap_bid_l5': features.vwap_bid_l5,
                        'vwap_ask_l5': features.vwap_ask_l5,
                        'bid_slope': features.bid_slope,
                        'ask_slope': features.ask_slope
                    }
                    features_list.append(feature_dict)
                    
            # Log progress
            if logger.isEnabledFor(logging.INFO):
                logger.info(f"Processed {min(i + chunk_size, len(order_book_data))} / {len(order_book_data)} order books")
        
        return pd.DataFrame(features_list)
    
    def get_feature_statistics(self) -> Dict[str, Any]:
        """Get extraction statistics"""
        stats = self.extraction_stats.copy()
        
        if self.enable_caching and self._feature_cache:
            stats['cache_size'] = len(self._feature_cache)
            stats['cache_hit_rate'] = (stats['cache_hits'] / 
                                     max(stats['total_extractions'], 1)) * 100
        
        return stats
    
    def clear_cache(self):
        """Clear feature cache to free memory"""
        if self.enable_caching and self._feature_cache:
            self._feature_cache.clear()
            logger.info("Feature cache cleared")
