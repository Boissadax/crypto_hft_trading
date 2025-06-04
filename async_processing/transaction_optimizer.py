#!/usr/bin/env python3
"""
Transaction Cost Optimization Module
Implements advanced transaction cost minimization strategies for HFT trading
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class OptimizedTransactionCosts:
    """Enhanced transaction cost parameters with optimization features."""
    base_maker_fee: float = 0.0001
    base_taker_fee: float = 0.0002
    base_slippage_bps: float = 0.5
    
    # Volume-based fee tiers
    volume_thresholds: List[float] = None
    maker_fee_tiers: List[float] = None
    taker_fee_tiers: List[float] = None
    
    # Time-based cost adjustments
    low_volume_hours: List[int] = None  # Hours with typically lower liquidity
    high_volume_hours: List[int] = None  # Hours with typically higher liquidity
    
    # Market impact parameters
    market_impact_coefficient: float = 0.1  # sqrt(volume) impact
    temporary_impact_decay: float = 0.8  # How quickly temporary impact decays
    
    def __post_init__(self):
        if self.volume_thresholds is None:
            self.volume_thresholds = [0, 100000, 500000, 1000000]  # USD thresholds
        if self.maker_fee_tiers is None:
            self.maker_fee_tiers = [0.0001, 0.00008, 0.00005, 0.00003]
        if self.taker_fee_tiers is None:
            self.taker_fee_tiers = [0.0002, 0.00015, 0.0001, 0.00008]
        if self.low_volume_hours is None:
            self.low_volume_hours = [0, 1, 2, 3, 4, 5, 22, 23]  # UTC hours
        if self.high_volume_hours is None:
            self.high_volume_hours = [8, 9, 10, 13, 14, 15, 16, 17]  # UTC hours

class TransactionCostOptimizer:
    """Optimizes transaction costs through intelligent order management."""
    
    def __init__(self, cost_params: OptimizedTransactionCosts):
        self.cost_params = cost_params
        self.recent_trades = []  # Track recent trades for volume calculations
        self.market_impact_history = {}  # Track market impact by symbol
        
    def get_optimal_order_parameters(self, symbol: str, side: str, quantity: float, 
                                   current_time: datetime, order_book_state) -> Dict:
        """
        Determine optimal order parameters to minimize transaction costs.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Order quantity in base currency
            current_time: Current timestamp
            order_book_state: Current order book state
            
        Returns:
            Dictionary with optimal order parameters
        """
        
        # Calculate current spread and liquidity
        spread_info = self._analyze_spread_conditions(order_book_state)
        liquidity_info = self._analyze_liquidity_conditions(order_book_state, current_time)
        
        # Determine optimal order type
        order_type = self._choose_optimal_order_type(spread_info, liquidity_info, quantity)
        
        # Calculate optimal order size fragmentation
        order_fragments = self._optimize_order_fragmentation(quantity, order_book_state, side)
        
        # Calculate timing optimization
        timing_delay = self._calculate_optimal_timing_delay(symbol, current_time, order_book_state)
        
        # Estimate total transaction costs
        cost_estimate = self._estimate_transaction_costs(
            symbol, side, quantity, order_type, current_time, order_book_state
        )
        
        return {
            'order_type': order_type,
            'order_fragments': order_fragments,
            'timing_delay_seconds': timing_delay,
            'estimated_cost': cost_estimate,
            'spread_bps': spread_info['spread_bps'],
            'liquidity_score': liquidity_info['liquidity_score'],
            'optimal_price': self._get_optimal_price(order_book_state, side, order_type)
        }
    
    def _analyze_spread_conditions(self, order_book_state) -> Dict:
        """Analyze current spread conditions."""
        if not order_book_state or 1 not in order_book_state.bids or 1 not in order_book_state.asks:
            return {'spread_bps': 999, 'spread_tight': False}
        
        bid_price, _ = order_book_state.bids[1]
        ask_price, _ = order_book_state.asks[1]
        mid_price = (bid_price + ask_price) / 2
        spread = ask_price - bid_price
        spread_bps = (spread / mid_price) * 10000
        
        return {
            'spread_bps': spread_bps,
            'spread_tight': spread_bps < 2.0,  # Consider tight if < 2bps
            'bid_price': bid_price,
            'ask_price': ask_price,
            'mid_price': mid_price
        }
    
    def _analyze_liquidity_conditions(self, order_book_state, current_time: datetime) -> Dict:
        """Analyze current liquidity conditions."""
        if not order_book_state:
            return {'liquidity_score': 0.0, 'depth_score': 0.0}
        
        # Calculate depth at multiple levels
        total_bid_volume = sum(vol for price, vol in order_book_state.bids.values())
        total_ask_volume = sum(vol for price, vol in order_book_state.asks.values())
        total_volume = total_bid_volume + total_ask_volume
        
        # Time-based liquidity adjustment
        hour = current_time.hour
        if hour in self.cost_params.high_volume_hours:
            time_multiplier = 1.2
        elif hour in self.cost_params.low_volume_hours:
            time_multiplier = 0.8
        else:
            time_multiplier = 1.0
        
        liquidity_score = min(total_volume * time_multiplier / 1000, 1.0)  # Normalize to 0-1
        depth_score = min(len(order_book_state.bids) + len(order_book_state.asks), 10) / 10
        
        return {
            'liquidity_score': liquidity_score,
            'depth_score': depth_score,
            'total_volume': total_volume,
            'time_multiplier': time_multiplier
        }
    
    def _choose_optimal_order_type(self, spread_info: Dict, liquidity_info: Dict, quantity: float) -> str:
        """Choose optimal order type to minimize costs."""
        
        # For very tight spreads and good liquidity, prefer limit orders
        if (spread_info['spread_tight'] and 
            liquidity_info['liquidity_score'] > 0.6 and 
            liquidity_info['depth_score'] > 0.5):
            return 'limit'
        
        # For large orders in good liquidity, prefer market orders to ensure execution
        if quantity > 10000 and liquidity_info['liquidity_score'] > 0.7:
            return 'market'
        
        # For wide spreads, prefer limit orders to avoid paying full spread
        if spread_info['spread_bps'] > 5.0:
            return 'limit'
        
        # Default to market for speed and certainty
        return 'market'
    
    def _optimize_order_fragmentation(self, total_quantity: float, order_book_state, side: str) -> List[float]:
        """Optimize order size fragmentation to minimize market impact."""
        
        if not order_book_state or total_quantity < 1000:  # Don't fragment small orders
            return [total_quantity]
        
        # Analyze available liquidity at each level
        relevant_levels = order_book_state.asks if side == 'buy' else order_book_state.bids
        
        fragments = []
        remaining_quantity = total_quantity
        
        # Fragment based on available liquidity at each level
        for level, (price, volume) in relevant_levels.items():
            if remaining_quantity <= 0:
                break
            
            # Take up to 70% of available volume at each level to avoid excessive impact
            fragment_size = min(remaining_quantity, volume * 0.7)
            if fragment_size >= 100:  # Minimum fragment size
                fragments.append(fragment_size)
                remaining_quantity -= fragment_size
        
        # If there's remaining quantity, add it as final fragment
        if remaining_quantity > 0:
            fragments.append(remaining_quantity)
        
        # Ensure we don't create too many tiny fragments
        if len(fragments) > 5:
            # Consolidate smallest fragments
            fragments.sort(reverse=True)
            fragments = fragments[:4] + [sum(fragments[4:])]
        
        return fragments
    
    def _calculate_optimal_timing_delay(self, symbol: str, current_time: datetime, 
                                      order_book_state) -> float:
        """Calculate optimal timing delay to reduce market impact."""
        
        # Check recent market impact for this symbol
        recent_impact = self.market_impact_history.get(symbol, 0.0)
        
        # If recent impact is high, suggest a delay
        if recent_impact > 0.1:  # 10bps impact threshold
            return min(recent_impact * 30, 120)  # Max 2 minute delay
        
        # Check order book stability (simplified)
        spread_info = self._analyze_spread_conditions(order_book_state)
        if spread_info['spread_bps'] > 10:  # Very wide spread
            return 30  # 30 second delay
        
        return 0  # No delay needed
    
    def _estimate_transaction_costs(self, symbol: str, side: str, quantity: float, 
                                  order_type: str, current_time: datetime, 
                                  order_book_state) -> Dict:
        """Estimate total transaction costs for the order."""
        
        # Get current volume tier
        recent_volume = self._calculate_recent_volume()
        fee_tier = self._get_fee_tier(recent_volume)
        
        # Base fees
        if order_type == 'limit':
            base_fee = self.cost_params.maker_fee_tiers[fee_tier]
        else:
            base_fee = self.cost_params.taker_fee_tiers[fee_tier]
        
        # Estimate market impact
        market_impact = self._estimate_market_impact(quantity, order_book_state, side)
        
        # Estimate slippage
        slippage = self.cost_params.base_slippage_bps / 10000
        if order_type == 'market':
            slippage *= 1.5  # Higher slippage for market orders
        
        # Time-based adjustments
        hour = current_time.hour
        if hour in self.cost_params.low_volume_hours:
            slippage *= 1.3  # Higher slippage during low volume
            market_impact *= 1.2
        elif hour in self.cost_params.high_volume_hours:
            slippage *= 0.8  # Lower slippage during high volume
            market_impact *= 0.9
        
        total_cost_bps = (base_fee + market_impact + slippage) * 10000
        
        return {
            'base_fee_bps': base_fee * 10000,
            'market_impact_bps': market_impact * 10000,
            'slippage_bps': slippage * 10000,
            'total_cost_bps': total_cost_bps,
            'total_cost_usd': quantity * total_cost_bps / 10000
        }
    
    def _calculate_recent_volume(self) -> float:
        """Calculate recent trading volume for fee tier determination."""
        cutoff_time = datetime.now() - timedelta(days=30)  # 30-day volume
        recent_trades = [t for t in self.recent_trades if t['timestamp'] > cutoff_time]
        return sum(t['value'] for t in recent_trades)
    
    def _get_fee_tier(self, volume: float) -> int:
        """Get fee tier based on volume."""
        for i, threshold in enumerate(self.cost_params.volume_thresholds):
            if volume < threshold:
                return max(0, i - 1)
        return len(self.cost_params.volume_thresholds) - 1
    
    def _estimate_market_impact(self, quantity: float, order_book_state, side: str) -> float:
        """Estimate market impact of the order."""
        if not order_book_state:
            return 0.01  # 1% default impact
        
        # Simple square-root impact model
        spread_info = self._analyze_spread_conditions(order_book_state)
        
        # Calculate available liquidity
        relevant_levels = order_book_state.asks if side == 'buy' else order_book_state.bids
        total_liquidity = sum(vol for price, vol in relevant_levels.values())
        
        if total_liquidity == 0:
            return 0.05  # 5% impact if no liquidity
        
        # Square root impact model
        liquidity_ratio = quantity / total_liquidity
        impact = self.cost_params.market_impact_coefficient * np.sqrt(liquidity_ratio)
        
        # Adjust based on spread
        spread_adjustment = min(spread_info['spread_bps'] / 100, 2.0)  # Cap at 2x
        
        return min(impact * spread_adjustment, 0.02)  # Cap at 2%
    
    def _get_optimal_price(self, order_book_state, side: str, order_type: str) -> float:
        """Get optimal price for the order."""
        if not order_book_state or 1 not in order_book_state.bids or 1 not in order_book_state.asks:
            return None
        
        bid_price, _ = order_book_state.bids[1]
        ask_price, _ = order_book_state.asks[1]
        mid_price = (bid_price + ask_price) / 2
        
        if order_type == 'market':
            return ask_price if side == 'buy' else bid_price
        else:  # limit order
            if side == 'buy':
                return min(bid_price, mid_price - (ask_price - bid_price) * 0.1)
            else:
                return max(ask_price, mid_price + (ask_price - bid_price) * 0.1)
    
    def record_trade(self, symbol: str, quantity: float, price: float, 
                    actual_costs: Dict, timestamp: datetime):
        """Record executed trade for future optimization."""
        trade_record = {
            'timestamp': timestamp,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'value': quantity * price,
            'actual_costs': actual_costs
        }
        
        self.recent_trades.append(trade_record)
        
        # Update market impact history
        impact = actual_costs.get('market_impact_bps', 0) / 10000
        self.market_impact_history[symbol] = impact
        
        # Keep only recent trades (last 1000)
        if len(self.recent_trades) > 1000:
            self.recent_trades = self.recent_trades[-1000:]
