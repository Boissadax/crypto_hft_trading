"""
Asynchronous Lead-Lag Detection Module

Detects lead-lag relationships in high-frequency order book events
without temporal synchronization, using event-based analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque, defaultdict
import logging
from scipy import stats
import warnings

from .event_processor import AsyncEventProcessor, OrderBookEvent, OrderBookState

logger = logging.getLogger(__name__)

@dataclass
class LeadLagSignal:
    """Represents a lead-lag signal between two assets."""
    timestamp: datetime
    leader_symbol: str
    follower_symbol: str
    signal_strength: float  # -1 to 1, strength of lead-lag relationship
    confidence: float  # 0 to 1, confidence in the signal
    lag_microseconds: int  # Lag time in microseconds
    feature_type: str  # Type of feature that triggered the signal
    
class AsyncLeadLagDetector:
    """
    Detects lead-lag relationships in asynchronous order book events.
    Uses event-driven analysis without temporal binning.
    """
    
    def __init__(self,
                 symbols: List[str],
                 max_lag_ms: int = 1000,  # Maximum lag to consider (1 second)
                 min_price_change: float = 0.0001,  # Minimum price change to consider
                 signal_decay_ms: int = 5000):  # Signal decay time (5 seconds)
        """
        Initialize the async lead-lag detector.
        
        Args:
            symbols: List of symbols to analyze
            max_lag_ms: Maximum lag time to consider in milliseconds
            min_price_change: Minimum relative price change to trigger analysis
            signal_decay_ms: Time for signals to decay in milliseconds
        """
        self.symbols = symbols
        self.max_lag_ms = max_lag_ms
        self.min_price_change = min_price_change
        self.signal_decay_ms = signal_decay_ms
        
        # Event buffers for each symbol
        self.price_events: Dict[str, deque] = {symbol: deque() for symbol in symbols}
        self.spread_events: Dict[str, deque] = {symbol: deque() for symbol in symbols}
        self.volume_events: Dict[str, deque] = {symbol: deque() for symbol in symbols}
        
        # Signal storage
        self.lead_lag_signals: List[LeadLagSignal] = []
        
        # Deduplication tracking
        self.recent_signals: Dict[str, datetime] = {}  # Signal hash -> last timestamp
        self.signal_cooldown_ms = 2000  # 2 seconds minimum between similar signals
        
        # Statistics tracking
        self.detection_stats = defaultdict(int)
        
        logger.info(f"Initialized AsyncLeadLagDetector for {symbols}")
        logger.info(f"Max lag: {max_lag_ms}ms, Min price change: {min_price_change}")
    
    def process_event_stream(self, event_processor: AsyncEventProcessor,
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None) -> List[LeadLagSignal]:
        """
        Process event stream and detect lead-lag relationships with progress tracking.
        
        This method preserves the asynchronous nature of events while detecting
        cross-crypto lead-lag patterns without any temporal synchronization.
        
        Args:
            event_processor: Async event processor with loaded events
            start_time: Start time for analysis
            end_time: End time for analysis
            
        Returns:
            List of detected lead-lag signals
        """
        import time
        start_processing_time = time.time()
        
        logger.info("🔍 Starting async lead-lag detection")
        logger.info("   IMPORTANT: No timestamp synchronization - pure event-driven analysis!")
        
        # Try to import tqdm for progress tracking
        try:
            from tqdm import tqdm
            TQDM_AVAILABLE = True
        except ImportError:
            TQDM_AVAILABLE = False
            logger.info("💡 Install tqdm for detailed progress: pip install tqdm")
        
        event_count = 0
        signals_detected = 0
        
        # Get total event count for progress tracking
        total_events = len([e for e in event_processor.get_event_iterator(start_time, end_time)])
        logger.info(f"📊 Processing {total_events:,} events for lead-lag analysis")
        
        # Create progress bar
        if TQDM_AVAILABLE:
            event_iterator = tqdm(event_processor.get_event_iterator(start_time, end_time),
                                total=total_events,
                                desc="Analyzing lead-lag patterns",
                                unit="event",
                                postfix={'signals': 0, 'signal_rate': 0})
        else:
            event_iterator = event_processor.get_event_iterator(start_time, end_time)
        
        # Process events chronologically (preserving asynchronous timing)
        for event in event_iterator:
            event_count += 1
            
            # Update order book state
            event_processor.update_orderbook_state(event)
            
            # Extract features from current state
            current_state = event_processor.order_book_states[event.symbol]
            features = self._extract_event_features(current_state)
            
            # Store features in buffers
            self._update_feature_buffers(event.symbol, event.timestamp, features)
            
            # Detect lead-lag relationships
            new_signals = self._detect_lead_lag_patterns(event.timestamp)
            
            # Filter out duplicate signals
            unique_signals = []
            for signal in new_signals:
                if not self._is_duplicate_signal(signal):
                    unique_signals.append(signal)
                else:
                    self.detection_stats['duplicates_filtered'] += 1
            
            self.lead_lag_signals.extend(unique_signals)
            signals_detected += len(unique_signals)
            
            # Update progress information
            if TQDM_AVAILABLE and event_count > 0:
                signal_rate = (signals_detected / event_count) * 1000  # signals per 1000 events
                event_iterator.set_postfix({
                    'signals': signals_detected,
                    'signal_rate': f"{signal_rate:.1f}/1k"
                })
            
            # Clean old events from buffers
            self._clean_old_events(event.timestamp)
            
            # Progress logging for non-tqdm users
            if not TQDM_AVAILABLE and event_count % 10000 == 0:
                signal_rate = (signals_detected / event_count) * 1000
                logger.info(f"📈 Processed {event_count:,} events, detected {signals_detected} signals (rate: {signal_rate:.1f}/1k events)")
        
        # Final performance summary
        end_processing_time = time.time()
        total_time = end_processing_time - start_processing_time
        processing_speed = event_count / total_time if total_time > 0 else 0
        
        logger.info(f"✅ Lead-lag detection completed!")
        logger.info(f"   Events processed: {event_count:,}")
        logger.info(f"   Signals detected: {len(self.lead_lag_signals):,}")
        logger.info(f"   Processing time: {total_time:.2f}s")
        logger.info(f"   Processing speed: {processing_speed:,.0f} events/second")
        logger.info(f"   Signal detection rate: {(len(self.lead_lag_signals)/event_count)*100:.4f}% of events")
        logger.info("🎯 Asynchronous lead-lag analysis completed - no temporal binning used!")
        
        return self.lead_lag_signals
    
    def _extract_event_features(self, state: OrderBookState) -> Dict[str, float]:
        """Extract relevant features from order book state."""
        features = {}
        
        # Price features
        mid_price = state.get_mid_price()
        if mid_price is not None:
            features['mid_price'] = mid_price
            
            # Price changes (calculated in buffer update)
            features['price_available'] = True
        
        # Spread features
        spread = state.get_spread()
        if spread is not None:
            features['spread'] = spread
            if mid_price is not None:
                features['spread_bps'] = (spread / mid_price) * 10000
        
        # Volume imbalance
        volume_imbalance = state.get_volume_imbalance()
        if volume_imbalance is not None:
            features['volume_imbalance'] = volume_imbalance
        
        # Depth features
        if 1 in state.bids and 1 in state.asks:
            bid_price, bid_qty = state.bids[1]
            ask_price, ask_qty = state.asks[1]
            
            features['best_bid'] = bid_price
            features['best_ask'] = ask_price
            features['bid_qty'] = bid_qty
            features['ask_qty'] = ask_qty
            
            # Multi-level features
            total_bid_qty = sum(qty for _, qty in state.bids.values())
            total_ask_qty = sum(qty for _, qty in state.asks.values())
            
            if total_bid_qty + total_ask_qty > 0:
                features['total_volume_imbalance'] = ((total_bid_qty - total_ask_qty) / 
                                                    (total_bid_qty + total_ask_qty))
        
        return features
    
    def _update_feature_buffers(self, symbol: str, timestamp: datetime, 
                              features: Dict[str, float]) -> None:
        """Update feature buffers with new data point."""
        
        # Update price events
        if 'mid_price' in features:
            # Calculate price changes if we have previous data
            price_change = None
            price_return = None
            
            if self.price_events[symbol]:
                prev_timestamp, prev_price, _, _ = self.price_events[symbol][-1]
                price_change = features['mid_price'] - prev_price
                if prev_price > 0:
                    price_return = price_change / prev_price
            
            self.price_events[symbol].append((
                timestamp, 
                features['mid_price'], 
                price_change, 
                price_return
            ))
        
        # Update spread events
        if 'spread' in features:
            self.spread_events[symbol].append((timestamp, features['spread']))
        
        # Update volume events
        if 'volume_imbalance' in features:
            self.volume_events[symbol].append((timestamp, features['volume_imbalance']))
    
    def _generate_signal_hash(self, signal: LeadLagSignal) -> str:
        """Generate a hash key for signal deduplication."""
        return f"{signal.leader_symbol}_{signal.follower_symbol}_{signal.feature_type}_{int(signal.confidence*1000)}"
    
    def _is_duplicate_signal(self, signal: LeadLagSignal) -> bool:
        """Check if signal is a duplicate of a recent signal."""
        signal_hash = self._generate_signal_hash(signal)
        
        if signal_hash in self.recent_signals:
            time_since_last = (signal.timestamp - self.recent_signals[signal_hash]).total_seconds() * 1000
            if time_since_last < self.signal_cooldown_ms:
                return True
        
        # Update tracking
        self.recent_signals[signal_hash] = signal.timestamp
        return False
    
    def _detect_lead_lag_patterns(self, current_time: datetime) -> List[LeadLagSignal]:
        """Detect lead-lag patterns across symbol pairs or within single symbol features."""
        signals = []
        
        if len(self.symbols) >= 2:
            # Multi-symbol analysis: analyze all symbol pairs
            for i, symbol1 in enumerate(self.symbols):
                for symbol2 in self.symbols[i+1:]:
                    
                    # Price-based lead-lag
                    price_signals = self._detect_price_lead_lag(symbol1, symbol2, current_time)
                    signals.extend(price_signals)
                    
                    # Spread-based lead-lag
                    spread_signals = self._detect_spread_lead_lag(symbol1, symbol2, current_time)
                    signals.extend(spread_signals)
                    
                    # Volume-based lead-lag
                    volume_signals = self._detect_volume_lead_lag(symbol1, symbol2, current_time)
                    signals.extend(volume_signals)
        else:
            # Single symbol analysis: analyze intra-symbol lead-lag relationships
            for symbol in self.symbols:
                intra_signals = self._detect_intra_symbol_lead_lag(symbol, current_time)
                signals.extend(intra_signals)
        
        return signals
    
    def _detect_price_lead_lag(self, symbol1: str, symbol2: str, 
                             current_time: datetime) -> List[LeadLagSignal]:
        """Detect price-based lead-lag relationships."""
        signals = []
        
        # Get recent price events
        events1 = list(self.price_events[symbol1])
        events2 = list(self.price_events[symbol2])
        
        if len(events1) < 2 or len(events2) < 2:
            return signals
        
        # Look for significant price movements in symbol1
        for i, (ts1, price1, change1, return1) in enumerate(events1[-10:], -10):  # Last 10 events
            if return1 is None or abs(return1) < self.min_price_change:
                continue
            
            # Look for corresponding movement in symbol2 within lag window
            lag_start = ts1
            lag_end = ts1 + timedelta(milliseconds=self.max_lag_ms)
            
            for ts2, price2, change2, return2 in events2:
                if ts2 < lag_start or ts2 > lag_end:
                    continue
                
                if return2 is None:
                    continue
                
                # Check if movements are in same direction
                if (return1 > 0 and return2 > 0) or (return1 < 0 and return2 < 0):
                    # Calculate correlation strength
                    lag_microseconds = int((ts2 - ts1).total_seconds() * 1000000)
                    
                    # Simple correlation based on magnitude similarity
                    magnitude_ratio = min(abs(return1), abs(return2)) / max(abs(return1), abs(return2))
                    signal_strength = magnitude_ratio * np.sign(return1)
                    
                    # Confidence based on magnitude and timing
                    confidence = magnitude_ratio * (1 - lag_microseconds / (self.max_lag_ms * 1000))
                    
                    if confidence > 0.2:  # Seuil abaissé pour plus de signaux
                        signal = LeadLagSignal(
                            timestamp=current_time,
                            leader_symbol=symbol1,
                            follower_symbol=symbol2,
                            signal_strength=signal_strength,
                            confidence=confidence,
                            lag_microseconds=lag_microseconds,
                            feature_type='price'
                        )
                        signals.append(signal)
                        self.detection_stats['price_signals'] += 1
        
        return signals
    
    def _detect_spread_lead_lag(self, symbol1: str, symbol2: str,
                              current_time: datetime) -> List[LeadLagSignal]:
        """Detect spread-based lead-lag relationships."""
        signals = []
        
        events1 = list(self.spread_events[symbol1])
        events2 = list(self.spread_events[symbol2])
        
        if len(events1) < 3 or len(events2) < 3:
            return signals
        
        # Calculate spread changes
        for i in range(len(events1) - 1, max(0, len(events1) - 10), -1):
            ts1, spread1 = events1[i]
            prev_ts1, prev_spread1 = events1[i-1]
            
            spread_change1 = (spread1 - prev_spread1) / prev_spread1 if prev_spread1 > 0 else 0
            
            if abs(spread_change1) < 0.005:  # 0.5% minimum spread change (abaissé)
                continue
            
            # Look for corresponding spread changes
            lag_start = ts1
            lag_end = ts1 + timedelta(milliseconds=self.max_lag_ms)
            
            for j in range(len(events2) - 1):
                ts2, spread2 = events2[j]
                if ts2 < lag_start or ts2 > lag_end:
                    continue
                
                if j == 0:
                    continue
                
                prev_ts2, prev_spread2 = events2[j-1]
                spread_change2 = (spread2 - prev_spread2) / prev_spread2 if prev_spread2 > 0 else 0
                
                # Check correlation
                if (spread_change1 > 0 and spread_change2 > 0) or (spread_change1 < 0 and spread_change2 < 0):
                    lag_microseconds = int((ts2 - ts1).total_seconds() * 1000000)
                    magnitude_ratio = min(abs(spread_change1), abs(spread_change2)) / max(abs(spread_change1), abs(spread_change2))
                    
                    signal_strength = magnitude_ratio * np.sign(spread_change1)
                    confidence = magnitude_ratio * (1 - lag_microseconds / (self.max_lag_ms * 1000))
                    
                    if confidence > 0.2:
                        signal = LeadLagSignal(
                            timestamp=current_time,
                            leader_symbol=symbol1,
                            follower_symbol=symbol2,
                            signal_strength=signal_strength,
                            confidence=confidence,
                            lag_microseconds=lag_microseconds,
                            feature_type='spread'
                        )
                        signals.append(signal)
                        self.detection_stats['spread_signals'] += 1
        
        return signals
    
    def _detect_volume_lead_lag(self, symbol1: str, symbol2: str,
                              current_time: datetime) -> List[LeadLagSignal]:
        """Detect volume imbalance-based lead-lag relationships."""
        signals = []
        
        events1 = list(self.volume_events[symbol1])
        events2 = list(self.volume_events[symbol2])
        
        if len(events1) < 5 or len(events2) < 5:
            return signals
        
        # Look for volume imbalance patterns
        recent_events1 = events1[-5:]  # Last 5 events
        
        for ts1, vol_imb1 in recent_events1:
            if abs(vol_imb1) < 0.05:  # Minimum 5% imbalance (abaissé)
                continue
            
            # Look for similar patterns in symbol2
            lag_start = ts1
            lag_end = ts1 + timedelta(milliseconds=self.max_lag_ms)
            
            for ts2, vol_imb2 in events2:
                if ts2 < lag_start or ts2 > lag_end:
                    continue
                
                # Check if imbalances are correlated
                correlation = vol_imb1 * vol_imb2  # Positive if same direction
                
                if correlation > 0 and abs(vol_imb2) > 0.05:  # Same direction, significant
                    lag_microseconds = int((ts2 - ts1).total_seconds() * 1000000)
                    
                    signal_strength = min(abs(vol_imb1), abs(vol_imb2)) * np.sign(correlation)
                    confidence = signal_strength * (1 - lag_microseconds / (self.max_lag_ms * 1000))
                    
                    if confidence > 0.1:
                        signal = LeadLagSignal(
                            timestamp=current_time,
                            leader_symbol=symbol1,
                            follower_symbol=symbol2,
                            signal_strength=signal_strength,
                            confidence=confidence,
                            lag_microseconds=lag_microseconds,
                            feature_type='volume'
                        )
                        signals.append(signal)
                        self.detection_stats['volume_signals'] += 1
        
        return signals
    
    def _detect_intra_symbol_lead_lag(self, symbol: str, current_time: datetime) -> List[LeadLagSignal]:
        """
        Detect lead-lag relationships within a single symbol's features.
        Analyzes relationships between bid/ask, different levels, and feature types.
        """
        signals = []
        
        # Get recent events
        price_events = list(self.price_events[symbol])
        spread_events = list(self.spread_events[symbol])
        volume_events = list(self.volume_events[symbol])
        
        if len(price_events) < 5:
            return signals
        
        # 1. Price momentum vs spread changes
        # Look for price movements that precede spread changes
        for i in range(len(price_events) - 1, max(0, len(price_events) - 5), -1):
            ts_price, price, price_change, price_return = price_events[i]
            
            if price_return is None or abs(price_return) < self.min_price_change:
                continue
            
            # Look for spread changes within lag window
            lag_start = ts_price
            lag_end = ts_price + timedelta(milliseconds=self.max_lag_ms)
            
            for ts_spread, spread in spread_events:
                if ts_spread < lag_start or ts_spread > lag_end:
                    continue
                
                # Calculate spread change from previous
                prev_spread = None
                for prev_ts, prev_sp in reversed(spread_events):
                    if prev_ts < ts_spread:
                        prev_spread = prev_sp
                        break
                
                if prev_spread is None or prev_spread == 0:
                    continue
                
                spread_change = (spread - prev_spread) / prev_spread
                
                # Check if price movement predicts spread change direction
                # Rising prices often lead to tighter spreads, falling prices to wider spreads
                expected_spread_direction = -np.sign(price_return)  # Inverse relationship
                actual_spread_direction = np.sign(spread_change)
                
                if expected_spread_direction == actual_spread_direction and abs(spread_change) > 0.01:
                    lag_microseconds = int((ts_spread - ts_price).total_seconds() * 1000000)
                    
                    signal_strength = min(abs(price_return), abs(spread_change)) * np.sign(price_return)
                    confidence = 0.3 + 0.4 * min(abs(price_return) / 0.001, 1.0)  # Scale with magnitude
                    confidence *= (1 - lag_microseconds / (self.max_lag_ms * 1000))  # Decay with lag
                    
                    if confidence > 0.2:
                        signal = LeadLagSignal(
                            timestamp=current_time,
                            leader_symbol=f"{symbol}_price",
                            follower_symbol=f"{symbol}_spread",
                            signal_strength=signal_strength,
                            confidence=confidence,
                            lag_microseconds=lag_microseconds,
                            feature_type='price_spread'
                        )
                        signals.append(signal)
                        self.detection_stats['intra_price_spread'] += 1
        
        # 2. Volume imbalance vs price changes
        # Look for volume imbalances that precede price movements
        if len(volume_events) >= 3:
            for ts_vol, vol_imbalance in volume_events[-5:]:  # Last 5 volume events
                if abs(vol_imbalance) < 0.1:  # Minimum 10% imbalance
                    continue
                
                # Look for price changes within lag window
                lag_start = ts_vol
                lag_end = ts_vol + timedelta(milliseconds=self.max_lag_ms)
                
                for ts_price, price, price_change, price_return in price_events:
                    if ts_price < lag_start or ts_price > lag_end:
                        continue
                    
                    if price_return is None:
                        continue
                    
                    # Check if volume imbalance predicts price direction
                    # Positive imbalance (more bids) should lead to price increases
                    expected_price_direction = np.sign(vol_imbalance)
                    actual_price_direction = np.sign(price_return)
                    
                    if expected_price_direction == actual_price_direction and abs(price_return) > self.min_price_change:
                        lag_microseconds = int((ts_price - ts_vol).total_seconds() * 1000000)
                        
                        signal_strength = min(abs(vol_imbalance), abs(price_return) / 0.001) * np.sign(vol_imbalance)
                        confidence = 0.4 + 0.3 * min(abs(vol_imbalance), 1.0)
                        confidence *= (1 - lag_microseconds / (self.max_lag_ms * 1000))
                        
                        if confidence > 0.25:
                            signal = LeadLagSignal(
                                timestamp=current_time,
                                leader_symbol=f"{symbol}_volume",
                                follower_symbol=f"{symbol}_price",
                                signal_strength=signal_strength,
                                confidence=confidence,
                                lag_microseconds=lag_microseconds,
                                feature_type='volume_price'
                            )
                            signals.append(signal)
                            self.detection_stats['intra_volume_price'] += 1
        
        # 3. Mean reversion signals
        # Look for extreme price movements that may revert
        if len(price_events) >= 10:
            recent_returns = [ret for _, _, _, ret in price_events[-10:] if ret is not None]
            if len(recent_returns) >= 5:
                # Calculate volatility from recent returns
                volatility = np.std(recent_returns)
                
                for ts_price, price, price_change, price_return in price_events[-3:]:
                    if price_return is None or volatility == 0:
                        continue
                    
                    # Check if return is extreme (> 2 standard deviations)
                    z_score = price_return / volatility
                    
                    if abs(z_score) > 2.0:  # Extreme movement
                        # Generate mean reversion signal
                        signal_strength = -np.sign(price_return) * min(abs(z_score) / 2.0, 1.0)
                        confidence = 0.2 + 0.3 * min((abs(z_score) - 2.0) / 2.0, 1.0)
                        
                        if confidence > 0.15:
                            signal = LeadLagSignal(
                                timestamp=current_time,
                                leader_symbol=f"{symbol}_extrememove",
                                follower_symbol=f"{symbol}_reversion",
                                signal_strength=signal_strength,
                                confidence=confidence,
                                lag_microseconds=int(self.max_lag_ms * 500),  # Medium-term reversion
                                feature_type='mean_reversion'
                            )
                            signals.append(signal)
                            self.detection_stats['intra_mean_reversion'] += 1
        
        return signals
    
    def _clean_old_events(self, current_time: datetime) -> None:
        """Remove events older than the maximum lag window."""
        cutoff_time = current_time - timedelta(milliseconds=self.max_lag_ms * 2)
        
        for symbol in self.symbols:
            # Clean price events
            while (self.price_events[symbol] and 
                   self.price_events[symbol][0][0] < cutoff_time):
                self.price_events[symbol].popleft()
            
            # Clean spread events
            while (self.spread_events[symbol] and 
                   self.spread_events[symbol][0][0] < cutoff_time):
                self.spread_events[symbol].popleft()
            
            # Clean volume events
            while (self.volume_events[symbol] and 
                   self.volume_events[symbol][0][0] < cutoff_time):
                self.volume_events[symbol].popleft()
    
    def get_signal_statistics(self) -> Dict[str, any]:
        """Get statistics about detected signals."""
        if not self.lead_lag_signals:
            return {}
        
        signals_df = pd.DataFrame([
            {
                'timestamp': s.timestamp,
                'leader': s.leader_symbol,
                'follower': s.follower_symbol,
                'strength': s.signal_strength,
                'confidence': s.confidence,
                'lag_ms': s.lag_microseconds / 1000,
                'feature_type': s.feature_type
            }
            for s in self.lead_lag_signals
        ])
        
        stats = {
            'total_signals': len(self.lead_lag_signals),
            'avg_confidence': signals_df['confidence'].mean(),
            'avg_lag_ms': signals_df['lag_ms'].mean(),
            'signal_types': signals_df['feature_type'].value_counts().to_dict(),
            'leader_frequency': signals_df['leader'].value_counts().to_dict(),
            'detection_stats': dict(self.detection_stats)
        }
        
        # Pair-wise statistics
        signals_df['pair'] = signals_df['leader'] + '_' + signals_df['follower']
        stats['pair_statistics'] = {}
        
        for pair in signals_df['pair'].unique():
            pair_data = signals_df[signals_df['pair'] == pair]
            stats['pair_statistics'][pair] = {
                'count': len(pair_data),
                'avg_confidence': pair_data['confidence'].mean(),
                'avg_lag_ms': pair_data['lag_ms'].mean(),
                'avg_strength': pair_data['strength'].mean()
            }
        
        return stats
    
    def filter_signals_by_confidence(self, min_confidence: float = 0.5) -> List[LeadLagSignal]:
        """Filter signals by minimum confidence threshold."""
        return [s for s in self.lead_lag_signals if s.confidence >= min_confidence]
    
    def get_signals_in_timerange(self, start_time: datetime, 
                               end_time: datetime) -> List[LeadLagSignal]:
        """Get signals within a specific time range."""
        return [s for s in self.lead_lag_signals 
                if start_time <= s.timestamp < end_time]
