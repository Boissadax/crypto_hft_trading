"""
Asynchronous Event Stream Processor for High-Frequency Crypto Trading

This module processes raw order book events without synchronization,
maintaining the exact timing and order of market events for lead-lag analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Iterator, Tuple, Optional, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from collections import deque, defaultdict
import heapq
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class OrderBookEvent:
    """Represents a single order book update event."""
    timestamp: datetime
    symbol: str
    event_type: str  # 'update', 'insert', 'delete'
    side: str  # 'bid' or 'ask'
    level: int  # 1-based order book level
    price: float
    quantity: float
    sequence_id: Optional[int] = None
    
    def __lt__(self, other):
        """Enable priority queue ordering by timestamp."""
        return self.timestamp < other.timestamp

@dataclass
class OrderBookState:
    """Current state of an order book at a given moment."""
    timestamp: datetime
    symbol: str
    bids: Dict[int, Tuple[float, float]] = field(default_factory=dict)  # level -> (price, qty)
    asks: Dict[int, Tuple[float, float]] = field(default_factory=dict)  # level -> (price, qty)
    
    def get_mid_price(self) -> Optional[float]:
        """Calculate mid price from best bid/ask."""
        if 1 in self.bids and 1 in self.asks:
            bid_price, _ = self.bids[1]
            ask_price, _ = self.asks[1]
            return (bid_price + ask_price) / 2
        return None
    
    def get_spread(self) -> Optional[float]:
        """Calculate bid-ask spread."""
        if 1 in self.bids and 1 in self.asks:
            bid_price, _ = self.bids[1]
            ask_price, _ = self.asks[1]
            return ask_price - bid_price
        return None
    
    def get_volume_imbalance(self) -> Optional[float]:
        """Calculate volume imbalance at best level."""
        if 1 in self.bids and 1 in self.asks:
            _, bid_qty = self.bids[1]
            _, ask_qty = self.asks[1]
            total_qty = bid_qty + ask_qty
            if total_qty > 0:
                return (bid_qty - ask_qty) / total_qty
        return None

class AsyncEventProcessor:
    """
    Processes order book events asynchronously without temporal synchronization.
    Maintains exact event timing for proper lead-lag analysis.
    """
    
    def __init__(self, 
                 symbols: List[str],
                 max_levels: int = 5,
                 event_buffer_size: int = 100000):
        """
        Initialize the async event processor.
        
        Args:
            symbols: List of crypto symbols to process
            max_levels: Maximum order book depth levels to track
            event_buffer_size: Maximum events to buffer in memory
        """
        self.symbols = symbols
        self.max_levels = max_levels
        self.event_buffer_size = event_buffer_size
        
        # Event storage
        self.event_stream: List[OrderBookEvent] = []
        self.order_book_states: Dict[str, OrderBookState] = {}
        
        # Initialize order book states
        for symbol in symbols:
            self.order_book_states[symbol] = OrderBookState(
                timestamp=datetime.min,
                symbol=symbol
            )
        
        # Event buffer for real-time processing
        self.event_buffer = deque(maxlen=event_buffer_size)
        
        logger.info(f"Initialized AsyncEventProcessor for {len(symbols)} symbols")
    
    def load_events_from_data(self, raw_data: Dict[str, pd.DataFrame]) -> None:
        """
        Load order book events from raw data files with progress tracking.
        Converts traditional order book snapshots to event stream while preserving
        exact timing and asynchronous nature.
        """
        import time
        start_time = time.time()
        
        logger.info("🔄 Converting order book snapshots to asynchronous event stream")
        logger.info("   IMPORTANT: No timestamp synchronization - preserving original timing!")
        
        # Try to import tqdm for progress tracking
        try:
            from tqdm import tqdm
            TQDM_AVAILABLE = True
        except ImportError:
            TQDM_AVAILABLE = False
            logger.info("💡 Install tqdm for detailed progress: pip install tqdm")
        
        all_events = []
        total_snapshots = sum(len(df) for df in raw_data.values())
        
        # Progress tracking across all symbols
        if TQDM_AVAILABLE:
            symbol_iterator = tqdm(raw_data.items(), 
                                 desc="Processing symbols",
                                 unit="symbol")
        else:
            symbol_iterator = raw_data.items()
            
        for symbol, df in symbol_iterator:
            logger.info(f"📊 Processing {len(df)} snapshots for {symbol}")
            
            # Sort by timestamp to ensure chronological order
            df_sorted = df.sort_values('datetime')
            
            # Convert snapshots to events with progress tracking
            symbol_events = self._snapshots_to_events(symbol, df_sorted)
            all_events.extend(symbol_events)
            
            conversion_rate = len(symbol_events) / len(df) if len(df) > 0 else 0
            logger.info(f"✅ Generated {len(symbol_events)} events for {symbol} (ratio: {conversion_rate:.2f})")
        
        # Sort all events by timestamp - critical for proper lead-lag analysis
        logger.info("🔄 Sorting events chronologically (preserving microsecond precision)...")
        all_events.sort(key=lambda x: x.timestamp)
        self.event_stream = all_events
        
        # Performance summary
        end_time = time.time()
        total_events = len(self.event_stream)
        processing_speed = total_snapshots / (end_time - start_time) if (end_time - start_time) > 0 else 0
        
        logger.info(f"✅ Event stream creation completed!")
        logger.info(f"   Total snapshots processed: {total_snapshots:,}")
        logger.info(f"   Total events generated: {total_events:,}")
        logger.info(f"   Processing time: {end_time - start_time:.2f}s")
        logger.info(f"   Processing speed: {processing_speed:,.0f} snapshots/second")
        logger.info(f"   Conversion ratio: {total_events/total_snapshots:.3f} events/snapshot")
        logger.info(f"   Time range: {self.event_stream[0].timestamp} to {self.event_stream[-1].timestamp}")
        logger.info("🎯 Asynchronous event stream ready - no temporal synchronization applied!")
    
    def _snapshots_to_events(self, symbol: str, df: pd.DataFrame) -> List[OrderBookEvent]:
        """
        Convert order book snapshots to events by detecting changes with progress tracking.
        
        This method preserves the exact asynchronous timing of order book updates
        without any temporal synchronization or rounding.
        """
        # Try to import tqdm for progress tracking
        try:
            from tqdm import tqdm
            TQDM_AVAILABLE = True
        except ImportError:
            TQDM_AVAILABLE = False
        
        events = []
        previous_state = None
        
        # Create progress bar for snapshot processing
        if TQDM_AVAILABLE:
            df_iterator = tqdm(df.iterrows(), 
                             total=len(df),
                             desc=f"Converting {symbol} snapshots",
                             unit="snapshot",
                             leave=False)
        else:
            df_iterator = df.iterrows()
            
        processed_snapshots = 0
        events_generated = 0
        
        for idx, row in df_iterator:
            processed_snapshots += 1
            
            # The datetime is now the index after our fix
            timestamp = idx if isinstance(idx, datetime) else row.get('datetime', idx)
            current_state = self._extract_orderbook_state(symbol, row, timestamp)
            
            if previous_state is not None:
                # Detect changes and create events
                level_events = self._detect_level_changes(previous_state, current_state)
                events.extend(level_events)
                events_generated += len(level_events)
                
                # Update progress postfix for tqdm
                if TQDM_AVAILABLE:
                    df_iterator.set_postfix({
                        'events_generated': events_generated,
                        'events_per_snapshot': f"{events_generated/processed_snapshots:.2f}"
                    })
            
            previous_state = current_state
            
            # Progress logging for non-tqdm users
            if not TQDM_AVAILABLE and processed_snapshots % 1000 == 0:
                logger.info(f"   Processed {processed_snapshots}/{len(df)} snapshots for {symbol}")
        
        logger.info(f"   {symbol}: {events_generated} events from {processed_snapshots} snapshots")
        return events
    
    def _extract_orderbook_state(self, symbol: str, row: pd.Series, timestamp: datetime) -> OrderBookState:
        """Extract order book state from a data row."""
        state = OrderBookState(timestamp=timestamp, symbol=symbol)
        
        # Extract bid levels
        for level in range(1, self.max_levels + 1):
            price_col = f'bid_price_{level}'
            qty_col = f'bid_quantity_{level}'
            
            if price_col in row and qty_col in row and pd.notna(row[price_col]):
                state.bids[level] = (row[price_col], row[qty_col])
        
        # Extract ask levels
        for level in range(1, self.max_levels + 1):
            price_col = f'ask_price_{level}'
            qty_col = f'ask_quantity_{level}'
            
            if price_col in row and qty_col in row and pd.notna(row[price_col]):
                state.asks[level] = (row[price_col], row[qty_col])
        
        return state
    
    def _detect_level_changes(self, prev_state: OrderBookState, 
                            curr_state: OrderBookState) -> List[OrderBookEvent]:
        """Detect changes between two order book states and create events."""
        events = []
        
        # Check bid changes
        for level in range(1, self.max_levels + 1):
            prev_bid = prev_state.bids.get(level)
            curr_bid = curr_state.bids.get(level)
            
            if prev_bid != curr_bid:
                if curr_bid is not None:
                    event_type = 'update' if prev_bid is not None else 'insert'
                    events.append(OrderBookEvent(
                        timestamp=curr_state.timestamp,
                        symbol=curr_state.symbol,
                        event_type=event_type,
                        side='bid',
                        level=level,
                        price=curr_bid[0],
                        quantity=curr_bid[1]
                    ))
                elif prev_bid is not None:
                    events.append(OrderBookEvent(
                        timestamp=curr_state.timestamp,
                        symbol=curr_state.symbol,
                        event_type='delete',
                        side='bid',
                        level=level,
                        price=prev_bid[0],
                        quantity=0.0
                    ))
        
        # Check ask changes
        for level in range(1, self.max_levels + 1):
            prev_ask = prev_state.asks.get(level)
            curr_ask = curr_state.asks.get(level)
            
            if prev_ask != curr_ask:
                if curr_ask is not None:
                    event_type = 'update' if prev_ask is not None else 'insert'
                    events.append(OrderBookEvent(
                        timestamp=curr_state.timestamp,
                        symbol=curr_state.symbol,
                        event_type=event_type,
                        side='ask',
                        level=level,
                        price=curr_ask[0],
                        quantity=curr_ask[1]
                    ))
                elif prev_ask is not None:
                    events.append(OrderBookEvent(
                        timestamp=curr_state.timestamp,
                        symbol=curr_state.symbol,
                        event_type='delete',
                        side='ask',
                        level=level,
                        price=prev_ask[0],
                        quantity=0.0
                    ))
        
        return events
    
    def get_event_iterator(self, 
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None) -> Iterator[OrderBookEvent]:
        """
        Get an iterator over events in chronological order.
        
        Args:
            start_time: Start time filter (inclusive)
            end_time: End time filter (exclusive)
            
        Yields:
            OrderBookEvent: Events in chronological order
        """
        for event in self.event_stream:
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp >= end_time:
                break
            yield event
    
    def update_orderbook_state(self, event: OrderBookEvent) -> None:
        """Update the order book state with a new event."""
        state = self.order_book_states[event.symbol]
        
        # Update timestamp
        state.timestamp = event.timestamp
        
        # Apply the event
        if event.side == 'bid':
            if event.event_type == 'delete' or event.quantity == 0:
                state.bids.pop(event.level, None)
            else:
                state.bids[event.level] = (event.price, event.quantity)
        else:  # ask
            if event.event_type == 'delete' or event.quantity == 0:
                state.asks.pop(event.level, None)
            else:
                state.asks[event.level] = (event.price, event.quantity)
    
    def get_orderbook_snapshot(self, symbol: str, 
                             timestamp: datetime) -> Optional[OrderBookState]:
        """
        Get order book state at a specific timestamp by replaying events.
        """
        # Reset to initial state
        temp_state = OrderBookState(timestamp=datetime.min, symbol=symbol)
        
        # Replay events up to timestamp
        for event in self.event_stream:
            if event.timestamp > timestamp:
                break
            if event.symbol == symbol:
                self._apply_event_to_state(temp_state, event)
        
        if temp_state.timestamp <= timestamp:
            temp_state.timestamp = timestamp
            return temp_state
        
        return None
    
    def _apply_event_to_state(self, state: OrderBookState, event: OrderBookEvent) -> None:
        """Apply an event to a given order book state."""
        state.timestamp = event.timestamp
        
        if event.side == 'bid':
            if event.event_type == 'delete' or event.quantity == 0:
                state.bids.pop(event.level, None)
            else:
                state.bids[event.level] = (event.price, event.quantity)
        else:  # ask
            if event.event_type == 'delete' or event.quantity == 0:
                state.asks.pop(event.level, None)
            else:
                state.asks[event.level] = (event.price, event.quantity)
    
    def get_event_statistics(self) -> Dict[str, Dict]:
        """Get statistics about the event stream."""
        stats = {}
        
        for symbol in self.symbols:
            symbol_events = [e for e in self.event_stream if e.symbol == symbol]
            
            if symbol_events:
                event_types = defaultdict(int)
                for event in symbol_events:
                    event_types[event.event_type] += 1
                
                time_deltas = []
                for i in range(1, len(symbol_events)):
                    delta = (symbol_events[i].timestamp - symbol_events[i-1].timestamp).total_seconds()
                    time_deltas.append(delta)
                
                stats[symbol] = {
                    'total_events': len(symbol_events),
                    'event_types': dict(event_types),
                    'avg_time_between_events': np.mean(time_deltas) if time_deltas else 0,
                    'min_time_between_events': np.min(time_deltas) if time_deltas else 0,
                    'max_time_between_events': np.max(time_deltas) if time_deltas else 0,
                    'start_time': symbol_events[0].timestamp,
                    'end_time': symbol_events[-1].timestamp
                }
        
        return stats
