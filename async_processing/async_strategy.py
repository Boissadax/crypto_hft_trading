"""
Asynchronous Trading Strategy with Transaction Costs

Implements event-driven trading strategy based on lead-lag signals
with realistic transaction cost modeling and net alpha calculation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
import logging
from enum import Enum

from .event_processor import AsyncEventProcessor, OrderBookEvent, OrderBookState
from .lead_lag_detector import AsyncLeadLagDetector, LeadLagSignal

logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

@dataclass
class Order:
    """Represents a trading order."""
    timestamp: datetime
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    price: Optional[float] = None  # For limit orders
    order_id: Optional[str] = None

@dataclass
class Fill:
    """Represents an order execution."""
    timestamp: datetime
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    slippage: float

@dataclass
class Position:
    """Represents a position in a symbol."""
    symbol: str
    quantity: float = 0.0
    average_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

@dataclass
class TransactionCosts:
    """Transaction cost parameters."""
    maker_fee: float = 0.0001  # 0.01% maker fee
    taker_fee: float = 0.0002  # 0.02% taker fee
    slippage_bps: float = 0.5  # 0.5 bps slippage
    min_commission: float = 0.01  # Minimum commission per trade

class AsyncTradingStrategy:
    """
    Event-driven trading strategy based on asynchronous lead-lag signals.
    Includes realistic transaction cost modeling.
    """
    
    def __init__(self,
                 symbols: List[str],
                 initial_capital: float = 100000.0,
                 position_size: float = 0.1,  # 10% of capital per position
                 max_positions: int = 2,  # Maximum concurrent positions
                 signal_threshold: float = 0.5,  # Minimum signal confidence
                 transaction_costs: Optional[TransactionCosts] = None):
        """
        Initialize the async trading strategy.
        
        Args:
            symbols: List of symbols to trade
            initial_capital: Starting capital
            position_size: Position size as fraction of capital
            max_positions: Maximum number of concurrent positions
            signal_threshold: Minimum signal confidence to act on
            transaction_costs: Transaction cost parameters
        """
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position_size = position_size
        self.max_positions = max_positions
        self.signal_threshold = signal_threshold
        
        if transaction_costs is None:
            self.transaction_costs = TransactionCosts()
        else:
            self.transaction_costs = transaction_costs
        
        # Trading state
        self.positions: Dict[str, Position] = {symbol: Position(symbol) for symbol in symbols}
        self.orders: List[Order] = []
        self.fills: List[Fill] = []
        self.trades_log: List[Dict] = []
        
        # Performance tracking
        self.portfolio_values: List[Tuple[datetime, float]] = []
        self.drawdowns: List[Tuple[datetime, float]] = []
        self.gross_pnl: float = 0.0
        self.net_pnl: float = 0.0
        self.total_commissions: float = 0.0
        self.total_slippage: float = 0.0
        
        # Signal tracking
        self.active_signals: deque = deque(maxlen=1000)
        self.signal_performance: List[Dict] = []
        
        logger.info(f"Initialized AsyncTradingStrategy for {symbols}")
        logger.info(f"Initial capital: ${initial_capital:,.2f}")
        logger.info(f"Transaction costs - Maker: {transaction_costs.maker_fee:.4f}, "
                   f"Taker: {transaction_costs.taker_fee:.4f}")
    
    def run_backtest(self,
                    event_processor: AsyncEventProcessor,
                    lead_lag_detector: AsyncLeadLagDetector,
                    start_time: Optional[datetime] = None,
                    end_time: Optional[datetime] = None) -> Dict:
        """
        Run complete backtest with event-driven strategy.
        
        Args:
            event_processor: Loaded event processor
            lead_lag_detector: Lead-lag detector with signals
            start_time: Backtest start time
            end_time: Backtest end time
            
        Returns:
            Dictionary with backtest results
        """
        logger.info("Starting asynchronous backtest")
        
        # Filter signals for backtest period
        signals = lead_lag_detector.get_signals_in_timerange(start_time, end_time)
        signal_idx = 0
        
        event_count = 0
        trade_count = 0
        
        # Process events chronologically
        for event in event_processor.get_event_iterator(start_time, end_time):
            event_count += 1
            
            # Update order book state
            event_processor.update_orderbook_state(event)
            
            # Process any signals at this timestamp
            while (signal_idx < len(signals) and 
                   signals[signal_idx].timestamp <= event.timestamp):
                
                signal = signals[signal_idx]
                
                # Log signal evaluation
                base_follower = self._extract_base_symbol(signal.follower_symbol)
                should_act = self._should_act_on_signal(signal)
                
                logger.info(f"===== SIGNAL EVALUATION =====")
                logger.info(f"Signal #{signal_idx + 1}/{len(signals)}")
                logger.info(f"Timestamp: {signal.timestamp}")
                logger.info(f"Leader: {signal.leader_symbol}")
                logger.info(f"Follower: {signal.follower_symbol} -> {base_follower}")
                logger.info(f"Confidence: {signal.confidence:.3f}")
                logger.info(f"Signal strength: {signal.signal_strength:.3f}")
                logger.info(f"Should act: {should_act}")
                logger.info(f"Available symbols: {self.symbols}")
                logger.info(f"Current positions: {[(s, abs(p.quantity) > 0) for s, p in self.positions.items()]}")
                
                if should_act:
                    logger.info(f"GENERATING ORDERS for signal...")
                    orders = self._generate_orders_from_signal(signal, event.timestamp)
                    logger.info(f"Generated {len(orders)} orders")
                    
                    # Execute orders
                    for order in orders:
                        logger.info(f"Executing order: {order.side.value} {order.symbol}")
                        fill = self._execute_order(order, event_processor.order_book_states)
                        if fill:
                            self.fills.append(fill)
                            trade_count += 1
                            logger.info(f"✅ TRADE EXECUTED: {fill.side.value} {fill.quantity:.4f} "
                                       f"{fill.symbol} at {fill.price:.6f}")
                        else:
                            logger.warning(f"❌ ORDER EXECUTION FAILED for {order.symbol}")
                else:
                    logger.info(f"Signal rejected - no action taken")
                
                logger.info(f"==============================")
                signal_idx += 1
            
            # Update portfolio value periodically
            if event_count % 1000 == 0:
                self._update_portfolio_value(event.timestamp, event_processor.order_book_states)
            
            # Log progress
            if event_count % 50000 == 0:
                logger.info(f"Processed {event_count} events, executed {trade_count} trades")
        
        # Final portfolio update
        if event_processor.event_stream:
            final_time = event_processor.event_stream[-1].timestamp
            self._update_portfolio_value(final_time, event_processor.order_book_states)
        
        # Calculate final results
        results = self._calculate_backtest_results()
        
        logger.info(f"Backtest completed: {event_count} events, {trade_count} trades")
        logger.info(f"Net PnL: ${results['net_pnl']:,.2f}")
        logger.info(f"Net return: {results['net_return']:.2%}")
        
        return results
    
    def run_backtest_with_learned_patterns(self,
                                          event_processor: AsyncEventProcessor,
                                          learned_signals: List[LeadLagSignal],
                                          start_time: Optional[datetime] = None,
                                          end_time: Optional[datetime] = None) -> Dict:
        """
        Run backtest using learned lead-lag patterns to generate synthetic signals.
        
        This method simulates using historical patterns to make trading decisions
        in the out-of-sample period.
        
        Args:
            event_processor: Loaded event processor
            learned_signals: Signals learned from in-sample period
            start_time: Backtest start time
            end_time: Backtest end time
            
        Returns:
            Dictionary with backtest results
        """
        logger.info("Starting backtest with learned lead-lag patterns")
        
        # Analyze the learned signals to understand patterns
        pattern_analysis = self._analyze_learned_patterns(learned_signals)
        logger.info(f"Learned pattern analysis: {pattern_analysis}")
        
        event_count = 0
        trade_count = 0
        
        # Track recent price movements to generate synthetic signals
        price_history = defaultdict(deque)  # symbol -> price history
        last_prices = {}  # symbol -> last_price
        last_signal_time = defaultdict(lambda: datetime(2020, 1, 1))  # symbol -> last signal timestamp (safe minimum)
        signal_generation_count = defaultdict(int)  # Track signals per symbol to avoid spam
        signal_generation_count = defaultdict(int)  # Track signals per symbol to avoid spam
        
        # Process events chronologically
        for event in event_processor.get_event_iterator(start_time, end_time):
            event_count += 1
            
            # Update order book state
            event_processor.update_orderbook_state(event)
            
            # Track price movements
            ob_state = event_processor.order_book_states.get(event.symbol)
            if ob_state:
                current_price = ob_state.get_mid_price()
                if current_price:
                    # Log price tracking for first few events and debug pricing
                    if event_count <= 10:
                        logger.info(f"Event #{event_count}: {event.symbol} price = {current_price:.6f} (type: {event.event_type})")
                        logger.info(f"  Event details: side={getattr(event, 'side', 'N/A')}, "
                                   f"price={getattr(event, 'price', 'N/A')}, "
                                   f"quantity={getattr(event, 'quantity', 'N/A')}")
                        if event.symbol in last_prices:
                            price_change_debug = (current_price - last_prices[event.symbol]) / last_prices[event.symbol]
                            logger.info(f"  Previous price: {last_prices[event.symbol]:.6f}, change: {price_change_debug:.6f}")
                    
                    # Update price history
                    price_history[event.symbol].append((event.timestamp, current_price))
                    if len(price_history[event.symbol]) > 100:  # Keep last 100 prices
                        price_history[event.symbol].popleft()
                    
                    # Check for price movements that match learned patterns
                    if event.symbol in last_prices:
                        price_change = (current_price - last_prices[event.symbol]) / last_prices[event.symbol]
                        
                        # Debug: Log all price changes to see distribution
                        if event_count % 10 == 0:  # Log every 10th event
                            logger.debug(f"Event #{event_count}: {event.symbol} price_change = {price_change:.8f}")
                        
                        # Log significant price changes
                        if abs(price_change) > 0.00005:  # Lower threshold for debugging
                            logger.info(f"Price change detected: {event.symbol} = {current_price:.6f}, "
                                       f"change = {price_change:.8f} ({price_change*100:.6f}%)")
                        elif abs(price_change) > 0.000001:  # Even lower threshold
                            logger.debug(f"Tiny price change: {event.symbol} = {current_price:.6f}, "
                                        f"change = {price_change:.8f}")
                        
                        # Generate synthetic signals based on learned patterns with smart timing
                        # Avoid generating too many signals too quickly
                        time_since_last_signal = (event.timestamp - last_signal_time[event.symbol]).total_seconds()
                        min_signal_interval = 5.0  # Minimum 5 seconds between signals
                        
                        synthetic_signals = []
                        
                        # Only generate signals if enough time has passed or there's significant price movement
                        should_generate_signal = (
                            time_since_last_signal > min_signal_interval or 
                            abs(price_change) > 0.0001 or  # Significant price movement
                            signal_generation_count[event.symbol] == 0  # First signal
                        )
                        
                        if should_generate_signal:
                            if abs(price_change) > 0.000001:  # Any detectable price change
                                synthetic_signals = self._generate_synthetic_signals(
                                    event, current_price, price_change, pattern_analysis
                                )
                            
                            # If no price-based signals and enough time passed, try microstructure
                            if not synthetic_signals and time_since_last_signal > min_signal_interval * 2:
                                logger.info(f"No price changes detected, trying microstructure-based signals...")
                                synthetic_signals = self._generate_microstructure_signals(
                                    event, event_processor.order_book_states, pattern_analysis
                                )
                            
                            # Only use periodic signals as absolute last resort
                            if (not synthetic_signals and 
                                time_since_last_signal > min_signal_interval * 4 and 
                                signal_generation_count[event.symbol] < 2):  # Limit periodic signals
                                logger.info(f"Generating periodic test signal...")
                                synthetic_signals = self._generate_periodic_signals(
                                    event, current_price, pattern_analysis
                                )
                        else:
                            logger.debug(f"Skipping signal generation: time_since_last={time_since_last_signal:.1f}s < {min_signal_interval}s")
                        
                        if synthetic_signals:
                            logger.info(f"Generated {len(synthetic_signals)} synthetic signals")
                            last_signal_time[event.symbol] = event.timestamp
                            signal_generation_count[event.symbol] += len(synthetic_signals)
                        
                        for signal in synthetic_signals:
                            # Process synthetic signal
                            base_follower = self._extract_base_symbol(signal.follower_symbol)
                            should_act = self._should_act_on_signal(signal)
                            
                            logger.info(f"===== SYNTHETIC SIGNAL =====")
                            logger.info(f"Event #{event_count}")
                            logger.info(f"Trigger: {event.symbol} price change: {price_change:.4f}")
                            logger.info(f"Signal timestamp: {signal.timestamp}")
                            logger.info(f"Leader: {signal.leader_symbol}")
                            logger.info(f"Follower: {signal.follower_symbol} -> {base_follower}")
                            logger.info(f"Confidence: {signal.confidence:.3f}")
                            logger.info(f"Signal strength: {signal.signal_strength:.3f}")
                            logger.info(f"Should act: {should_act}")
                            
                            if should_act:
                                logger.info(f"GENERATING ORDERS for synthetic signal...")
                                orders = self._generate_orders_from_signal(signal, event.timestamp)
                                logger.info(f"Generated {len(orders)} orders")
                                
                                # Execute orders
                                for order in orders:
                                    logger.info(f"Executing order: {order.side.value} {order.symbol}")
                                    fill = self._execute_order(order, event_processor.order_book_states)
                                    if fill:
                                        self.fills.append(fill)
                                        trade_count += 1
                                        logger.info(f"✅ TRADE EXECUTED: {fill.side.value} {fill.quantity:.4f} "
                                                   f"{fill.symbol} at {fill.price:.6f}")
                                    else:
                                        logger.warning(f"❌ ORDER EXECUTION FAILED for {order.symbol}")
                            else:
                                logger.info(f"Synthetic signal rejected - no action taken")
                            
                            logger.info(f"==============================")
                    
                    last_prices[event.symbol] = current_price
            
            # Update portfolio value periodically
            if event_count % 1000 == 0:
                self._update_portfolio_value(event.timestamp, event_processor.order_book_states)
            
            # Log progress
            if event_count % 50000 == 0:
                logger.info(f"Processed {event_count} events, executed {trade_count} trades")
        
        # Final portfolio update
        if event_processor.event_stream:
            final_time = event_processor.event_stream[-1].timestamp
            self._update_portfolio_value(final_time, event_processor.order_book_states)
        
        # Calculate final results
        results = self._calculate_backtest_results()
        
        logger.info(f"Learned patterns backtest completed: {event_count} events, {trade_count} trades")
        logger.info(f"Net PnL: ${results['net_pnl']:,.2f}")
        logger.info(f"Net return: {results['net_return']:.2%}")
        
        return results
    
    def _extract_base_symbol(self, feature_symbol: str) -> str:
        """Extract the base trading symbol from feature-derived symbol names."""
        # Remove feature suffixes like '_extrememove', '_reversion', etc.
        base_symbol = feature_symbol
        
        # List of known feature suffixes to remove
        suffixes = ['_extrememove', '_reversion', '_spread', '_volume', '_momentum', 
                   '_lead', '_lag', '_signal', '_feature', '_delta', '_change']
        
        for suffix in suffixes:
            if base_symbol.endswith(suffix):
                base_symbol = base_symbol[:-len(suffix)]
                break  # Only remove the first matching suffix
        
        # Additional cleanup - remove any trailing underscores or numbers
        base_symbol = base_symbol.rstrip('_0123456789')
        
        # Ensure the symbol is in our tradeable universe
        if base_symbol not in self.symbols:
            # Try to find a match in our symbols list
            for symbol in self.symbols:
                if symbol in base_symbol or base_symbol in symbol:
                    return symbol
            # If no match found, return the original symbol
            logger.warning(f"Could not map feature symbol '{feature_symbol}' to tradeable symbol. "
                          f"Extracted '{base_symbol}', available symbols: {self.symbols}")
        
        return base_symbol
    
    def _should_act_on_signal(self, signal: LeadLagSignal) -> bool:
        """Determine if we should act on a lead-lag signal with adaptive enhanced criteria."""
        
        # Log the signal evaluation for debugging
        logger.debug(f"Evaluating signal: confidence={signal.confidence:.3f}, "
                    f"threshold={self.signal_threshold:.3f}")
        
        # Adaptive confidence filtering based on signal type and market conditions
        if signal.feature_type == 'microstructure':
            min_confidence = 0.20  # Lower threshold for microstructure signals
        elif signal.feature_type == 'periodic_test':
            min_confidence = 0.15  # Lowest threshold for test signals
        elif signal.feature_type == 'price':
            min_confidence = 0.25  # Medium threshold for price signals
        elif signal.feature_type == 'volume':
            min_confidence = 0.22  # Medium-low for volume signals
        else:
            min_confidence = 0.25  # Default threshold
        
        # Dynamic threshold adjustment based on signal strength
        if abs(signal.signal_strength) > 0.8:
            min_confidence *= 0.9  # Reduce threshold for very strong signals
        elif abs(signal.signal_strength) > 0.6:
            min_confidence *= 0.95  # Slightly reduce for strong signals
        elif abs(signal.signal_strength) < 0.3:
            min_confidence *= 1.1  # Increase threshold for weak signals
            
        if signal.confidence < min_confidence:
            logger.debug(f"Signal rejected: confidence {signal.confidence:.3f} < adjusted threshold {min_confidence:.3f}")
            return False
        
        # Extract and validate the follower symbol
        base_follower_symbol = self._extract_base_symbol(signal.follower_symbol)
        if base_follower_symbol not in self.positions:
            logger.debug(f"Signal rejected: symbol '{base_follower_symbol}' not in tradeable universe")
            return False
            
        # Enhanced position management with improved capital allocation
        follower_position = self.positions[base_follower_symbol]
        current_position_size = abs(follower_position.quantity) * follower_position.average_price if follower_position.average_price > 0 else 0
        
        # Dynamic maximum position calculation based on signal quality and market conditions
        base_max_position = self.current_capital * self.position_size
        
        # Adjust max position based on signal quality
        if signal.confidence > 0.4:
            max_position_value = base_max_position * 2.5  # 25% for high-confidence signals
        elif signal.confidence > 0.3:
            max_position_value = base_max_position * 2.0  # 20% for medium-confidence signals
        else:
            max_position_value = base_max_position * 1.5  # 15% for lower-confidence signals
        
        # Check if we can add to position (improved scaling logic)
        if current_position_size > 0:
            existing_direction = 1 if follower_position.quantity > 0 else -1
            signal_direction = 1 if signal.signal_strength > 0 else -1
            
            # Allow position scaling with relaxed criteria
            scale_threshold = 0.25 if signal.feature_type == 'microstructure' else 0.35
            
            if (signal.confidence > scale_threshold and 
                signal_direction == existing_direction and 
                current_position_size < max_position_value * 0.8):  # Allow scaling up to 80% of limit
                logger.info(f"Adding to existing position: confidence={signal.confidence:.3f}, "
                           f"current=${current_position_size:.2f}, max=${max_position_value:.2f}")
            elif signal_direction != existing_direction and signal.confidence > 0.35:
                # Allow position reversal for high-confidence opposing signals
                logger.info(f"Position reversal signal: confidence={signal.confidence:.3f}")
            else:
                logger.debug(f"Signal rejected: position rules - existing size=${current_position_size:.2f}, "
                           f"same direction={signal_direction == existing_direction}, "
                           f"confidence={signal.confidence:.3f}")
                return False
        
        # Adaptive signal strength threshold based on market volatility proxy
        strength_threshold = 0.08  # Base threshold
        if signal.feature_type == 'microstructure':
            strength_threshold = 0.05  # Lower for microstructure
        elif signal.feature_type == 'volume':
            strength_threshold = 0.06  # Lower for volume signals
        
        if abs(signal.signal_strength) < strength_threshold:
            logger.debug(f"Signal rejected: signal strength {signal.signal_strength:.3f} < threshold {strength_threshold:.3f}")
            return False
        
        # Check if we already have maximum total positions across all symbols
        active_positions = sum(1 for pos in self.positions.values() if abs(pos.quantity) > 0)
        if active_positions >= self.max_positions and current_position_size == 0:
            logger.debug(f"Signal rejected: max positions reached ({active_positions}/{self.max_positions})")
            return False
        
        logger.info(f"Signal ACCEPTED: {signal.follower_symbol} -> {base_follower_symbol}, "
                   f"confidence={signal.confidence:.3f}, strength={signal.signal_strength:.3f}")
        return True
    
    def _generate_orders_from_signal(self, signal: LeadLagSignal, 
                                   current_time: datetime) -> List[Order]:
        """Generate trading orders from a lead-lag signal with enhanced position sizing."""
        orders = []
        
        # Extract base symbol for trading
        base_follower_symbol = self._extract_base_symbol(signal.follower_symbol)
        
        # Verify we can trade this symbol
        if base_follower_symbol not in self.symbols:
            logger.warning(f"Cannot trade symbol {base_follower_symbol} - not in tradeable universe")
            return orders
        
        # Determine trade direction based on signal
        if signal.signal_strength > 0:
            side = OrderSide.BUY
        else:
            side = OrderSide.SELL
        
        # Enhanced adaptive position sizing based on signal quality and market conditions
        base_position_value = self.current_capital * self.position_size
        
        # Advanced confidence multiplier with non-linear scaling
        if signal.confidence > 0.5:
            confidence_multiplier = 2.0 + (signal.confidence - 0.5) * 2.0  # 2.0x to 3.0x for high confidence
        elif signal.confidence > 0.3:
            confidence_multiplier = 1.0 + (signal.confidence - 0.3) * 5.0  # 1.0x to 2.0x for medium confidence
        else:
            confidence_multiplier = 0.5 + signal.confidence * 1.67  # 0.5x to 1.0x for low confidence
        
        # Signal strength multiplier with enhanced scaling
        strength_multiplier = min(abs(signal.signal_strength) * 1.2, 1.8)  # More aggressive, cap at 1.8x
        
        # Adaptive signal type multiplier
        if signal.feature_type == 'microstructure':
            type_multiplier = 0.9  # Higher multiplier for microstructure signals
        elif signal.feature_type == 'volume':
            type_multiplier = 0.85  # Good multiplier for volume signals
        elif signal.feature_type == 'price':
            type_multiplier = 1.0  # Full size for price-based signals
        elif signal.feature_type == 'periodic_test':
            type_multiplier = 0.5  # Smaller for test signals
        else:
            type_multiplier = 0.8  # Conservative default
        
        # Smart position scaling logic
        existing_position = self.positions[base_follower_symbol]
        current_exposure = abs(existing_position.quantity) * existing_position.average_price if existing_position.average_price > 0 else 0
        
        if current_exposure > 0:
            # More sophisticated add-on logic
            if signal.confidence > 0.4:
                add_on_multiplier = 0.7  # Larger add-on for high confidence
            elif signal.confidence > 0.3:
                add_on_multiplier = 0.5  # Medium add-on
            else:
                add_on_multiplier = 0.3  # Small add-on for low confidence
        else:
            add_on_multiplier = 1.0  # Full size for new positions
        
        # Market regime multiplier (could be enhanced with volatility calculation)
        market_multiplier = 1.1  # Slightly aggressive baseline
        
        # Calculate final position value with all multipliers
        adjusted_position_value = (base_position_value * 
                                 confidence_multiplier * 
                                 strength_multiplier * 
                                 type_multiplier * 
                                 add_on_multiplier * 
                                 market_multiplier)
        
        # Ensure minimum viable trade size
        min_trade_value = self.current_capital * 0.005  # 0.5% minimum
        adjusted_position_value = max(adjusted_position_value, min_trade_value)
        
        # Create order for follower symbol
        order = Order(
            timestamp=current_time,
            symbol=base_follower_symbol,
            side=side,
            quantity=adjusted_position_value,  # Will be converted to units later
            order_type=OrderType.MARKET,
            order_id=f"signal_{len(self.orders)}"
        )
        orders.append(order)
        
        logger.info(f"Enhanced order generation: {side.value} {base_follower_symbol}")
        logger.info(f"  Base value: ${base_position_value:.2f}")
        logger.info(f"  Confidence multiplier: {confidence_multiplier:.2f} (conf={signal.confidence:.3f})")
        logger.info(f"  Strength multiplier: {strength_multiplier:.2f}")
        logger.info(f"  Type multiplier: {type_multiplier:.2f} ({signal.feature_type})")
        logger.info(f"  Add-on multiplier: {add_on_multiplier:.2f}")
        logger.info(f"  Final value: ${adjusted_position_value:.2f}")
        
        # Log the signal and decision
        self.signal_performance.append({
            'signal_timestamp': signal.timestamp,
            'order_timestamp': current_time,
            'leader': signal.leader_symbol,
            'follower': signal.follower_symbol,
            'base_follower': base_follower_symbol,
            'signal_strength': signal.signal_strength,
            'confidence': signal.confidence,
            'feature_type': signal.feature_type,
            'side': side.value,
            'position_value': adjusted_position_value,
            'confidence_multiplier': confidence_multiplier,
            'strength_multiplier': strength_multiplier,
            'type_multiplier': type_multiplier
        })
        
        return orders
    
    def _execute_order(self, order: Order, 
                      order_book_states: Dict[str, OrderBookState]) -> Optional[Fill]:
        """Execute an order and return fill information."""
        
        # Get current order book state
        ob_state = order_book_states.get(order.symbol)
        if not ob_state:
            logger.warning(f"No order book state for {order.symbol}")
            return None
        
        # Determine execution price
        if order.order_type == OrderType.MARKET:
            if order.side == OrderSide.BUY:
                # Buy at ask price (taker)
                if 1 in ob_state.asks:
                    execution_price = ob_state.asks[1][0]
                    fee_rate = self.transaction_costs.taker_fee
                else:
                    logger.warning(f"No ask price available for {order.symbol}")
                    return None
            else:
                # Sell at bid price (taker)
                if 1 in ob_state.bids:
                    execution_price = ob_state.bids[1][0]
                    fee_rate = self.transaction_costs.taker_fee
                else:
                    logger.warning(f"No bid price available for {order.symbol}")
                    return None
        else:
            # Limit order logic (simplified)
            execution_price = order.price
            fee_rate = self.transaction_costs.maker_fee
        
        # Calculate quantity in units
        if execution_price <= 0:
            logger.warning(f"Invalid execution price {execution_price} for {order.symbol}")
            return None
        
        quantity_units = order.quantity / execution_price
        
        # Apply slippage
        slippage_amount = execution_price * (self.transaction_costs.slippage_bps / 10000)
        if order.side == OrderSide.BUY:
            execution_price += slippage_amount
        else:
            execution_price -= slippage_amount
        
        # Calculate costs
        trade_value = quantity_units * execution_price
        commission = max(trade_value * fee_rate, self.transaction_costs.min_commission)
        
        # Create fill
        fill = Fill(
            timestamp=order.timestamp,
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=quantity_units,
            price=execution_price,
            commission=commission,
            slippage=slippage_amount * quantity_units
        )
        
        # Update position
        self._update_position(fill)
        
        # Update capital
        if order.side == OrderSide.BUY:
            self.current_capital -= (trade_value + commission)
        else:
            self.current_capital += (trade_value - commission)
        
        # Update totals
        self.total_commissions += commission
        self.total_slippage += fill.slippage
        
        # Log trade
        self.trades_log.append({
            'timestamp': fill.timestamp,
            'symbol': fill.symbol,
            'side': fill.side.value,
            'quantity': fill.quantity,
            'price': fill.price,
            'value': trade_value,
            'commission': commission,
            'slippage': fill.slippage,
            'capital_after': self.current_capital
        })
        
        return fill
    
    def _update_position(self, fill: Fill) -> None:
        """Update position based on fill."""
        position = self.positions[fill.symbol]
        
        if fill.side == OrderSide.BUY:
            # Calculate new average price
            total_quantity = position.quantity + fill.quantity
            if total_quantity != 0:
                total_cost = (position.quantity * position.average_price + 
                             fill.quantity * fill.price)
                position.average_price = total_cost / total_quantity
            position.quantity += fill.quantity
        else:
            # Selling - realize PnL
            if position.quantity > 0:
                realized_pnl = fill.quantity * (fill.price - position.average_price)
                position.realized_pnl += realized_pnl
                self.gross_pnl += realized_pnl
            
            position.quantity -= fill.quantity
            
            # If position closed, reset average price
            if abs(position.quantity) < 1e-8:
                position.quantity = 0.0
                position.average_price = 0.0
    
    def _update_portfolio_value(self, timestamp: datetime,
                              order_book_states: Dict[str, OrderBookState]) -> None:
        """Update portfolio value and calculate unrealized PnL."""
        
        total_value = self.current_capital
        
        # Add value of positions
        for symbol, position in self.positions.items():
            if abs(position.quantity) > 0:
                ob_state = order_book_states.get(symbol)
                if ob_state:
                    current_price = ob_state.get_mid_price()
                    if current_price:
                        position_value = position.quantity * current_price
                        unrealized_pnl = position.quantity * (current_price - position.average_price)
                        position.unrealized_pnl = unrealized_pnl
                        total_value += position_value
        
        self.portfolio_values.append((timestamp, total_value))
        
        # Calculate drawdown
        if self.portfolio_values:
            peak_value = max(value for _, value in self.portfolio_values)
            drawdown = (total_value - peak_value) / peak_value
            self.drawdowns.append((timestamp, drawdown))
    
    def _calculate_backtest_results(self) -> Dict:
        """Calculate comprehensive backtest results."""
        
        if not self.portfolio_values:
            return {}
        
        # Calculate returns
        initial_value = self.initial_capital
        final_value = self.portfolio_values[-1][1]
        
        gross_return = (final_value - initial_value) / initial_value
        net_pnl = final_value - initial_value
        net_return = net_pnl / initial_value
        
        # Calculate performance metrics
        portfolio_df = pd.DataFrame(self.portfolio_values, columns=['timestamp', 'value'])
        portfolio_df['return'] = portfolio_df['value'].pct_change()
        
        # Sharpe ratio (annualized)
        if len(portfolio_df) > 1:
            avg_return = portfolio_df['return'].mean()
            std_return = portfolio_df['return'].std()
            # Assuming we have sub-second data, approximate annual Sharpe
            trading_days = 252
            periods_per_day = 86400  # Approximate for sub-second data
            annual_factor = np.sqrt(trading_days * periods_per_day)
            
            if std_return > 0:
                sharpe_ratio = (avg_return / std_return) * annual_factor
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
        
        # Maximum drawdown
        if self.drawdowns:
            max_drawdown = min(dd for _, dd in self.drawdowns)
        else:
            max_drawdown = 0.0
        
        # Trade statistics
        if self.trades_log:
            trades_df = pd.DataFrame(self.trades_log)
            num_trades = len(trades_df)
            avg_trade_value = trades_df['value'].mean()
            
            # Win rate calculation (simplified)
            profitable_trades = sum(1 for trade in self.trades_log 
                                  if trade['side'] == 'sell')  # Simplified
            win_rate = profitable_trades / num_trades if num_trades > 0 else 0
        else:
            num_trades = 0
            avg_trade_value = 0
            win_rate = 0
        
        # Transaction cost analysis
        total_costs = self.total_commissions + abs(self.total_slippage)
        cost_ratio = total_costs / self.initial_capital
        
        # Calculate net alpha (excess return after costs)
        # For simplicity, assume benchmark return is 0 (risk-free rate)
        net_alpha = net_return
        
        results = {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'gross_pnl': gross_return * self.initial_capital,
            'net_pnl': net_pnl,
            'gross_return': gross_return,
            'net_return': net_return,
            'net_alpha': net_alpha,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_commissions': self.total_commissions,
            'total_slippage': abs(self.total_slippage),
            'total_transaction_costs': total_costs,
            'cost_ratio': cost_ratio,
            'num_trades': num_trades,
            'avg_trade_value': avg_trade_value,
            'win_rate': win_rate,
            'portfolio_values': self.portfolio_values,
            'trades_log': self.trades_log,
            'signal_performance': self.signal_performance
        }
        
        return results
    
    def _analyze_learned_patterns(self, signals: List[LeadLagSignal]) -> Dict:
        """Analyze learned signals to extract trading patterns."""
        if not signals:
            return {}
        
        # Group signals by leader-follower pairs
        pattern_stats = defaultdict(list)
        
        for signal in signals:
            key = f"{signal.leader_symbol}->{signal.follower_symbol}"
            pattern_stats[key].append({
                'strength': signal.signal_strength,
                'confidence': signal.confidence,
                'lag_ms': signal.lag_microseconds / 1000,
                'feature_type': signal.feature_type
            })
        
        # Calculate statistics for each pattern
        patterns = {}
        for key, signal_list in pattern_stats.items():
            patterns[key] = {
                'count': len(signal_list),
                'avg_strength': np.mean([s['strength'] for s in signal_list]),
                'avg_confidence': np.mean([s['confidence'] for s in signal_list]),
                'avg_lag_ms': np.mean([s['lag_ms'] for s in signal_list]),
                'feature_types': list(set([s['feature_type'] for s in signal_list]))
            }
        
        return patterns
    
    def _generate_synthetic_signals(self, event: OrderBookEvent, current_price: float, 
                                   price_change: float, pattern_analysis: Dict) -> List[LeadLagSignal]:
        """Generate synthetic signals based on learned patterns and current price action."""
        synthetic_signals = []
        
        logger.debug(f"Checking price change {price_change:.6f} for {event.symbol}")
        
        # Only generate signals for significant price movements
        if abs(price_change) < 0.00005:  # 0.005% minimum movement (lowered for testing)
            logger.debug(f"Price change {price_change:.8f} too small, skipping")
            return synthetic_signals
        
        logger.info(f"Significant price change detected: {event.symbol} moved {price_change:.6f}")
        
        # Look for patterns where this symbol acts as a leader
        for pattern_key, stats in pattern_analysis.items():
            leader_symbol, follower_symbol = pattern_key.split('->')
            
            logger.debug(f"Checking pattern: {pattern_key}")
            logger.debug(f"Event symbol: {event.symbol}, Leader pattern: {leader_symbol}")
            
            # Check if current event symbol matches a known leader pattern
            # The learned pattern has "XBT_EUR_demo_extrememove" as leader, but our events are "XBT_EUR_demo"
            # Enhanced matching: check if event symbol is contained in leader pattern or vice versa
            event_base = self._extract_base_symbol(event.symbol)  # XBT_EUR_demo
            leader_base = self._extract_base_symbol(leader_symbol)  # XBT_EUR_demo
            
            logger.debug(f"Comparing: event_base='{event_base}' vs leader_base='{leader_base}'")
            
            if (event.symbol in leader_symbol or 
                leader_symbol.startswith(event.symbol) or
                event_base == leader_base or
                event_base in leader_symbol):
                logger.info(f"Found matching pattern for {event.symbol}: {pattern_key}")
                logger.info(f"  Pattern stats: count={stats['count']}, confidence={stats['avg_confidence']:.3f}")
                
                # Generate synthetic signal with learned characteristics
                signal = LeadLagSignal(
                    timestamp=event.timestamp,
                    leader_symbol=leader_symbol,
                    follower_symbol=follower_symbol,
                    signal_strength=stats['avg_strength'] * np.sign(price_change),
                    confidence=stats['avg_confidence'],
                    lag_microseconds=int(stats['avg_lag_ms'] * 1000),
                    feature_type=stats['feature_types'][0] if stats['feature_types'] else 'synthetic'
                )
                synthetic_signals.append(signal)
                
                logger.info(f"Generated synthetic signal: {event.symbol} -> {follower_symbol}, "
                           f"price_change={price_change:.6f}, confidence={signal.confidence:.3f}")
            else:
                logger.debug(f"No match: event_base='{event_base}' != leader_base='{leader_base}'")
        
        if not synthetic_signals:
            logger.debug(f"No patterns matched for {event.symbol}")
            logger.debug(f"Available patterns: {list(pattern_analysis.keys())}")
        
        return synthetic_signals
    
    def _generate_periodic_signals(self, event: OrderBookEvent, current_price: float,
                                  pattern_analysis: Dict) -> List[LeadLagSignal]:
        """Generate periodic signals for testing when no market activity is detected."""
        synthetic_signals = []
        
        # Generate a test signal occasionally to ensure the system works
        for pattern_key, stats in pattern_analysis.items():
            leader_symbol, follower_symbol = pattern_key.split('->')
            
            # Check if event symbol matches pattern
            event_base = self._extract_base_symbol(event.symbol)
            leader_base = self._extract_base_symbol(leader_symbol)
            
            if event_base == leader_base or event_base in leader_symbol:
                # Create a weak periodic signal
                signal = LeadLagSignal(
                    timestamp=event.timestamp,
                    leader_symbol=leader_symbol,
                    follower_symbol=follower_symbol,
                    signal_strength=0.5,  # Moderate strength
                    confidence=stats['avg_confidence'] * 0.6,  # Lower confidence
                    lag_microseconds=int(stats['avg_lag_ms'] * 1000),
                    feature_type='periodic_test'
                )
                synthetic_signals.append(signal)
                
                logger.info(f"Generated periodic test signal: {event.symbol} -> {follower_symbol}, "
                           f"confidence={signal.confidence:.3f}")
                break  # Only generate one signal per event
        
        return synthetic_signals

    def _generate_microstructure_signals(self, event: OrderBookEvent, 
                                        order_book_states: Dict[str, OrderBookState],
                                        pattern_analysis: Dict) -> List[LeadLagSignal]:
        """Generate signals based on enhanced order book microstructure analysis."""
        synthetic_signals = []
        
        ob_state = order_book_states.get(event.symbol)
        if not ob_state:
            return synthetic_signals
        
        # Get order book metrics
        # Extract best bid/ask with volumes
        if 1 in ob_state.bids:
            bid_price, bid_volume = ob_state.bids[1]
        else:
            bid_price, bid_volume = None, 0
        if 1 in ob_state.asks:
            ask_price, ask_volume = ob_state.asks[1]
        else:
            ask_price, ask_volume = None, 0
        
        if not bid_price or not ask_price:
            return synthetic_signals
        
        # Basic spread metrics
        spread = ask_price - bid_price
        mid_price = (bid_price + ask_price) / 2
        rel_spread = spread / mid_price if mid_price > 0 else 0
        
        # Volume imbalance analysis
        total_volume = bid_volume + ask_volume
        volume_imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
        
        # Calculate multi-level depth (up to 5 levels)
        bid_depth = sum(vol for _, vol in ob_state.bids.values() if len(ob_state.bids) <= 5)
        ask_depth = sum(vol for _, vol in ob_state.asks.values() if len(ob_state.asks) <= 5)
        total_depth = bid_depth + ask_depth
        depth_imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0
        
        # Enhanced signal detection
        signal_strength = 0.0
        signal_reasons = []
        
        # 1. Spread-based signals (enhanced thresholds)
        if rel_spread < 0.00005:  # Ultra-tight spread - strong signal
            signal_strength += 0.8
            signal_reasons.append(f"ultra_tight_spread({rel_spread:.6f})")
        elif rel_spread < 0.0001:  # Very tight spread
            signal_strength += 0.5
            signal_reasons.append(f"tight_spread({rel_spread:.6f})")
        elif rel_spread > 0.002:  # Very wide spread - negative signal
            signal_strength -= 0.6
            signal_reasons.append(f"wide_spread({rel_spread:.6f})")
        
        # 2. Volume imbalance signals
        if abs(volume_imbalance) > 0.6:  # Strong imbalance
            imbalance_signal = 0.7 * (1 if volume_imbalance > 0 else -1)
            signal_strength += imbalance_signal
            signal_reasons.append(f"volume_imbalance({volume_imbalance:.3f})")
        elif abs(volume_imbalance) > 0.3:  # Moderate imbalance
            imbalance_signal = 0.4 * (1 if volume_imbalance > 0 else -1)
            signal_strength += imbalance_signal
            signal_reasons.append(f"mod_volume_imbalance({volume_imbalance:.3f})")
        
        # 3. Depth imbalance signals
        if abs(depth_imbalance) > 0.5:  # Strong depth imbalance
            depth_signal = 0.5 * (1 if depth_imbalance > 0 else -1)
            signal_strength += depth_signal
            signal_reasons.append(f"depth_imbalance({depth_imbalance:.3f})")
        
        # 4. Combined microstructure score
        if len(signal_reasons) >= 2:  # Multiple factors - boost signal
            signal_strength *= 1.2
            signal_reasons.append("multi_factor_boost")
        
        # Normalize signal strength
        signal_strength = max(-1.0, min(1.0, signal_strength))
        
        # Generate signal if conditions are met (lowered threshold for enhanced signals)
        if abs(signal_strength) > 0.25:
            for pattern_key, stats in pattern_analysis.items():
                leader_symbol, follower_symbol = pattern_key.split('->')
                
                # Check if event symbol matches pattern
                event_base = self._extract_base_symbol(event.symbol)
                leader_base = self._extract_base_symbol(leader_symbol)
                
                if event_base == leader_base or event_base in leader_symbol:
                    # Enhanced confidence calculation
                    base_confidence = stats['avg_confidence'] * 0.85  # Slightly higher confidence
                    
                    # Boost confidence for multiple signal factors
                    if len(signal_reasons) >= 2:
                        base_confidence *= 1.1
                    
                    # Boost confidence for strong signals
                    if abs(signal_strength) > 0.6:
                        base_confidence *= 1.15
                    
                    signal = LeadLagSignal(
                        timestamp=event.timestamp,
                        leader_symbol=leader_symbol,
                        follower_symbol=follower_symbol,
                        signal_strength=signal_strength,
                        confidence=min(0.95, base_confidence),  # Cap at 95%
                        lag_microseconds=int(stats['avg_lag_ms'] * 1000),
                        feature_type='microstructure'
                    )
                    synthetic_signals.append(signal)
                    
                    logger.info(f"Generated enhanced microstructure signal: {event.symbol} -> {follower_symbol}")
                    logger.info(f"  Signal strength: {signal_strength:.3f}, Confidence: {signal.confidence:.3f}")
                    logger.info(f"  Factors: {', '.join(signal_reasons)}")
                    logger.info(f"  Spread: {rel_spread:.6f}, Vol_imbal: {volume_imbalance:.3f}, Depth_imbal: {depth_imbalance:.3f}")
                    break  # Only generate one signal per event
        
        return synthetic_signals
    
    def _generate_periodic_signals(self, event: OrderBookEvent, current_price: float,
                                  pattern_analysis: Dict) -> List[LeadLagSignal]:
        """Generate periodic signals for testing when no market activity is detected."""
        synthetic_signals = []
        
        # Generate a test signal occasionally to ensure the system works
        for pattern_key, stats in pattern_analysis.items():
            leader_symbol, follower_symbol = pattern_key.split('->')
            
            # Check if event symbol matches pattern
            event_base = self._extract_base_symbol(event.symbol)
            leader_base = self._extract_base_symbol(leader_symbol)
            
            if event_base == leader_base or event_base in leader_symbol:
                # Create a weak periodic signal
                signal = LeadLagSignal(
                    timestamp=event.timestamp,
                    leader_symbol=leader_symbol,
                    follower_symbol=follower_symbol,
                    signal_strength=0.5,  # Moderate strength
                    confidence=stats['avg_confidence'] * 0.6,  # Lower confidence
                    lag_microseconds=int(stats['avg_lag_ms'] * 1000),
                    feature_type='periodic_test'
                )
                synthetic_signals.append(signal)
                
                logger.info(f"Generated periodic test signal: {event.symbol} -> {follower_symbol}, "
                           f"confidence={signal.confidence:.3f}")
                break  # Only generate one signal per event
        
        return synthetic_signals

    def get_position_summary(self) -> Dict[str, Dict]:
        """Get current position summary."""
        summary = {}
        for symbol, position in self.positions.items():
            summary[symbol] = {
                'quantity': position.quantity,
                'average_price': position.average_price,
                'unrealized_pnl': position.unrealized_pnl,
                'realized_pnl': position.realized_pnl
            }
        return summary
    
    def export_trades_to_csv(self, filename: str) -> None:
        """Export trade log to CSV file."""
        if self.trades_log:
            trades_df = pd.DataFrame(self.trades_log)
            trades_df.to_csv(filename, index=False)
            logger.info(f"Exported {len(self.trades_log)} trades to {filename}")
        else:
            logger.warning("No trades to export")
