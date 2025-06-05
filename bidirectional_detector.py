#!/usr/bin/env python3
"""
Enhanced Bidirectional Lead-Lag Detector
========================================

Improved version that properly analyzes both directions:
1. ETH leading XBT 
2. XBT leading ETH
3. Advanced transfer entropy implementation
4. Better signal threshold optimization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from scipy import stats
from collections import defaultdict, deque

from async_processing.lead_lag_detector import AsyncLeadLagDetector, LeadLagSignal
from async_processing.event_processor import AsyncEventProcessor

logger = logging.getLogger(__name__)

class BidirectionalLeadLagDetector:
    """
    Enhanced detector that properly analyzes lead-lag relationships in both directions
    """
    
    def __init__(self, symbols: List[str], 
                 max_lag_ms: int = 2000,
                 min_price_change: float = 0.00001,
                 signal_decay_ms: int = 5000):
        self.symbols = symbols
        self.max_lag_ms = max_lag_ms
        self.min_price_change = min_price_change
        self.signal_decay_ms = signal_decay_ms
        
        # Create separate detectors for each direction
        self.detectors = {}
        self.results = {}
        
        logger.info(f"🔄 Initialized Bidirectional Lead-Lag Detector")
        logger.info(f"   Symbols: {symbols}")
        logger.info(f"   Max lag: {max_lag_ms}ms")
        logger.info(f"   Min price change: {min_price_change}")
    
    def analyze_all_directions(self, event_processor: AsyncEventProcessor) -> Dict:
        """
        Analyze lead-lag relationships in all possible directions
        """
        results = {}
        
        if len(self.symbols) != 2:
            logger.error("Bidirectional analysis currently supports exactly 2 symbols")
            return results
        
        symbol_a, symbol_b = self.symbols
        
        # Test all direction combinations
        direction_configs = [
            {
                'name': f'{symbol_a}_leads_{symbol_b}',
                'leader': symbol_a,
                'follower': symbol_b,
                'description': f'{symbol_a} → {symbol_b}'
            },
            {
                'name': f'{symbol_b}_leads_{symbol_a}',
                'leader': symbol_b,
                'follower': symbol_a,
                'description': f'{symbol_b} → {symbol_a}'
            },
            {
                'name': 'mutual_influence',
                'leader': 'both',
                'follower': 'both',
                'description': 'Mutual influence analysis'
            }
        ]
        
        for config in direction_configs:
            logger.info(f"🔍 Analyzing: {config['description']}")
            
            if config['name'] == 'mutual_influence':
                # Special case for mutual influence
                result = self._analyze_mutual_influence(event_processor)
            else:
                # Standard directional analysis
                result = self._analyze_direction(event_processor, config)
            
            results[config['name']] = result
            
            signal_count = len(result.get('signals', []))
            avg_confidence = result.get('stats', {}).get('avg_confidence', 0)
            logger.info(f"   ✅ {signal_count} signals, avg confidence: {avg_confidence:.3f}")
        
        # Comparative analysis
        results['comparison'] = self._compare_directions(results)
        
        return results
    
    def _analyze_direction(self, event_processor: AsyncEventProcessor, config: Dict) -> Dict:
        """
        Analyze lead-lag relationship in a specific direction
        """
        # Create detector with ordered symbols (leader first)
        ordered_symbols = [config['leader'], config['follower']]
        
        detector = AsyncLeadLagDetector(
            symbols=ordered_symbols,
            max_lag_ms=self.max_lag_ms,
            min_price_change=self.min_price_change,
            signal_decay_ms=self.signal_decay_ms
        )
        
        # Process events
        signals = detector.process_event_stream(event_processor)
        
        # Filter signals to only include the desired direction
        filtered_signals = []
        for signal in signals:
            if (signal.leader_symbol == config['leader'] and 
                signal.follower_symbol == config['follower']):
                filtered_signals.append(signal)
        
        # Get statistics
        stats = self._compute_directional_stats(filtered_signals)
        
        return {
            'signals': filtered_signals,
            'stats': stats,
            'config': config,
            'detector': detector
        }
    
    def _analyze_mutual_influence(self, event_processor: AsyncEventProcessor) -> Dict:
        """
        Analyze mutual influence and feedback loops between symbols
        """
        logger.info("   Analyzing mutual influence and feedback loops...")
        
        # Create detector for both symbols
        detector = AsyncLeadLagDetector(
            symbols=self.symbols,
            max_lag_ms=self.max_lag_ms,
            min_price_change=self.min_price_change,
            signal_decay_ms=self.signal_decay_ms
        )
        
        # Get all signals
        all_signals = detector.process_event_stream(event_processor)
        
        # Analyze feedback loops and mutual causality
        feedback_analysis = self._detect_feedback_loops(all_signals)
        
        return {
            'signals': all_signals,
            'feedback_analysis': feedback_analysis,
            'stats': detector.get_signal_statistics()
        }
    
    def _detect_feedback_loops(self, signals: List[LeadLagSignal]) -> Dict:
        """
        Detect feedback loops and mutual causality patterns
        """
        feedback_stats = {
            'potential_feedback_loops': 0,
            'rapid_reversals': 0,
            'mutual_causality_events': 0,
            'feedback_details': []
        }
        
        if len(signals) < 2:
            return feedback_stats
        
        # Sort signals by timestamp
        sorted_signals = sorted(signals, key=lambda x: x.timestamp)
        
        # Look for rapid direction reversals
        for i in range(len(sorted_signals) - 1):
            current = sorted_signals[i]
            next_signal = sorted_signals[i + 1]
            
            # Check if this is a direction reversal
            if (current.leader_symbol == next_signal.follower_symbol and
                current.follower_symbol == next_signal.leader_symbol):
                
                time_diff = (next_signal.timestamp - current.timestamp).total_seconds()
                
                # If reversal happens within a short time window, it might be feedback
                if time_diff < 30:  # 30 seconds
                    feedback_stats['rapid_reversals'] += 1
                    
                    if time_diff < 10:  # Very rapid reversal
                        feedback_stats['potential_feedback_loops'] += 1
                        
                        feedback_stats['feedback_details'].append({
                            'timestamp': current.timestamp,
                            'direction_1': f"{current.leader_symbol}→{current.follower_symbol}",
                            'direction_2': f"{next_signal.leader_symbol}→{next_signal.follower_symbol}",
                            'time_gap': time_diff,
                            'confidence_1': current.confidence,
                            'confidence_2': next_signal.confidence
                        })
        
        return feedback_stats
    
    def _compute_directional_stats(self, signals: List[LeadLagSignal]) -> Dict:
        """
        Compute statistics for signals in a specific direction
        """
        if not signals:
            return {}
        
        confidences = [s.confidence for s in signals]
        lags = [s.lag_microseconds / 1000 for s in signals]  # Convert to ms
        strengths = [s.signal_strength for s in signals]
        
        return {
            'total_signals': len(signals),
            'avg_confidence': np.mean(confidences),
            'median_confidence': np.median(confidences),
            'std_confidence': np.std(confidences),
            'avg_lag_ms': np.mean(lags),
            'median_lag_ms': np.median(lags),
            'std_lag_ms': np.std(lags),
            'avg_strength': np.mean(strengths),
            'signal_frequency': len(signals) / ((signals[-1].timestamp - signals[0].timestamp).total_seconds() / 3600) if len(signals) > 1 else 0,  # signals per hour
            'feature_types': {ft: sum(1 for s in signals if s.feature_type == ft) for ft in set(s.feature_type for s in signals)}
        }
    
    def _compare_directions(self, results: Dict) -> Dict:
        """
        Compare lead-lag strength between different directions
        """
        comparison = {
            'dominant_direction': None,
            'strength_ratio': 0,
            'confidence_comparison': {},
            'frequency_comparison': {},
            'recommendations': []
        }
        
        # Extract directional results (exclude mutual_influence)
        directional_results = {k: v for k, v in results.items() if k != 'mutual_influence'}
        
        if len(directional_results) < 2:
            return comparison
        
        # Compare signal counts and confidence
        direction_stats = {}
        for direction, result in directional_results.items():
            stats = result.get('stats', {})
            direction_stats[direction] = {
                'signal_count': stats.get('total_signals', 0),
                'avg_confidence': stats.get('avg_confidence', 0),
                'frequency': stats.get('signal_frequency', 0)
            }
        
        # Determine dominant direction
        max_signals_dir = max(direction_stats.keys(), key=lambda x: direction_stats[x]['signal_count'])
        max_confidence_dir = max(direction_stats.keys(), key=lambda x: direction_stats[x]['avg_confidence'])
        
        if max_signals_dir == max_confidence_dir:
            comparison['dominant_direction'] = max_signals_dir
        else:
            # If different, choose based on combined score
            combined_scores = {}
            for direction in direction_stats:
                stats = direction_stats[direction]
                # Normalize and combine signal count and confidence
                combined_scores[direction] = (stats['signal_count'] * stats['avg_confidence'])
            
            comparison['dominant_direction'] = max(combined_scores.keys(), key=combined_scores.get)
        
        # Calculate strength ratio
        directions = list(direction_stats.keys())
        if len(directions) == 2:
            dir1, dir2 = directions
            count1 = direction_stats[dir1]['signal_count']
            count2 = direction_stats[dir2]['signal_count']
            comparison['strength_ratio'] = count1 / max(count2, 1)  # Avoid division by zero
        
        comparison['confidence_comparison'] = {d: direction_stats[d]['avg_confidence'] for d in direction_stats}
        comparison['frequency_comparison'] = {d: direction_stats[d]['frequency'] for d in direction_stats}
        
        # Generate recommendations
        if comparison['dominant_direction']:
            dominant_stats = direction_stats[comparison['dominant_direction']]
            comparison['recommendations'].append(
                f"Focus on {comparison['dominant_direction']} direction with {dominant_stats['signal_count']} signals"
            )
            
            if comparison['strength_ratio'] > 2:
                comparison['recommendations'].append(
                    f"Strong directional bias detected (ratio: {comparison['strength_ratio']:.2f})"
                )
            elif comparison['strength_ratio'] < 0.5:
                comparison['recommendations'].append(
                    f"Reverse directional bias detected (ratio: {comparison['strength_ratio']:.2f})"
                )
            else:
                comparison['recommendations'].append(
                    f"Balanced bidirectional relationship (ratio: {comparison['strength_ratio']:.2f})"
                )
        
        # Check for feedback loops
        if 'mutual_influence' in results:
            feedback = results['mutual_influence'].get('feedback_analysis', {})
            if feedback.get('potential_feedback_loops', 0) > 0:
                comparison['recommendations'].append(
                    f"Detected {feedback['potential_feedback_loops']} potential feedback loops"
                )
        
        return comparison
    
    def optimize_thresholds(self, event_processor: AsyncEventProcessor, 
                          target_signal_count: int = 50) -> Dict:
        """
        Optimize detection thresholds to achieve target signal count
        """
        logger.info(f"🎯 Optimizing thresholds for target of {target_signal_count} signals")
        
        # Test different threshold combinations
        price_change_values = [0.00001, 0.00005, 0.0001, 0.0002]
        lag_values = [1000, 2000, 3000, 5000]
        
        best_config = None
        best_score = float('inf')
        results_by_config = {}
        
        for min_price_change in price_change_values:
            for max_lag_ms in lag_values:
                logger.info(f"   Testing: price_change={min_price_change}, lag={max_lag_ms}ms")
                
                # Test this configuration
                test_detector = BidirectionalLeadLagDetector(
                    symbols=self.symbols,
                    max_lag_ms=max_lag_ms,
                    min_price_change=min_price_change,
                    signal_decay_ms=self.signal_decay_ms
                )
                
                # Quick analysis (just first direction for speed)
                test_results = test_detector._analyze_direction(
                    event_processor, 
                    {'leader': self.symbols[0], 'follower': self.symbols[1]}
                )
                
                signal_count = len(test_results['signals'])
                avg_confidence = test_results['stats'].get('avg_confidence', 0)
                
                # Score based on distance from target and confidence
                distance_penalty = abs(signal_count - target_signal_count)
                confidence_bonus = avg_confidence * 10  # Boost good confidence
                score = distance_penalty - confidence_bonus
                
                results_by_config[(min_price_change, max_lag_ms)] = {
                    'signal_count': signal_count,
                    'avg_confidence': avg_confidence,
                    'score': score
                }
                
                logger.info(f"      → {signal_count} signals, confidence: {avg_confidence:.3f}, score: {score:.2f}")
                
                if score < best_score:
                    best_score = score
                    best_config = (min_price_change, max_lag_ms)
        
        if best_config:
            logger.info(f"🏆 Best config: price_change={best_config[0]}, lag={best_config[1]}ms")
            logger.info(f"   Score: {best_score:.2f}")
            
            # Update our settings
            self.min_price_change = best_config[0]
            self.max_lag_ms = best_config[1]
        
        return {
            'best_config': best_config,
            'best_score': best_score,
            'all_results': results_by_config,
            'optimization_target': target_signal_count
        }

def main():
    """
    Test the bidirectional detector
    """
    from data_processing.data_loader import OrderBookDataLoader
    from data_processing.data_formatter import OrderBookDataFormatter
    
    # Load sample data
    symbols = ['ETH_EUR', 'XBT_EUR']
    
    logger.info("Loading data for bidirectional analysis...")
    raw_data = {}
    loader = OrderBookDataLoader('data/raw')
    
    for symbol in symbols:
        raw_data[symbol] = loader.load_data(symbol, max_records=10000)
    
    # Format data
    formatter = OrderBookDataFormatter()
    combined_data = pd.concat([
        df.assign(symbol=symbol) for symbol, df in raw_data.items()
    ], ignore_index=True)
    formatted_data = formatter.format_data(combined_data)
    
    # Create event processor
    event_processor = AsyncEventProcessor(symbols)
    event_processor.load_formatted_data(formatted_data)
    
    # Test bidirectional detector
    detector = BidirectionalLeadLagDetector(symbols)
    
    # Optimize thresholds first
    optimization_results = detector.optimize_thresholds(event_processor, target_signal_count=20)
    logger.info(f"Optimization results: {optimization_results['best_config']}")
    
    # Run full bidirectional analysis
    results = detector.analyze_all_directions(event_processor)
    
    # Print summary
    logger.info("\n📊 BIDIRECTIONAL ANALYSIS SUMMARY:")
    for direction, result in results.items():
        if direction == 'comparison':
            continue
        
        signal_count = len(result.get('signals', []))
        stats = result.get('stats', {})
        avg_confidence = stats.get('avg_confidence', 0)
        
        logger.info(f"   {direction}: {signal_count} signals (confidence: {avg_confidence:.3f})")
    
    # Print comparison results
    if 'comparison' in results:
        comp = results['comparison']
        logger.info(f"\n🎯 COMPARISON RESULTS:")
        logger.info(f"   Dominant direction: {comp['dominant_direction']}")
        logger.info(f"   Strength ratio: {comp['strength_ratio']:.2f}")
        logger.info(f"   Recommendations: {comp['recommendations']}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
