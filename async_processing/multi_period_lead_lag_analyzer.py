"""
Multi-Period Lead-Lag Analysis Module

Orchestrates lead-lag detection across multiple temporal periods (DATA_0, DATA_1, DATA_2)
for comprehensive temporal validation and pattern stability analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging
from pathlib import Path
import json
from collections import defaultdict
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from async_processing.lead_lag_detector import AsyncLeadLagDetector, LeadLagSignal
from async_processing.event_processor import AsyncEventProcessor
from data_processing.multi_period_loader import MultiPeriodDataLoader

logger = logging.getLogger(__name__)

class MultiPeriodLeadLagAnalyzer:
    """
    Analyzes lead-lag relationships across multiple temporal periods.
    
    This class provides:
    1. Period-by-period analysis with consistent methodology
    2. Cross-period pattern validation and stability assessment
    3. Temporal evolution tracking of lead-lag relationships
    4. Out-of-sample validation of detected patterns
    """
    
    def __init__(self, 
                 data_dir: str,
                 symbols: List[str] = None,
                 max_lag_ms: int = 1000,
                 min_price_change: float = 0.0001):
        """
        Initialize multi-period lead-lag analyzer.
        
        Args:
            data_dir: Path to data directory containing DATA_0, DATA_1, DATA_2
            symbols: List of symbols to analyze (default: ['ETH_EUR', 'XBT_EUR'])
            max_lag_ms: Maximum lag time in milliseconds
            min_price_change: Minimum price change threshold
        """
        self.data_dir = Path(data_dir)
        self.symbols = symbols or ['ETH_EUR', 'XBT_EUR']
        self.max_lag_ms = max_lag_ms
        self.min_price_change = min_price_change
        
        # Period-specific detectors
        self.detectors: Dict[str, AsyncLeadLagDetector] = {}
        
        # Results storage
        self.period_results: Dict[str, List[LeadLagSignal]] = {}
        self.cross_period_analysis: Dict = {}
        
        # Initialize multi-period loader
        self.multi_loader = MultiPeriodDataLoader(str(self.data_dir))
        
        logger.info(f"Initialized MultiPeriodLeadLagAnalyzer for periods: DATA_0, DATA_1, DATA_2")
        logger.info(f"Symbols: {self.symbols}, Max lag: {max_lag_ms}ms")
    
    def analyze_all_periods(self, 
                          sample_size_per_period: int = 100000,
                          save_results: bool = True) -> Dict[str, List[LeadLagSignal]]:
        """
        Analyze lead-lag patterns across all three periods.
        
        Args:
            sample_size_per_period: Number of samples to analyze per period
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary mapping period_id to list of detected signals
        """
        logger.info("🚀 Starting multi-period lead-lag analysis")
        
        periods = ['DATA_0', 'DATA_1', 'DATA_2']
        
        for period_id in periods:
            logger.info(f"📊 Analyzing period {period_id}")
            
            # Initialize detector for this period
            detector = AsyncLeadLagDetector(
                symbols=self.symbols,
                max_lag_ms=self.max_lag_ms,
                min_price_change=self.min_price_change
            )
            self.detectors[period_id] = detector
            
            # Load period data
            period_data = self._load_period_data(period_id, sample_size_per_period)
            
            if period_data is None:
                logger.warning(f"Failed to load data for period {period_id}")
                continue
            
            # Create event processor for this period
            event_processor = self._create_event_processor(period_data, period_id)
            
            # Detect lead-lag patterns
            signals = detector.process_event_stream(
                event_processor,
                period_id=period_id
            )
            
            self.period_results[period_id] = signals
            
            logger.info(f"✅ Period {period_id}: Detected {len(signals)} lead-lag signals")
        
        # Perform cross-period analysis
        self._analyze_cross_period_patterns()
        
        if save_results:
            self._save_results()
        
        return self.period_results
    
    def _load_period_data(self, period_id: str, sample_size: int) -> Optional[Dict[str, pd.DataFrame]]:
        """Load and sample data for a specific period."""
        try:
            # Load data returns a dictionary of symbol -> DataFrame
            period_data_dict = self.multi_loader.load_period_data(
                period=period_id,
                symbols=self.symbols,
                max_records=sample_size
            )
            
            if not period_data_dict:
                logger.error(f"No data loaded for period {period_id}")
                return None
            
            # Validate data
            total_records = 0
            for symbol, df in period_data_dict.items():
                if not df.empty:
                    total_records += len(df)
            
            if total_records == 0:
                logger.error(f"No valid data for period {period_id}")
                return None
            
            logger.info(f"Loaded {total_records:,} records for period {period_id}")
            return period_data_dict
            
        except Exception as e:
            logger.error(f"Error loading period {period_id}: {e}")
            return None
    
    def _create_event_processor(self, symbol_data: Dict[str, pd.DataFrame], period_id: str) -> AsyncEventProcessor:
        """Create an event processor for the given period data."""
        event_processor = AsyncEventProcessor(self.symbols)
        
        # Load events from the symbol-organized data directly
        event_processor.load_events_from_data(symbol_data)
        
        logger.info(f"Created event processor with {len(event_processor.event_stream)} events for {period_id}")
        return event_processor
    
    def _analyze_cross_period_patterns(self):
        """Analyze patterns that persist across multiple periods."""
        logger.info("🔍 Analyzing cross-period pattern stability")
        
        # Pattern consistency analysis
        self.cross_period_analysis = {
            'pattern_stability': self._calculate_pattern_stability(),
            'temporal_evolution': self._analyze_temporal_evolution(),
            'validation_metrics': self._calculate_validation_metrics()
        }
        
        logger.info("✅ Cross-period analysis completed")
    
    def _calculate_pattern_stability(self) -> Dict:
        """Calculate how stable patterns are across periods."""
        pattern_counts = defaultdict(lambda: defaultdict(int))
        
        # Count pattern occurrences by period
        for period_id, signals in self.period_results.items():
            for signal in signals:
                # Create pattern key: leader-follower-feature_type
                pattern_key = f"{signal.leader_symbol}-{signal.follower_symbol}-{signal.feature_type}"
                pattern_counts[pattern_key][period_id] += 1
        
        # Calculate stability metrics
        stability_metrics = {}
        for pattern_key, period_counts in pattern_counts.items():
            periods_present = len(period_counts)
            total_occurrences = sum(period_counts.values())
            
            stability_metrics[pattern_key] = {
                'periods_present': periods_present,
                'total_occurrences': total_occurrences,
                'stability_score': periods_present / 3.0,  # 3 periods total
                'period_distribution': dict(period_counts)
            }
        
        return stability_metrics
    
    def _analyze_temporal_evolution(self) -> Dict:
        """Analyze how signal characteristics evolve over time."""
        evolution_metrics = {}
        
        periods = ['DATA_0', 'DATA_1', 'DATA_2']
        
        for metric in ['confidence', 'signal_strength', 'lag_microseconds']:
            period_values = {}
            
            for period_id in periods:
                if period_id in self.period_results:
                    values = [getattr(signal, metric) for signal in self.period_results[period_id]]
                    if values:
                        period_values[period_id] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'median': np.median(values),
                            'count': len(values)
                        }
            
            evolution_metrics[metric] = period_values
        
        return evolution_metrics
    
    def _calculate_validation_metrics(self) -> Dict:
        """Calculate validation metrics for out-of-sample performance."""
        validation_metrics = {}
        
        # Use DATA_0 as training, DATA_1 as validation, DATA_2 as test
        if all(period in self.period_results for period in ['DATA_0', 'DATA_1', 'DATA_2']):
            
            train_patterns = self._extract_pattern_signatures(self.period_results['DATA_0'])
            val_patterns = self._extract_pattern_signatures(self.period_results['DATA_1'])
            test_patterns = self._extract_pattern_signatures(self.period_results['DATA_2'])
            
            # Pattern persistence rates
            train_to_val_persistence = len(train_patterns & val_patterns) / len(train_patterns) if train_patterns else 0
            val_to_test_persistence = len(val_patterns & test_patterns) / len(val_patterns) if val_patterns else 0
            train_to_test_persistence = len(train_patterns & test_patterns) / len(train_patterns) if train_patterns else 0
            
            validation_metrics = {
                'train_to_validation_persistence': train_to_val_persistence,
                'validation_to_test_persistence': val_to_test_persistence,
                'train_to_test_persistence': train_to_test_persistence,
                'pattern_counts': {
                    'train': len(train_patterns),
                    'validation': len(val_patterns),
                    'test': len(test_patterns)
                }
            }
        
        return validation_metrics
    
    def _extract_pattern_signatures(self, signals: List[LeadLagSignal]) -> set:
        """Extract unique pattern signatures from signals."""
        signatures = set()
        for signal in signals:
            # Create signature based on leader-follower pair and feature type
            sig = f"{signal.leader_symbol}-{signal.follower_symbol}-{signal.feature_type}"
            signatures.add(sig)
        return signatures
    
    def _save_results(self):
        """Save analysis results to disk."""
        results_dir = Path("results/lead_lag_analysis")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save period-specific results
        for period_id, signals in self.period_results.items():
            period_file = results_dir / f"lead_lag_signals_{period_id}_{timestamp}.json"
            
            # Convert signals to serializable format
            signals_data = []
            for signal in signals:
                signals_data.append({
                    'timestamp': signal.timestamp.isoformat(),
                    'leader_symbol': signal.leader_symbol,
                    'follower_symbol': signal.follower_symbol,
                    'signal_strength': signal.signal_strength,
                    'confidence': signal.confidence,
                    'lag_microseconds': signal.lag_microseconds,
                    'feature_type': signal.feature_type,
                    'period_id': signal.period_id
                })
            
            with open(period_file, 'w') as f:
                json.dump(signals_data, f, indent=2)
            
            logger.info(f"Saved {len(signals)} signals for {period_id} to {period_file}")
        
        # Save cross-period analysis
        cross_period_file = results_dir / f"cross_period_analysis_{timestamp}.json"
        with open(cross_period_file, 'w') as f:
            json.dump(self.cross_period_analysis, f, indent=2, default=str)
        
        logger.info(f"Saved cross-period analysis to {cross_period_file}")
    
    def get_summary_report(self) -> Dict:
        """Generate a comprehensive summary report."""
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'periods_analyzed': list(self.period_results.keys()),
            'total_signals_by_period': {
                period: len(signals) for period, signals in self.period_results.items()
            },
            'cross_period_analysis': self.cross_period_analysis,
            'detector_statistics': {}
        }
        
        # Add detector statistics for each period
        for period_id, detector in self.detectors.items():
            if hasattr(detector, 'detection_stats'):
                report['detector_statistics'][period_id] = dict(detector.detection_stats)
        
        return report
    
    def plot_analysis_results(self, save_plots: bool = True):
        """Generate visualization plots for the multi-period analysis."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set up the plotting style
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Multi-Period Lead-Lag Analysis Results', fontsize=16)
            
            # Plot 1: Signal counts by period
            periods = list(self.period_results.keys())
            signal_counts = [len(self.period_results[p]) for p in periods]
            
            axes[0, 0].bar(periods, signal_counts, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            axes[0, 0].set_title('Signal Counts by Period')
            axes[0, 0].set_ylabel('Number of Signals')
            
            # Plot 2: Average confidence by period
            avg_confidences = []
            for period in periods:
                confidences = [s.confidence for s in self.period_results[period]]
                avg_confidences.append(np.mean(confidences) if confidences else 0)
            
            axes[0, 1].bar(periods, avg_confidences, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            axes[0, 1].set_title('Average Signal Confidence by Period')
            axes[0, 1].set_ylabel('Average Confidence')
            
            # Plot 3: Pattern stability
            if 'pattern_stability' in self.cross_period_analysis:
                stability_data = self.cross_period_analysis['pattern_stability']
                stability_scores = [data['stability_score'] for data in stability_data.values()]
                
                axes[1, 0].hist(stability_scores, bins=10, alpha=0.7, color='skyblue')
                axes[1, 0].set_title('Pattern Stability Distribution')
                axes[1, 0].set_xlabel('Stability Score (0-1)')
                axes[1, 0].set_ylabel('Number of Patterns')
            
            # Plot 4: Validation metrics
            if 'validation_metrics' in self.cross_period_analysis:
                val_metrics = self.cross_period_analysis['validation_metrics']
                metric_names = ['Train→Val', 'Val→Test', 'Train→Test']
                persistence_rates = [
                    val_metrics.get('train_to_validation_persistence', 0),
                    val_metrics.get('validation_to_test_persistence', 0),
                    val_metrics.get('train_to_test_persistence', 0)
                ]
                
                axes[1, 1].bar(metric_names, persistence_rates, color=['orange', 'green', 'red'])
                axes[1, 1].set_title('Pattern Persistence Across Periods')
                axes[1, 1].set_ylabel('Persistence Rate')
                axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if save_plots:
                plots_dir = Path("results/lead_lag_analysis/plots")
                plots_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_file = plots_dir / f"multi_period_analysis_{timestamp}.png"
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                logger.info(f"Saved analysis plots to {plot_file}")
            
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib/Seaborn not available. Install with: pip install matplotlib seaborn")
        except Exception as e:
            logger.error(f"Error creating plots: {e}")
