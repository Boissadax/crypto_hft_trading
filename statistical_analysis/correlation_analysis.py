"""
Cross-Correlation Analysis Module

Implements advanced cross-correlation analysis for lead-lag detection:
- Time-domain cross-correlation
- Frequency-domain cross-correlation
- Partial cross-correlation
- Dynamic cross-correlation with sliding windows
- Statistical significance testing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from scipy import signal, stats
from scipy.signal import correlate, correlation_lags
from scipy.fftpack import fft, ifft
import warnings
warnings.filterwarnings('ignore')

@dataclass
class CrossCorrelationResult:
    """Results from cross-correlation analysis."""
    lags: np.ndarray
    correlation: np.ndarray
    max_correlation: float
    optimal_lag: int
    p_values: Optional[np.ndarray]
    confidence_intervals: Optional[np.ndarray]
    is_significant: bool
    method: str
    additional_info: Dict[str, Any]

class CrossCorrelationAnalyzer:
    """
    Advanced cross-correlation analysis for lead-lag detection.
    
    Provides multiple methods for correlation analysis with proper
    statistical validation and significance testing.
    """
    
    def __init__(self,
                 max_lag: int = 100,
                 significance_level: float = 0.05,
                 detrend_method: str = 'linear'):
        """
        Initialize cross-correlation analyzer.
        
        Args:
            max_lag: Maximum lag to analyze
            significance_level: Statistical significance threshold
            detrend_method: Method for detrending ('linear', 'constant', None)
        """
        self.max_lag = max_lag
        self.significance_level = significance_level
        self.detrend_method = detrend_method
        
    def cross_correlation(self,
                         x: np.ndarray,
                         y: np.ndarray,
                         mode: str = 'full',
                         normalize: bool = True) -> CrossCorrelationResult:
        """
        Compute cross-correlation between two time series.
        
        Args:
            x: First time series
            y: Second time series
            mode: Correlation mode ('full', 'valid', 'same')
            normalize: Whether to normalize correlation
            
        Returns:
            Cross-correlation results
        """
        # Preprocess data
        x_clean, y_clean = self._preprocess_data(x, y)
        
        # Compute cross-correlation
        correlation = correlate(y_clean, x_clean, mode=mode)
        lags = correlation_lags(len(x_clean), len(y_clean), mode=mode)
        
        # Limit to specified max_lag
        if mode == 'full':
            center_idx = len(correlation) // 2
            start_idx = max(0, center_idx - self.max_lag)
            end_idx = min(len(correlation), center_idx + self.max_lag + 1)
            
            correlation = correlation[start_idx:end_idx]
            lags = lags[start_idx:end_idx]
        
        # Normalize if requested
        if normalize:
            correlation = correlation / (np.std(x_clean) * np.std(y_clean) * len(x_clean))
        
        # Find optimal lag
        max_idx = np.argmax(np.abs(correlation))
        optimal_lag = lags[max_idx]
        max_correlation = correlation[max_idx]
        
        # Statistical significance testing
        p_values = self._compute_correlation_p_values(correlation, len(x_clean))
        confidence_intervals = self._compute_confidence_intervals(correlation, len(x_clean))
        
        is_significant = p_values[max_idx] < self.significance_level if p_values is not None else False
        
        return CrossCorrelationResult(
            lags=lags,
            correlation=correlation,
            max_correlation=max_correlation,
            optimal_lag=optimal_lag,
            p_values=p_values,
            confidence_intervals=confidence_intervals,
            is_significant=is_significant,
            method='time_domain',
            additional_info={
                'mode': mode,
                'normalized': normalize,
                'sample_size': len(x_clean),
                'max_lag_tested': self.max_lag
            }
        )
    
    def frequency_domain_correlation(self,
                                   x: np.ndarray,
                                   y: np.ndarray) -> CrossCorrelationResult:
        """
        Compute cross-correlation in frequency domain using FFT.
        
        Args:
            x: First time series
            y: Second time series
            
        Returns:
            Cross-correlation results from frequency domain
        """
        x_clean, y_clean = self._preprocess_data(x, y)
        
        # Pad to power of 2 for efficient FFT
        n = len(x_clean)
        n_fft = 2 ** int(np.ceil(np.log2(2 * n - 1)))
        
        # FFT-based cross-correlation
        X = fft(x_clean, n_fft)
        Y = fft(y_clean, n_fft)
        
        # Cross-power spectral density
        cross_psd = X * np.conj(Y)
        
        # Inverse FFT to get correlation
        correlation = np.real(ifft(cross_psd))
        
        # Rearrange and truncate
        correlation = np.concatenate([correlation[n:], correlation[:n]])
        
        # Create lag array
        lags = np.arange(-n + 1, n)
        
        # Limit to max_lag
        mask = np.abs(lags) <= self.max_lag
        lags = lags[mask]
        correlation = correlation[mask]
        
        # Normalize
        correlation = correlation / (np.std(x_clean) * np.std(y_clean) * n)
        
        # Find optimal lag
        max_idx = np.argmax(np.abs(correlation))
        optimal_lag = lags[max_idx]
        max_correlation = correlation[max_idx]
        
        # Statistical testing
        p_values = self._compute_correlation_p_values(correlation, n)
        confidence_intervals = self._compute_confidence_intervals(correlation, n)
        
        is_significant = p_values[max_idx] < self.significance_level if p_values is not None else False
        
        return CrossCorrelationResult(
            lags=lags,
            correlation=correlation,
            max_correlation=max_correlation,
            optimal_lag=optimal_lag,
            p_values=p_values,
            confidence_intervals=confidence_intervals,
            is_significant=is_significant,
            method='frequency_domain',
            additional_info={
                'n_fft': n_fft,
                'sample_size': n,
                'max_lag_tested': self.max_lag
            }
        )
    
    def partial_cross_correlation(self,
                                x: np.ndarray,
                                y: np.ndarray,
                                z: np.ndarray) -> CrossCorrelationResult:
        """
        Compute partial cross-correlation between x and y, controlling for z.
        
        Args:
            x: First time series
            y: Second time series  
            z: Control time series
            
        Returns:
            Partial cross-correlation results
        """
        # Preprocess all series
        x_clean = self._preprocess_single_series(x)
        y_clean = self._preprocess_single_series(y)
        z_clean = self._preprocess_single_series(z)
        
        # Ensure same length
        min_len = min(len(x_clean), len(y_clean), len(z_clean))
        x_clean = x_clean[:min_len]
        y_clean = y_clean[:min_len]
        z_clean = z_clean[:min_len]
        
        # Remove linear dependence on z
        x_residual = self._remove_linear_dependence(x_clean, z_clean)
        y_residual = self._remove_linear_dependence(y_clean, z_clean)
        
        # Compute cross-correlation of residuals
        result = self.cross_correlation(x_residual, y_residual)
        result.method = 'partial_correlation'
        result.additional_info['controlled_for'] = 'z_series'
        
        return result
    
    def dynamic_cross_correlation(self,
                                x: np.ndarray,
                                y: np.ndarray,
                                window_size: int = 100,
                                step_size: int = 10) -> Dict[str, Any]:
        """
        Compute dynamic cross-correlation using sliding windows.
        
        Args:
            x: First time series
            y: Second time series
            window_size: Size of sliding window
            step_size: Step size for window sliding
            
        Returns:
            Dynamic cross-correlation results
        """
        x_clean, y_clean = self._preprocess_data(x, y)
        
        n_samples = len(x_clean)
        n_windows = (n_samples - window_size) // step_size + 1
        
        # Storage for results
        time_centers = []
        correlations = []
        optimal_lags = []
        max_correlations = []
        p_values = []
        
        for i in range(n_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size
            
            if end_idx > n_samples:
                break
                
            # Extract window
            x_window = x_clean[start_idx:end_idx]
            y_window = y_clean[start_idx:end_idx]
            
            # Compute cross-correlation for this window
            result = self.cross_correlation(x_window, y_window, mode='full')
            
            time_centers.append(start_idx + window_size // 2)
            correlations.append(result.correlation)
            optimal_lags.append(result.optimal_lag)
            max_correlations.append(result.max_correlation)
            
            if result.p_values is not None:
                max_idx = np.argmax(np.abs(result.correlation))
                p_values.append(result.p_values[max_idx])
            else:
                p_values.append(np.nan)
        
        return {
            'time_centers': np.array(time_centers),
            'correlations': correlations,
            'optimal_lags': np.array(optimal_lags),
            'max_correlations': np.array(max_correlations),
            'p_values': np.array(p_values),
            'window_size': window_size,
            'step_size': step_size,
            'n_windows': len(time_centers),
            'lags': result.lags,  # Lag structure (same for all windows)
            'method': 'dynamic_correlation'
        }
    
    def comprehensive_correlation_analysis(self,
                                         x: np.ndarray,
                                         y: np.ndarray,
                                         x_name: str = 'X',
                                         y_name: str = 'Y') -> Dict[str, Any]:
        """
        Perform comprehensive cross-correlation analysis.
        
        Args:
            x: First time series
            y: Second time series
            x_name: Name for first series
            y_name: Name for second series
            
        Returns:
            Comprehensive correlation analysis results
        """
        results = {
            'series_names': (x_name, y_name),
            'data_info': {
                'x_length': len(x),
                'y_length': len(y),
                'x_mean': np.mean(x),
                'y_mean': np.mean(y),
                'x_std': np.std(x),
                'y_std': np.std(y)
            },
            'analyses': {}
        }
        
        # Time-domain cross-correlation
        results['analyses']['time_domain'] = self.cross_correlation(x, y)
        
        # Frequency-domain cross-correlation
        results['analyses']['frequency_domain'] = self.frequency_domain_correlation(x, y)
        
        # Dynamic cross-correlation
        window_size = min(len(x) // 10, 200)  # Adaptive window size
        step_size = max(1, window_size // 10)
        
        if len(x) > window_size * 2:  # Only if sufficient data
            results['analyses']['dynamic'] = self.dynamic_cross_correlation(
                x, y, window_size, step_size
            )
        
        # Summary and comparison
        results['summary'] = self._summarize_correlation_results(results['analyses'])
        
        return results
    
    def _preprocess_data(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess time series data."""
        # Remove NaN values
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]
        
        # Ensure same length
        min_len = min(len(x_clean), len(y_clean))
        x_clean = x_clean[:min_len]
        y_clean = y_clean[:min_len]
        
        # Detrend if specified
        if self.detrend_method == 'linear':
            x_clean = signal.detrend(x_clean, type='linear')
            y_clean = signal.detrend(y_clean, type='linear')
        elif self.detrend_method == 'constant':
            x_clean = signal.detrend(x_clean, type='constant')
            y_clean = signal.detrend(y_clean, type='constant')
        
        return x_clean, y_clean
    
    def _preprocess_single_series(self, x: np.ndarray) -> np.ndarray:
        """Preprocess single time series."""
        # Remove NaN values
        x_clean = x[~np.isnan(x)]
        
        # Detrend if specified
        if self.detrend_method == 'linear':
            x_clean = signal.detrend(x_clean, type='linear')
        elif self.detrend_method == 'constant':
            x_clean = signal.detrend(x_clean, type='constant')
        
        return x_clean
    
    def _remove_linear_dependence(self, y: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Remove linear dependence of y on x."""
        # Simple linear regression: y = a*x + b + residual
        A = np.vstack([x, np.ones(len(x))]).T
        coeffs, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
        
        # Return residuals
        return y - (coeffs[0] * x + coeffs[1])
    
    def _compute_correlation_p_values(self, correlation: np.ndarray, n: int) -> Optional[np.ndarray]:
        """Compute p-values for correlation coefficients."""
        try:
            # Standard error under null hypothesis
            se = 1.0 / np.sqrt(n - 3)
            
            # Fisher z-transform
            z_scores = np.arctanh(np.abs(correlation)) / se
            
            # Two-tailed p-values
            p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
            
            return p_values
        except:
            return None
    
    def _compute_confidence_intervals(self, correlation: np.ndarray, n: int) -> Optional[np.ndarray]:
        """Compute confidence intervals for correlation coefficients."""
        try:
            # Standard error
            se = 1.0 / np.sqrt(n - 3)
            
            # Critical value for confidence interval
            alpha = self.significance_level
            z_critical = stats.norm.ppf(1 - alpha / 2)
            
            # Fisher z-transform
            z_r = np.arctanh(correlation)
            
            # Confidence interval in z-space
            z_lower = z_r - z_critical * se
            z_upper = z_r + z_critical * se
            
            # Transform back to correlation space
            ci_lower = np.tanh(z_lower)
            ci_upper = np.tanh(z_upper)
            
            return np.column_stack([ci_lower, ci_upper])
        except:
            return None
    
    def _summarize_correlation_results(self, analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize correlation analysis results."""
        summary = {
            'consensus_lag': None,
            'consensus_correlation': None,
            'method_agreement': {},
            'significant_results': [],
            'recommendations': []
        }
        
        # Collect significant results
        significant_lags = []
        significant_correlations = []
        
        for method_name, result in analyses.items():
            if method_name == 'dynamic':
                # Handle dynamic results differently
                significant_windows = np.sum(result['p_values'] < self.significance_level)
                if significant_windows > len(result['p_values']) // 2:
                    median_lag = np.median(result['optimal_lags'])
                    median_corr = np.median(result['max_correlations'])
                    summary['significant_results'].append({
                        'method': method_name,
                        'median_lag': median_lag,
                        'median_correlation': median_corr,
                        'significant_windows': significant_windows,
                        'total_windows': len(result['p_values'])
                    })
                    significant_lags.append(median_lag)
                    significant_correlations.append(median_corr)
            else:
                if result.is_significant:
                    summary['significant_results'].append({
                        'method': method_name,
                        'optimal_lag': result.optimal_lag,
                        'max_correlation': result.max_correlation,
                        'p_value': result.p_values[np.argmax(np.abs(result.correlation))] if result.p_values is not None else None
                    })
                    significant_lags.append(result.optimal_lag)
                    significant_correlations.append(result.max_correlation)
        
        # Determine consensus
        if significant_lags:
            summary['consensus_lag'] = int(np.median(significant_lags))
            summary['consensus_correlation'] = np.median(significant_correlations)
            
            # Method agreement
            lag_std = np.std(significant_lags)
            corr_std = np.std(significant_correlations)
            
            summary['method_agreement'] = {
                'lag_std': lag_std,
                'correlation_std': corr_std,
                'high_agreement': lag_std < 2.0 and corr_std < 0.1
            }
        
        return summary
