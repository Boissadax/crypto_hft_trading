"""
Regime Detection Module

Implements regime detection and structural break analysis:
- Hidden Markov Models for regime identification
- Structural break tests (Chow, CUSUM, etc.)
- Volatility regime detection
- Dynamic regime detection with online updates
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from scipy import stats
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

@dataclass
class RegimeChange:
    """Information about detected regime change."""
    timestamp: int
    confidence: float
    regime_before: int
    regime_after: int
    statistic: float
    p_value: Optional[float]
    method: str

@dataclass
class RegimeInfo:
    """Information about a detected regime."""
    regime_id: int
    start_time: int
    end_time: int
    duration: int
    mean_value: float
    volatility: float
    characteristics: Dict[str, Any]

class RegimeDetector:
    """
    Advanced regime detection for financial time series.
    
    Implements multiple methods for detecting structural breaks
    and regime changes in time series data.
    """
    
    def __init__(self,
                 min_regime_length: int = 50,
                 significance_level: float = 0.05,
                 n_regimes: Optional[int] = None):
        """
        Initialize regime detector.
        
        Args:
            min_regime_length: Minimum length for a regime
            significance_level: Statistical significance threshold
            n_regimes: Number of regimes (auto-detect if None)
        """
        self.min_regime_length = min_regime_length
        self.significance_level = significance_level
        self.n_regimes = n_regimes
        
    def hmm_regime_detection(self,
                           data: np.ndarray,
                           n_states: Optional[int] = None,
                           features: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detect regimes using Hidden Markov Model approach.
        
        Args:
            data: Time series data or feature matrix
            n_states: Number of hidden states (auto-detect if None)
            features: Names of features if data is multivariate
            
        Returns:
            HMM regime detection results
        """
        # Prepare data
        if data.ndim == 1:
            # Convert to returns and volatility features
            returns = np.diff(np.log(data + 1e-10))  # Avoid log(0)
            volatility = pd.Series(returns).rolling(window=20).std().fillna(method='bfill').values
            X = np.column_stack([returns[19:], volatility[19:]])  # Align after rolling window
        else:
            X = data
            
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Determine optimal number of states
        if n_states is None:
            n_states = self._select_optimal_states(X_scaled)
        
        # Fit Gaussian Mixture Model as HMM approximation
        gmm = GaussianMixture(
            n_components=n_states,
            covariance_type='full',
            random_state=42,
            max_iter=200
        )
        
        try:
            gmm.fit(X_scaled)
            states = gmm.predict(X_scaled)
            probabilities = gmm.predict_proba(X_scaled)
            
            # Detect regime changes
            regime_changes = self._detect_state_changes(states, probabilities)
            
            # Characterize regimes
            regimes = self._characterize_regimes(data, states, X_scaled)
            
            return {
                'states': states,
                'probabilities': probabilities,
                'regime_changes': regime_changes,
                'regimes': regimes,
                'n_regimes': n_states,
                'aic': gmm.aic(X_scaled),
                'bic': gmm.bic(X_scaled),
                'log_likelihood': gmm.score(X_scaled),
                'method': 'hmm_gmm',
                'features_used': features or ['returns', 'volatility']
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'method': 'hmm_gmm',
                'states': None,
                'regime_changes': [],
                'regimes': []
            }
    
    def structural_break_test(self,
                            data: np.ndarray,
                            test_type: str = 'cusum') -> Dict[str, Any]:
        """
        Perform structural break tests.
        
        Args:
            data: Time series data
            test_type: Type of test ('cusum', 'chow', 'bai_perron')
            
        Returns:
            Structural break test results
        """
        if test_type == 'cusum':
            return self._cusum_test(data)
        elif test_type == 'chow':
            return self._chow_test(data)
        elif test_type == 'bai_perron':
            return self._bai_perron_test(data)
        else:
            raise ValueError(f"Unknown test type: {test_type}")
    
    def volatility_regime_detection(self,
                                  returns: np.ndarray,
                                  window_size: int = 50) -> Dict[str, Any]:
        """
        Detect volatility regimes in return series.
        
        Args:
            returns: Return time series
            window_size: Window for volatility estimation
            
        Returns:
            Volatility regime detection results
        """
        # Calculate rolling volatility
        volatility = pd.Series(returns).rolling(window=window_size).std().fillna(method='bfill').values
        
        # Detect volatility regimes using clustering
        vol_reshaped = volatility.reshape(-1, 1)
        
        # Try different numbers of clusters
        n_clusters_range = range(2, min(6, len(volatility) // self.min_regime_length + 1))
        best_score = -np.inf
        best_clustering = None
        best_n_clusters = 2
        
        for n_clusters in n_clusters_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(vol_reshaped)
            
            # Calculate silhouette-like score
            if len(np.unique(labels)) > 1:
                score = self._calculate_clustering_score(vol_reshaped, labels)
                if score > best_score:
                    best_score = score
                    best_clustering = kmeans
                    best_n_clusters = n_clusters
        
        if best_clustering is None:
            return {
                'error': 'Could not find valid clustering',
                'method': 'volatility_regime'
            }
        
        # Get final clustering
        regime_labels = best_clustering.predict(vol_reshaped)
        centroids = best_clustering.cluster_centers_.flatten()
        
        # Sort regimes by volatility level
        sorted_indices = np.argsort(centroids)
        regime_mapping = {old: new for new, old in enumerate(sorted_indices)}
        regime_labels = np.array([regime_mapping[label] for label in regime_labels])
        
        # Detect regime changes
        regime_changes = []
        for i in range(1, len(regime_labels)):
            if regime_labels[i] != regime_labels[i-1]:
                # Calculate confidence based on distance to cluster centers
                current_vol = volatility[i]
                distances = np.abs(centroids[sorted_indices] - current_vol)
                min_dist = np.min(distances)
                confidence = 1.0 / (1.0 + min_dist)
                
                regime_changes.append(RegimeChange(
                    timestamp=i,
                    confidence=confidence,
                    regime_before=regime_labels[i-1],
                    regime_after=regime_labels[i],
                    statistic=min_dist,
                    p_value=None,
                    method='volatility_clustering'
                ))
        
        # Characterize volatility regimes
        regimes = []
        current_regime = regime_labels[0]
        start_time = 0
        
        for i in range(1, len(regime_labels)):
            if regime_labels[i] != current_regime or i == len(regime_labels) - 1:
                end_time = i if regime_labels[i] != current_regime else i + 1
                
                regime_data = returns[start_time:end_time]
                regime_vol = volatility[start_time:end_time]
                
                regimes.append(RegimeInfo(
                    regime_id=current_regime,
                    start_time=start_time,
                    end_time=end_time,
                    duration=end_time - start_time,
                    mean_value=np.mean(regime_vol),
                    volatility=np.std(regime_vol),
                    characteristics={
                        'volatility_level': ['low', 'medium', 'high'][current_regime] if best_n_clusters <= 3 else f'regime_{current_regime}',
                        'mean_return': np.mean(regime_data),
                        'return_volatility': np.std(regime_data),
                        'skewness': stats.skew(regime_data) if len(regime_data) > 3 else 0,
                        'kurtosis': stats.kurtosis(regime_data) if len(regime_data) > 3 else 0
                    }
                ))
                
                current_regime = regime_labels[i]
                start_time = i
        
        return {
            'regime_labels': regime_labels,
            'volatility': volatility,
            'regime_changes': regime_changes,
            'regimes': regimes,
            'n_regimes': best_n_clusters,
            'centroids': centroids[sorted_indices],
            'clustering_score': best_score,
            'method': 'volatility_regime',
            'window_size': window_size
        }
    
    def comprehensive_regime_analysis(self,
                                    data: np.ndarray,
                                    include_volatility: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive regime analysis using multiple methods.
        
        Args:
            data: Time series data
            include_volatility: Whether to include volatility regime analysis
            
        Returns:
            Comprehensive regime analysis results
        """
        results = {
            'data_info': {
                'length': len(data),
                'mean': np.mean(data),
                'std': np.std(data),
                'min': np.min(data),
                'max': np.max(data)
            },
            'analyses': {}
        }
        
        # HMM regime detection
        results['analyses']['hmm'] = self.hmm_regime_detection(data)
        
        # Structural break tests
        results['analyses']['cusum'] = self.structural_break_test(data, 'cusum')
        results['analyses']['chow'] = self.structural_break_test(data, 'chow')
        
        # Volatility regime analysis (if requested and data looks like prices)
        if include_volatility and len(data) > 100:
            returns = np.diff(np.log(data + 1e-10))
            results['analyses']['volatility'] = self.volatility_regime_detection(returns)
        
        # Summarize results
        results['summary'] = self._summarize_regime_results(results['analyses'])
        
        return results
    
    def _select_optimal_states(self, X: np.ndarray) -> int:
        """Select optimal number of states using information criteria."""
        max_states = min(6, len(X) // self.min_regime_length)
        
        if max_states < 2:
            return 2
        
        aic_scores = []
        bic_scores = []
        
        for n_states in range(2, max_states + 1):
            try:
                gmm = GaussianMixture(n_components=n_states, random_state=42, max_iter=100)
                gmm.fit(X)
                aic_scores.append(gmm.aic(X))
                bic_scores.append(gmm.bic(X))
            except:
                aic_scores.append(np.inf)
                bic_scores.append(np.inf)
        
        # Select based on BIC (more conservative)
        optimal_states = np.argmin(bic_scores) + 2
        return optimal_states
    
    def _detect_state_changes(self, states: np.ndarray, probabilities: np.ndarray) -> List[RegimeChange]:
        """Detect regime changes from state sequence."""
        changes = []
        
        for i in range(1, len(states)):
            if states[i] != states[i-1]:
                # Calculate confidence based on probability difference
                prob_before = probabilities[i-1, states[i-1]]
                prob_after = probabilities[i, states[i]]
                confidence = (prob_before + prob_after) / 2
                
                changes.append(RegimeChange(
                    timestamp=i,
                    confidence=confidence,
                    regime_before=states[i-1],
                    regime_after=states[i],
                    statistic=abs(prob_after - prob_before),
                    p_value=None,
                    method='hmm_state_change'
                ))
        
        return changes
    
    def _characterize_regimes(self, original_data: np.ndarray, states: np.ndarray, features: np.ndarray) -> List[RegimeInfo]:
        """Characterize detected regimes."""
        regimes = []
        unique_states = np.unique(states)
        
        for state in unique_states:
            state_mask = states == state
            state_indices = np.where(state_mask)[0]
            
            if len(state_indices) == 0:
                continue
            
            # Find contiguous segments for this state
            segments = []
            start = state_indices[0]
            
            for i in range(1, len(state_indices)):
                if state_indices[i] != state_indices[i-1] + 1:
                    # End of segment
                    segments.append((start, state_indices[i-1] + 1))
                    start = state_indices[i]
            
            # Add last segment
            segments.append((start, state_indices[-1] + 1))
            
            # Characterize each segment
            for start_time, end_time in segments:
                if end_time - start_time >= self.min_regime_length:
                    regime_data = original_data[start_time:end_time]
                    regime_features = features[state_mask]
                    
                    regimes.append(RegimeInfo(
                        regime_id=state,
                        start_time=start_time,
                        end_time=end_time,
                        duration=end_time - start_time,
                        mean_value=np.mean(regime_data),
                        volatility=np.std(regime_data),
                        characteristics={
                            'feature_means': np.mean(regime_features, axis=0),
                            'feature_stds': np.std(regime_features, axis=0),
                            'skewness': stats.skew(regime_data) if len(regime_data) > 3 else 0,
                            'kurtosis': stats.kurtosis(regime_data) if len(regime_data) > 3 else 0,
                            'trend': np.polyfit(range(len(regime_data)), regime_data, 1)[0] if len(regime_data) > 1 else 0
                        }
                    ))
        
        return regimes
    
    def _cusum_test(self, data: np.ndarray) -> Dict[str, Any]:
        """Perform CUSUM test for structural breaks."""
        n = len(data)
        
        # Calculate cumulative sum of deviations from mean
        mean_val = np.mean(data)
        cusum = np.cumsum(data - mean_val)
        
        # Standardize
        std_val = np.std(data)
        cusum_standardized = cusum / (std_val * np.sqrt(n))
        
        # Find peaks in absolute CUSUM
        peaks, properties = find_peaks(np.abs(cusum_standardized), height=0.5, distance=self.min_regime_length)
        
        # Critical value for CUSUM test (approximate)
        critical_value = 1.36  # 5% significance level
        
        # Identify significant breaks
        breaks = []
        for peak in peaks:
            if np.abs(cusum_standardized[peak]) > critical_value:
                p_value = 2 * (1 - stats.norm.cdf(np.abs(cusum_standardized[peak])))
                
                breaks.append(RegimeChange(
                    timestamp=peak,
                    confidence=1 - p_value,
                    regime_before=0,  # Before break
                    regime_after=1,   # After break
                    statistic=cusum_standardized[peak],
                    p_value=p_value,
                    method='cusum'
                ))
        
        return {
            'cusum': cusum_standardized,
            'breaks': breaks,
            'critical_value': critical_value,
            'max_cusum': np.max(np.abs(cusum_standardized)),
            'is_significant': np.max(np.abs(cusum_standardized)) > critical_value,
            'method': 'cusum'
        }
    
    def _chow_test(self, data: np.ndarray) -> Dict[str, Any]:
        """Perform Chow test for structural breaks."""
        n = len(data)
        breaks = []
        
        # Test potential break points
        for breakpoint in range(self.min_regime_length, n - self.min_regime_length):
            # Split data
            data1 = data[:breakpoint]
            data2 = data[breakpoint:]
            
            # Fit simple linear models (trend + constant)
            t1 = np.arange(len(data1))
            t2 = np.arange(len(data2))
            
            # Full model
            t_full = np.arange(n)
            X_full = np.column_stack([t_full, np.ones(n)])
            beta_full, residuals_full, _, _ = np.linalg.lstsq(X_full, data, rcond=None)
            rss_full = np.sum((data - X_full @ beta_full) ** 2)
            
            # Restricted models
            X1 = np.column_stack([t1, np.ones(len(data1))])
            X2 = np.column_stack([t2, np.ones(len(data2))])
            
            try:
                beta1, _, _, _ = np.linalg.lstsq(X1, data1, rcond=None)
                beta2, _, _, _ = np.linalg.lstsq(X2, data2, rcond=None)
                
                rss1 = np.sum((data1 - X1 @ beta1) ** 2)
                rss2 = np.sum((data2 - X2 @ beta2) ** 2)
                rss_restricted = rss1 + rss2
                
                # Chow statistic
                k = 2  # Number of parameters
                f_stat = ((rss_full - rss_restricted) / k) / (rss_restricted / (n - 2 * k))
                p_value = 1 - stats.f.cdf(f_stat, k, n - 2 * k)
                
                if p_value < self.significance_level:
                    breaks.append(RegimeChange(
                        timestamp=breakpoint,
                        confidence=1 - p_value,
                        regime_before=0,
                        regime_after=1,
                        statistic=f_stat,
                        p_value=p_value,
                        method='chow'
                    ))
                    
            except np.linalg.LinAlgError:
                continue
        
        return {
            'breaks': breaks,
            'method': 'chow',
            'n_potential_breaks': len(breaks)
        }
    
    def _bai_perron_test(self, data: np.ndarray) -> Dict[str, Any]:
        """Simplified Bai-Perron test for multiple structural breaks."""
        # This is a simplified version - full implementation would be more complex
        n = len(data)
        max_breaks = min(5, n // (2 * self.min_regime_length))
        
        if max_breaks < 1:
            return {'breaks': [], 'method': 'bai_perron', 'error': 'Insufficient data for break detection'}
        
        # Use BIC-like criterion to select breaks
        best_breaks = []
        best_bic = np.inf
        
        for n_breaks in range(1, max_breaks + 1):
            # Simple approach: divide data into equal segments
            segment_length = n // (n_breaks + 1)
            candidate_breaks = [i * segment_length for i in range(1, n_breaks + 1)]
            
            # Calculate BIC for this segmentation
            bic = self._calculate_segmentation_bic(data, candidate_breaks)
            
            if bic < best_bic:
                best_bic = bic
                best_breaks = candidate_breaks
        
        # Convert to RegimeChange objects
        regime_changes = []
        for i, breakpoint in enumerate(best_breaks):
            regime_changes.append(RegimeChange(
                timestamp=breakpoint,
                confidence=0.95,  # Placeholder
                regime_before=i,
                regime_after=i + 1,
                statistic=best_bic,
                p_value=None,
                method='bai_perron'
            ))
        
        return {
            'breaks': regime_changes,
            'method': 'bai_perron',
            'bic': best_bic,
            'n_breaks': len(best_breaks)
        }
    
    def _calculate_clustering_score(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Calculate clustering quality score."""
        if len(np.unique(labels)) <= 1:
            return -np.inf
        
        # Within-cluster sum of squares
        wcss = 0
        for label in np.unique(labels):
            cluster_data = data[labels == label]
            if len(cluster_data) > 0:
                center = np.mean(cluster_data, axis=0)
                wcss += np.sum((cluster_data - center) ** 2)
        
        # Between-cluster sum of squares
        overall_center = np.mean(data, axis=0)
        bcss = 0
        for label in np.unique(labels):
            cluster_data = data[labels == label]
            if len(cluster_data) > 0:
                center = np.mean(cluster_data, axis=0)
                bcss += len(cluster_data) * np.sum((center - overall_center) ** 2)
        
        # Return ratio (higher is better)
        return bcss / (wcss + 1e-10)
    
    def _calculate_segmentation_bic(self, data: np.ndarray, breakpoints: List[int]) -> float:
        """Calculate BIC for a given segmentation."""
        breakpoints = [0] + breakpoints + [len(data)]
        n = len(data)
        k = len(breakpoints) - 1  # Number of segments
        
        log_likelihood = 0
        for i in range(len(breakpoints) - 1):
            segment_data = data[breakpoints[i]:breakpoints[i+1]]
            if len(segment_data) > 1:
                segment_var = np.var(segment_data)
                if segment_var > 0:
                    log_likelihood += len(segment_data) * np.log(2 * np.pi * segment_var) / 2
                    log_likelihood += np.sum((segment_data - np.mean(segment_data)) ** 2) / (2 * segment_var)
        
        # BIC = 2 * log_likelihood + k * log(n)
        bic = 2 * log_likelihood + k * np.log(n)
        return bic
    
    def _summarize_regime_results(self, analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize regime detection results across methods."""
        summary = {
            'detected_regimes': {},
            'structural_breaks': [],
            'consensus': {},
            'recommendations': []
        }
        
        # Collect all detected breaks
        all_breaks = []
        
        for method_name, result in analyses.items():
            if 'error' in result:
                continue
                
            method_breaks = []
            
            if method_name == 'hmm' and 'regime_changes' in result:
                method_breaks = [change.timestamp for change in result['regime_changes']]
                summary['detected_regimes']['hmm'] = {
                    'n_regimes': result.get('n_regimes', 0),
                    'breaks': method_breaks
                }
            
            elif method_name in ['cusum', 'chow', 'bai_perron'] and 'breaks' in result:
                method_breaks = [change.timestamp for change in result['breaks']]
                summary['structural_breaks'].extend([{
                    'method': method_name,
                    'timestamp': change.timestamp,
                    'p_value': change.p_value,
                    'confidence': change.confidence
                } for change in result['breaks']])
            
            elif method_name == 'volatility' and 'regime_changes' in result:
                method_breaks = [change.timestamp for change in result['regime_changes']]
                summary['detected_regimes']['volatility'] = {
                    'n_regimes': result.get('n_regimes', 0),
                    'breaks': method_breaks
                }
            
            all_breaks.extend(method_breaks)
        
        # Find consensus breaks (breaks detected by multiple methods)
        if all_breaks:
            # Group nearby breaks (within tolerance)
            tolerance = 10
            consensus_breaks = []
            
            all_breaks_sorted = sorted(all_breaks)
            current_group = [all_breaks_sorted[0]]
            
            for break_point in all_breaks_sorted[1:]:
                if break_point - current_group[-1] <= tolerance:
                    current_group.append(break_point)
                else:
                    if len(current_group) > 1:  # Multiple methods agree
                        consensus_breaks.append({
                            'timestamp': int(np.median(current_group)),
                            'methods_agreeing': len(current_group),
                            'timestamp_range': (min(current_group), max(current_group))
                        })
                    current_group = [break_point]
            
            # Check last group
            if len(current_group) > 1:
                consensus_breaks.append({
                    'timestamp': int(np.median(current_group)),
                    'methods_agreeing': len(current_group),
                    'timestamp_range': (min(current_group), max(current_group))
                })
            
            summary['consensus']['breaks'] = consensus_breaks
            summary['consensus']['total_breaks'] = len(all_breaks)
            summary['consensus']['consensus_breaks'] = len(consensus_breaks)
        
        return summary
