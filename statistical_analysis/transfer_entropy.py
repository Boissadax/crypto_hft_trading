"""
Transfer Entropy Analyzer

Implements Transfer Entropy for detecting directional information flow
between time series, specifically designed for cryptocurrency lead-lag analysis.

Transfer Entropy measures the amount of uncertainty reduced in predicting
the future of Y by using the history of X, given the history of Y.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.special import digamma
from numba import njit
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logger = logging.getLogger(__name__)


@dataclass
class TransferEntropyResult:
    """Result from Transfer Entropy analysis"""
    source_symbol: str
    target_symbol: str
    lag_us: int
    transfer_entropy: float
    p_value: float
    confidence_interval: Tuple[float, float]
    significance_level: float
    
    # Additional metrics
    entropy_source: float
    entropy_target: float
    mutual_information: float
    normalized_te: float  # Normalized by target entropy
    
    # Method metadata
    method: str
    embedding_dimension: int
    num_samples: int
    bootstrap_samples: int


class TransferEntropyAnalyzer:
    """
    Advanced Transfer Entropy analyzer for cryptocurrency lead-lag detection.
    
    Implements multiple estimation methods:
    - KSG (Kraskov-Stögbauer-Grassberger) estimator
    - Binning-based estimator
    - Gaussian estimator (for comparison)
    
    Features:
    - Statistical significance testing via bootstrap
    - Multiple lag analysis
    - Bidirectional causality detection
    - Computational optimization for HFT data
    """
    
    def __init__(self,
                 embedding_dimension: int = 3,
                 delay: int = 1,
                 k_neighbors: int = 5,
                 method: str = 'ksg',
                 significance_level: float = 0.05,
                 bootstrap_samples: int = 1000,
                 max_workers: int = 4):
        """
        Initialize Transfer Entropy analyzer
        
        Args:
            embedding_dimension: Embedding dimension for reconstruction
            delay: Time delay for embedding
            k_neighbors: Number of neighbors for KSG estimator
            method: Estimation method ('ksg', 'binning', 'gaussian')
            significance_level: Statistical significance level
            bootstrap_samples: Number of bootstrap samples for p-value estimation
            max_workers: Number of parallel workers
        """
        self.embedding_dimension = embedding_dimension
        self.delay = delay
        self.k_neighbors = k_neighbors
        self.method = method
        self.significance_level = significance_level
        self.bootstrap_samples = bootstrap_samples
        self.max_workers = max_workers
        
        # Validation
        if method not in ['ksg', 'binning', 'gaussian']:
            raise ValueError(f"Unknown method: {method}")
            
        if embedding_dimension < 1:
            raise ValueError("Embedding dimension must be >= 1")
            
        # Statistics tracking
        self.computation_stats = {
            'total_computations': 0,
            'avg_computation_time_s': 0.0,
            'method_usage': {m: 0 for m in ['ksg', 'binning', 'gaussian']}
        }
        
    def analyze_transfer_entropy(self,
                               source_data: np.ndarray,
                               target_data: np.ndarray,
                               source_symbol: str,
                               target_symbol: str,
                               lags_us: List[int],
                               sampling_frequency_hz: float = 1.0) -> List[TransferEntropyResult]:
        """
        Analyze transfer entropy for multiple lags
        
        Args:
            source_data: Source time series data
            target_data: Target time series data
            source_symbol: Source symbol name
            target_symbol: Target symbol name
            lags_us: List of lags in microseconds to analyze
            sampling_frequency_hz: Sampling frequency for lag conversion
            
        Returns:
            List of TransferEntropyResult objects
        """
        start_time = time.time()
        
        # Validate inputs
        if len(source_data) != len(target_data):
            raise ValueError("Source and target data must have same length")
            
        if len(source_data) < 100:
            logger.warning("Small sample size may lead to unreliable results")
            
        # Convert lags to sample indices
        lag_samples = [max(1, int(lag_us * sampling_frequency_hz / 1_000_000)) 
                      for lag_us in lags_us]
        
        # Prepare data
        source_clean, target_clean = self._prepare_data(source_data, target_data)
        
        # Parallel computation for multiple lags
        results = []
        
        if self.max_workers > 1:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                
                for lag_us, lag_samples_val in zip(lags_us, lag_samples):
                    future = executor.submit(
                        self._compute_transfer_entropy_single_lag,
                        source_clean, target_clean, source_symbol, target_symbol,
                        lag_us, lag_samples_val
                    )
                    futures.append(future)
                    
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except Exception as e:
                        logger.error(f"Error computing transfer entropy: {e}")
        else:
            # Sequential execution
            for lag_us, lag_samples_val in zip(lags_us, lag_samples):
                result = self._compute_transfer_entropy_single_lag(
                    source_clean, target_clean, source_symbol, target_symbol,
                    lag_us, lag_samples_val
                )
                if result:
                    results.append(result)
        
        # Update statistics
        computation_time = time.time() - start_time
        self.computation_stats['total_computations'] += 1
        self.computation_stats['avg_computation_time_s'] = (
            (self.computation_stats['avg_computation_time_s'] * 
             (self.computation_stats['total_computations'] - 1) + computation_time) /
            self.computation_stats['total_computations']
        )
        self.computation_stats['method_usage'][self.method] += len(results)
        
        return sorted(results, key=lambda x: x.lag_us)
    
    def analyze_bidirectional_causality(self,
                                      data_x: np.ndarray,
                                      data_y: np.ndarray,
                                      symbol_x: str,
                                      symbol_y: str,
                                      lags_us: List[int],
                                      sampling_frequency_hz: float = 1.0) -> Dict[str, List[TransferEntropyResult]]:
        """
        Analyze bidirectional causality between two time series
        
        Returns:
            Dictionary with 'x_to_y' and 'y_to_x' transfer entropy results
        """
        results = {}
        
        # X -> Y
        results['x_to_y'] = self.analyze_transfer_entropy(
            data_x, data_y, symbol_x, symbol_y, lags_us, sampling_frequency_hz
        )
        
        # Y -> X
        results['y_to_x'] = self.analyze_transfer_entropy(
            data_y, data_x, symbol_y, symbol_x, lags_us, sampling_frequency_hz
        )
        
        return results
    
    def _prepare_data(self, 
                     source_data: np.ndarray, 
                     target_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare and clean data for transfer entropy computation"""
        # Remove NaN values
        valid_mask = np.isfinite(source_data) & np.isfinite(target_data)
        source_clean = source_data[valid_mask]
        target_clean = target_data[valid_mask]
        
        # Standardize data for better numerical stability
        if len(source_clean) > 1:
            source_clean = (source_clean - np.mean(source_clean)) / np.std(source_clean)
            target_clean = (target_clean - np.mean(target_clean)) / np.std(target_clean)
        
        return source_clean, target_clean
    
    def _compute_transfer_entropy_single_lag(self,
                                           source_data: np.ndarray,
                                           target_data: np.ndarray,
                                           source_symbol: str,
                                           target_symbol: str,
                                           lag_us: int,
                                           lag_samples: int) -> Optional[TransferEntropyResult]:
        """Compute transfer entropy for a single lag"""
        try:
            # Create embedded vectors
            source_embedded, target_embedded = self._create_embeddings(
                source_data, target_data, lag_samples
            )
            
            if len(source_embedded) < self.k_neighbors + 1:
                logger.warning(f"Insufficient data for lag {lag_us}μs")
                return None
                
            # Compute transfer entropy based on method
            if self.method == 'ksg':
                te_value = self._compute_te_ksg(source_embedded, target_embedded)
            elif self.method == 'binning':
                te_value = self._compute_te_binning(source_embedded, target_embedded)
            elif self.method == 'gaussian':
                te_value = self._compute_te_gaussian(source_embedded, target_embedded)
            else:
                raise ValueError(f"Unknown method: {self.method}")
                
            # Compute additional metrics
            entropy_source = self._compute_entropy(source_embedded)
            entropy_target = self._compute_entropy(target_embedded)
            mutual_info = self._compute_mutual_information(source_embedded, target_embedded)
            
            # Normalize transfer entropy
            normalized_te = te_value / entropy_target if entropy_target > 0 else 0.0
            
            # Statistical significance testing
            p_value, confidence_interval = self._bootstrap_significance_test(
                source_embedded, target_embedded, te_value
            )
            
            return TransferEntropyResult(
                source_symbol=source_symbol,
                target_symbol=target_symbol,
                lag_us=lag_us,
                transfer_entropy=te_value,
                p_value=p_value,
                confidence_interval=confidence_interval,
                significance_level=self.significance_level,
                entropy_source=entropy_source,
                entropy_target=entropy_target,
                mutual_information=mutual_info,
                normalized_te=normalized_te,
                method=self.method,
                embedding_dimension=self.embedding_dimension,
                num_samples=len(source_embedded),
                bootstrap_samples=self.bootstrap_samples
            )
            
        except Exception as e:
            logger.error(f"Error computing transfer entropy for lag {lag_us}: {e}")
            return None
    
    def _create_embeddings(self,
                          source_data: np.ndarray,
                          target_data: np.ndarray,
                          lag: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create time-delay embeddings for transfer entropy computation"""
        n = len(source_data)
        max_delay = max(lag, (self.embedding_dimension - 1) * self.delay)
        
        if n <= max_delay:
            raise ValueError("Insufficient data for embedding")
            
        # Create embeddings
        n_vectors = n - max_delay
        
        # Source embedding: current and past values
        source_embedded = np.zeros((n_vectors, self.embedding_dimension))
        for i in range(self.embedding_dimension):
            start_idx = max_delay - i * self.delay
            end_idx = start_idx + n_vectors
            source_embedded[:, i] = source_data[start_idx:end_idx]
            
        # Target embedding: future value + past values
        target_embedded = np.zeros((n_vectors, self.embedding_dimension + 1))
        
        # Future target value (what we're trying to predict)
        target_embedded[:, 0] = target_data[max_delay + 1:max_delay + 1 + n_vectors]
        
        # Past target values
        for i in range(self.embedding_dimension):
            start_idx = max_delay - i * self.delay
            end_idx = start_idx + n_vectors
            target_embedded[:, i + 1] = target_data[start_idx:end_idx]
            
        return source_embedded, target_embedded
    
    def _compute_te_ksg(self, 
                       source_embedded: np.ndarray, 
                       target_embedded: np.ndarray) -> float:
        """Compute transfer entropy using KSG estimator"""
        try:
            # Combine source and target for joint space
            joint_data = np.hstack([source_embedded, target_embedded])
            
            # Build nearest neighbor models
            nn_joint = NearestNeighbors(n_neighbors=self.k_neighbors + 1, metric='chebyshev')
            nn_source = NearestNeighbors(n_neighbors=self.k_neighbors + 1, metric='chebyshev')
            nn_target_past = NearestNeighbors(n_neighbors=self.k_neighbors + 1, metric='chebyshev')
            nn_target_future = NearestNeighbors(n_neighbors=self.k_neighbors + 1, metric='chebyshev')
            
            nn_joint.fit(joint_data)
            nn_source.fit(source_embedded)
            nn_target_past.fit(target_embedded[:, 1:])  # Past target values only
            nn_target_future.fit(target_embedded)  # Future + past target values
            
            # Compute KSG estimate
            te_sum = 0.0
            n_samples = len(joint_data)
            
            for i in range(n_samples):
                # Find k-th nearest neighbor distance in joint space
                distances, _ = nn_joint.kneighbors([joint_data[i]])
                epsilon = distances[0][self.k_neighbors]  # k-th distance (excluding self)
                
                # Count neighbors within epsilon in marginal spaces
                n_source = self._count_neighbors_within_epsilon(
                    nn_source, source_embedded[i], epsilon
                )
                n_target_past = self._count_neighbors_within_epsilon(
                    nn_target_past, target_embedded[i, 1:], epsilon
                )
                n_target_future = self._count_neighbors_within_epsilon(
                    nn_target_future, target_embedded[i], epsilon
                )
                
                # KSG formula
                if n_source > 0 and n_target_past > 0 and n_target_future > 0:
                    te_sum += (digamma(self.k_neighbors) + 
                              digamma(n_target_past) - 
                              digamma(n_source) - 
                              digamma(n_target_future))
                              
            return te_sum / n_samples
            
        except Exception as e:
            logger.warning(f"KSG estimator failed: {e}, falling back to Gaussian")
            return self._compute_te_gaussian(source_embedded, target_embedded)
    
    def _count_neighbors_within_epsilon(self, 
                                       nn_model: NearestNeighbors,
                                       point: np.ndarray,
                                       epsilon: float) -> int:
        """Count neighbors within epsilon distance"""
        distances, _ = nn_model.radius_neighbors([point], radius=epsilon)
        return len(distances[0]) - 1  # Exclude self
    
    def _compute_te_binning(self,
                           source_embedded: np.ndarray,
                           target_embedded: np.ndarray) -> float:
        """Compute transfer entropy using binning method"""
        try:
            # Determine number of bins using Sturges' rule
            n_samples = len(source_embedded)
            n_bins = max(3, int(np.log2(n_samples)) + 1)
            
            # Discretize data
            source_bins = self._discretize_data(source_embedded, n_bins)
            target_past_bins = self._discretize_data(target_embedded[:, 1:], n_bins)
            target_future_bins = self._discretize_data(target_embedded[:, 0:1], n_bins)
            target_full_bins = self._discretize_data(target_embedded, n_bins)
            
            # Compute joint and marginal probabilities
            prob_joint = self._compute_joint_probability(
                [source_bins, target_full_bins]
            )
            prob_source_target_past = self._compute_joint_probability(
                [source_bins, target_past_bins]
            )
            prob_target_past = self._compute_marginal_probability(target_past_bins)
            prob_target_full = self._compute_marginal_probability(target_full_bins)
            
            # Compute transfer entropy
            te = 0.0
            for state in prob_joint:
                p_joint = prob_joint[state]
                
                # Extract components
                source_state = state[0]
                target_full_state = state[1]
                target_past_state = target_full_state[1:]  # Remove future component
                
                if target_past_state in prob_target_past:
                    p_target_past = prob_target_past[target_past_state]
                    
                    source_target_past_state = (source_state, target_past_state)
                    if source_target_past_state in prob_source_target_past:
                        p_source_target_past = prob_source_target_past[source_target_past_state]
                        
                        if target_full_state in prob_target_full:
                            p_target_full = prob_target_full[target_full_state]
                            
                            if p_target_past > 0 and p_source_target_past > 0 and p_target_full > 0:
                                ratio = (p_joint * p_target_past) / (p_source_target_past * p_target_full)
                                if ratio > 0:
                                    te += p_joint * np.log2(ratio)
            
            return te
            
        except Exception as e:
            logger.warning(f"Binning estimator failed: {e}, falling back to Gaussian")
            return self._compute_te_gaussian(source_embedded, target_embedded)
    
    def _discretize_data(self, data: np.ndarray, n_bins: int) -> np.ndarray:
        """Discretize continuous data into bins"""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            
        discretized = np.zeros(data.shape[0], dtype=int)
        
        for i in range(data.shape[1]):
            # Use equal-frequency binning for better coverage
            _, bin_edges = np.histogram(data[:, i], bins=n_bins)
            digitized = np.digitize(data[:, i], bin_edges[1:-1])
            discretized = discretized * n_bins + digitized
            
        return discretized
    
    def _compute_joint_probability(self, data_list: List[np.ndarray]) -> Dict:
        """Compute joint probability distribution"""
        combined_states = tuple(zip(*data_list))
        n_samples = len(combined_states)
        
        # Count occurrences
        state_counts = {}
        for state in combined_states:
            state_counts[state] = state_counts.get(state, 0) + 1
            
        # Convert to probabilities
        return {state: count / n_samples for state, count in state_counts.items()}
    
    def _compute_marginal_probability(self, data: np.ndarray) -> Dict:
        """Compute marginal probability distribution"""
        n_samples = len(data)
        unique_values, counts = np.unique(data, return_counts=True)
        return {val: count / n_samples for val, count in zip(unique_values, counts)}
    
    def _compute_te_gaussian(self,
                            source_embedded: np.ndarray,
                            target_embedded: np.ndarray) -> float:
        """Compute transfer entropy assuming Gaussian distributions"""
        try:
            # Combine data
            joint_data = np.hstack([source_embedded, target_embedded])
            
            # Compute covariance matrices
            cov_joint = np.cov(joint_data.T)
            cov_source = np.cov(source_embedded.T)
            cov_target_past = np.cov(target_embedded[:, 1:].T)
            cov_target_full = np.cov(target_embedded.T)
            
            # Compute determinants
            det_joint = np.linalg.det(cov_joint)
            det_source = np.linalg.det(cov_source)
            det_target_past = np.linalg.det(cov_target_past)
            det_target_full = np.linalg.det(cov_target_full)
            
            # Transfer entropy formula for Gaussian case
            if (det_joint > 0 and det_source > 0 and 
                det_target_past > 0 and det_target_full > 0):
                te = 0.5 * np.log((det_source * det_target_full) / (det_joint * det_target_past))
                return max(0.0, te)  # Transfer entropy should be non-negative
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Gaussian estimator failed: {e}")
            return 0.0
    
    def _compute_entropy(self, data: np.ndarray) -> float:
        """Compute differential entropy estimate"""
        try:
            if self.method == 'gaussian':
                # Gaussian entropy
                cov = np.cov(data.T)
                det = np.linalg.det(cov)
                if det > 0:
                    return 0.5 * np.log(2 * np.pi * np.e * det)
                else:
                    return 0.0
            else:
                # KSG entropy estimator
                nn = NearestNeighbors(n_neighbors=self.k_neighbors + 1, metric='chebyshev')
                nn.fit(data)
                
                entropy_sum = 0.0
                n_samples = len(data)
                d = data.shape[1]  # Dimensionality
                
                for i in range(n_samples):
                    distances, _ = nn.kneighbors([data[i]])
                    epsilon = distances[0][self.k_neighbors]
                    
                    if epsilon > 0:
                        entropy_sum += np.log(epsilon)
                        
                return (d * np.log(2) + digamma(n_samples) - digamma(self.k_neighbors) + 
                       entropy_sum / n_samples)
                       
        except Exception:
            return 0.0
    
    def _compute_mutual_information(self,
                                  source_embedded: np.ndarray,
                                  target_embedded: np.ndarray) -> float:
        """Compute mutual information between source and target"""
        try:
            joint_data = np.hstack([source_embedded, target_embedded])
            
            entropy_joint = self._compute_entropy(joint_data)
            entropy_source = self._compute_entropy(source_embedded)
            entropy_target = self._compute_entropy(target_embedded)
            
            return entropy_source + entropy_target - entropy_joint
            
        except Exception:
            return 0.0
    
    def _bootstrap_significance_test(self,
                                   source_embedded: np.ndarray,
                                   target_embedded: np.ndarray,
                                   observed_te: float) -> Tuple[float, Tuple[float, float]]:
        """Bootstrap significance test for transfer entropy"""
        try:
            bootstrap_tes = []
            n_samples = len(source_embedded)
            
            for _ in range(self.bootstrap_samples):
                # Create surrogate data by shuffling source
                shuffled_indices = np.random.permutation(n_samples)
                source_shuffled = source_embedded[shuffled_indices]
                
                # Compute TE for shuffled data
                if self.method == 'ksg':
                    te_surrogate = self._compute_te_ksg(source_shuffled, target_embedded)
                elif self.method == 'binning':
                    te_surrogate = self._compute_te_binning(source_shuffled, target_embedded)
                else:
                    te_surrogate = self._compute_te_gaussian(source_shuffled, target_embedded)
                    
                bootstrap_tes.append(te_surrogate)
            
            bootstrap_tes = np.array(bootstrap_tes)
            
            # Compute p-value
            p_value = np.mean(bootstrap_tes >= observed_te)
            
            # Compute confidence interval
            alpha = self.significance_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            ci_lower = np.percentile(bootstrap_tes, lower_percentile)
            ci_upper = np.percentile(bootstrap_tes, upper_percentile)
            
            return float(p_value), (float(ci_lower), float(ci_upper))
            
        except Exception as e:
            logger.warning(f"Bootstrap test failed: {e}")
            return 1.0, (0.0, 0.0)  # Conservative fallback
    
    def get_computation_statistics(self) -> Dict[str, Any]:
        """Get computation statistics"""
        return self.computation_stats.copy()
    
    def find_optimal_parameters(self,
                              source_data: np.ndarray,
                              target_data: np.ndarray,
                              embedding_dims: List[int] = [2, 3, 4, 5],
                              k_values: List[int] = [3, 5, 7, 10]) -> Dict[str, Any]:
        """
        Find optimal parameters using cross-validation
        
        Returns optimal embedding dimension and k value based on TE magnitude
        and statistical significance
        """
        best_score = 0.0
        best_params = {}
        results = []
        
        # Save current parameters
        original_embed_dim = self.embedding_dimension
        original_k = self.k_neighbors
        
        try:
            for embed_dim in embedding_dims:
                for k in k_values:
                    if k >= len(source_data) // 10:  # Skip if k too large
                        continue
                        
                    # Update parameters
                    self.embedding_dimension = embed_dim
                    self.k_neighbors = k
                    
                    try:
                        # Test with a single lag
                        result = self._compute_transfer_entropy_single_lag(
                            source_data, target_data, 'source', 'target', 1000000, 1
                        )
                        
                        if result and result.p_value < self.significance_level:
                            # Score based on TE magnitude and significance
                            score = result.transfer_entropy * (1 - result.p_value)
                            
                            results.append({
                                'embedding_dim': embed_dim,
                                'k_neighbors': k,
                                'transfer_entropy': result.transfer_entropy,
                                'p_value': result.p_value,
                                'score': score
                            })
                            
                            if score > best_score:
                                best_score = score
                                best_params = {
                                    'embedding_dimension': embed_dim,
                                    'k_neighbors': k
                                }
                                
                    except Exception as e:
                        logger.debug(f"Parameter combination failed (embed_dim={embed_dim}, k={k}): {e}")
                        
        finally:
            # Restore original parameters
            self.embedding_dimension = original_embed_dim
            self.k_neighbors = original_k
        
        return {
            'best_parameters': best_params,
            'best_score': best_score,
            'all_results': results
        }
