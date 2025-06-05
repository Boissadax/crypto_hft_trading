"""
Causality Testing Module

Implements various causality tests for lead-lag analysis:
- Granger causality tests
- VAR-based causality tests
- Conditional independence tests
- Nonlinear causality tests
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from scipy import stats
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mutual_info_score
import warnings
warnings.filterwarnings('ignore')

@dataclass
class CausalityResult:
    """Results from causality tests."""
    test_type: str
    direction: str  # 'X->Y', 'Y->X', or 'bidirectional'
    statistic: float
    p_value: float
    is_significant: bool
    confidence_level: float
    lag_order: int
    additional_info: Dict[str, Any]

class CausalityTester:
    """
    Comprehensive causality testing framework.
    
    Implements multiple causality tests to determine lead-lag relationships
    between time series with proper statistical validation.
    """
    
    def __init__(self, 
                 max_lags: int = 10,
                 significance_level: float = 0.05,
                 ic_criterion: str = 'aic'):
        """
        Initialize causality tester.
        
        Args:
            max_lags: Maximum number of lags to test
            significance_level: Statistical significance threshold
            ic_criterion: Information criterion for lag selection ('aic', 'bic', 'hqic')
        """
        self.max_lags = max_lags
        self.significance_level = significance_level
        self.ic_criterion = ic_criterion
        
    def granger_causality_test(self, 
                              x: np.ndarray, 
                              y: np.ndarray,
                              max_lags: Optional[int] = None) -> Dict[str, CausalityResult]:
        """
        Perform Granger causality tests in both directions.
        
        Args:
            x: First time series
            y: Second time series
            max_lags: Maximum lags to test (overrides instance default)
            
        Returns:
            Dictionary with causality results for both directions
        """
        if max_lags is None:
            max_lags = self.max_lags
            
        # Prepare data
        data = pd.DataFrame({'x': x, 'y': y}).dropna()
        
        results = {}
        
        # Test X -> Y causality
        try:
            gc_test = grangercausalitytests(data[['y', 'x']], max_lags, verbose=False)
            
            # Extract best result based on information criterion
            best_lag = self._select_optimal_lag(gc_test)
            test_result = gc_test[best_lag][0]
            
            # Use F-test statistic
            f_stat = test_result['ssr_ftest'][0]
            p_value = test_result['ssr_ftest'][1]
            
            results['X->Y'] = CausalityResult(
                test_type='Granger Causality',
                direction='X->Y',
                statistic=f_stat,
                p_value=p_value,
                is_significant=p_value < self.significance_level,
                confidence_level=1 - self.significance_level,
                lag_order=best_lag,
                additional_info={
                    'ssr_ftest': test_result['ssr_ftest'],
                    'ssr_chi2test': test_result['ssr_chi2test'],
                    'lrtest': test_result['lrtest'],
                    'params_ftest': test_result['params_ftest']
                }
            )
        except Exception as e:
            results['X->Y'] = self._create_error_result('Granger Causality', 'X->Y', str(e))
        
        # Test Y -> X causality
        try:
            gc_test = grangercausalitytests(data[['x', 'y']], max_lags, verbose=False)
            
            best_lag = self._select_optimal_lag(gc_test)
            test_result = gc_test[best_lag][0]
            
            f_stat = test_result['ssr_ftest'][0]
            p_value = test_result['ssr_ftest'][1]
            
            results['Y->X'] = CausalityResult(
                test_type='Granger Causality',
                direction='Y->X',
                statistic=f_stat,
                p_value=p_value,
                is_significant=p_value < self.significance_level,
                confidence_level=1 - self.significance_level,
                lag_order=best_lag,
                additional_info={
                    'ssr_ftest': test_result['ssr_ftest'],
                    'ssr_chi2test': test_result['ssr_chi2test'],
                    'lrtest': test_result['lrtest'],
                    'params_ftest': test_result['params_ftest']
                }
            )
        except Exception as e:
            results['Y->X'] = self._create_error_result('Granger Causality', 'Y->X', str(e))
            
        return results
    
    def var_causality_test(self,
                          x: np.ndarray,
                          y: np.ndarray,
                          lag_order: Optional[int] = None) -> Dict[str, CausalityResult]:
        """
        Perform VAR-based causality tests.
        
        Args:
            x: First time series
            y: Second time series
            lag_order: VAR lag order (auto-selected if None)
            
        Returns:
            Dictionary with VAR causality results
        """
        # Prepare data
        data = np.column_stack([x, y])
        data = pd.DataFrame(data, columns=['x', 'y']).dropna()
        
        try:
            # Fit VAR model
            var_model = VAR(data)
            
            if lag_order is None:
                # Select optimal lag order
                lag_order = var_model.select_order(maxlags=self.max_lags).selected_orders[self.ic_criterion]
            
            var_fit = var_model.fit(lag_order)
            
            results = {}
            
            # Test X -> Y causality
            causality_test_xy = var_fit.test_causality('y', 'x', kind='f')
            results['X->Y'] = CausalityResult(
                test_type='VAR Causality',
                direction='X->Y',
                statistic=causality_test_xy.statistic,
                p_value=causality_test_xy.pvalue,
                is_significant=causality_test_xy.pvalue < self.significance_level,
                confidence_level=1 - self.significance_level,
                lag_order=lag_order,
                additional_info={
                    'critical_value': causality_test_xy.critical_value,
                    'df': causality_test_xy.df,
                    'summary': str(causality_test_xy)
                }
            )
            
            # Test Y -> X causality
            causality_test_yx = var_fit.test_causality('x', 'y', kind='f')
            results['Y->X'] = CausalityResult(
                test_type='VAR Causality',
                direction='Y->X',
                statistic=causality_test_yx.statistic,
                p_value=causality_test_yx.pvalue,
                is_significant=causality_test_yx.pvalue < self.significance_level,
                confidence_level=1 - self.significance_level,
                lag_order=lag_order,
                additional_info={
                    'critical_value': causality_test_yx.critical_value,
                    'df': causality_test_yx.df,
                    'summary': str(causality_test_yx)
                }
            )
            
        except Exception as e:
            results = {
                'X->Y': self._create_error_result('VAR Causality', 'X->Y', str(e)),
                'Y->X': self._create_error_result('VAR Causality', 'Y->X', str(e))
            }
            
        return results
    
    def nonlinear_causality_test(self,
                                x: np.ndarray,
                                y: np.ndarray,
                                embedding_dim: int = 3,
                                n_neighbors: int = 5) -> Dict[str, CausalityResult]:
        """
        Perform nonlinear causality tests using mutual information.
        
        Args:
            x: First time series
            y: Second time series
            embedding_dim: Embedding dimension for phase space reconstruction
            n_neighbors: Number of neighbors for MI estimation
            
        Returns:
            Dictionary with nonlinear causality results
        """
        results = {}
        
        try:
            # Create embedded vectors
            x_embedded = self._embed_time_series(x, embedding_dim)
            y_embedded = self._embed_time_series(y, embedding_dim)
            
            # Test X -> Y causality
            mi_xy = self._conditional_mutual_information(
                x_embedded, y[embedding_dim:], y_embedded[:-1]
            )
            
            # Bootstrap for significance testing
            p_value_xy = self._bootstrap_mi_test(
                x_embedded, y[embedding_dim:], y_embedded[:-1], mi_xy
            )
            
            results['X->Y'] = CausalityResult(
                test_type='Nonlinear Causality (MI)',
                direction='X->Y',
                statistic=mi_xy,
                p_value=p_value_xy,
                is_significant=p_value_xy < self.significance_level,
                confidence_level=1 - self.significance_level,
                lag_order=embedding_dim,
                additional_info={
                    'embedding_dim': embedding_dim,
                    'n_neighbors': n_neighbors,
                    'method': 'conditional_mutual_information'
                }
            )
            
            # Test Y -> X causality
            mi_yx = self._conditional_mutual_information(
                y_embedded, x[embedding_dim:], x_embedded[:-1]
            )
            
            p_value_yx = self._bootstrap_mi_test(
                y_embedded, x[embedding_dim:], x_embedded[:-1], mi_yx
            )
            
            results['Y->X'] = CausalityResult(
                test_type='Nonlinear Causality (MI)',
                direction='Y->X',
                statistic=mi_yx,
                p_value=p_value_yx,
                is_significant=p_value_yx < self.significance_level,
                confidence_level=1 - self.significance_level,
                lag_order=embedding_dim,
                additional_info={
                    'embedding_dim': embedding_dim,
                    'n_neighbors': n_neighbors,
                    'method': 'conditional_mutual_information'
                }
            )
            
        except Exception as e:
            results = {
                'X->Y': self._create_error_result('Nonlinear Causality', 'X->Y', str(e)),
                'Y->X': self._create_error_result('Nonlinear Causality', 'Y->X', str(e))
            }
            
        return results
    
    def comprehensive_causality_analysis(self,
                                       x: np.ndarray,
                                       y: np.ndarray,
                                       x_name: str = 'X',
                                       y_name: str = 'Y') -> Dict[str, Any]:
        """
        Perform comprehensive causality analysis using multiple methods.
        
        Args:
            x: First time series
            y: Second time series
            x_name: Name for first series
            y_name: Name for second series
            
        Returns:
            Comprehensive causality analysis results
        """
        results = {
            'series_names': (x_name, y_name),
            'data_info': {
                'x_length': len(x),
                'y_length': len(y),
                'x_mean': np.mean(x),
                'y_mean': np.mean(y),
                'x_std': np.std(x),
                'y_std': np.std(y),
                'correlation': np.corrcoef(x, y)[0, 1]
            },
            'tests': {}
        }
        
        # Granger causality test
        results['tests']['granger'] = self.granger_causality_test(x, y)
        
        # VAR causality test
        results['tests']['var'] = self.var_causality_test(x, y)
        
        # Nonlinear causality test
        results['tests']['nonlinear'] = self.nonlinear_causality_test(x, y)
        
        # Summary of significant results
        results['summary'] = self._summarize_causality_results(results['tests'])
        
        return results
    
    def _select_optimal_lag(self, gc_results: Dict) -> int:
        """Select optimal lag based on information criterion."""
        lags = list(gc_results.keys())
        
        if self.ic_criterion == 'aic':
            # Use the lag with minimum AIC (if available)
            # For now, use lag 1 as default
            return min(lags)
        else:
            return min(lags)
    
    def _embed_time_series(self, x: np.ndarray, dim: int) -> np.ndarray:
        """Create time-delayed embedding of time series."""
        n = len(x)
        embedded = np.zeros((n - dim + 1, dim))
        
        for i in range(dim):
            embedded[:, i] = x[i:n - dim + 1 + i]
            
        return embedded
    
    def _conditional_mutual_information(self,
                                      x: np.ndarray,
                                      y: np.ndarray,
                                      z: np.ndarray) -> float:
        """Estimate conditional mutual information I(X;Y|Z)."""
        # Discretize variables for MI estimation
        n_bins = max(10, int(np.sqrt(len(x))))
        
        x_discrete = pd.cut(x.flatten(), bins=n_bins, labels=False)
        y_discrete = pd.cut(y.flatten(), bins=n_bins, labels=False)
        z_discrete = pd.cut(z.flatten(), bins=n_bins, labels=False)
        
        # Handle NaN values
        mask = ~(np.isnan(x_discrete) | np.isnan(y_discrete) | np.isnan(z_discrete))
        x_discrete = x_discrete[mask]
        y_discrete = y_discrete[mask]
        z_discrete = z_discrete[mask]
        
        if len(x_discrete) == 0:
            return 0.0
        
        # Calculate conditional MI: I(X;Y|Z) = I(X,Z;Y) - I(Z;Y)
        try:
            mi_xyz = mutual_info_score(
                np.column_stack([x_discrete, z_discrete]).astype(str),
                y_discrete.astype(str)
            )
            mi_zy = mutual_info_score(z_discrete.astype(str), y_discrete.astype(str))
            return max(0, mi_xyz - mi_zy)
        except:
            return 0.0
    
    def _bootstrap_mi_test(self,
                          x: np.ndarray,
                          y: np.ndarray,
                          z: np.ndarray,
                          observed_mi: float,
                          n_bootstrap: int = 100) -> float:
        """Bootstrap test for mutual information significance."""
        null_mi_values = []
        
        for _ in range(n_bootstrap):
            # Permute x to break causality
            x_perm = np.random.permutation(x)
            null_mi = self._conditional_mutual_information(x_perm, y, z)
            null_mi_values.append(null_mi)
        
        # Calculate p-value
        null_mi_values = np.array(null_mi_values)
        p_value = np.mean(null_mi_values >= observed_mi)
        
        return p_value
    
    def _create_error_result(self, test_type: str, direction: str, error_msg: str) -> CausalityResult:
        """Create error result for failed tests."""
        return CausalityResult(
            test_type=test_type,
            direction=direction,
            statistic=np.nan,
            p_value=1.0,
            is_significant=False,
            confidence_level=1 - self.significance_level,
            lag_order=0,
            additional_info={'error': error_msg}
        )
    
    def _summarize_causality_results(self, test_results: Dict) -> Dict[str, Any]:
        """Summarize causality test results across all methods."""
        summary = {
            'significant_relationships': [],
            'consensus': None,
            'strength': {},
            'recommendations': []
        }
        
        # Collect significant results
        for test_name, directions in test_results.items():
            for direction, result in directions.items():
                if result.is_significant:
                    summary['significant_relationships'].append({
                        'test': test_name,
                        'direction': direction,
                        'p_value': result.p_value,
                        'statistic': result.statistic
                    })
        
        # Determine consensus
        xy_votes = sum(1 for test_name, directions in test_results.items() 
                      for direction, result in directions.items()
                      if direction == 'X->Y' and result.is_significant)
        
        yx_votes = sum(1 for test_name, directions in test_results.items()
                      for direction, result in directions.items() 
                      if direction == 'Y->X' and result.is_significant)
        
        if xy_votes > yx_votes:
            summary['consensus'] = 'X->Y'
        elif yx_votes > xy_votes:
            summary['consensus'] = 'Y->X'
        elif xy_votes == yx_votes and xy_votes > 0:
            summary['consensus'] = 'bidirectional'
        else:
            summary['consensus'] = 'no_causality'
        
        # Calculate strength scores
        for direction in ['X->Y', 'Y->X']:
            p_values = [result.p_value for test_name, directions in test_results.items()
                       for dir_name, result in directions.items()
                       if dir_name == direction and not np.isnan(result.p_value)]
            
            if p_values:
                # Fisher's method for combining p-values
                chi2_stat = -2 * np.sum(np.log(p_values))
                combined_p = 1 - stats.chi2.cdf(chi2_stat, 2 * len(p_values))
                summary['strength'][direction] = 1 - combined_p
            else:
                summary['strength'][direction] = 0.0
        
        return summary
