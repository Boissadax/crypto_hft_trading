"""
Data Preparation Module

Handles data preparation for machine learning with proper temporal considerations:
- Temporal train/test splits to avoid look-ahead bias
- Feature matrix construction from synchronized data
- Target variable creation for lead-lag prediction
- Data preprocessing and normalization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

@dataclass
class DataSplit:
    """Container for train/test data splits."""
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    train_timestamps: np.ndarray
    test_timestamps: np.ndarray
    feature_names: List[str]
    scaler: Optional[Any] = None
    
@dataclass
class TargetConfig:
    """Configuration for target variable creation."""
    prediction_horizon_us: int = 1_000_000  # 1 second
    target_type: str = 'direction'  # 'direction', 'magnitude', 'probability'
    threshold: float = 0.0001  # Minimum price change threshold
    use_returns: bool = True
    smoothing_window: int = 1  # No smoothing by default

class TemporalSplit:
    """
    Temporal data splitting that respects time ordering.
    
    Ensures no future information leaks into training data.
    """
    
    def __init__(self,
                 train_ratio: float = 0.7,
                 validation_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 gap_ratio: float = 0.01):
        """
        Initialize temporal splitter.
        
        Args:
            train_ratio: Proportion of data for training
            validation_ratio: Proportion for validation
            test_ratio: Proportion for testing
            gap_ratio: Gap between train/test to avoid leakage
        """
        if abs(train_ratio + validation_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
            
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.gap_ratio = gap_ratio
        
    def split(self, timestamps: np.ndarray, 
              features: np.ndarray,
              targets: np.ndarray) -> Dict[str, DataSplit]:
        """
        Split data temporally.
        
        Args:
            timestamps: Array of timestamps
            features: Feature matrix
            targets: Target vector
            
        Returns:
            Dictionary with train/val/test splits
        """
        n_samples = len(timestamps)
        
        # Calculate split indices
        train_end = int(n_samples * self.train_ratio)
        gap_size = int(n_samples * self.gap_ratio)
        val_start = train_end + gap_size
        val_end = val_start + int(n_samples * self.validation_ratio)
        test_start = val_end + gap_size
        
        # Ensure we don't exceed array bounds
        val_end = min(val_end, n_samples)
        test_start = min(test_start, n_samples)
        
        splits = {}
        
        # Training split
        splits['train'] = DataSplit(
            X_train=features[:train_end],
            X_test=None,
            y_train=targets[:train_end],
            y_test=None,
            train_timestamps=timestamps[:train_end],
            test_timestamps=None,
            feature_names=[]
        )
        
        # Validation split
        if val_start < val_end:
            splits['validation'] = DataSplit(
                X_train=features[:train_end],  # Training data for validation
                X_test=features[val_start:val_end],
                y_train=targets[:train_end],
                y_test=targets[val_start:val_end],
                train_timestamps=timestamps[:train_end],
                test_timestamps=timestamps[val_start:val_end],
                feature_names=[]
            )
        
        # Test split
        if test_start < n_samples:
            splits['test'] = DataSplit(
                X_train=features[:train_end],  # Training data for test
                X_test=features[test_start:],
                y_train=targets[:train_end],
                y_test=targets[test_start:],
                train_timestamps=timestamps[:train_end],
                test_timestamps=timestamps[test_start:],
                feature_names=[]
            )
        
        return splits

class DataPreparator:
    """
    Comprehensive data preparation for lead-lag learning.
    
    Handles feature engineering, target creation, and preprocessing
    with proper temporal considerations.
    """
    
    def __init__(self,
                 target_config: TargetConfig = None,
                 scaler_type: str = 'robust',
                 feature_selection: bool = True,
                 max_features: int = 50):
        """
        Initialize data preparator.
        
        Args:
            target_config: Target variable configuration
            scaler_type: Type of scaler ('standard', 'robust', 'minmax')
            feature_selection: Whether to perform feature selection
            max_features: Maximum number of features to select
        """
        self.target_config = target_config or TargetConfig()
        self.scaler_type = scaler_type
        self.feature_selection = feature_selection
        self.max_features = max_features
        
        # Initialize scaler
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
            
        self.feature_selector = None
        self.selected_features = None
        
    def prepare_features_and_targets(self,
                                   synchronized_data: List,
                                   primary_symbol: str,
                                   secondary_symbol: str,
                                   causality_results: Dict[str, Any] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Prepare feature matrix and target variables from synchronized data.
        
        Args:
            synchronized_data: List of SynchronizedDataPoint objects
            primary_symbol: Primary symbol for analysis
            secondary_symbol: Secondary symbol for analysis
            causality_results: Results from causality analysis
            
        Returns:
            Tuple of (features, targets, timestamps, feature_names)
        """
        if not synchronized_data:
            return np.array([]), np.array([]), np.array([]), []
        
        # Extract features and timestamps
        features_list = []
        timestamps = []
        
        for point in synchronized_data:
            # Combine features from both symbols
            combined_features = {}
            
            # Primary symbol features
            if primary_symbol in point.symbols:
                for k, v in point.symbols[primary_symbol].items():
                    combined_features[f'{primary_symbol}_{k}'] = v
            
            # Secondary symbol features  
            if secondary_symbol in point.symbols:
                for k, v in point.symbols[secondary_symbol].items():
                    combined_features[f'{secondary_symbol}_{k}'] = v
            
            # Cross-symbol features (already computed in synchronization)
            # Add causality-based features if available
            if causality_results:
                combined_features.update(self._create_causality_features(causality_results))
            
            features_list.append(combined_features)
            timestamps.append(point.timestamp_us)
        
        # Convert to DataFrame for easier manipulation
        features_df = pd.DataFrame(features_list)
        feature_names = list(features_df.columns)
        
        # Handle missing values
        features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Create targets
        targets = self._create_targets(
            features_df, timestamps, primary_symbol, secondary_symbol
        )
        
        # Convert to numpy arrays
        features_array = features_df.values
        timestamps_array = np.array(timestamps)
        
        # Remove samples where targets are NaN
        valid_mask = ~np.isnan(targets)
        features_array = features_array[valid_mask]
        targets = targets[valid_mask]
        timestamps_array = timestamps_array[valid_mask]
        
        return features_array, targets, timestamps_array, feature_names
    
    def preprocess_data(self,
                       X_train: np.ndarray,
                       X_test: np.ndarray,
                       y_train: np.ndarray,
                       feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Preprocess features with scaling and selection.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training targets (for feature selection)
            feature_names: List of feature names
            
        Returns:
            Tuple of (X_train_processed, X_test_processed, selected_feature_names)
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Feature selection
        if self.feature_selection and len(feature_names) > self.max_features:
            # Use mutual information for feature selection
            n_features = min(self.max_features, X_train_scaled.shape[1])
            
            self.feature_selector = SelectKBest(
                score_func=mutual_info_classif if self._is_classification_target(y_train) else f_classif,
                k=n_features
            )
            
            X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
            X_test_selected = self.feature_selector.transform(X_test_scaled)
            
            # Get selected feature names
            selected_indices = self.feature_selector.get_support(indices=True)
            selected_feature_names = [feature_names[i] for i in selected_indices]
            
        else:
            X_train_selected = X_train_scaled
            X_test_selected = X_test_scaled
            selected_feature_names = feature_names
        
        self.selected_features = selected_feature_names
        
        return X_train_selected, X_test_selected, selected_feature_names
    
    def _create_targets(self,
                       features_df: pd.DataFrame,
                       timestamps: List[int],
                       primary_symbol: str,
                       secondary_symbol: str) -> np.ndarray:
        """Create target variables based on configuration."""
        
        # Get price columns
        primary_price_col = f'{primary_symbol}_mid_price'
        secondary_price_col = f'{secondary_symbol}_mid_price'
        
        if primary_price_col not in features_df.columns or secondary_price_col not in features_df.columns:
            # Fallback to any price-like column
            price_cols = [col for col in features_df.columns if 'price' in col.lower()]
            if len(price_cols) >= 2:
                primary_price_col = price_cols[0]
                secondary_price_col = price_cols[1]
            else:
                raise ValueError("No suitable price columns found for target creation")
        
        primary_prices = features_df[primary_price_col].values
        secondary_prices = features_df[secondary_price_col].values
        
        # Calculate future returns based on prediction horizon
        horizon_samples = max(1, self.target_config.prediction_horizon_us // 1_000_000)  # Assuming 1s sampling
        
        if self.target_config.use_returns:
            # Use returns instead of raw prices
            primary_returns = np.diff(np.log(primary_prices + 1e-10))
            secondary_returns = np.diff(np.log(secondary_prices + 1e-10))
            
            # Pad with NaN to maintain length
            primary_returns = np.concatenate([[np.nan], primary_returns])
            secondary_returns = np.concatenate([[np.nan], secondary_returns])
        else:
            primary_returns = primary_prices
            secondary_returns = secondary_prices
        
        # Create future targets
        if self.target_config.target_type == 'direction':
            # Predict direction of price movement
            future_primary = np.roll(primary_returns, -horizon_samples)
            current_primary = primary_returns
            
            price_change = future_primary - current_primary
            targets = np.where(price_change > self.target_config.threshold, 1, 0)
            
        elif self.target_config.target_type == 'magnitude':
            # Predict magnitude of price change
            future_primary = np.roll(primary_returns, -horizon_samples)
            current_primary = primary_returns
            
            targets = future_primary - current_primary
            
        elif self.target_config.target_type == 'probability':
            # Predict probability of significant move
            future_primary = np.roll(primary_returns, -horizon_samples)
            current_primary = primary_returns
            
            price_change = np.abs(future_primary - current_primary)
            targets = np.where(price_change > self.target_config.threshold, 1, 0)
        
        else:
            raise ValueError(f"Unknown target type: {self.target_config.target_type}")
        
        # Set last horizon_samples to NaN (no future data available)
        targets[-horizon_samples:] = np.nan
        
        # Apply smoothing if requested
        if self.target_config.smoothing_window > 1:
            targets = pd.Series(targets).rolling(
                window=self.target_config.smoothing_window, 
                center=True
            ).mean().values
        
        return targets
    
    def _create_causality_features(self, causality_results: Dict[str, Any]) -> Dict[str, float]:
        """Create features based on causality analysis results."""
        features = {}
        
        # Transfer entropy features
        if 'transfer_entropy' in causality_results:
            te_results = causality_results['transfer_entropy']
            features['te_xy_strength'] = te_results.get('xy_strength', 0.0)
            features['te_yx_strength'] = te_results.get('yx_strength', 0.0)
            features['te_net_flow'] = te_results.get('net_information_flow', 0.0)
            features['te_dominant_direction'] = float(te_results.get('dominant_direction', 0))
        
        # Granger causality features
        if 'granger' in causality_results:
            granger_results = causality_results['granger']
            for direction, result in granger_results.items():
                features[f'granger_{direction.lower()}_significant'] = float(result.is_significant)
                features[f'granger_{direction.lower()}_pvalue'] = result.p_value
        
        # Cross-correlation features
        if 'correlation' in causality_results:
            corr_results = causality_results['correlation']
            features['optimal_lag'] = float(corr_results.get('optimal_lag', 0))
            features['max_correlation'] = corr_results.get('max_correlation', 0.0)
        
        return features
    
    def _is_classification_target(self, targets: np.ndarray) -> bool:
        """Check if targets represent a classification problem."""
        unique_values = np.unique(targets[~np.isnan(targets)])
        return len(unique_values) <= 10 and all(isinstance(v, (int, np.integer)) or v.is_integer() for v in unique_values)
    
    def create_temporal_splits(self,
                              features: np.ndarray,
                              targets: np.ndarray,
                              timestamps: np.ndarray,
                              feature_names: List[str]) -> Dict[str, DataSplit]:
        """
        Create temporal splits for training and testing.
        
        Args:
            features: Feature matrix
            targets: Target vector
            timestamps: Timestamp vector
            feature_names: List of feature names
            
        Returns:
            Dictionary with train/validation/test splits
        """
        splitter = TemporalSplit()
        splits = splitter.split(timestamps, features, targets)
        
        # Add feature names to all splits
        for split_name, split_data in splits.items():
            split_data.feature_names = feature_names
            
        return splits
    
    def get_feature_importance_scores(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores from feature selection."""
        if self.feature_selector is None or self.selected_features is None:
            return None
            
        scores = self.feature_selector.scores_
        selected_indices = self.feature_selector.get_support(indices=True)
        
        importance_dict = {}
        for i, idx in enumerate(selected_indices):
            feature_name = self.selected_features[i]
            importance_dict[feature_name] = scores[idx]
            
        return importance_dict
