"""
Model Training Module

Implements model training for lead-lag strategies:
- Transfer entropy-based models
- Traditional machine learning models
- Ensemble methods combining multiple approaches
- Hyperparameter optimization
- Model persistence and loading
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import joblib
from pathlib import Path

# ML models
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import warnings
warnings.filterwarnings('ignore')

@dataclass
class ModelConfig:
    """Configuration for model training."""
    model_type: str = 'random_forest'  # 'random_forest', 'gradient_boosting', 'logistic', 'svm', 'mlp'
    problem_type: str = 'classification'  # 'classification', 'regression'
    hyperparameter_tuning: bool = True
    cv_folds: int = 5
    random_state: int = 42
    n_jobs: int = -1
    
@dataclass
class ModelResults:
    """Results from model training."""
    model: Any
    train_score: float
    validation_score: float
    test_score: Optional[float]
    feature_importance: Optional[Dict[str, float]]
    predictions: Optional[np.ndarray]
    probabilities: Optional[np.ndarray]
    best_params: Optional[Dict[str, Any]]
    training_time: float
    metadata: Dict[str, Any]

class BaseModel(ABC):
    """Base class for all models."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> ModelResults:
        """Fit the model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Predict probabilities (for classification models)."""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        return None
    
    def save_model(self, filepath: str):
        """Save model to disk."""
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath: str):
        """Load model from disk."""
        self.model = joblib.load(filepath)
        self.is_fitted = True

class TransferEntropyModel(BaseModel):
    """
    Model that incorporates transfer entropy analysis for predictions.
    
    Uses transfer entropy features combined with traditional features
    to make lead-lag predictions.
    """
    
    def __init__(self, config: ModelConfig, te_weight: float = 0.3):
        super().__init__(config)
        self.te_weight = te_weight
        self.base_model = None
        self.te_features = None
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            te_features: Optional[Dict[str, float]] = None) -> ModelResults:
        """
        Fit transfer entropy model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            te_features: Transfer entropy features
        """
        import time
        start_time = time.time()
        
        self.te_features = te_features
        
        # Create base model
        self.base_model = self._create_base_model()
        
        # Augment features with transfer entropy information
        X_train_augmented = self._augment_features(X_train, te_features)
        X_val_augmented = self._augment_features(X_val, te_features) if X_val is not None else None
        
        # Train model
        if self.config.hyperparameter_tuning and X_val is not None:
            # Use validation set for hyperparameter tuning
            self.model = self._tune_hyperparameters(X_train_augmented, y_train, X_val_augmented, y_val)
        else:
            self.model = self.base_model
            self.model.fit(X_train_augmented, y_train)
        
        self.is_fitted = True
        
        # Calculate scores
        train_pred = self.model.predict(X_train_augmented)
        train_score = self._calculate_score(y_train, train_pred)
        
        val_score = None
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val_augmented)
            val_score = self._calculate_score(y_val, val_pred)
        
        # Feature importance
        feature_importance = self._get_feature_importance()
        
        training_time = time.time() - start_time
        
        return ModelResults(
            model=self.model,
            train_score=train_score,
            validation_score=val_score,
            test_score=None,
            feature_importance=feature_importance,
            predictions=None,
            probabilities=None,
            best_params=getattr(self.model, 'best_params_', None),
            training_time=training_time,
            metadata={
                'te_weight': self.te_weight,
                'te_features_used': te_features is not None,
                'model_type': 'transfer_entropy'
            }
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_augmented = self._augment_features(X, self.te_features)
        return self.model.predict(X_augmented)
    
    def _augment_features(self, X: np.ndarray, te_features: Optional[Dict[str, float]]) -> np.ndarray:
        """Augment features with transfer entropy information."""
        if te_features is None:
            return X
        
        # Create TE feature vector
        te_feature_vector = np.array(list(te_features.values()))
        
        # Broadcast TE features to match number of samples
        n_samples = X.shape[0]
        te_features_broadcasted = np.tile(te_feature_vector, (n_samples, 1))
        
        # Concatenate with original features
        X_augmented = np.concatenate([X, te_features_broadcasted], axis=1)
        
        return X_augmented
    
    def _create_base_model(self):
        """Create base model based on configuration."""
        if self.config.model_type == 'random_forest':
            if self.config.problem_type == 'classification':
                return RandomForestClassifier(
                    random_state=self.config.random_state,
                    n_jobs=self.config.n_jobs
                )
            else:
                return RandomForestRegressor(
                    random_state=self.config.random_state,
                    n_jobs=self.config.n_jobs
                )
        
        elif self.config.model_type == 'gradient_boosting':
            if self.config.problem_type == 'classification':
                return GradientBoostingClassifier(random_state=self.config.random_state)
            else:
                return GradientBoostingRegressor(random_state=self.config.random_state)
        
        elif self.config.model_type == 'logistic':
            return LogisticRegression(random_state=self.config.random_state)
        
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
    
    def _tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray):
        """Tune hyperparameters using validation set."""
        param_grids = self._get_param_grids()
        
        if self.config.model_type in param_grids:
            param_grid = param_grids[self.config.model_type]
            
            grid_search = GridSearchCV(
                self.base_model,
                param_grid,
                cv=self.config.cv_folds,
                n_jobs=self.config.n_jobs,
                scoring='accuracy' if self.config.problem_type == 'classification' else 'r2'
            )
            
            grid_search.fit(X_train, y_train)
            return grid_search.best_estimator_
        
        # Fallback to default model
        self.base_model.fit(X_train, y_train)
        return self.base_model
    
    def _get_param_grids(self) -> Dict[str, Dict]:
        """Get parameter grids for hyperparameter tuning."""
        return {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            'logistic': {
                'C': [0.01, 0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        }
    
    def _calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate appropriate score based on problem type."""
        if self.config.problem_type == 'classification':
            return accuracy_score(y_true, y_pred)
        else:
            return r2_score(y_true, y_pred)
    
    def _get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from the trained model."""
        if hasattr(self.model, 'feature_importances_'):
            # For tree-based models
            importances = self.model.feature_importances_
            
            # Create feature names (original + TE features)
            n_original_features = len(importances) - (len(self.te_features) if self.te_features else 0)
            feature_names = [f'feature_{i}' for i in range(n_original_features)]
            
            if self.te_features:
                feature_names.extend([f'te_{name}' for name in self.te_features.keys()])
            
            return dict(zip(feature_names, importances))
        
        elif hasattr(self.model, 'coef_'):
            # For linear models
            coefs = np.abs(self.model.coef_).flatten()
            n_original_features = len(coefs) - (len(self.te_features) if self.te_features else 0)
            feature_names = [f'feature_{i}' for i in range(n_original_features)]
            
            if self.te_features:
                feature_names.extend([f'te_{name}' for name in self.te_features.keys()])
            
            return dict(zip(feature_names, coefs))
        
        return None

class ModelTrainer:
    """
    Main model training orchestrator.
    
    Handles training of multiple models and ensemble creation.
    """
    
    def __init__(self, 
                 models_config: Dict[str, ModelConfig] = None,
                 enable_ensemble: bool = True):
        """
        Initialize model trainer.
        
        Args:
            models_config: Dictionary of model configurations
            enable_ensemble: Whether to create ensemble models
        """
        self.models_config = models_config or self._get_default_configs()
        self.enable_ensemble = enable_ensemble
        self.trained_models = {}
        self.ensemble_weights = {}
        
    def train_all_models(self,
                        X_train: np.ndarray,
                        y_train: np.ndarray,
                        X_val: np.ndarray,
                        y_val: np.ndarray,
                        feature_names: List[str],
                        te_features: Optional[Dict[str, float]] = None) -> Dict[str, ModelResults]:
        """
        Train all configured models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            feature_names: List of feature names
            te_features: Transfer entropy features
            
        Returns:
            Dictionary of training results
        """
        results = {}
        
        for model_name, config in self.models_config.items():
            print(f"Training {model_name}...")
            
            if model_name == 'transfer_entropy':
                model = TransferEntropyModel(config)
                result = model.fit(X_train, y_train, X_val, y_val, te_features)
            else:
                model = self._create_standard_model(config)
                result = model.fit(X_train, y_train, X_val, y_val)
            
            self.trained_models[model_name] = model
            results[model_name] = result
            
            print(f"{model_name} - Train Score: {result.train_score:.4f}, "
                  f"Val Score: {result.validation_score:.4f}")
        
        # Create ensemble if enabled
        if self.enable_ensemble and len(results) > 1:
            ensemble_result = self._create_ensemble(X_val, y_val, results)
            results['ensemble'] = ensemble_result
        
        return results
    
    def predict_with_models(self,
                           X: np.ndarray,
                           model_names: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Make predictions with trained models.
        
        Args:
            X: Features for prediction
            model_names: List of model names to use (all if None)
            
        Returns:
            Dictionary of predictions
        """
        if model_names is None:
            model_names = list(self.trained_models.keys())
        
        predictions = {}
        
        for model_name in model_names:
            if model_name in self.trained_models:
                model = self.trained_models[model_name]
                predictions[model_name] = model.predict(X)
        
        # Ensemble prediction
        if 'ensemble' in model_names and len(predictions) > 1:
            ensemble_pred = self._ensemble_predict(X, predictions)
            predictions['ensemble'] = ensemble_pred
        
        return predictions
    
    def _get_default_configs(self) -> Dict[str, ModelConfig]:
        """Get default model configurations."""
        return {
            'random_forest': ModelConfig(
                model_type='random_forest',
                problem_type='classification',
                hyperparameter_tuning=True
            ),
            'gradient_boosting': ModelConfig(
                model_type='gradient_boosting', 
                problem_type='classification',
                hyperparameter_tuning=True
            ),
            'logistic_regression': ModelConfig(
                model_type='logistic',
                problem_type='classification',
                hyperparameter_tuning=True
            ),
            'transfer_entropy': ModelConfig(
                model_type='random_forest',
                problem_type='classification',
                hyperparameter_tuning=True
            )
        }
    
    def _create_standard_model(self, config: ModelConfig) -> BaseModel:
        """Create a standard model instance."""
        
        class StandardModel(BaseModel):
            def fit(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> ModelResults:
                import time
                start_time = time.time()
                
                # Create base model
                if config.model_type == 'random_forest':
                    if config.problem_type == 'classification':
                        self.model = RandomForestClassifier(
                            random_state=config.random_state,
                            n_jobs=config.n_jobs
                        )
                    else:
                        self.model = RandomForestRegressor(
                            random_state=config.random_state,
                            n_jobs=config.n_jobs
                        )
                
                elif config.model_type == 'gradient_boosting':
                    if config.problem_type == 'classification':
                        self.model = GradientBoostingClassifier(random_state=config.random_state)
                    else:
                        self.model = GradientBoostingRegressor(random_state=config.random_state)
                
                elif config.model_type == 'logistic':
                    self.model = LogisticRegression(random_state=config.random_state)
                
                # Train model
                self.model.fit(X_train, y_train)
                self.is_fitted = True
                
                # Calculate scores
                train_pred = self.model.predict(X_train)
                if config.problem_type == 'classification':
                    train_score = accuracy_score(y_train, train_pred)
                    val_score = accuracy_score(y_val, self.model.predict(X_val)) if X_val is not None else None
                else:
                    train_score = r2_score(y_train, train_pred)
                    val_score = r2_score(y_val, self.model.predict(X_val)) if X_val is not None else None
                
                # Feature importance
                feature_importance = None
                if hasattr(self.model, 'feature_importances_'):
                    feature_importance = dict(enumerate(self.model.feature_importances_))
                
                training_time = time.time() - start_time
                
                return ModelResults(
                    model=self.model,
                    train_score=train_score,
                    validation_score=val_score,
                    test_score=None,
                    feature_importance=feature_importance,
                    predictions=None,
                    probabilities=None,
                    best_params=None,
                    training_time=training_time,
                    metadata={'model_type': config.model_type}
                )
            
            def predict(self, X: np.ndarray) -> np.ndarray:
                if not self.is_fitted:
                    raise ValueError("Model must be fitted before making predictions")
                return self.model.predict(X)
        
        return StandardModel(config)
    
    def _create_ensemble(self,
                        X_val: np.ndarray,
                        y_val: np.ndarray,
                        results: Dict[str, ModelResults]) -> ModelResults:
        """Create ensemble model from individual models."""
        # Calculate ensemble weights based on validation performance
        val_scores = {}
        for name, result in results.items():
            if result.validation_score is not None:
                val_scores[name] = result.validation_score
        
        if not val_scores:
            return None
        
        # Normalize weights
        total_score = sum(val_scores.values())
        self.ensemble_weights = {name: score / total_score for name, score in val_scores.items()}
        
        # Calculate ensemble performance on validation set
        ensemble_pred = self._ensemble_predict(X_val, {name: self.trained_models[name].predict(X_val) 
                                                      for name in val_scores.keys()})
        
        # Calculate score
        val_score = accuracy_score(y_val, ensemble_pred) if results[list(results.keys())[0]].metadata.get('problem_type') == 'classification' else r2_score(y_val, ensemble_pred)
        
        return ModelResults(
            model=None,  # Ensemble doesn't have a single model
            train_score=np.mean([r.train_score for r in results.values()]),
            validation_score=val_score,
            test_score=None,
            feature_importance=None,
            predictions=ensemble_pred,
            probabilities=None,
            best_params=self.ensemble_weights,
            training_time=sum([r.training_time for r in results.values()]),
            metadata={'model_type': 'ensemble', 'ensemble_weights': self.ensemble_weights}
        )
    
    def _ensemble_predict(self, X: np.ndarray, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Make ensemble predictions using weighted voting."""
        if not self.ensemble_weights:
            # Equal weights
            weights = {name: 1.0 / len(predictions) for name in predictions.keys()}
        else:
            weights = self.ensemble_weights
        
        # Weighted average for regression, weighted voting for classification
        ensemble_pred = np.zeros(len(X))
        
        for name, pred in predictions.items():
            if name in weights:
                ensemble_pred += weights[name] * pred
        
        # For classification, round to nearest integer
        problem_type = list(self.models_config.values())[0].problem_type
        if problem_type == 'classification':
            ensemble_pred = np.round(ensemble_pred).astype(int)
        
        return ensemble_pred
    
    def save_models(self, directory: str):
        """Save all trained models to directory."""
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        for name, model in self.trained_models.items():
            filepath = Path(directory) / f"{name}_model.joblib"
            model.save_model(str(filepath))
        
        # Save ensemble weights
        if self.ensemble_weights:
            weights_path = Path(directory) / "ensemble_weights.joblib"
            joblib.dump(self.ensemble_weights, weights_path)
    
    def load_models(self, directory: str):
        """Load models from directory."""
        directory_path = Path(directory)
        
        for config_name in self.models_config.keys():
            filepath = directory_path / f"{config_name}_model.joblib"
            if filepath.exists():
                if config_name == 'transfer_entropy':
                    model = TransferEntropyModel(self.models_config[config_name])
                else:
                    model = self._create_standard_model(self.models_config[config_name])
                
                model.load_model(str(filepath))
                self.trained_models[config_name] = model
        
        # Load ensemble weights
        weights_path = directory_path / "ensemble_weights.joblib"
        if weights_path.exists():
            self.ensemble_weights = joblib.load(weights_path)
