"""
Machine learning models for cryptocurrency trading.
Implements various ML algorithms for price prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
from sklearn.ensemble import (
    RandomForestClassifier, 
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    ExtraTreesClassifier,
    AdaBoostClassifier
)
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger(__name__)

class MLModelManager:
    """
    Manages machine learning models for cryptocurrency trading.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.models = {}
        self.scalers = {}
        self.best_models = {}
        self.model_performance = {}
        
    def get_classification_models(self) -> Dict[str, Any]:
        """
        Get classification models for direction prediction.
        
        Returns:
            Dictionary of classification models
        """
        models = {
            'RandomForest': RandomForestClassifier(
                random_state=42,
                n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            ),
            'LightGBM': lgb.LGBMClassifier(
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                random_state=42
            ),
            'ExtraTrees': ExtraTreesClassifier(
                random_state=42,
                n_jobs=-1
            ),
            'SVM': SVC(
                random_state=42,
                probability=True
            ),
            'LogisticRegression': LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        }
        return models
    
    def get_regression_models(self) -> Dict[str, Any]:
        """
        Get regression models for return prediction.
        
        Returns:
            Dictionary of regression models
        """
        models = {
            'RandomForest': RandomForestRegressor(
                random_state=42,
                n_jobs=-1
            ),
            'XGBoost': xgb.XGBRegressor(
                random_state=42,
                n_jobs=-1
            ),
            'LightGBM': lgb.LGBMRegressor(
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'GradientBoosting': GradientBoostingRegressor(
                random_state=42
            ),
            'SVM': SVR(),
            'LinearRegression': LinearRegression()
        }
        return models
    
    def get_hyperparameter_grids(self, model_type: str = 'classification') -> Dict[str, Dict]:
        """
        Get hyperparameter grids for model tuning.
        
        Args:
            model_type: Type of models ('classification' or 'regression')
            
        Returns:
            Dictionary of hyperparameter grids
        """
        if model_type == 'classification':
            return {
                'RandomForest': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                },
                'XGBoost': {
                    'n_estimators': [100, 200],
                    'max_depth': [6, 10],
                    'learning_rate': [0.1, 0.2],
                    'subsample': [0.8, 1.0]
                },
                'LightGBM': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20],
                    'learning_rate': [0.1, 0.2],
                    'num_leaves': [31, 50]
                },
                'GradientBoosting': {
                    'n_estimators': [100, 200],
                    'max_depth': [5, 10],
                    'learning_rate': [0.1, 0.2]
                },
                'SVM': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                },
                'LogisticRegression': {
                    'C': [0.1, 1, 10],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            }
        else:  # regression
            return {
                'RandomForest': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                },
                'XGBoost': {
                    'n_estimators': [100, 200],
                    'max_depth': [6, 10],
                    'learning_rate': [0.1, 0.2]
                },
                'LightGBM': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20],
                    'learning_rate': [0.1, 0.2]
                },
                'SVM': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear']
                }
            }
    
    def train_model_with_grid_search(self,
                                   X_train: pd.DataFrame,
                                   y_train: pd.Series,
                                   model_name: str,
                                   model_type: str = 'classification',
                                   cv_folds: int = 5) -> Tuple[Any, Dict]:
        """
        Train a model with grid search hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training targets
            model_name: Name of the model
            model_type: Type of model ('classification' or 'regression')
            cv_folds: Number of cross-validation folds
            
        Returns:
            Tuple of (best_model, best_params)
        """
        logger.info("Training %s with grid search", model_name)
        
        # Get model and parameter grid
        if model_type == 'classification':
            models = self.get_classification_models()
            scoring = 'accuracy'
        else:
            models = self.get_regression_models()
            scoring = 'neg_mean_squared_error'
            
        if model_name not in models:
            raise ValueError(f"Model {model_name} not found")
            
        model = models[model_name]
        param_grid = self.get_hyperparameter_grids(model_type).get(model_name, {})
        
        # Use TimeSeriesSplit for time series data
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # Grid search
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=tscv,
            scoring=scoring,
            n_jobs=-1,
            verbose=0
        )
        
        # Scale features if necessary
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Fit grid search
        grid_search.fit(X_train_scaled, y_train)
        
        # Store scaler and best model
        self.scalers[model_name] = scaler
        self.best_models[model_name] = grid_search.best_estimator_
        
        logger.info("Best parameters for %s: %s", model_name, grid_search.best_params_)
        logger.info("Best CV score for %s: %.4f", model_name, grid_search.best_score_)
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def evaluate_model(self,
                      model: Any,
                      X_test: pd.DataFrame,
                      y_test: pd.Series,
                      model_name: str,
                      model_type: str = 'classification') -> Dict[str, float]:
        """
        Evaluate a trained model.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            model_name: Name of the model
            model_type: Type of model ('classification' or 'regression')
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Scale test features
        if model_name in self.scalers:
            X_test_scaled = self.scalers[model_name].transform(X_test)
        else:
            X_test_scaled = X_test
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        if model_type == 'classification':
            # Classification metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
            
            # Get prediction probabilities if available
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test_scaled)
                if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                    metrics['confidence'] = np.mean(np.max(y_pred_proba, axis=1))
                else:
                    metrics['confidence'] = np.nan
            else:
                metrics['confidence'] = np.nan
                
        else:
            # Regression metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
        
        return metrics
    
    def train_all_models(self,
                        X_train: pd.DataFrame,
                        y_train: pd.Series,
                        X_test: pd.DataFrame,
                        y_test: pd.Series,
                        model_type: str = 'classification') -> pd.DataFrame:
        """
        Train and evaluate all models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            model_type: Type of models ('classification' or 'regression')
            
        Returns:
            DataFrame with model performance comparison
        """
        results = []
        
        if model_type == 'classification':
            models = self.get_classification_models()
        else:
            models = self.get_regression_models()
        
        for model_name in models.keys():
            try:
                logger.info("Training %s", model_name)
                
                # Train with grid search
                best_model, best_params = self.train_model_with_grid_search(
                    X_train, y_train, model_name, model_type
                )
                
                # Evaluate on test set
                metrics = self.evaluate_model(
                    best_model, X_test, y_test, model_name, model_type
                )
                
                # Store results
                result = {'model': model_name}
                result.update(metrics)
                result.update({'best_params': str(best_params)})
                results.append(result)
                
                # Store performance
                self.model_performance[model_name] = metrics
                
            except (ValueError, RuntimeError) as e:
                logger.error("Error training %s: %s", model_name, str(e))
                continue
        
        results_df = pd.DataFrame(results)
        
        # Sort by primary metric
        if model_type == 'classification':
            results_df = results_df.sort_values('accuracy', ascending=False)
        else:
            results_df = results_df.sort_values('r2', ascending=False)
        
        return results_df
    
    def get_feature_importance(self, model_name: str, feature_names: List[str]) -> pd.DataFrame:
        """
        Get feature importance from a trained model.
        
        Args:
            model_name: Name of the trained model
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance
        """
        if model_name not in self.best_models:
            raise ValueError(f"Model {model_name} not found in trained models")
        
        model = self.best_models[model_name]
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For linear models, use absolute coefficients
            if len(model.coef_.shape) > 1:
                importance = np.abs(model.coef_[0])
            else:
                importance = np.abs(model.coef_)
        else:
            logger.warning("Model %s does not have feature importance", model_name)
            return pd.DataFrame()
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def predict(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using a trained model.
        
        Args:
            model_name: Name of the trained model
            X: Features for prediction
            
        Returns:
            Predictions array
        """
        if model_name not in self.best_models:
            raise ValueError(f"Model {model_name} not found in trained models")
        
        model = self.best_models[model_name]
        
        # Scale features if scaler exists
        if model_name in self.scalers:
            X_scaled = self.scalers[model_name].transform(X)
        else:
            X_scaled = X
        
        return model.predict(X_scaled)
    
    def predict_proba(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities using a trained model.
        
        Args:
            model_name: Name of the trained model
            X: Features for prediction
            
        Returns:
            Prediction probabilities array
        """
        if model_name not in self.best_models:
            raise ValueError(f"Model {model_name} not found in trained models")
        
        model = self.best_models[model_name]
        
        if not hasattr(model, 'predict_proba'):
            raise ValueError(f"Model {model_name} does not support probability prediction")
        
        # Scale features if scaler exists
        if model_name in self.scalers:
            X_scaled = self.scalers[model_name].transform(X)
        else:
            X_scaled = X
        
        return model.predict_proba(X_scaled)
