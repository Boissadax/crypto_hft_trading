"""
Model selection and hyperparameter tuning pipeline.
Implements rolling window model selection similar to the SGX approach.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import time
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from .ml_models import MLModelManager

logger = logging.getLogger(__name__)

class RollingModelSelection:
    """
    Implements rolling window model selection for time series prediction.
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 window_size_seconds: int = 1800,  # 30 minutes
                 prediction_horizon_seconds: float = 1.0,
                 update_frequency_seconds: int = 60):  # 1 minute
        """
        Initialize the rolling model selection.
        
        Args:
            config: Configuration dictionary
            window_size_seconds: Size of training window in seconds
            prediction_horizon_seconds: How far ahead to predict
            update_frequency_seconds: How often to retrain models
        """
        self.config = config
        self.window_size_seconds = window_size_seconds
        self.prediction_horizon_seconds = prediction_horizon_seconds
        self.update_frequency_seconds = update_frequency_seconds
        
        self.model_manager = MLModelManager(config)
        
        # Storage for results
        self.rolling_results = []
        self.model_performance_history = {}
        self.best_model_history = []
        self.predictions_history = []
        
    def prepare_rolling_data(self, 
                           features_df: pd.DataFrame,
                           targets_df: pd.DataFrame,
                           symbol: str) -> List[Dict]:
        """
        Prepare data for rolling window analysis.
        
        Args:
            features_df: DataFrame with features
            targets_df: DataFrame with targets
            symbol: Symbol to predict
            
        Returns:
            List of dictionaries with train/test splits
        """
        # Merge features and targets
        data = features_df.merge(targets_df, on='timestamp', how='inner')
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        # Convert timestamp to datetime for easier manipulation
        data['datetime'] = pd.to_datetime(data['timestamp'], unit='s')
        
        # Find the target column for direction prediction
        target_col = f'target_{symbol}_{self.prediction_horizon_seconds}s_direction'
        if target_col not in data.columns:
            raise ValueError(f"Target column {target_col} not found")
        
        # Create rolling windows
        rolling_data = []
        
        # Convert time windows to number of rows (assuming 100ms frequency)
        window_size_rows = int(self.window_size_seconds * 10)  # 10 rows per second
        update_frequency_rows = int(self.update_frequency_seconds * 10)
        
        start_idx = window_size_rows
        end_idx = len(data) - int(self.prediction_horizon_seconds * 10)
        
        for i in range(start_idx, end_idx, update_frequency_rows):
            # Training window
            train_start = i - window_size_rows
            train_end = i
            
            # Test window (next prediction_horizon)
            test_start = i
            test_end = i + int(self.prediction_horizon_seconds * 10)
            
            if test_end >= len(data):
                break
            
            train_data = data.iloc[train_start:train_end].copy()
            test_data = data.iloc[test_start:test_end].copy()
            
            # Remove timestamp and datetime columns for training
            feature_cols = [col for col in train_data.columns 
                          if col not in ['timestamp', 'datetime'] and 
                          not col.startswith('target_')]
            
            X_train = train_data[feature_cols]
            y_train = train_data[target_col]
            X_test = test_data[feature_cols]
            y_test = test_data[target_col]
            
            # Skip if not enough data or too many NaN values
            if (len(X_train) < 50 or 
                X_train.isnull().sum().sum() > len(X_train) * len(feature_cols) * 0.1 or
                y_train.isnull().sum() > len(y_train) * 0.1):
                continue
            
            rolling_data.append({
                'window_id': len(rolling_data),
                'timestamp': data.iloc[i]['timestamp'],
                'datetime': data.iloc[i]['datetime'],
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test,
                'feature_cols': feature_cols
            })
        
        logger.info(f"Created {len(rolling_data)} rolling windows")
        return rolling_data
    
    def run_rolling_selection(self, 
                            rolling_data: List[Dict],
                            models_to_test: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Run rolling model selection across all windows.
        
        Args:
            rolling_data: List of rolling window data
            models_to_test: List of model names to test (None for all)
            
        Returns:
            DataFrame with rolling results
        """
        if models_to_test is None:
            models_to_test = list(self.model_manager.get_classification_models().keys())
        
        logger.info(f"Running rolling model selection with {len(models_to_test)} models")
        
        for window_data in rolling_data:
            window_id = window_data['window_id']
            timestamp = window_data['timestamp']
            
            logger.info(f"Processing window {window_id} at {window_data['datetime']}")
            
            start_time = time.time()
            
            # Train and evaluate all models for this window
            window_results = self._evaluate_window(window_data, models_to_test)
            
            # Store results
            for model_name, metrics in window_results.items():
                result = {
                    'window_id': window_id,
                    'timestamp': timestamp,
                    'datetime': window_data['datetime'],
                    'model': model_name,
                    **metrics
                }
                self.rolling_results.append(result)
            
            # Find best model for this window
            best_model = max(window_results.keys(), 
                           key=lambda k: window_results[k].get('accuracy', 0))
            
            self.best_model_history.append({
                'window_id': window_id,
                'timestamp': timestamp,
                'datetime': window_data['datetime'],
                'best_model': best_model,
                'best_accuracy': window_results[best_model]['accuracy']
            })
            
            # Store predictions from best model
            if 'predictions' in window_results[best_model]:
                self.predictions_history.append({
                    'window_id': window_id,
                    'timestamp': timestamp,
                    'datetime': window_data['datetime'],
                    'model': best_model,
                    'predictions': window_results[best_model]['predictions'],
                    'true_values': window_results[best_model]['true_values']
                })
            
            elapsed_time = time.time() - start_time
            logger.info(f"Window {window_id} completed in {elapsed_time:.2f}s. "
                       f"Best model: {best_model} (accuracy: {window_results[best_model]['accuracy']:.4f})")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.rolling_results)
        return results_df
    
    def _evaluate_window(self, 
                        window_data: Dict,
                        models_to_test: List[str]) -> Dict[str, Dict]:
        """
        Evaluate all models for a single window.
        
        Args:
            window_data: Data for current window
            models_to_test: List of model names to test
            
        Returns:
            Dictionary with results for each model
        """
        X_train = window_data['X_train']
        y_train = window_data['y_train']
        X_test = window_data['X_test']
        y_test = window_data['y_test']
        
        # Fill NaN values
        X_train = X_train.fillna(method='ffill').fillna(0)
        X_test = X_test.fillna(method='ffill').fillna(0)
        y_train = y_train.fillna(0)
        y_test = y_test.fillna(0)
        
        window_results = {}
        
        for model_name in models_to_test:
            try:
                # Train model with grid search
                best_model, best_params = self.model_manager.train_model_with_grid_search(
                    X_train, y_train, model_name, model_type='classification', cv_folds=3
                )
                
                # Evaluate on test set
                metrics = self.model_manager.evaluate_model(
                    best_model, X_test, y_test, model_name, model_type='classification'
                )
                
                # Get predictions
                predictions = self.model_manager.predict(model_name, X_test)
                
                # Add additional metrics
                metrics['predictions'] = predictions.tolist()
                metrics['true_values'] = y_test.tolist()
                metrics['best_params'] = best_params
                
                window_results[model_name] = metrics
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name} in window {window_data['window_id']}: {str(e)}")
                window_results[model_name] = {
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'confidence': 0.0,
                    'error': str(e)
                }
        
        return window_results
    
    def get_model_performance_summary(self) -> pd.DataFrame:
        """
        Get summary of model performance across all windows.
        
        Returns:
            DataFrame with performance summary by model
        """
        if not self.rolling_results:
            return pd.DataFrame()
        
        results_df = pd.DataFrame(self.rolling_results)
        
        # Group by model and calculate statistics
        summary = results_df.groupby('model').agg({
            'accuracy': ['mean', 'std', 'min', 'max'],
            'precision': ['mean', 'std'],
            'recall': ['mean', 'std'],
            'f1_score': ['mean', 'std'],
            'confidence': ['mean', 'std']
        }).round(4)
        
        # Flatten column names
        summary.columns = ['_'.join(col).strip() for col in summary.columns]
        
        # Sort by mean accuracy
        summary = summary.sort_values('accuracy_mean', ascending=False)
        
        return summary
    
    def get_best_model_over_time(self) -> pd.DataFrame:
        """
        Get the best performing model over time.
        
        Returns:
            DataFrame showing best model selection over time
        """
        return pd.DataFrame(self.best_model_history)
    
    def calculate_trading_performance(self, 
                                    price_data: pd.DataFrame,
                                    symbol: str,
                                    transaction_cost: float = 0.001) -> Dict[str, float]:
        """
        Calculate trading performance based on model predictions.
        
        Args:
            price_data: DataFrame with price data
            symbol: Symbol being traded
            transaction_cost: Transaction cost as fraction of trade value
            
        Returns:
            Dictionary with trading performance metrics
        """
        if not self.predictions_history:
            return {}
        
        total_return = 0.0
        num_trades = 0
        correct_predictions = 0
        total_predictions = 0
        
        for pred_data in self.predictions_history:
            predictions = pred_data['predictions']
            true_values = pred_data['true_values']
            timestamp = pred_data['timestamp']
            
            # Find corresponding price data
            price_mask = abs(price_data['timestamp'] - timestamp) < 1.0  # 1 second tolerance
            if not price_mask.any():
                continue
            
            price_row = price_data[price_mask].iloc[0]
            current_price = price_row[f'{symbol}_mid']
            
            # Calculate returns for each prediction
            for pred, true_val in zip(predictions, true_values):
                total_predictions += 1
                
                if pred == true_val:
                    correct_predictions += 1
                
                # Simulate trading based on prediction
                if pred == 1:  # Predicted price increase
                    # Buy
                    trade_return = (self.prediction_horizon_seconds / 1.0) * 0.001  # Simplified return
                    if true_val == 1:  # Correct prediction
                        total_return += trade_return - transaction_cost
                    else:  # Wrong prediction
                        total_return -= trade_return + transaction_cost
                    num_trades += 1
                # If pred == 0 (predicted decrease), we don't trade
        
        if total_predictions == 0:
            return {}
        
        accuracy = correct_predictions / total_predictions
        avg_return_per_trade = total_return / num_trades if num_trades > 0 else 0.0
        
        return {
            'total_return': total_return,
            'num_trades': num_trades,
            'accuracy': accuracy,
            'avg_return_per_trade': avg_return_per_trade,
            'total_predictions': total_predictions,
            'sharpe_ratio': total_return / abs(total_return) if total_return != 0 else 0.0
        }
