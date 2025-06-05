"""
Learning Module

Implements machine learning components for lead-lag strategy:
- Train/test data preparation with proper temporal splits
- Feature selection and engineering pipelines
- Model training and validation
- Performance evaluation and backtesting
- Transfer entropy-based strategy learning
"""

from .data_preparation import DataPreparator, TemporalSplit
from .model_training import ModelTrainer, TransferEntropyModel
from .validation import ModelValidator, BacktestValidator
from .pipeline import LearningPipeline

__all__ = [
    'DataPreparator',
    'TemporalSplit', 
    'ModelTrainer',
    'TransferEntropyModel',
    'ModelValidator',
    'BacktestValidator',
    'LearningPipeline'
]
