"""
Enhanced Data Persistence Manager

Provides advanced data persistence with compression, validation, and metadata tracking
for the crypto HFT trading system.
"""

import pandas as pd
import numpy as np
import pickle
import gzip
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import pyarrow.parquet as pq
import pyarrow as pa

logger = logging.getLogger(__name__)

class EnhancedDataPersistence:
    """Advanced data persistence with compression and validation."""
    
    def __init__(self, base_path: str = "data/processed"):
        """
        Initialize enhanced persistence manager.
        
        Args:
            base_path: Base directory for all persistent data
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.cache_dir = self.base_path / "cache"
        self.features_dir = self.base_path / "features"  
        self.models_dir = self.base_path / "models"
        self.metadata_dir = self.base_path / "metadata"
        
        for dir_path in [self.cache_dir, self.features_dir, self.models_dir, self.metadata_dir]:
            dir_path.mkdir(exist_ok=True)
            
        # Metadata tracking
        self.metadata_file = self.metadata_dir / "persistence_metadata.json"
        self.load_metadata()
        
    def load_metadata(self):
        """Load persistence metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                'datasets': {},
                'features': {},
                'models': {},
                'created': datetime.now().isoformat()
            }
    
    def save_metadata(self):
        """Save persistence metadata."""
        self.metadata['last_updated'] = datetime.now().isoformat()
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def save_features_optimized(self, 
                               features: pd.DataFrame,
                               dataset_name: str,
                               compression: str = 'snappy',
                               engine: str = 'pyarrow') -> Dict[str, Any]:
        """
        Save features with optimal compression and validation.
        
        Args:
            features: Feature DataFrame
            dataset_name: Unique name for the dataset
            compression: Compression algorithm ('snappy', 'gzip', 'brotli')
            engine: Parquet engine ('pyarrow', 'fastparquet')
            
        Returns:
            Dictionary with save statistics
        """
        file_path = self.features_dir / f"{dataset_name}_features.parquet"
        
        # Calculate checksum before saving
        data_hash = self._calculate_dataframe_hash(features)
        
        # Save with optimal settings
        features.to_parquet(
            file_path,
            compression=compression,
            engine=engine,
            index=False
        )
        
        # Get file stats
        file_size = file_path.stat().st_size
        compression_ratio = len(features) * features.shape[1] * 8 / file_size  # Rough estimate
        
        # Update metadata
        self.metadata['features'][dataset_name] = {
            'file_path': str(file_path),
            'shape': features.shape,
            'columns': list(features.columns),
            'dtypes': features.dtypes.to_dict(),
            'size_bytes': file_size,
            'compression': compression,
            'engine': engine,
            'data_hash': data_hash,
            'compression_ratio': compression_ratio,
            'created': datetime.now().isoformat()
        }
        
        self.save_metadata()
        
        logger.info(f"Features saved: {dataset_name}")
        logger.info(f"  Shape: {features.shape}")
        logger.info(f"  Size: {file_size / 1024 / 1024:.2f} MB")
        logger.info(f"  Compression ratio: {compression_ratio:.1f}x")
        
        return self.metadata['features'][dataset_name]
    
    def load_features_validated(self, dataset_name: str) -> Optional[pd.DataFrame]:
        """
        Load features with validation.
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            Feature DataFrame or None if validation fails
        """
        if dataset_name not in self.metadata['features']:
            logger.warning(f"Dataset {dataset_name} not found in metadata")
            return None
        
        metadata = self.metadata['features'][dataset_name]
        file_path = Path(metadata['file_path'])
        
        if not file_path.exists():
            logger.error(f"Feature file not found: {file_path}")
            return None
        
        try:
            # Load data
            features = pd.read_parquet(file_path)
            
            # Validate against metadata
            if features.shape != tuple(metadata['shape']):
                logger.warning(f"Shape mismatch for {dataset_name}: "
                             f"expected {metadata['shape']}, got {features.shape}")
            
            # Validate data integrity
            current_hash = self._calculate_dataframe_hash(features)
            if current_hash != metadata['data_hash']:
                logger.warning(f"Data integrity check failed for {dataset_name}")
            
            logger.info(f"Features loaded and validated: {dataset_name}")
            return features
            
        except Exception as e:
            logger.error(f"Error loading features {dataset_name}: {e}")
            return None
    
    def save_compressed_pickle(self, 
                              data: Any, 
                              name: str,
                              compress: bool = True) -> Dict[str, Any]:
        """
        Save data as compressed pickle.
        
        Args:
            data: Data to save
            name: Unique name for the data
            compress: Whether to use gzip compression
            
        Returns:
            Dictionary with save statistics
        """
        if compress:
            file_path = self.cache_dir / f"{name}.pkl.gz"
            with gzip.open(file_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            file_path = self.cache_dir / f"{name}.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        file_size = file_path.stat().st_size
        
        # Update metadata
        self.metadata['datasets'][name] = {
            'file_path': str(file_path),
            'compressed': compress,
            'size_bytes': file_size,
            'type': str(type(data).__name__),
            'created': datetime.now().isoformat()
        }
        
        self.save_metadata()
        
        logger.info(f"Data saved: {name} ({file_size / 1024:.1f} KB)")
        
        return self.metadata['datasets'][name]
    
    def load_compressed_pickle(self, name: str) -> Optional[Any]:
        """
        Load compressed pickle data.
        
        Args:
            name: Name of the data to load
            
        Returns:
            Loaded data or None if not found
        """
        if name not in self.metadata['datasets']:
            logger.warning(f"Dataset {name} not found")
            return None
        
        metadata = self.metadata['datasets'][name]
        file_path = Path(metadata['file_path'])
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None
        
        try:
            if metadata['compressed']:
                with gzip.open(file_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            
            logger.info(f"Data loaded: {name}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading {name}: {e}")
            return None
    
    def save_model_bundle(self, 
                         model: Any,
                         model_name: str,
                         metadata: Dict[str, Any],
                         scaler: Optional[Any] = None,
                         feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Save complete model bundle with metadata.
        
        Args:
            model: Trained model
            model_name: Unique model name
            metadata: Model metadata
            scaler: Feature scaler (optional)
            feature_names: List of feature names (optional)
            
        Returns:
            Dictionary with save statistics
        """
        bundle_dir = self.models_dir / model_name
        bundle_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = bundle_dir / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save scaler if provided
        if scaler is not None:
            scaler_path = bundle_dir / "scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
        
        # Save feature names
        if feature_names is not None:
            features_path = bundle_dir / "feature_names.json"
            with open(features_path, 'w') as f:
                json.dump(feature_names, f)
        
        # Save metadata
        bundle_metadata = {
            **metadata,
            'model_name': model_name,
            'has_scaler': scaler is not None,
            'has_feature_names': feature_names is not None,
            'bundle_path': str(bundle_dir),
            'created': datetime.now().isoformat()
        }
        
        metadata_path = bundle_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(bundle_metadata, f, indent=2, default=str)
        
        # Update global metadata
        self.metadata['models'][model_name] = bundle_metadata
        self.save_metadata()
        
        logger.info(f"Model bundle saved: {model_name}")
        return bundle_metadata
    
    def load_model_bundle(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Load complete model bundle.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Dictionary with model, scaler, features, metadata
        """
        if model_name not in self.metadata['models']:
            logger.warning(f"Model {model_name} not found")
            return None
        
        bundle_metadata = self.metadata['models'][model_name]
        bundle_dir = Path(bundle_metadata['bundle_path'])
        
        if not bundle_dir.exists():
            logger.error(f"Model bundle directory not found: {bundle_dir}")
            return None
        
        try:
            bundle = {'metadata': bundle_metadata}
            
            # Load model
            model_path = bundle_dir / "model.pkl"
            with open(model_path, 'rb') as f:
                bundle['model'] = pickle.load(f)
            
            # Load scaler if available
            if bundle_metadata['has_scaler']:
                scaler_path = bundle_dir / "scaler.pkl"
                with open(scaler_path, 'rb') as f:
                    bundle['scaler'] = pickle.load(f)
            
            # Load feature names if available
            if bundle_metadata['has_feature_names']:
                features_path = bundle_dir / "feature_names.json"
                with open(features_path, 'r') as f:
                    bundle['feature_names'] = json.load(f)
            
            logger.info(f"Model bundle loaded: {model_name}")
            return bundle
            
        except Exception as e:
            logger.error(f"Error loading model bundle {model_name}: {e}")
            return None
    
    def _calculate_dataframe_hash(self, df: pd.DataFrame) -> str:
        """Calculate hash of DataFrame for integrity checking."""
        # Create a string representation of the DataFrame
        df_string = df.to_csv(index=False)
        return hashlib.md5(df_string.encode()).hexdigest()
    
    def get_storage_summary(self) -> Dict[str, Any]:
        """Get comprehensive storage summary."""
        total_size = 0
        file_count = 0
        
        for path in [self.cache_dir, self.features_dir, self.models_dir]:
            for file_path in path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1
        
        return {
            'total_size_mb': total_size / 1024 / 1024,
            'total_files': file_count,
            'features_count': len(self.metadata['features']),
            'models_count': len(self.metadata['models']),
            'datasets_count': len(self.metadata['datasets']),
            'base_path': str(self.base_path),
            'metadata': self.metadata
        }
    
    def cleanup_old_data(self, max_age_days: int = 30) -> Dict[str, int]:
        """Clean up old cached data."""
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        removed_count = {'features': 0, 'models': 0, 'datasets': 0}
        
        # Clean features
        for name, metadata in list(self.metadata['features'].items()):
            created = datetime.fromisoformat(metadata['created'])
            if created < cutoff_date:
                file_path = Path(metadata['file_path'])
                if file_path.exists():
                    file_path.unlink()
                del self.metadata['features'][name]
                removed_count['features'] += 1
        
        # Clean datasets
        for name, metadata in list(self.metadata['datasets'].items()):
            created = datetime.fromisoformat(metadata['created'])
            if created < cutoff_date:
                file_path = Path(metadata['file_path'])
                if file_path.exists():
                    file_path.unlink()
                del self.metadata['datasets'][name]
                removed_count['datasets'] += 1
        
        # Clean models (more conservative - only remove very old ones)
        model_cutoff = datetime.now() - timedelta(days=max_age_days * 2)
        for name, metadata in list(self.metadata['models'].items()):
            created = datetime.fromisoformat(metadata['created'])
            if created < model_cutoff:
                bundle_dir = Path(metadata['bundle_path'])
                if bundle_dir.exists():
                    import shutil
                    shutil.rmtree(bundle_dir)
                del self.metadata['models'][name]
                removed_count['models'] += 1
        
        self.save_metadata()
        
        logger.info(f"Cleanup completed: {removed_count}")
        return removed_count