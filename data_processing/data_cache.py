"""
Data Caching System for HFT Trading Pipeline

Provides efficient caching of processed data to avoid reprocessing
large datasets on every run.
"""

import pickle
import pandas as pd
import numpy as np
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DataCache:
    """Caches processed data to avoid reprocessing on every run."""
    
    def __init__(self, cache_dir: str = "data/processed"):
        """
        Initialize the data cache.
        
        Args:
            cache_dir: Directory to store cached data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache metadata
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.load_metadata()
    
    def load_metadata(self):
        """Load cache metadata."""
        if self.metadata_file.exists():
            import json
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
    
    def save_metadata(self):
        """Save cache metadata."""
        import json
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def _generate_cache_key(self, data_path: str, symbols: list, 
                          processing_params: Dict[str, Any]) -> str:
        """Generate a unique cache key based on data and parameters."""
        # Include file modification times in hash
        file_info = {}
        for symbol in symbols:
            file_path = Path(data_path) / f"{symbol}.csv"
            if file_path.exists():
                file_info[symbol] = file_path.stat().st_mtime
        
        # Create hash of all relevant parameters
        cache_data = {
            'data_path': str(data_path),
            'symbols': sorted(symbols),
            'processing_params': processing_params,
            'file_info': file_info
        }
        
        cache_string = str(sorted(cache_data.items()))
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def get_processed_data(self, data_path: str, symbols: list, 
                          processing_params: Dict[str, Any]) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Get processed data from cache if available and valid.
        
        Args:
            data_path: Path to raw data
            symbols: List of symbols
            processing_params: Processing parameters
            
        Returns:
            Cached data if available, None otherwise
        """
        cache_key = self._generate_cache_key(data_path, symbols, processing_params)
        cache_file = self.cache_dir / f"processed_data_{cache_key}.pkl"
        
        if not cache_file.exists():
            logger.info(f"❌ No cached data found for key: {cache_key[:8]}...")
            return None
        
        # Check if cache is still valid
        if cache_key in self.metadata:
            cache_age = datetime.now() - datetime.fromisoformat(self.metadata[cache_key]['created'])
            if cache_age > timedelta(days=7):  # Cache expires after 7 days
                logger.info(f"⏰ Cache expired (age: {cache_age}), will reprocess")
                return None
        
        try:
            logger.info(f"✅ Loading processed data from cache ({cache_key[:8]}...)")
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            logger.info(f"📊 Cached data loaded: {sum(len(df) for df in data.values()):,} total records")
            return data
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to load cache: {e}")
            return None
    
    def save_processed_data(self, data: Dict[str, pd.DataFrame], 
                           data_path: str, symbols: list, 
                           processing_params: Dict[str, Any]) -> None:
        """
        Save processed data to cache.
        
        Args:
            data: Processed data to cache
            data_path: Path to raw data
            symbols: List of symbols
            processing_params: Processing parameters
        """
        cache_key = self._generate_cache_key(data_path, symbols, processing_params)
        cache_file = self.cache_dir / f"processed_data_{cache_key}.pkl"
        
        try:
            logger.info(f"💾 Saving processed data to cache ({cache_key[:8]}...)")
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Update metadata
            self.metadata[cache_key] = {
                'created': datetime.now().isoformat(),
                'symbols': symbols,
                'total_records': sum(len(df) for df in data.values()),
                'file_size_mb': cache_file.stat().st_size / 1024 / 1024
            }
            self.save_metadata()
            
            logger.info(f"✅ Data cached successfully ({self.metadata[cache_key]['file_size_mb']:.1f} MB)")
            
        except Exception as e:
            logger.error(f"❌ Failed to save cache: {e}")
    
    def get_event_stream(self, cache_key_suffix: str) -> Optional[list]:
        """Get cached event stream."""
        cache_file = self.cache_dir / f"event_stream_{cache_key_suffix}.pkl"
        
        if not cache_file.exists():
            return None
        
        try:
            logger.info(f"✅ Loading event stream from cache")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"⚠️  Failed to load event stream cache: {e}")
            return None
    
    def save_event_stream(self, events: list, cache_key_suffix: str) -> None:
        """Save event stream to cache."""
        cache_file = self.cache_dir / f"event_stream_{cache_key_suffix}.pkl"
        
        try:
            logger.info(f"💾 Saving event stream to cache ({len(events):,} events)")
            with open(cache_file, 'wb') as f:
                pickle.dump(events, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.info(f"✅ Event stream cached successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to save event stream cache: {e}")
    
    def clear_old_cache(self, max_age_days: int = 30) -> None:
        """Clear old cache files."""
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        removed_count = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            if cache_file.stat().st_mtime < cutoff_date.timestamp():
                cache_file.unlink()
                removed_count += 1
        
        # Clean metadata
        old_keys = []
        for key, meta in self.metadata.items():
            if datetime.fromisoformat(meta['created']) < cutoff_date:
                old_keys.append(key)
        
        for key in old_keys:
            del self.metadata[key]
        
        if removed_count > 0:
            self.save_metadata()
            logger.info(f"🧹 Cleaned {removed_count} old cache files")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about current cache."""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            'cache_dir': str(self.cache_dir),
            'total_files': len(cache_files),
            'total_size_mb': total_size / 1024 / 1024,
            'metadata_entries': len(self.metadata),
            'oldest_cache': min((datetime.fromisoformat(m['created']) 
                               for m in self.metadata.values()), default=None),
            'newest_cache': max((datetime.fromisoformat(m['created']) 
                               for m in self.metadata.values()), default=None)
        }
