"""
Data Cache System for HFT Engine v3

Efficient caching system for large CSV datasets with automatic invalidation.
Converts CSV files to Parquet format with streaming to avoid memory overload.
Provides fingerprinting to detect changes and avoid reprocessing.
"""

from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import hashlib
import os
import json
import datetime as dt
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Configuration
RAW_ROOT = Path("raw_data")
PROCESSED_ROOT = Path("processed_data")
PROCESSED_ROOT.mkdir(exist_ok=True)

def _fingerprint(csv_path: Path) -> str:
    """
    Return SHA1 hash of file size + mtime for cheap change detection.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        SHA1 hash string combining file size and modification time
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
    stat = csv_path.stat()
    s = f"{stat.st_size}-{int(stat.st_mtime)}"
    return hashlib.sha1(s.encode()).hexdigest()

def ensure_processed(dataset: str) -> Dict[str, Path]:
    """
    Guarantee there is a parquet version of ETH_EUR + XBT_EUR for `dataset`.
    
    Creates processed parquet files from raw CSV data if they don't exist or
    if the source CSV files have changed. Uses streaming to handle large files
    efficiently without loading everything into memory.
    
    Args:
        dataset: Dataset identifier (e.g., "DATA_0", "DATA_1", "DATA_2")
        
    Returns:
        Dictionary mapping symbol to parquet file path:
        {"ETH": path_to_eth_parquet, "XBT": path_to_xbt_parquet}
        
    Raises:
        FileNotFoundError: If raw CSV files are missing
        ValueError: If CSV structure is invalid
    """
    raw_dir = RAW_ROOT / dataset          # e.g. raw_data/DATA_0
    proc_dir = PROCESSED_ROOT / dataset   # e.g. processed_data/DATA_0
    proc_dir.mkdir(exist_ok=True)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    # Load or initialize fingerprint mapping
    mapping_file = proc_dir / ".fingerprints.json"
    old_fingerprints = {}
    if mapping_file.exists():
        try:
            old_fingerprints = json.loads(mapping_file.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not load fingerprints file: {e}")
            old_fingerprints = {}

    result = {}
    
    for symbol in ("ETH_EUR", "XBT_EUR"):
        csv_path = raw_dir / f"{symbol}.csv"
        parquet_path = proc_dir / f"{symbol}.parquet"
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Required CSV file not found: {csv_path}")
            
        # Check if conversion is needed
        current_fp = _fingerprint(csv_path)
        needs_conversion = (
            old_fingerprints.get(symbol) != current_fp or 
            not parquet_path.exists()
        )
        
        if needs_conversion:
            logger.info(f"Converting {symbol} from CSV to Parquet (streaming)...")
            start_time = dt.datetime.now()
            
            _convert_csv_to_parquet(csv_path, parquet_path)
            
            elapsed = (dt.datetime.now() - start_time).total_seconds()
            logger.info(f"Conversion completed for {symbol} in {elapsed:.1f}s")
            
            # Update fingerprint
            old_fingerprints[symbol] = current_fp
        else:
            logger.info(f"Using cached Parquet for {symbol}")
            
        # Map symbol to short name for return dict
        short_symbol = symbol.split("_")[0]  # "ETH_EUR" -> "ETH"
        result[short_symbol] = parquet_path

    # Save updated fingerprints
    try:
        mapping_file.write_text(json.dumps(old_fingerprints, indent=2))
    except OSError as e:
        logger.warning(f"Could not save fingerprints file: {e}")

    logger.info(f"Data cache ready for dataset {dataset}")
    return result

def _convert_csv_to_parquet(csv_path: Path, parquet_path: Path) -> None:
    """
    Convert CSV to Parquet using streaming to avoid memory overload.
    
    Args:
        csv_path: Source CSV file path
        parquet_path: Destination Parquet file path
    """
    # Define schema for consistent data types
    schema = pa.schema([
        ("price", pa.float64()),
        ("volume", pa.float64()),
        ("timestamp", pa.float64()),
        ("side", pa.dictionary(pa.int8(), pa.string())),
        ("level", pa.int8())
    ])
    
    # Stream CSV in chunks to avoid memory issues
    chunk_size = 5_000_000  # 5M rows per chunk - adjust based on available RAM
    
    try:
        csv_reader = pd.read_csv(
            csv_path,
            chunksize=chunk_size,
            dtype={
                "price": "float64",
                "volume": "float64", 
                "timestamp": "float64",
                "side": "category",
                "level": "int8"
            }
        )
        
        # Write to Parquet with streaming
        with pq.ParquetWriter(parquet_path, schema, compression="zstd") as writer:
            total_rows = 0
            for chunk_num, chunk in enumerate(csv_reader):
                # Convert chunk to Arrow table
                table = pa.Table.from_pandas(chunk, schema=schema, preserve_index=False)
                writer.write_table(table)
                
                total_rows += len(chunk)
                if chunk_num % 10 == 0:  # Log progress every 10 chunks
                    logger.debug(f"Processed {total_rows:,} rows...")
                    
        logger.info(f"Successfully converted {total_rows:,} rows to Parquet")
        
    except Exception as e:
        # Clean up partial file on error
        if parquet_path.exists():
            parquet_path.unlink()
        raise ValueError(f"Failed to convert {csv_path} to Parquet: {e}")

def get_cache_info(dataset: str) -> Dict[str, any]:
    """
    Get information about cached data for a dataset.
    
    Args:
        dataset: Dataset identifier
        
    Returns:
        Dictionary with cache information
    """
    proc_dir = PROCESSED_ROOT / dataset
    info = {
        "dataset": dataset,
        "processed_dir": str(proc_dir),
        "exists": proc_dir.exists(),
        "files": {}
    }
    
    if proc_dir.exists():
        for symbol in ("ETH_EUR", "XBT_EUR"):
            parquet_path = proc_dir / f"{symbol}.parquet"
            if parquet_path.exists():
                stat = parquet_path.stat()
                info["files"][symbol] = {
                    "path": str(parquet_path),
                    "size_mb": stat.st_size / (1024 * 1024),
                    "modified": dt.datetime.fromtimestamp(stat.st_mtime).isoformat()
                }
    
    return info

def clear_cache(dataset: Optional[str] = None) -> None:
    """
    Clear cached data for a dataset or all datasets.
    
    Args:
        dataset: Specific dataset to clear, or None to clear all
    """
    if dataset:
        proc_dir = PROCESSED_ROOT / dataset
        if proc_dir.exists():
            import shutil
            shutil.rmtree(proc_dir)
            logger.info(f"Cleared cache for dataset {dataset}")
    else:
        if PROCESSED_ROOT.exists():
            import shutil
            shutil.rmtree(PROCESSED_ROOT)
            PROCESSED_ROOT.mkdir(exist_ok=True)
            logger.info("Cleared all cached data")
