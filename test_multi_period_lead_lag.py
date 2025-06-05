#!/usr/bin/env python3
"""
Test script for multi-period lead-lag analysis functionality.

This script validates the integration between the multi-period data loader
and the enhanced lead-lag detector across the three temporal periods.
"""

import sys
import logging
from pathlib import Path

# Add the project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from async_processing.multi_period_lead_lag_analyzer import MultiPeriodLeadLagAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_multi_period_lead_lag_analysis():
    """Test the multi-period lead-lag analysis functionality."""
    
    logger.info("🧪 Starting multi-period lead-lag analysis test")
    
    # Configuration
    data_dir = "data/raw"
    symbols = ['ETH_EUR', 'XBT_EUR']
    sample_size_per_period = 50000  # Smaller sample for testing
    
    try:
        # Initialize the analyzer
        analyzer = MultiPeriodLeadLagAnalyzer(
            data_dir=data_dir,
            symbols=symbols,
            max_lag_ms=2000,  # 2 seconds max lag
            min_price_change=0.0001
        )
        
        logger.info("✅ Analyzer initialized successfully")
        
        # Run the analysis
        logger.info("🚀 Running multi-period analysis...")
        results = analyzer.analyze_all_periods(
            sample_size_per_period=sample_size_per_period,
            save_results=True
        )
        
        # Display results summary
        logger.info("📊 Analysis Results Summary:")
        for period_id, signals in results.items():
            logger.info(f"   {period_id}: {len(signals)} signals detected")
        
        # Get detailed report
        report = analyzer.get_summary_report()
        
        logger.info("📈 Cross-Period Analysis:")
        if 'validation_metrics' in report['cross_period_analysis']:
            val_metrics = report['cross_period_analysis']['validation_metrics']
            logger.info(f"   Train→Validation persistence: {val_metrics.get('train_to_validation_persistence', 0):.2%}")
            logger.info(f"   Validation→Test persistence: {val_metrics.get('validation_to_test_persistence', 0):.2%}")
            logger.info(f"   Train→Test persistence: {val_metrics.get('train_to_test_persistence', 0):.2%}")
        
        # Try to generate plots
        try:
            analyzer.plot_analysis_results(save_plots=True)
            logger.info("📊 Analysis plots generated successfully")
        except Exception as e:
            logger.warning(f"Could not generate plots: {e}")
        
        logger.info("✅ Multi-period lead-lag analysis test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_data_structure():
    """Validate that the expected data structure exists."""
    
    logger.info("🔍 Validating data structure...")
    
    data_dir = Path("data/raw")
    expected_periods = ['DATA_0', 'DATA_1', 'DATA_2']
    expected_files = ['ETH_EUR.csv', 'XBT_EUR.csv']
    
    validation_passed = True
    
    for period in expected_periods:
        period_dir = data_dir / period
        if not period_dir.exists():
            logger.error(f"❌ Missing period directory: {period_dir}")
            validation_passed = False
            continue
            
        logger.info(f"✅ Found period directory: {period}")
        
        for file_name in expected_files:
            file_path = period_dir / file_name
            if not file_path.exists():
                logger.error(f"❌ Missing file: {file_path}")
                validation_passed = False
            else:
                file_size = file_path.stat().st_size / (1024 * 1024)  # MB
                logger.info(f"   {file_name}: {file_size:.1f} MB")
    
    if validation_passed:
        logger.info("✅ Data structure validation passed")
    else:
        logger.error("❌ Data structure validation failed")
    
    return validation_passed

def main():
    """Main test function."""
    
    logger.info("🚀 Starting multi-period lead-lag test suite")
    
    # Step 1: Validate data structure
    if not validate_data_structure():
        logger.error("❌ Data structure validation failed. Cannot proceed with tests.")
        return False
    
    # Step 2: Test multi-period analysis
    if not test_multi_period_lead_lag_analysis():
        logger.error("❌ Multi-period analysis test failed.")
        return False
    
    logger.info("🎉 All tests passed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
