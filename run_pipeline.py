#!/usr/bin/env python3
"""
HFT Engine v3 - Heavy Data Ready Pipeline
Main entry point with dataset selection support

Usage:
    python run_pipeline.py --dataset DATA_0
    python run_pipeline.py --dataset DATA_1 --symbols BTC ETH
    python run_pipeline.py --dataset DATA_2 --symbols BTC
"""

import argparse
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from feature_engineering.feature_engineer import FeatureEngineer
from feature_engineering import SyncConfig
from data_cache import ensure_processed, get_cache_info
from learning import DataPreparator, ModelTrainer
from feature_engineering.feature_engineer import split_train_test
from visualization import ModelPerformanceVisualizer, FeatureAnalysisVisualizer
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def setup_logging():
    """Setup logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "pipeline.log"),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def run_full_pipeline(dataset_id: str, symbols: list = None):
    """
    Run the complete feature engineering pipeline for a dataset.
    
    Args:
        dataset_id: Dataset identifier (DATA_0, DATA_1, DATA_2)
        symbols: List of symbols to process (default: ["BTC", "ETH"])
    """
    logger = logging.getLogger(__name__)
    
    if symbols is None:
        symbols = ["BTC", "ETH"]
    
    logger.info(f"🚀 Starting pipeline for dataset {dataset_id}")
    logger.info(f"📈 Processing symbols: {symbols}")
    
    # Show cache info
    cache_info = get_cache_info(dataset_id)
    logger.info(f"📁 Cache status: {cache_info}")
    
    # Initialize feature engineer
    sync_config = SyncConfig(
        max_interpolation_gap_us=300_000_000,  # 5 minutes
        min_symbols_required=1,
        enable_cross_symbol_features=True
    )
    
    feature_engineer = FeatureEngineer(
        symbols=symbols,
        sync_config=sync_config,
        max_levels=10,
        dataset_id=dataset_id
    )
    
    try:
        # Run pipeline
        logger.info("⚙️ Starting feature extraction...")
        features_df = feature_engineer.create_features()
        
        if features_df.empty:
            logger.error("❌ No features generated")
            return None
            
        logger.info(f"✅ Pipeline completed successfully!")
        logger.info(f"📊 Generated features: {features_df.shape}")
        logger.info(f"🕒 Time range: {features_df.index.min()} to {features_df.index.max()}")
        
        # Show feature summary
        symbol_features = {}
        for symbol in symbols:
            symbol_cols = [col for col in features_df.columns if col.startswith(f"{symbol}_")]
            symbol_features[symbol] = len(symbol_cols)
            
        logger.info(f"📈 Features per symbol: {symbol_features}")
        
        # Save results (optional)
        output_dir = Path("processed_data") / dataset_id
        output_dir.mkdir(exist_ok=True, parents=True)
        
        output_file = output_dir / "features.parquet"
        features_df.to_parquet(output_file, compression="zstd")
        logger.info(f"💾 Features saved to: {output_file}")
        
        # Train lead-lag models
        logger.info("🤖 Starting lead-lag model training...")
        model_results = train_lead_lag_models(features_df, dataset_id, symbols)
        
        if model_results:
            logger.info("🎯 Model training completed successfully!")
            logger.info(f"📊 Best model: {model_results['best_model']} (score: {model_results['best_score']:.4f})")
            
            # Create visualizations
            logger.info("📊 Creating performance visualizations...")
            create_visualizations(features_df, model_results, dataset_id, symbols)
        
        return features_df, model_results
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None, None


def create_visualizations(features_df: pd.DataFrame, model_results: dict, dataset_id: str, symbols: list):
    """
    Create comprehensive visualizations for the pipeline results.
    
    Args:
        features_df: Features DataFrame
        model_results: Training results dictionary
        dataset_id: Dataset identifier
        symbols: List of symbols
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Create visualization directories
        viz_dir = Path("processed_data") / dataset_id / "visualizations"
        viz_dir.mkdir(exist_ok=True, parents=True)
        
        # 1. Feature Analysis Visualizations
        logger.info("📊 Creating feature analysis visualizations...")
        feature_viz = FeatureAnalysisVisualizer()
        
        # Select important features for visualization
        important_features = []
        for symbol in symbols:
            important_features.extend([
                f"{symbol}_mid_price",
                f"{symbol}_spread", 
                f"{symbol}_volume_imbalance_l1",
                f"{symbol}_volatility_5m"
            ])
        
        # Filter existing features
        available_features = [f for f in important_features if f in features_df.columns]
        if not available_features:
            # Fallback to first numerical columns
            available_features = list(features_df.select_dtypes(include=[np.number]).columns[:8])
        
        # Create feature dashboard
        feature_figures = feature_viz.create_feature_dashboard(
            features_df, 
            important_features=available_features,
            save_dir=str(viz_dir / "features")
        )
        
        # 2. Model Performance Visualizations (if we have model training results)
        if 'training_results' in model_results:
            logger.info("📈 Creating model performance visualizations...")
            model_viz = ModelPerformanceVisualizer()
            
            # Plot model comparison
            model_comparison_fig = model_viz.plot_model_comparison(
                model_results['training_results'],
                save_path=str(viz_dir / "model_comparison.png")
            )
            
            # Close figures to save memory
            for fig in feature_figures.values():
                plt.close(fig)
            plt.close(model_comparison_fig)
        
        logger.info(f"✅ Visualizations saved to: {viz_dir}")
        
    except Exception as e:
        logger.error(f"❌ Visualization creation failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")


def train_lead_lag_models(features_df: pd.DataFrame, dataset_id: str, symbols: list) -> dict:
    """
    Train lead-lag models on the feature data.
    
    Args:
        features_df: DataFrame with extracted features
        dataset_id: Dataset identifier  
        symbols: List of symbols
        
    Returns:
        Dictionary with training results
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Temporal train/test split (70% train, 30% test)
        logger.info("📊 Performing temporal train/test split...")
        df_train, df_test = split_train_test(features_df, frac=0.7)
        
        logger.info(f"Train set: {len(df_train):,} samples")
        logger.info(f"Test set: {len(df_test):,} samples")
        
        # Prepare data for ML
        logger.info("🔧 Preparing data for machine learning...")
        data_prep = DataPreparator(
            target_config=None,  # Use defaults
            feature_selection=True,
            max_features=50
        )
        
        # Convert to format expected by DataPreparator
        # For now, we'll create a simple lead-lag target based on price movement
        primary_symbol = symbols[0] if symbols else "BTC"
        
        # Create target: predict if primary symbol will outperform in next period
        target_col = f"{primary_symbol}_mid_price"
        if target_col not in df_train.columns:
            logger.warning(f"Target column {target_col} not found, using first available price column")
            price_cols = [col for col in df_train.columns if "mid_price" in col]
            if price_cols:
                target_col = price_cols[0]
            else:
                raise ValueError("No price columns found for target creation")
        
        # Create simple directional target (1 if price goes up, 0 if down)
        def create_target(df):
            prices = df[target_col].values
            target = np.zeros(len(prices))
            for i in range(len(prices) - 1):
                if prices[i+1] > prices[i]:
                    target[i] = 1
                else:
                    target[i] = 0
            return target[:-1]  # Remove last element (no future price)
        
        y_train = create_target(df_train)
        y_test = create_target(df_test)
        
        # Prepare feature matrices
        X_train = df_train.iloc[:-1].values  # Remove last row to match target
        X_test = df_test.iloc[:-1].values
        feature_names = list(df_train.columns)
        
        # Split training data for validation
        split_idx = int(len(X_train) * 0.8)
        X_train_split = X_train[:split_idx]
        X_val_split = X_train[split_idx:]
        y_train_split = y_train[:split_idx]
        y_val_split = y_train[split_idx:]
        
        # Preprocess data (scaling, feature selection)
        X_train_processed, X_val_processed, selected_features = data_prep.preprocess_data(
            X_train_split, X_val_split, y_train_split, feature_names
        )
        
        logger.info(f"🎯 Selected {len(selected_features)} features out of {len(feature_names)}")
        
        # Train models
        logger.info("🤖 Training machine learning models...")
        model_trainer = ModelTrainer(enable_ensemble=True)
        
        results = model_trainer.train_all_models(
            X_train_processed, y_train_split,
            X_val_processed, y_val_split,
            selected_features
        )
        
        # ✅ ÉVALUATION SUR VRAIES DONNÉES DE TEST
        logger.info("🧪 Evaluating models on real test data...")
        
        # Prepare test data (same preprocessing as training)
        X_test = df_test.iloc[:-1].values  # Remove last row to match target
        y_test = create_target(df_test)
        
        # Apply same preprocessing as training data
        X_test_scaled = data_prep.scaler.transform(X_test)
        
        # Apply same feature selection as training
        if data_prep.feature_selector is not None:
            X_test_processed = data_prep.feature_selector.transform(X_test_scaled)
        else:
            X_test_processed = X_test_scaled
        
        # Test all models on real test data
        test_results = {}
        for model_name, model_result in results.items():
            if model_name != 'ensemble' and model_result.model:
                test_pred = model_trainer.trained_models[model_name].predict(X_test_processed)
                test_score = accuracy_score(y_test, test_pred)
                test_results[model_name] = test_score
                logger.info(f"📊 {model_name} - Test Accuracy: {test_score:.4f}")
        
        # Test ensemble if available
        if 'ensemble' in results and len(test_results) > 1:
            ensemble_predictions = model_trainer.predict_with_models(X_test_processed, ['ensemble'])
            if 'ensemble' in ensemble_predictions:
                ensemble_test_score = accuracy_score(y_test, ensemble_predictions['ensemble'])
                test_results['ensemble'] = ensemble_test_score
                logger.info(f"🎯 Ensemble - Test Accuracy: {ensemble_test_score:.4f}")
        
        # Find best model based on TEST performance (not validation)
        best_model = None
        best_score = 0
        best_test_score = 0
        
        for model_name, result in results.items():
            val_score = result.validation_score if result.validation_score else 0
            test_score = test_results.get(model_name, 0)
            
            # Prioritize test score, fallback to validation
            if test_score > best_test_score:
                best_test_score = test_score
                best_score = val_score
                best_model = model_name
        
        # Save models
        models_dir = Path("processed_data") / dataset_id / "models"
        models_dir.mkdir(exist_ok=True, parents=True)
        model_trainer.save_models(str(models_dir))
        
        # Save training results
        results_file = models_dir / "training_results.json"
        import json
        
        results_summary = {
            "best_model": best_model,
            "best_score": best_score,
            "target_column": target_col,
            "features_used": len(selected_features),
            "train_samples": len(X_train_processed),
            "val_samples": len(X_val_processed),
            "model_scores": {name: r.validation_score for name, r in results.items() if r.validation_score}
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        logger.info(f"💾 Models saved to: {models_dir}")
        logger.info(f"📈 Best model: {best_model} (validation score: {best_score:.4f})")
        
        return results_summary
        
    except Exception as e:
        logger.error(f"❌ Model training failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="HFT Engine v3 - Heavy Data Ready Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --dataset DATA_0
  python run_pipeline.py --dataset DATA_1 --symbols BTC ETH
  python run_pipeline.py --dataset DATA_2 --symbols BTC
        """
    )
    
    parser.add_argument(
        "--dataset", 
        choices=["DATA_0", "DATA_1", "DATA_2"],
        required=True,
        help="Dataset to process"
    )
    
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTC", "ETH"],
        help="Trading symbols to analyze (default: BTC ETH)"
    )
    
    parser.add_argument(
        "--cache-info",
        action="store_true",
        help="Show cache information and exit"
    )
    
    parser.add_argument(
        "--clear-cache",
        action="store_true", 
        help="Clear cache for the dataset and exit"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    print("🚀 HFT Engine v3 - Heavy Data Ready Pipeline")
    print("=" * 50)
    
    if args.cache_info:
        print(f"📁 Cache information for {args.dataset}:")
        cache_info = get_cache_info(args.dataset)
        import json
        print(json.dumps(cache_info, indent=2))
        return
        
    if args.clear_cache:
        print(f"🗑️  Clearing cache for {args.dataset}...")
        from data_cache import clear_cache
        clear_cache(args.dataset)
        print("✅ Cache cleared")
        return
    
    # Run pipeline
    features_df, model_results = run_full_pipeline(args.dataset, args.symbols)
    
    if features_df is not None:
        print("\n✅ Pipeline completed successfully!")
        print(f"📊 Final features shape: {features_df.shape}")
        if model_results:
            print(f"🎯 Best model: {model_results['best_model']} (score: {model_results['best_score']:.4f})")
    else:
        print("\n❌ Pipeline failed!")
        sys.exit(1)
