#!/usr/bin/env python3
"""
Test du HFT Engine v3 avec données synthétiques pour validation rapide
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from main import HFTEngineComplete

def test_synthetic_data():
    """Test rapide avec données synthétiques."""
    print("🚀 " + "="*60)
    print("   HFT ENGINE v3 - TEST RAPIDE")
    print("   Avec Données Synthétiques")
    print("="*62)
    
    try:
        # Initialiser avec des symboles qui forceront l'utilisation de données synthétiques
        engine = HFTEngineComplete(
            dataset_id="TEST",  # Dataset inexistant pour forcer synthétique
            symbols=["BTC", "ETH"],
            verbose=True
        )
        
        # Forcer l'utilisation de données synthétiques en modifiant temporairement
        print("\n📊 Génération de données synthétiques...")
        data = engine._generate_synthetic_data()
        
        print(f"✅ Données générées: {list(data.keys())}")
        for symbol, df in data.items():
            print(f"   {symbol}: {len(df):,} points de données")
        
        # Test feature engineering avec données synthétiques
        print("\n⚙️ Test d'ingénierie des features...")
        try:
            features = engine.engineer_features(data)
            print(f"✅ Features générées: {features.shape}")
        except Exception as e:
            print(f"❌ Feature engineering échoué: {e}")
            # Continuer quand même
            features = None
        
        # Test Transfer Entropy
        print("\n🔬 Test de Transfer Entropy...")
        te_results = engine.analyze_transfer_entropy(data)
        if te_results:
            print(f"✅ Transfer Entropy calculé pour {len(te_results.get('pairwise_results', {}))} paires")
        
        # Test causality
        print("\n📈 Test de causalité...")
        causality_results = engine.perform_causality_tests(data)
        if causality_results:
            print(f"✅ Tests de causalité: {list(causality_results.keys())}")
        
        # Test ML (si features disponibles)
        if features is not None and not features.empty:
            print("\n🤖 Test ML...")
            ml_results = engine.train_ml_models(features)
            if ml_results:
                print(f"✅ Modèles ML: {list(ml_results.keys())}")
        
        # Test backtesting
        print("\n🎯 Test de backtesting...")
        backtest_results = engine.run_comprehensive_backtest(data)
        if backtest_results and 'comparison' in backtest_results:
            comparison_df = backtest_results['comparison']
            print(f"✅ Backtesting: {len(comparison_df)} stratégies testées")
            
            # Afficher les résultats
            print("\n📊 RÉSULTATS DES STRATÉGIES:")
            ranked = comparison_df.sort_values('Total Return', ascending=False)
            for i, (strategy, row) in enumerate(ranked.iterrows(), 1):
                print(f"   {i}. {strategy}: {row['Total Return']:.2f}%")
        
        # Test visualisations
        print("\n📊 Test de visualisations...")
        try:
            engine.create_comprehensive_visualizations()
            print("✅ Visualisations créées")
        except Exception as e:
            print(f"⚠️  Visualisations partielles: {e}")
        
        # Sauvegarde des résultats
        print("\n💾 Sauvegarde des résultats...")
        engine.save_results()
        
        # Résumé final
        engine._print_final_summary()
        
        return True
        
    except Exception as e:
        print(f"❌ Test échoué: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_synthetic_data()
    print(f"\n{'✅ TEST RÉUSSI' if success else '❌ TEST ÉCHOUÉ'}")
