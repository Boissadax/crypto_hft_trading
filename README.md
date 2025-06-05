# HFT Engine v3 - Transfer Entropy Based Trading System

## Overview

This is a completely refactored High-Frequency Trading (HFT) engine focused on **Transfer Entropy-based lead-lag analysis** for cryptocurrency markets. The system uses advanced statistical methods and machine learning to identify and exploit causal relationships between cryptocurrency pairs.

## ✨ Fonctionnalités

- **🔥 Signaux Sub-Seconde**: Détection de patterns lead-lag entre cryptos
- **⚡ Traitement Optimisé**: Streaming de données sans limite mémoire
- **💾 Cache Intelligent**: Conversion CSV→Parquet avec détection de changements
- **📊 Analytics Avancés**: Métriques de performance professionnelles
- **🎯 Stratégie Lead-Lag**: Algorithme optimisé pour corrélations crypto

## 🏗️ Architecture Simplifiée

```
📊 Données CSV → 🔄 Moteur HFT → 📈 Stratégie Lead-Lag → 📊 Résultats
```

**Composants Essentiels:**
- **OptimizedTradingEngine**: Moteur de trading principal
- **OptimizedLeadLagStrategy**: Stratégie de corrélation crypto
- **OptimizedDataHandler**: Gestion optimisée des données
- **PerformanceTracker**: Suivi des performances

## 🚀 Utilisation Rapide

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Lancement du Notebook

```bash
jupyter notebook HFT_Algorithm_Analysis.ipynb
```

### 3. Exécution Simple

```bash
python simple_main.py
```

## 📊 Format des Données

Vos fichiers CSV doivent contenir les colonnes suivantes:

```csv
timestamp,price,volume,side,level
2024-01-01 09:00:00.000,50000.50,1.5,bid,1
2024-01-01 09:00:00.000,50000.75,2.0,ask,1
```

**Colonnes requises:**
- `timestamp`: Horodatage des événements
- `price`: Prix de l'ordre (float)  
- `volume`: Volume de l'ordre (float)
- `side`: 'bid' ou 'ask'
- `level`: Niveau du carnet (1=meilleur, 2=second, etc.)

## 📁 Structure du Projet

```
hft_engine_v3/
├── HFT_Algorithm_Analysis.ipynb    # 📓 Notebook principal
├── simple_main.py                  # 🚀 Point d'entrée simple
├── core/                          # 🏗️ Moteur de trading
├── data/                          # 📊 Gestion des données  
├── strategies/                    # 🎯 Stratégies de trading
├── portfolio/                     # 💰 Gestion du portefeuille
├── execution/                     # ⚡ Exécution des ordres
├── events/                        # 📡 Système d'événements
├── utils/                         # 🔧 Utilitaires
└── raw_data/                      # 📂 Données brutes
```

## 🎯 Utilisation du Notebook

Le notebook `HFT_Algorithm_Analysis.ipynb` permet une analyse interactive complète:

1. **Configuration**: Paramètres de l'algorithme
2. **Préparation**: Cache optimisé des données
3. **Backtest**: Exécution de la stratégie lead-lag
4. **Analyse**: Métriques de performance détaillées
5. **Visualisation**: Graphiques des résultats
6. **Recommandations**: Optimisations suggérées

## 🎯 Paramètres de la Stratégie Lead-Lag

```python
strategy = OptimizedLeadLagStrategy(
    symbols=['ETH', 'BTC'],
    lookback_window_ms=1000,     # Fenêtre d'analyse (1 seconde)
    signal_threshold=0.6,        # Seuil de signal (60%)
    max_position_size=0.1,       # Taille max de position (10%)
    min_spread_threshold=0.001   # Spread minimum (0.1%)
)
```

## 📊 Résultats Exemple

```
🏆 RÉSULTATS LEAD-LAG CRYPTO
============================================================
⏱️  Temps d'exécution: 45.67 secondes
📊 Événements traités: 2,500,000 à 54,742 evt/sec

📈 Performance de la Stratégie:
----------------------------------------
Rendement total:     +15.42%
Ratio de Sharpe:     1.234
Drawdown maximum:    -3.21%
Nombre de trades:    1,245

🥇 Stratégie Lead-Lag optimisée pour corrélations crypto
```

## ⚙️ Optimisation

### Configuration Haute Performance

```python
config = OptimizedEngineConfig(
    batch_size=10000,           # Batches plus larges
    enable_logging=False,       # Désactiver logs
    chunk_size=50000           # Chunks plus grands
)
```

## 🔧 Support

Pour toute question sur l'algorithme, consultez le notebook interactif ou les commentaires dans le code.

---

**Algorithme Lead-Lag Optimisé pour Trading Crypto** 🚀

## Key Features

### 🔬 Statistical Analysis
- **Transfer Entropy Analysis**: Comprehensive TE calculation with multiple methods
- **Causality Testing**: Granger causality, VAR causality, and nonlinear causality tests
- **Cross-Correlation Analysis**: Time-domain, frequency-domain, and dynamic correlations
- **Regime Detection**: HMM-based regime detection and structural break analysis

### 🤖 Machine Learning Integration
- **Data Preparation**: Temporal data splitting with no look-ahead bias
- **Feature Engineering**: Causality-based feature creation from synchronized data
- **Model Training**: Ensemble methods with Transfer Entropy features
- **Model Validation**: Comprehensive backtesting with statistical significance

### 📊 Benchmarking Framework
- **Performance Metrics**: Risk-adjusted returns (Sharpe, Sortino, Calmar)
- **Baseline Strategies**: Buy & Hold, Random, Momentum, Mean Reversion
- **Backtesting Engine**: Event-driven architecture with transaction costs
- **Statistical Testing**: Significance tests and performance comparison

### 🎯 Trading Strategy
- **Transfer Entropy Strategy**: Main strategy using lead-lag relationships
- **Multi-Symbol Support**: Simultaneous analysis of multiple cryptocurrency pairs
- **Regime Awareness**: Adaptive behavior based on market regime detection
- **Risk Management**: Position sizing and portfolio management

## Project Structure

```
hft_engine_v3/
├── statistical_analysis/          # Statistical methods and analysis
│   ├── transfer_entropy.py       # Transfer Entropy calculations
│   ├── causality_tests.py        # Granger and VAR causality tests
│   ├── correlation_analysis.py   # Cross-correlation analysis
│   └── regime_detection.py       # Market regime detection
├── feature_engineering/           # Data processing and feature creation
│   ├── order_book_features.py    # Order book feature extraction
│   ├── synchronization.py        # Asynchronous data synchronization
│   └── time_series_features.py   # Time series feature engineering
├── learning/                      # Machine learning pipeline
│   ├── data_preparation.py       # Data preparation and splitting
│   └── model_training.py         # Model training and validation
├── benchmark/                     # Benchmarking and performance analysis
│   ├── strategies.py             # Baseline benchmark strategies
│   ├── metrics.py                # Performance metrics calculation
│   └── backtesting.py            # Backtesting engine
├── strategy/                      # Main trading strategies
│   └── transfer_entropy_strategy.py  # Transfer Entropy strategy
├── notebooks/                     # Research and analysis notebooks
├── data_cache/                    # Cached data files
├── raw_data/                      # Raw market data
├── processed_data/                # Processed datasets
├── results/                       # Analysis results and outputs
├── logs/                          # System logs
└── models/                        # Trained ML models
```

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd hft_engine_v3
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the main analysis**:
```bash
python main.py
```

## Usage

### Quick Start

```python
from statistical_analysis import TransferEntropyAnalyzer
from strategy import TransferEntropyStrategy
from benchmark import BacktestEngine

# Initialize components
te_analyzer = TransferEntropyAnalyzer()
strategy = TransferEntropyStrategy(['BTC', 'ETH'])
backtest_engine = BacktestEngine()

# Run analysis
results = backtest_engine.run_backtest(strategy, data)
```

### Research Workflow

1. **Data Analysis**: Start with `notebooks/1_transfer_entropy_analysis.ipynb`
2. **Statistical Validation**: Use `notebooks/2_statistical_validation.ipynb`
3. **Feature Engineering**: Explore `notebooks/3_feature_engineering_analysis.ipynb`
4. **ML Pipeline**: Develop models in `notebooks/4_machine_learning_pipeline.ipynb`
5. **Strategy Development**: Test strategies in `notebooks/5_strategy_development.ipynb`
6. **Full Backtesting**: Complete analysis in `notebooks/6_comprehensive_backtesting.ipynb`

## Key Components

### Transfer Entropy Analysis
The system calculates Transfer Entropy between cryptocurrency pairs to identify lead-lag relationships:

```python
from statistical_analysis import TransferEntropyAnalyzer

analyzer = TransferEntropyAnalyzer()
te_result = analyzer.calculate_transfer_entropy(
    leader_data, follower_data, max_lag=5
)
```

### Statistical Validation
Comprehensive statistical testing ensures the reliability of identified relationships:

```python
from statistical_analysis import CausalityTester

tester = CausalityTester()
causality_result = tester.granger_causality_test(data, max_lag=5)
```

### Machine Learning Integration
The system uses ML models to refine trading signals:

```python
from learning import DataPreparator, ModelTrainer

preparator = DataPreparator()
trainer = ModelTrainer()

# Prepare data with temporal splitting
X, y = preparator.prepare_features_targets(features, targets)

# Train ensemble models
model = trainer.train_model(X, y, model_type='random_forest')
```

### Benchmarking
Comprehensive performance evaluation against baseline strategies:

```python
from benchmark import BacktestEngine, BuyHoldStrategy

engine = BacktestEngine()
baseline = BuyHoldStrategy('BTC')

results = engine.compare_strategies([strategy, baseline], data)
```

## Performance Metrics

The system provides comprehensive performance analysis:

- **Risk-Adjusted Returns**: Sharpe, Sortino, Calmar ratios
- **Drawdown Analysis**: Maximum drawdown and recovery periods
- **Trading Metrics**: Win rate, profit factor, trade frequency
- **Statistical Significance**: T-tests, Jarque-Bera, Ljung-Box tests
- **Causality Metrics**: Transfer Entropy values, p-values, confidence levels

## Configuration

Key parameters can be adjusted in the strategy initialization:

```python
strategy = TransferEntropyStrategy(
    symbols=['BTC', 'ETH', 'LTC'],
    te_threshold=0.1,              # Minimum TE for signals
    confidence_threshold=0.6,       # Minimum confidence for trades
    lookback_window=100,           # Analysis window size
    rebalance_frequency=50,        # TE recalculation frequency
    use_ml=True,                   # Enable ML refinement
    use_regime_detection=True      # Enable regime detection
)
```

## Research Papers and Methods

The system implements methods from cutting-edge research in:

- **Transfer Entropy**: Schreiber (2000), Marschinski & Kantz (2002)
- **Causality Testing**: Granger (1969), Geweke (1982)
- **Regime Detection**: Hamilton (1989), Bai & Perron (2003)
- **High-Frequency Trading**: Aldridge (2013), Narang (2013)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational and research purposes only. Cryptocurrency trading involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results.
