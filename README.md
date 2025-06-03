# Crypto High-Frequency Trading Strategy

## Overview
This project implements a high-frequency trading strategy for cryptocurrency order books, specifically analyzing lead-lag relationships between ETH/EUR and XBT/EUR markets.

## Project Structure
```
crypto_hft_trading/
├── data_processing/
│   ├── data_loader.py          # Raw data loading and preprocessing
│   ├── feature_extractor.py    # Order book feature extraction
│   └── synchronizer.py         # Asynchronous data synchronization
├── models/
│   ├── ml_models.py            # Machine learning models
│   ├── model_selection.py      # Model selection and hyperparameter tuning
│   └── predictor.py            # Real-time prediction interface
├── strategy/
│   ├── signal_generator.py     # Trading signal generation
│   ├── risk_manager.py         # Risk management
│   └── portfolio_manager.py    # Portfolio and position management
├── utils/
│   ├── metrics.py              # Performance metrics
│   └── visualization.py       # Plotting and analysis tools
├── config/
│   └── config.yaml             # Configuration parameters
└── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_feature_engineering.ipynb
    ├── 03_model_development.ipynb
    └── 04_strategy_backtesting.ipynb
```

## Key Features
- Asynchronous order book data processing
- Lead-lag signal detection between cryptocurrencies
- Machine learning-based prediction models
- Real-time trading signal generation
- Risk management and portfolio optimization

## Methodology
1. **Data Processing**: Handle asynchronous order book data with microsecond precision
2. **Feature Engineering**: Extract order book imbalance, depth changes, and cross-market signals
3. **Model Training**: Use ensemble methods for prediction of sub-second price movements
4. **Strategy Implementation**: Deploy automated trading signals based on model predictions

## Installation

1. Clonez ce dépôt :
```bash
git clone https://github.com/<votre-utilisateur>/crypto_hft_trading.git
cd crypto_hft_trading
```
2. Installez les dépendances Python :
```bash
pip install -r requirements.txt
```

## Reproductibilité
Pour garantir la reproductibilité des résultats :
- Fixez le seed global dans chaque notebook :
```python
import numpy as np
import random
np.random.seed(42)
random.seed(42)
```
- Pour scikit-learn, XGBoost, LightGBM, ajoutez `random_state=42` partout.

## Exécution
- Lancez les notebooks dans l'ordre :
  1. `01_data_exploration.ipynb`
  2. `02_feature_engineering.ipynb`
  3. `03_model_development.ipynb`

- Les données doivent être placées dans le dossier `data/`.
