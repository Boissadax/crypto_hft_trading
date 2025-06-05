# 🚀 Team Testing Guide - Crypto HFT Strategy

## 👥 Team Distribution (6 Members)

### 🎯 Strategy Overview
Chaque membre teste une variation différente de la stratégie sur les **mêmes données** pour évaluer l'impact des paramètres.

### 📋 Assignments

| Member | Focus | Strategy Type | Key Parameters |
|--------|-------|---------------|----------------|
| **1** | Conservative HFT | Low risk, high confidence | `signal_threshold: 0.7`, `position_size: 0.05` |
| **2** | Aggressive Multi-Pos | High frequency, multiple positions | `signal_threshold: 0.3`, `max_positions: 3` |
| **3** | Mean Reversion | Capture price reversions | Long signals, `signal_decay_ms: 10000` |
| **4** | Cost Sensitivity | Impact of transaction costs | Double fees, high slippage |
| **5** | Temporal Split | Different train/test ratios | `in_sample: 0.6`, `out_sample: 0.2` |
| **6** | Stress Test | Full dataset performance | No record limit, maximum data |

## 🏃‍♂️ How to Run

### Individual Test
```bash
# Each member runs their specific test
python team_runner.py --member 1  # For member 1
python team_runner.py --member 2  # For member 2
# ... etc
```

### All Tests at Once
```bash
# Run all 6 tests sequentially
python team_runner.py --all
```

### Compare Results
```bash
# Compare all team results
python team_runner.py --compare
```

## 📊 Expected Outcomes

### 🎯 Key Metrics to Compare:
- **Net Return** after transaction costs
- **Sharpe Ratio** (risk-adjusted performance)
- **Maximum Drawdown** (risk measure)
- **Number of Trades** (activity level)
- **Transaction Costs** (cost impact)

### 🧪 Hypotheses to Test:
1. **Conservative vs Aggressive**: Lower risk = lower return but better Sharpe?
2. **Cost Impact**: How much do transaction costs reduce alpha?
3. **Temporal Sensitivity**: How robust are the patterns over time?
4. **Scalability**: Does performance hold with larger datasets?

## 📈 Data Insights

### Current System Stats:
- **Data Reduction**: ~99% (intelligent, not random)
- **Timestamp Preservation**: Microsecond precision maintained
- **Processing Speed**: 55ms between events
- **Signal Quality**: 40% average confidence

### What Each Member Tests:
- **Different risk tolerances**
- **Various signal thresholds**
- **Alternative cost structures**
- **Temporal split sensitivity**
- **Scalability limits**

## 🔬 Scientific Approach

### Same Data, Different Parameters:
✅ Controls for data quality  
✅ Isolates parameter impact  
✅ Enables fair comparison  
✅ Tests strategy robustness  

### Parallel Execution Benefits:
- **6x faster** than sequential testing
- **Comprehensive coverage** of parameter space
- **Statistical significance** through multiple approaches
- **Risk assessment** across strategies

## 📁 Results Structure

```
results/
├── team_member_1/
│   └── Conservative_HFT_20250604_143022.yaml
├── team_member_2/
│   └── Aggressive_MultiPos_20250604_143055.yaml
├── team_member_3/
│   └── MeanReversion_Focus_20250604_143128.yaml
├── team_member_4/
│   └── HighCost_Analysis_20250604_143201.yaml
├── team_member_5/
│   └── TempSplit_Analysis_20250604_143234.yaml
└── team_member_6/
    └── MaxDataset_Stress_20250604_143307.yaml
```

## 🏆 Success Criteria

### Individual Success:
- **Positive net alpha** after costs
- **Sharpe ratio > 1.0**
- **Max drawdown < 15%**
- **Stable performance** across splits

### Team Success:
- **Identify best strategy** through comparison
- **Understand parameter sensitivity**
- **Validate strategy robustness**
- **Optimize for production deployment**

## ⚡ Performance Tips

### For Fast Execution:
1. Start with `max_records: 50000` for initial tests
2. Use `time_window_ms: 100` for balanced performance
3. Monitor memory usage during full dataset tests
4. Run tests during off-peak hours for better performance

### Troubleshooting:
- If memory issues: reduce `max_records`
- If too slow: increase `time_window_ms`
- If no signals: lower `signal_threshold`
- If too many trades: increase `signal_threshold`

---

**🎯 Goal**: Find the optimal strategy configuration through systematic parameter exploration across 6 team members using the same high-quality dataset.
