# 📈 AlphaGrid: Spillover Forecasting Using Temporal Graph Neural Networks

This project implements a deep learning system to model and forecast **cross-sector volatility spillovers** in financial markets. Using a dynamic edge-centric graph representation of sector relationships and various GNN architectures, I aimed to outperform traditional volatility modeling methods (e.g., GARCH(1, 1)).

## 🎯 Problem Statement

- **Objective**: Predict future changes in realized volatility (ΔRV) for a given sector, based on independent 11 GICS sectors
- **Challenge**: Volatility is not isolated — shocks in one sector (e.g., Energy) can ripple through others (e.g., Industrials)
- **Solution**: Use a **fully connected, directed graph** where:
  - **Nodes** = sectors
  - **Edges** = directional influence relationships
  - **Edge features** = learned representations of sector-pair interactions
  - **Target** = next-step ΔRV (realized volatility change) for a given sector

---

## 🎯 VolatilitySpikeLoss Function

Custom loss function combining three components:

- **Asymmetric Error Penalties**: τ² weight for under-prediction, τ weight for over-prediction
- **Volatility Spike Weighting**: 10x penalty for high volatility periods (>threshold), 5x for normal periods
- **Directional Accuracy**: Additional penalty when predicted and true values have opposite signs
```python
class VolatilitySpikeLoss(nn.Module):
    def __init__(self, tau=2, spike_threshold=1.0, direction_weight=2):
        # Custom loss implementation
```

---

## 🧠 Methodology

### 🧱 Graph Construction

- **Nodes**: 11 GICS sectors (Technology, Healthcare, Financials, etc.)
- **Edges**: Fully connected graph (121 total edges)

### 🔧 Features

- **Node features**: Aggregated market features per sector
- **Edge features**: Engineered pairwise interactions between sectors
- **Temporal windows**: Rolling windows to create sequences of temporal graphs

### 🧠 Models Implemented

- `TemporalDenseGNN`: MLP-style model over temporal edge embeddings
- `TemporalEdgeGNN`: Custom GNN with edge-updating mechanism
- **Ablation Study**: Systematic comparison of LSTM, GRU, and MLP temporal components

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- pandas, numpy, matplotlib, tqdm

### Installation
```sh
git clone https://github.com/Retrospek/AlphaGrid
cd AlphaGrid
pip install -r requirements.txt
```

---

## 📊 Results

### Performance Comparison

| Model | VolatilitySpikeLoss | Architecture |
|-------|---------------------|--------------|
| **AlphaGrid GNN** | 3.29 | Graph + Temporal |
| GARCH(1,1) Baseline | 3.36 | Traditional Econometric |

### Key Findings

- **Temporal mechanisms**: Ablation study found GRU component superior for financial time series
- **Cross-sector spillovers**: Successfully captured dynamic correlation patterns through learned edge embeddings

---

## 📏 Evaluation Metrics

- **Primary**: Custom VolatilitySpikeLoss (combines magnitude + directional accuracy)
- **Secondary**: Mean Squared Error (MSE), Mean Absolute Error (MAE)
- **Directional**: Sign accuracy for change prediction
- **Validation**: Walk-forward validation on out-of-sample data

---

## 🔬 Implementation Details

### Data Pipeline

1. **ETF Data Collection**: 11 sector ETFs with daily OHLCV data
2. **Volatility Calculation**: Realized volatility using high-frequency returns
3. **Feature Engineering**: Technical indicators, cross-sector correlations
4. **Graph Construction**: Dynamic correlation matrices → adjacency tensors

### Training Procedure

- **Optimizer**: Adam with learning rate 3.25e-5
- **Batch Size**: 32
- **Sequence Length**: Variable temporal window
- **Regularization**: Custom loss weighting + dropout

---

## 🎨 Customization

Edit the `SECTOR_MAPPING` in `frontend/index.html`:
```javascript
const SECTOR_MAPPING = {
    'XLK': 'Technology',
    'XLV': 'Healthcare',
    // Add your sectors here
};
```

---

## 🚀 Future Work

- [ ] Incorporate options market data for volatility surface modeling
- [ ] Extend to international sector ETFs for global spillover analysis
- [ ] Real-time deployment with streaming market data
- [ ] Attention mechanisms for interpretable sector influence weights

---

## 📧 Contact

**Arjun Mahableshwarkar**  
📧 arjun.mahableshwarkar@gmail.com  
🐙 [GitHub](https://github.com/Retrospek)  
💼 [LinkedIn](https://linkedin.com/in/arjun-mahableshwarkar)

---

*Built with PyTorch, Pandas, and a curiosity for quantitative finance* 🚀
