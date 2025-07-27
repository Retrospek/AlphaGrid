# 📈 AlphaGrid: Volatility Spillover Forecasting using Temporal Graph Neural Networks

This project implements a deep learning system to model and forecast **cross-sector volatility spillovers** in financial markets. Using a dynamic edge-centric graph representation of sector relationships and various GNN architectures, I aimed to outperform traditional volatility modeling methods (e.g., GARCH(1, 1)).

## 🔍 Problem Statement

- **Objective**: Predict future changes in realized volatility (ΔRV) for a given sector, based on interdependencies among all 11 GICS sectors.
- **Challenge**: Volatility is not isolated — shocks in one sector (e.g., Energy) can ripple through others (e.g., Industrials).
- **Solution**: Use a **fully connected, directed graph** where:
 - Nodes = sectors
 - Edges = directional influence
 - Edge features = learned representations of sector-pair interactions
 - Target = next-step %Δ in realized volatility for a given sector

---

## 🏗️ Architecture Details

### VolatilitySpikeLoss Function
Custom loss function combining three components:
- **Asymmetric Error Penalties**: τ² weight for under-prediction, τ weight for under-prediction
- **Volatility Spike Weighting**: 10x penalty for high volatility periods (>threshold), 5x for normal periods
- **Directional Accuracy**: Additional penalty when prediction and true value have opposite signs

\```python
class VolatilitySpikeLoss(nn.Module):
   def __init__(self, tau=2, spike_threshold=1.0, direction_weight=2):
       # Custom loss implementation
\```

### Model Architecture
- **Input Shape**: `[batch_size, sequence_length, 11_sectors, features]`
- **GNN Layers**: Edge-updating mechanism with learned embeddings for 121 cross-sector relationships
- **Temporal Component**: LSTM/GRU/MLP variants for time-series processing
- **Output**: 11-dimensional volatility change predictions

---

## 🧠 Methodology

### 🧱 Graph Construction
- **Nodes**: 11 GICS sectors (Technology, Healthcare, Financials, etc.)
- **Edges**: Fully connected directed graph (121 total edges)
- **Features**:
 - **Node features**: Aggregated market features per sector
 - **Edge features**: Engineered pairwise interactions between sectors
 - **Temporal**: Rolling windows to create sequences of dynamic graphs

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
\```bash
git clone https://github.com/Retrospek/AlphaGrid
cd AlphaGrid
pip install -r requirements.txt
\```

---

## 📊 Results

### Performance Comparison
| Model | VolatilitySpikeLoss| Architecture
|-------|--------------------|---------------------
| **AlphaGrid GNN** | 3.29 | Graph + Temporal
| GARCH(1,1) Baseline | 3.36 | Traditional Econometric

### Key Findings
- **Temporal mechanisms**: Ablation study proved GRU components superior for financial time series
- **Cross-sector spillovers**: Successfully captured dynamic correlation patterns through learned edge embeddings

---

## 📊 Evaluation Metrics

- **Primary**: Custom VolatilitySpikeLoss (combines magnitude + directional accuracy)
- **Secondary**: Mean Squared Error (MSE), Mean Absolute Error (MAE)
- **Directional**: Sign accuracy for volatility change predictions
- **Backtesting**: Walk-forward validation on out-of-sample data

---

## 🔬 Technical Implementation

### Data Pipeline
1. **ETF Data Collection**: 11 sector ETFs with daily OHLCV data
2. **Volatility Calculation**: Realized volatility using high-frequency returns
3. **Feature Engineering**: Technical indicators, cross-sector correlations
4. **Graph Construction**: Dynamic correlation matrices → adjacency tensors

### Training Procedure
- **Optimizer**: Adam with learning rate 3.25e-5
- **Batch Size**: 32 sequences
- **Sequence Length**: Variable temporal windows
- **Regularization**: Custom loss weighting + dropout

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

*Built with PyTorch, Pandas, and a curiousity for quantitative finance* 🚀
