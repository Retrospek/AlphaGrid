# 📈 Volatility Spillover Forecasting using Temporal Graph Neural Networks

This project implements a deep learning system to model and forecast **cross-sector volatility spillovers** in financial markets. Using a dynamic graph representation of sector relationships and various GNN architectures, we aim to outperform traditional volatility modeling methods (e.g., GARCH, DCC).

## 🔍 Problem Statement

- **Objective**: Predict future changes in realized volatility (ΔRV) for a given sector, based on interdependencies among all 11 GICS sectors.
- **Challenge**: Volatility is not isolated — shocks in one sector (e.g., Energy) can ripple through others (e.g., Industrials).
- **Solution**: Use a **fully connected, directed graph** where:
  - Nodes = sectors
  - Edges = directional influence
  - Edge features = learned representations of sector-pair interactions
  - Target = next-step %Δ in realized volatility for a given sector

---

## 🧠 Methodology

### 🧱 Graph Construction

- Nodes: 11 GICS sectors
- Edges: Fully connected (directed), representing influence from sector A → sector B
- Features:
  - **Node features**: Aggregated market features per sector
  - **Edge features**: Engineered pairwise interactions between sectors
  - **Temporal**: Time windows used to create sequences of graphs

### 🧠 Models Implemented

- `TemporalDenseGNN`: MLP-style model over temporal edge embeddings
- `TemporalEdgeGNN`: Custom GNN that updates edges over time

### 📊 Evaluation

- Metric: Mean Squared Error (MSE), Sharpe Ratio, Directional Accuracy
- Backtesting over walk-forward windows
- Benchmarked against traditional econometric models
