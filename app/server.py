from fastapi import FastAPI
import torch
from torch.utils.data import DataLoader

import numpy as np

import yfinance as yf

import datetime as dt

from Models.Architectures.edgeGNN import EdgeCentricNetwork
from DataManagement.data import TemporalFinancialDataset

# MODEL DEFINITION

device = 'cuda' if torch.cuda.is_available() else 'cpu'

static = False
temporal = True

num_sectors = 11
node_dim = 89
window = 30
feature_time_shift = 30

model = EdgeCentricNetwork(
    num_nodes=num_sectors,
    edge_dim=16,
    node_dim=node_dim,
    static=static,
    temporal=temporal,
    window=window,
    complex=False)

model_state_dict = "app/GRUTemporalEdgeCentricGNN_v2_state.pth"
model.load_state_dict(torch.load(model_state_dict, map_location=device))
model.to(device)
model.eval()

# Data DEFINITION
sector_mapping = {
            'XLK': 'Technology',
            'XLF': 'Financials', 
            'XLE': 'Energy',
            'XLV': 'Health Care',
            'XLI': 'Industrials',
            'XLP': 'Consumer Staples',
            'XLY': 'Consumer Discretionary',
            'XLU': 'Utilities',
            'XLB': 'Materials',
            'XLRE': 'Real Estate',
            'XLC': 'Communication Services'
        }

batch_size = 1 # Because we are doing inference for most current day only

today_date = dt.date.today()
look_back_date = today_date - dt.timedelta(days=365)

data = TemporalFinancialDataset(window_size=window, start_day=look_back_date)
#dataLoader = DataLoader(dataset=data, batch_size=batch_size, shuffle=False)

# APP Definition

app = FastAPI()

@app.get('/')
def home():
    return {'HeadLiner': "Cross-Sector Volatility Spillover Model"}

@app.post('/predict')
def predict():
    predicted_deltas = {}

    # Get the most recent window (last item in dataset)
    features = data.__getitem__(-1, inference=True).unsqueeze(0)
    print(f"Feature Shape: {features.shape}")

    #reshape into predicted inference shape
    with torch.no_grad():
        output = model(features)
        output = output.squeeze(0)
        output = output.squeeze(-1)
    # Map predictions to sectors
    for i, key in enumerate(sector_mapping.keys()):
        predicted_deltas[key] = output[i].item()

    return {"Predicted Deltas": predicted_deltas}