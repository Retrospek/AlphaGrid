import yfinance as yf
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from DataManagement.data_load import (
    get_mappings,
    download_sector_data,
    get_sector_data_separated,
    remove_nans, define_target,
    align_on_common_dates,
    check_download_uniformity
)

from DataManagement.metric_methods import (
    calculate_sector_momentums,
    calculate_sector_moving_averages,
    calculate_sector_relative_strengths,
    calculate_sector_realized_volatility,
)
class StaticFinancialDataset(Dataset):
    def __init__(self):
        super().__init__()

        self.sector_map, self.node_map, self.inv_node_map = get_mappings()
        self.num_sectors = len(self.sector_map.keys())
        self.momentums = [5, 10, 20, 30]
        self.moving_averages = [5, 10, 14, 20, 30]
        self.rsi_periods = [14]
        self.realized_volatility_periods = [5, 10, 20, 30]


        dataset = download_sector_data(sector_mapping=self.sector_map)
        check_download_uniformity(dataset, self.sector_map)
        dataset = get_sector_data_separated(sector_mapping=self.sector_map, original_data=dataset, original_columns=dataset.columns)

        dataset = calculate_sector_momentums(sector_separated_data=dataset, momentums=self.momentums)
        dataset = calculate_sector_moving_averages(sector_separated_data=dataset, moving_averages=self.moving_averages)
        dataset = calculate_sector_relative_strengths(sector_separated_data=dataset, rsi_periods=self.rsi_periods)
        dataset = calculate_sector_realized_volatility(sector_separated_data=dataset, periods=self.realized_volatility_periods)

        dataset = define_target(sector_separated_data=dataset)
        for key in dataset:
            print(f"{key} len: {len(dataset[key])}")


        dataset = remove_nans(sector_separated_data=dataset)
        dataset = align_on_common_dates(sector_separated_data=dataset)

        for key in dataset:
            print(f"{key} len: {len(dataset[key])}")
        
        # Rename Keys from 0 - 10 inclusive
        for key in list(dataset.keys()):
            int_key = self.inv_node_map[key] 
            dataset[int_key] = dataset.pop(key)

        # Each Row will be for one graph, and each "entry" is going to be node feature vectors
        feature_data = [] # Dim: Sector Number x # Dates x Dim_Node
        target_data = []

        removed_features = ["Date", "Sector", "Target"]

        self.columns = [col for col in dataset[0].columns if col not in removed_features]

        for i in range(self.num_sectors):
            vect_feature = dataset[i].drop(columns=removed_features).values
            vect_target = dataset[i]["Target"].values

            feature_data.append(vect_feature)
            target_data.append(vect_target)
        
        feature_data = torch.tensor(data=feature_data, dtype=torch.float32).transpose(0, 1) 
        target_data = torch.tensor(data=target_data, dtype=torch.float32).transpose(0, 1)
        print("Any NaNs in features?", torch.isnan(feature_data).any())
        print("Any Infs in features?", torch.isinf(feature_data).any())

        self.features = feature_data # Dates x Sector Number x Dim_Node
        self.targets = target_data

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
    
class TemporalFinancialDataset(Dataset):
    def __init__(self, window_size):
        super().__init__()

        self.window_size = window_size

        self.sector_map, self.node_map, self.inv_node_map = get_mappings()
        self.num_sectors = len(self.sector_map.keys())
        self.momentums = [5, 10, 20, 30]
        self.moving_averages = [5, 10, 14, 20, 30]
        self.rsi_periods = [14]
        self.realized_volatility_periods = [5, 10, 20, 30]


        dataset = download_sector_data(sector_mapping=self.sector_map)
        check_download_uniformity(dataset, self.sector_map)
        dataset = get_sector_data_separated(sector_mapping=self.sector_map, original_data=dataset, original_columns=dataset.columns)

        dataset = calculate_sector_momentums(sector_separated_data=dataset, momentums=self.momentums)
        dataset = calculate_sector_moving_averages(sector_separated_data=dataset, moving_averages=self.moving_averages)
        dataset = calculate_sector_relative_strengths(sector_separated_data=dataset, rsi_periods=self.rsi_periods)
        dataset = calculate_sector_realized_volatility(sector_separated_data=dataset, periods=self.realized_volatility_periods)

        dataset = define_target(sector_separated_data=dataset)
        for key in dataset:
            print(f"{key} len: {len(dataset[key])}")


        dataset = remove_nans(sector_separated_data=dataset)
        dataset = align_on_common_dates(sector_separated_data=dataset)

        for key in dataset:
            print(f"{key} len: {len(dataset[key])}")
        
        # Rename Keys from 0 - 10 inclusive
        for key in list(dataset.keys()):
            int_key = self.inv_node_map[key] 
            dataset[int_key] = dataset.pop(key)

        # Each Row will be for one graph, and each "entry" is going to be node feature vectors
        feature_data = [] # Dim: Sector Number x # Dates x Dim_Node
        target_data = []

        removed_features = ["Date", "Sector", "Target"]

        self.columns = [col for col in dataset[0].columns if col not in removed_features]

        for i in range(self.num_sectors):
            vect_feature = dataset[i].drop(columns=removed_features).values
            vect_target = dataset[i]["Target"].values

            feature_data.append(vect_feature)
            target_data.append(vect_target)
        
        feature_data = torch.tensor(data=feature_data, dtype=torch.float32).transpose(0, 1) 
        target_data = torch.tensor(data=target_data, dtype=torch.float32).transpose(0, 1)
        print("Any NaNs in features?", torch.isnan(feature_data).any())
        print("Any Infs in features?", torch.isinf(feature_data).any())

        self.features = feature_data # Dates x Sector Number x Dim_Node
        self.targets = target_data

    def __len__(self):
        return self.features.shape[0] - self.window_size + 1
    
    def __getitem__(self, idx):
        features = self.features[idx:idx+self.window_size]
        targets = self.targets[idx:idx+self.window_size]
        return features, targets