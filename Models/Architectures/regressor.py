import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class EdgetoNodeRegressionBlock(nn.Module):
    def __init__(self, num_nodes, edge_dim):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.edge_dim = edge_dim

        self.regression = nn.Sequential(
            nn.Linear(self.num_nodes * self.edge_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.regression(x) # Batch_size x num_nodes x 1