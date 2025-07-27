import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class StaticEdgeUpdater(nn.Module):
    def __init__(self, node_dim, edge_dim):
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim

        self.edge_updater = nn.Sequential(
            nn.Linear(2 * self.node_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.edge_dim)
        )
    
    def forward(self, x):
        #print(x.shape)
        return self.edge_updater(x)
    
class TemporalEdgeUpdater(nn.Module):
    def __init__(self, node_dim, edge_dim, window, num_nodes, complex):
        super().__init__()

        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.window = window

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.dropout_10 = nn.Dropout(0.1)
        self.dropout_20 = nn.Dropout(0.2)

        self.temporal_layers = nn.ModuleList([
            nn.GRU(input_size=2 * self.node_dim, hidden_size=64, batch_first=True),
            nn.GRU(input_size=64, hidden_size=self.edge_dim, batch_first=True)
        ])

        if not complex:
            self.refined_edges = nn.Sequential(
                nn.Linear(in_features=self.edge_dim, out_features=256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(in_features=256, out_features=128),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(in_features=128, out_features=64),
                nn.GELU(),
                nn.Dropout(0.05),
                nn.Linear(in_features=64, out_features=self.edge_dim),
                nn.GELU()
            )
        else:
            self.refined_edges = nn.Sequential(
            nn.Linear(in_features=self.edge_dim, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.GELU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.GELU(),
            nn.Linear(in_features=64, out_features=self.edge_dim)
            )

        """
        def window_creations(self, x):
        Batch_size, node_pair_num, Node_dim = x.shape
        print(f"Input Dim: {x.shape}")
        reshaped = x.unfold(dimension=0, size=self.window, step=1)
        print(f"Reshaped Unfolded: {reshaped.shape}")
        reshaped = reshaped.permute(0, 3, 1, 2)
        print(f"Reshaped Permuted: {reshaped.shape}")
        reshaped = reshaped.reshape((Batch_size - self.window + 1) * self.window, node_pair_num, Node_dim)
        print(f"Reshaped View: {reshaped.shape}")
        #reshaped = reshaped.contiguous().view(Batch_size - W + 1, node_pair_num, W * Node_dim)        
        #print(f"After View: {reshaped.shape}")
        #reshaped = x.permute(0, 2, 1, 3)
        #print(f"Reshaped Dim: {reshaped.shape}")
        return reshaped 
        # Batch Size - Window Size + 1 x Length of Window x Number of Nodes x x Feature vector Dim
        # """

    def forward(self, x):
        #x = self.window_creations(x)
        #print(f"Input Shape: {x.shape}")
        out = x

        for idx, lstm in enumerate(self.temporal_layers):
            out, _ = lstm(out)
            out = self.tanh(out)       
        #out dim: (batch_size, seq_len, hidden_size), so we take the final hidden output for all sequences and their corresponding sequence
        #print(f"Before Sliced Shape: {out.shape}")
        #out = out[:, -1, :]
        #print(f"After Sliced Shape {out.shape}")
        out = self.refined_edges(out)

        return out
    
class AttentionEdgeUpdater(nn.Module):
    def __init__(self, node_dim, edge_dim):
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim

        self.edge_updater = nn.Sequential(
            nn.Linear(2 * self.node_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.edge_dim)
        )
    
    def forward(self, x):
        return self.edge_updater(x)