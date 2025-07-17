import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Edge-Centric GNN Architecture

class EdgeCentricNetwork(nn.Module):
    def __init__(self, num_nodes, directed, edge_dim, node_dim) -> None:
        super().__init__()

        self.num_nodes = num_nodes # This also represents the number of sectors
        self.directed = directed
        self.edge_dim = edge_dim
        self.node_dim = node_dim
        
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

    def forward(self, x) -> torch.tensor:

        batch_size, num_nodes, dim_node = x.shape
        #print(f"Batch Size: {batch_size}")
        #print(f"Num Nodes: {num_nodes}")
        #print(f"Node Embed Dim: {dim_node}")
        # x: [batch_size, # Nodes, # Node Feature Vector Dim]
        start = x.unsqueeze(2)
        terminal = x.unsqueeze(1)

        if self.directed:
            # This will concatonate the node_embeddings together for a pair of two nodes in a directed graph
            # Now the last dim is dim_node * 2 because we concatonate the two node feature vectors
            pair_broadcasting = torch.cat(
                [start.expand(batch_size, num_nodes, num_nodes, dim_node),
                terminal.expand(batch_size, num_nodes, num_nodes, dim_node)], dim=3)
        
        B, N_start, N_out, N_Dim = pair_broadcasting.shape

        pair_data = pair_broadcasting.view(B, N_start * N_out, N_Dim)

        edge_embeddings = self.edge_updater(pair_data)
        edge_embeddings = edge_embeddings.view(B, N_start, N_out, self.edge_dim)

        incoming_edges = edge_embeddings.permute(0, 2, 1, 3) # B, N_out(Target Node), N_start, self.edge_dim

        aggregated_incoming_edges =  incoming_edges.reshape(B, N_out, N_start * self.edge_dim)

        delta = self.regression(aggregated_incoming_edges)

        return delta