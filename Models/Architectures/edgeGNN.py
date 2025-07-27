import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from Models.Architectures.encoders import (
    StaticEdgeUpdater,
    TemporalEdgeUpdater,
    AttentionEdgeUpdater)

from Models.Architectures.regressor import (
    EdgetoNodeRegressionBlock
)
# Edge-Centric GNN Architecture

class EdgeCentricNetwork(nn.Module):
    def __init__(self, num_nodes, edge_dim, node_dim, window=14, static=False, temporal=False, attention=False, complex=False) -> None:
        super().__init__()

        self.num_nodes = num_nodes # This also represents the number of sectors
        self.edge_dim = edge_dim
        self.node_dim = node_dim
        self.window = window

        self.static = static
        self.temporal = temporal
        self.attention = attention

        self.complex = complex

        if static:
            self.edge_updater = StaticEdgeUpdater(node_dim=self.node_dim, edge_dim=self.edge_dim)
        elif temporal:
            self.edge_updater = TemporalEdgeUpdater(node_dim=self.node_dim, edge_dim=self.edge_dim, window=self.window, num_nodes=self.num_nodes, complex=self.complex)
        elif attention:
            self.edge_updater = AttentionEdgeUpdater(node_dim=self.node_dim, edge_dim=self.edge_dim)
        else:
            self.edge_updater = StaticEdgeUpdater(node_dim=self.node_dim, edge_dim=self.edge_dim)

        self.regression = EdgetoNodeRegressionBlock(num_nodes=self.num_nodes, edge_dim=self.edge_dim)

    def forward(self, x) -> torch.Tensor:
        """
        Handles different data formats based on static/temporal/attention modes.
        Shapes:
            - Static:    [B, N, D]
            - Temporal:  [B, T, N, D]
        """
        if self.static or self.attention:
            return self.forward_static_or_attention(x)
        elif self.temporal:
            return self.forward_temporal(x)
        else:
            raise ValueError("Invalid configuration: no model mode selected.")


    def forward_static_or_attention(self, x):
        """
        Used when input is [B, N, D]. Also shared by attention models.
        """
        B, N, D = x.shape

        start = x.unsqueeze(2)  # [B, N, 1, D]
        terminal = x.unsqueeze(1)  # [B, 1, N, D]

        pair_broadcasting = torch.cat([
            start.expand(B, N, N, D),
            terminal.expand(B, N, N, D)
        ], dim=3)  # [B, N, N, 2D]

        pair_data = pair_broadcasting.view(B, N * N, 2 * D)

        edge_embeddings = self.edge_updater(pair_data)
        edge_embeddings = edge_embeddings.view(B, N, N, self.edge_dim)

        incoming_edges = edge_embeddings.permute(0, 2, 1, 3)  # [B, N_out, N_in, edge_dim]
        aggregated_incoming = incoming_edges.reshape(B, N, N * self.edge_dim)

        delta = self.regression(aggregated_incoming)
        return delta

    # Just copy pasted forward_static_or_attention and differed the input dimension handling type shit
    def forward_temporal(self, x):
        """
        Used when input is [B, T, N, D]
        """
        B, T, N, D = x.shape

        # Unroll time into batch dimension
        x_time_flat = x.view(B * T, N, D)

        start = x_time_flat.unsqueeze(2)  # [B*T, N, 1, D]
        terminal = x_time_flat.unsqueeze(1)  # [B*T, 1, N, D]

        pair_broadcasting = torch.cat([
            start.expand(B * T, N, N, D),
            terminal.expand(B * T, N, N, D)
        ], dim=3)  # [B*T, N, N, 2D]

        pair_data = pair_broadcasting.view(B * T, N * N, 2 * D)

        edge_embeddings = self.edge_updater(pair_data)  # should handle B*T as batch
        edge_embeddings = edge_embeddings.view(B * T, N, N, self.edge_dim)

        incoming_edges = edge_embeddings.permute(0, 2, 1, 3)  # [B*T, N_out, N_in, edge_dim]
        aggregated_incoming = incoming_edges.reshape(B * T, N, N * self.edge_dim)

        delta = self.regression(aggregated_incoming)

        delta = delta.view(B, T, N, -1)
        delta = delta[:, -1, :, :] # Get rid of previous timesteps. Take the last instead

        return delta