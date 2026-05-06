"""Graph Neural Network for deadlock risk prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from typing import Optional


class DeadlockGNN(nn.Module):
    """
    GNN that predicts deadlock risk score (0-1) for a wait-for graph.
    
    Architecture: GCN layers + global pooling + MLP head.
    """

    def __init__(
        self,
        in_channels: int = 3,  # [in_degree, out_degree, priority]
        hidden_channels: int = 64,
        out_channels: int = 1,  # risk score
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.dropout = dropout
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels),
            nn.Sigmoid(),
        )

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass. Returns risk score per graph (batch) or per node.
        For graph-level: data has batch vector; returns (batch_size, 1).
        For single graph: returns (1,) scalar.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        out = self.mlp(x)
        return out

    def predict_risk(self, x: torch.Tensor, edge_index: torch.Tensor) -> float:
        """
        Predict deadlock risk for a single graph.
        Returns scalar risk in [0, 1].
        """
        self.eval()
        with torch.no_grad():
            data = Data(x=x, edge_index=edge_index, batch=torch.zeros(x.size(0), dtype=torch.long))
            risk = self.forward(data).item()
        return risk
