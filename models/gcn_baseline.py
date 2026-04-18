"""
gcn_baseline.py
---------------
Graph Convolutional Network baseline for pocket prediction.

Architecture:
    GCNConv(27 → 128) + ReLU + Dropout(0.3)
    GCNConv(128 → 64) + ReLU + Dropout(0.3)
    Linear(64 → 1)    + Sigmoid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNBaseline(nn.Module):
    """
    Simple 2-layer GCN for pocket residue binary classification.

    Parameters
    ----------
    in_channels : int
        Number of input node features (default 27).
    hidden : int
        Hidden dimension (default 128).
    dropout : float
        Dropout probability (default 0.3).
    """

    def __init__(self, in_channels: int = 27, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden // 2)
        self.fc    = nn.Linear(hidden // 2, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.fc(x)
        return torch.sigmoid(x)
