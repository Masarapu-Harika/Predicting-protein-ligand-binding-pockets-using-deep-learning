"""
fagat.py
--------
Feature-Augmented Graph Attention Network (FA-GAT) for residue-level
binding pocket prediction.

Architecture:
    GATConv(27 → 64, heads=4)  + ReLU + Dropout(0.3)
    GATConv(256 → 64, heads=2) + ReLU + Dropout(0.3)   [256 = 64*4]
    Linear(128 → 1)            + Sigmoid                [128 = 64*2]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class FAGAT(nn.Module):
    """
    Feature-Augmented Graph Attention Network.

    Parameters
    ----------
    in_channels : int
        Number of input node features (default 27).
    hidden1 : int
        Per-head hidden dim for first GAT layer (default 64).
    heads1 : int
        Number of attention heads in first layer (default 4).
    hidden2 : int
        Per-head hidden dim for second GAT layer (default 64).
    heads2 : int
        Number of attention heads in second layer (default 2).
    dropout : float
        Dropout probability applied after each layer (default 0.3).
    """

    def __init__(
        self,
        in_channels: int = 27,
        hidden1: int = 64,
        heads1: int = 4,
        hidden2: int = 64,
        heads2: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout

        # Layer 1: in_channels → hidden1 * heads1
        self.gat1 = GATConv(
            in_channels, hidden1,
            heads=heads1, dropout=dropout, concat=True
        )

        # Layer 2: hidden1*heads1 → hidden2 * heads2
        self.gat2 = GATConv(
            hidden1 * heads1, hidden2,
            heads=heads2, dropout=dropout, concat=True
        )

        # Output: hidden2*heads2 → 1
        self.fc = nn.Linear(hidden2 * heads2, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Layer 1
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)

        # Layer 2
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        x = F.elu(x)

        # Output
        x = self.fc(x)
        return torch.sigmoid(x)
