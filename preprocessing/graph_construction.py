"""
graph_construction.py
---------------------
Constructs a PyTorch Geometric Data object from extracted residue features.

Edges are added between any two residues whose C-alpha atoms are within
DISTANCE_THRESHOLD angstroms (default = 8Å).
"""

import numpy as np
import torch
from torch_geometric.data import Data

DISTANCE_THRESHOLD = 8.0  # Angstroms


def build_graph(
    features: np.ndarray,
    coords: np.ndarray,
    labels: np.ndarray | None = None,
    threshold: float = DISTANCE_THRESHOLD,
) -> Data:
    """
    Build a PyG Data object from per-residue features.

    Parameters
    ----------
    features : np.ndarray, shape (N, 27)
        Node feature matrix from feature_extraction.
    coords : np.ndarray, shape (N, 3)
        Raw C-alpha coordinates used for distance computation.
    labels : np.ndarray or None, shape (N,)
        Binary labels (1 = pocket residue, 0 = non-pocket).
    threshold : float
        Distance cutoff in Angstroms for edge construction.

    Returns
    -------
    torch_geometric.data.Data
        Data object with x, edge_index, (optionally) y.
    """
    N = len(features)
    assert N > 0, "No residues found — cannot build graph."

    # Compute pairwise distances
    diff = coords[:, None, :] - coords[None, :, :]  # (N, N, 3)
    dist = np.sqrt((diff ** 2).sum(axis=-1))         # (N, N)

    # Find all pairs within threshold (excluding self-loops)
    src, dst = np.where((dist <= threshold) & (dist > 0))

    edge_index = torch.tensor(
        np.stack([src, dst], axis=0), dtype=torch.long
    )
    x = torch.tensor(features, dtype=torch.float32)

    if labels is not None:
        y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        return Data(x=x, edge_index=edge_index, y=y)

    return Data(x=x, edge_index=edge_index)
