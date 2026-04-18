"""
pocket_detection.py
-------------------
Post-process model predictions into discrete pocket clusters.

Steps:
  1. Filter residues with predicted probability > threshold (default 0.5)
  2. Run DBSCAN spatial clustering on C-alpha coordinates
  3. Compute centroid per cluster = pocket centre
  4. Return list of pocket dicts
"""

import numpy as np
from sklearn.cluster import DBSCAN


def detect_pockets(
    probs: np.ndarray,
    coords: np.ndarray,
    threshold: float = 0.5,
    eps: float = 6.0,
    min_samples: int = 3,
) -> list[dict]:
    """
    Cluster predicted pocket residues into discrete pockets.

    Parameters
    ----------
    probs  : np.ndarray, shape (N,)
        Per-residue sigmoid probabilities from the model.
    coords : np.ndarray, shape (N, 3)
        C-alpha coordinates.
    threshold : float
        Probability cutoff for pocket residue selection.
    eps : float
        DBSCAN neighbourhood radius in Angstroms.
    min_samples : int
        Minimum points per DBSCAN cluster.

    Returns
    -------
    list of dict
        Each dict has:
            index         : cluster id (0-based)
            residue_mask  : boolean mask over the original N residues
            center        : np.ndarray (3,) — centroid of cluster
            size          : int — number of residues
            mean_prob     : float — average model confidence
    """
    pocket_mask = probs > threshold
    pocket_coords = coords[pocket_mask]
    pocket_probs  = probs[pocket_mask]

    if len(pocket_coords) < min_samples:
        return []

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(pocket_coords)
    labels = db.labels_

    pockets = []
    unique_labels = sorted(set(labels) - {-1})
    for label in unique_labels:
        cl_mask   = labels == label
        cl_coords = pocket_coords[cl_mask]
        cl_probs  = pocket_probs[cl_mask]

        # Map back to original residue indices
        original_indices = np.where(pocket_mask)[0][cl_mask]
        full_mask = np.zeros(len(probs), dtype=bool)
        full_mask[original_indices] = True

        pockets.append({
            "index"       : label,
            "residue_mask": full_mask,
            "center"      : cl_coords.mean(axis=0),
            "size"        : int(cl_mask.sum()),
            "mean_prob"   : float(cl_probs.mean()),
        })

    # Sort pockets by confidence (highest first)
    pockets.sort(key=lambda p: p["mean_prob"], reverse=True)
    return pockets
