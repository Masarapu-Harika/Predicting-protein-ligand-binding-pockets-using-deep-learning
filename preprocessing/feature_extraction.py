"""
feature_extraction.py
---------------------
Builds a 27-dimensional feature vector for each protein residue.

Dimensions:
  [0:3]   — Normalized C-alpha XYZ coordinates
  [3:6]   — Hydrophobicity, charge, polarity (per-residue physicochemical)
  [6]     — Estimated SASA surrogate (solvent exposure approximation)
  [7:27]  — One-hot encoding of the 20 standard amino acid types
"""

import numpy as np

# Standard amino acids and their index for one-hot encoding
AA_LIST = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS',
    'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO',
    'SER', 'THR', 'TRP', 'TYR', 'VAL'
]
AA_INDEX = {aa: i for i, aa in enumerate(AA_LIST)}

# Physicochemical properties (hydrophobicity, charge, polarity)
# Source: Kyte-Doolittle scale & canonical reference tables
PHYSICOCHEMICAL = {
    'ALA': ( 1.8,  0.0, 0.0),
    'ARG': (-4.5,  1.0, 1.0),
    'ASN': (-3.5,  0.0, 1.0),
    'ASP': (-3.5, -1.0, 1.0),
    'CYS': ( 2.5,  0.0, 0.0),
    'GLN': (-3.5,  0.0, 1.0),
    'GLU': (-3.5, -1.0, 1.0),
    'GLY': (-0.4,  0.0, 0.0),
    'HIS': (-3.2,  0.5, 1.0),
    'ILE': ( 4.5,  0.0, 0.0),
    'LEU': ( 3.8,  0.0, 0.0),
    'LYS': (-3.9,  1.0, 1.0),
    'MET': ( 1.9,  0.0, 0.0),
    'PHE': ( 2.8,  0.0, 0.0),
    'PRO': (-1.6,  0.0, 0.0),
    'SER': (-0.8,  0.0, 1.0),
    'THR': (-0.7,  0.0, 1.0),
    'TRP': (-0.9,  0.0, 1.0),
    'TYR': (-1.3,  0.0, 1.0),
    'VAL': ( 4.2,  0.0, 0.0),
}


def _one_hot(res_name: str) -> np.ndarray:
    vec = np.zeros(20, dtype=np.float32)
    idx = AA_INDEX.get(res_name, -1)
    if idx >= 0:
        vec[idx] = 1.0
    return vec


def extract_features(residues: list[dict]) -> np.ndarray:
    """
    Build the (N, 27) feature matrix for a list of residue dicts.

    Parameters
    ----------
    residues : list of dict
        Output of `pdb_parser.parse_pdb`.

    Returns
    -------
    np.ndarray, shape (N, 27)
    """
    if not residues:
        return np.zeros((0, 27), dtype=np.float32)

    coords = np.array([r["coords"] for r in residues], dtype=np.float32)

    # Normalize coordinates to zero mean, unit std
    mean = coords.mean(axis=0)
    std  = coords.std(axis=0) + 1e-8
    coords_norm = (coords - mean) / std

    features = []
    for i, r in enumerate(residues):
        res_name = r["res_name"]
        phys = PHYSICOCHEMICAL.get(res_name, (0.0, 0.0, 0.0))

        # Simple SASA surrogate: normalized inverse of neighbour count within 10Å
        dists = np.linalg.norm(coords - coords[i], axis=1)
        neighbour_count = float(np.sum(dists < 10.0) - 1)  # exclude self
        sasa_surrogate = 1.0 / (1.0 + neighbour_count)

        feat = np.concatenate([
            coords_norm[i],                      # 3
            np.array(phys, dtype=np.float32),    # 3
            np.array([sasa_surrogate], dtype=np.float32),  # 1
            _one_hot(res_name),                  # 20
        ])
        features.append(feat)

    return np.array(features, dtype=np.float32)
