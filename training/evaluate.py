"""
evaluate.py
-----------
Load a trained checkpoint and run full evaluation on a given dataset split.

Metrics reported:
    Accuracy, Precision, Recall, F1-score, ROC-AUC
    DCC (Distance to Closest Centre) — per-protein pocket centre distance

Usage:
    python training/evaluate.py --data_dir <path> --checkpoint checkpoints/best_fagat.pt [--model fagat|gcn]
"""

import os
import sys
import argparse
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import ScPDBDataset
from models.fagat import FAGAT
from models.gcn_baseline import GCNBaseline


def load_model(model_name: str, checkpoint: str, device) -> torch.nn.Module:
    if model_name == "fagat":
        model = FAGAT()
    else:
        model = GCNBaseline()
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def dcc(pred_coords: np.ndarray, true_coords: np.ndarray) -> float:
    """
    Distance to Closest Centre (DCC).
    Computes the Euclidean distance between the centroid of predicted pocket
    residues and the centroid of true pocket residues.
    Returns NaN if either set is empty.
    """
    if len(pred_coords) == 0 or len(true_coords) == 0:
        return float("nan")
    pred_center = pred_coords.mean(axis=0)
    true_center = true_coords.mean(axis=0)
    return float(np.linalg.norm(pred_center - true_center))


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    all_probs, all_preds, all_labels = [], [], []
    dcc_scores = []

    for batch in loader:
        batch = batch.to(device)
        probs = model(batch).squeeze().cpu().numpy()
        preds = (probs > threshold).astype(int)
        labels = batch.y.squeeze().cpu().numpy().astype(int)

        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

        # DCC per graph in the batch
        # batch.coords is the raw C-alpha coordinates
        if hasattr(batch, "coords"):
            ptr = batch.ptr.cpu().numpy()
            coords = batch.coords.cpu().numpy()
            for i in range(len(ptr) - 1):
                s, e = ptr[i], ptr[i + 1]
                c   = coords[s:e]
                pr  = preds[s:e]
                lb  = labels[s:e]
                pred_c = c[pr == 1]
                true_c = c[lb == 1]
                d = dcc(pred_c, true_c)
                if not np.isnan(d):
                    dcc_scores.append(d)

    all_probs  = np.array(all_probs)
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = {
        "Accuracy" : accuracy_score(all_labels, all_preds),
        "Precision": precision_score(all_labels, all_preds, zero_division=0),
        "Recall"   : recall_score(all_labels, all_preds, zero_division=0),
        "F1"       : f1_score(all_labels, all_preds, zero_division=0),
        "ROC-AUC"  : roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else float("nan"),
        "DCC (Å)"  : np.mean(dcc_scores) if dcc_scores else float("nan"),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate pocket prediction model")
    parser.add_argument("--data_dir",   required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model",      default="fagat", choices=["fagat", "gcn"])
    parser.add_argument("--batch",      type=int, default=4)
    parser.add_argument("--threshold",  type=float, default=0.5,
                        help="Classification threshold (default 0.5). "
                             "Increase to reduce overprediction.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(args.model, args.checkpoint, device)

    dataset = ScPDBDataset(root=args.data_dir)
    loader  = DataLoader(dataset, batch_size=args.batch, shuffle=False)

    print(f"Evaluating {args.model.upper()} on {len(dataset)} graphs (threshold={args.threshold}) …")
    metrics = evaluate(model, loader, device, threshold=args.threshold)

    print("\n=== Evaluation Results ===")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.4f}")


if __name__ == "__main__":
    main()
