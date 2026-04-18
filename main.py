"""
main.py
-------
Command-line entry point for the FA-GAT binding pocket prediction pipeline.

Modes
-----
  train     — Build dataset and train model (FA-GAT or GCN)
  predict   — Run inference on a single PDB file and print pocket info
  evaluate  — Run full metrics evaluation on an entire dataset directory
  visualize — Generate 3D pocket visualization for a single PDB file

Examples
--------
  python main.py train     --data_dir D:\\scpdb --model fagat --epochs 100
  python main.py predict   --pdb_file  D:\\scpdb\\1abc\\protein.pdb --checkpoint checkpoints/best_fagat.pt
  python main.py evaluate  --data_dir D:\\scpdb --checkpoint checkpoints/best_fagat.pt
  python main.py visualize --pdb_file  D:\\scpdb\\1abc\\protein.pdb --checkpoint checkpoints/best_fagat.pt
"""

import os
import sys
import argparse
import numpy as np
import torch

# ── make the project root importable regardless of CWD ──────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from preprocessing.pdb_parser      import parse_pdb
from preprocessing.feature_extraction import extract_features
from preprocessing.graph_construction import build_graph
from models.fagat                  import FAGAT
from models.gcn_baseline           import GCNBaseline
from postprocessing.pocket_detection import detect_pockets
from visualization.visualize       import visualize_pockets


# ── helpers ──────────────────────────────────────────────────────────────────

def get_model(name: str) -> torch.nn.Module:
    return FAGAT() if name == "fagat" else GCNBaseline()


def load_checkpoint(model: torch.nn.Module, ckpt: str, device):
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def infer_single(pdb_path: str, model, device):
    """Run inference on a single PDB file. Returns (probs, coords, residues)."""
    residues = parse_pdb(pdb_path)
    if not residues:
        raise ValueError(f"No standard residues found in {pdb_path}")
    features = extract_features(residues)
    coords   = np.array([r["coords"] for r in residues], dtype=np.float32)
    graph    = build_graph(features, coords)
    graph    = graph.to(device)
    with torch.no_grad():
        probs = model(graph).squeeze().cpu().numpy()
    return probs, coords, residues


# ── sub-commands ─────────────────────────────────────────────────────────────

def cmd_train(args):
    """Delegates fully to training/train.py logic (avoids duplication)."""
    from training.train import main as train_main
    # Patch sys.argv so train_main can parse its own args
    sys.argv = [
        "train.py",
        "--data_dir", args.data_dir,
        "--model",    args.model,
        "--epochs",   str(args.epochs),
        "--lr",       str(args.lr),
        "--batch",    str(args.batch),
    ]
    train_main()


def cmd_predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = get_model(args.model)
    model  = load_checkpoint(model, args.checkpoint, device)

    print(f"Running inference on: {args.pdb_file}")
    probs, coords, residues = infer_single(args.pdb_file, model, device)

    pockets = detect_pockets(probs, coords)
    if not pockets:
        print("No pockets detected above threshold.")
        return

    print(f"\nDetected {len(pockets)} pocket(s):\n")
    for i, p in enumerate(pockets):
        cx, cy, cz = p["center"]
        print(
            f"  Pocket {i+1:2d} | residues: {p['size']:4d} | "
            f"mean_prob: {p['mean_prob']:.3f} | "
            f"centre: ({cx:.2f}, {cy:.2f}, {cz:.2f})"
        )


def cmd_evaluate(args):
    from training.evaluate import evaluate, load_model
    from data.dataset import ScPDBDataset
    from torch_geometric.loader import DataLoader

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model   = load_model(args.model, args.checkpoint, device)
    dataset = ScPDBDataset(root=args.data_dir)
    loader  = DataLoader(dataset, batch_size=args.batch, shuffle=False)

    print(f"Evaluating {args.model.upper()} on {len(dataset)} proteins …")
    metrics = evaluate(model, loader, device)
    print("\n=== Evaluation Results ===")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.4f}")


def cmd_visualize(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = get_model(args.model)
    model  = load_checkpoint(model, args.checkpoint, device)

    probs, coords, _ = infer_single(args.pdb_file, model, device)
    pockets = detect_pockets(probs, coords)

    out_png  = args.out_png
    out_html = args.out_html
    visualize_pockets(coords, pockets, out_png=out_png, out_html=out_html)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="fa-gat",
        description="FA-GAT Binding Pocket Prediction Pipeline",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # -- train --
    p_train = sub.add_parser("train", help="Train a pocket prediction model")
    p_train.add_argument("--data_dir", required=True)
    p_train.add_argument("--model",    default="fagat", choices=["fagat", "gcn"])
    p_train.add_argument("--epochs",   type=int,   default=100)
    p_train.add_argument("--lr",       type=float, default=1e-3)
    p_train.add_argument("--batch",    type=int,   default=4)

    # -- predict --
    p_pred = sub.add_parser("predict", help="Predict pockets for a single PDB file")
    p_pred.add_argument("--pdb_file",   required=True)
    p_pred.add_argument("--checkpoint", required=True)
    p_pred.add_argument("--model",      default="fagat", choices=["fagat", "gcn"])

    # -- evaluate --
    p_eval = sub.add_parser("evaluate", help="Evaluate model on a dataset")
    p_eval.add_argument("--data_dir",   required=True)
    p_eval.add_argument("--checkpoint", required=True)
    p_eval.add_argument("--model",      default="fagat", choices=["fagat", "gcn"])
    p_eval.add_argument("--batch",      type=int, default=4)

    # -- visualize --
    p_vis = sub.add_parser("visualize", help="Visualize predicted pockets for a PDB file")
    p_vis.add_argument("--pdb_file",   required=True)
    p_vis.add_argument("--checkpoint", required=True)
    p_vis.add_argument("--model",      default="fagat", choices=["fagat", "gcn"])
    p_vis.add_argument("--out_png",    default="pocket_visualization.png")
    p_vis.add_argument("--out_html",   default="pocket_visualization.html")

    args = parser.parse_args()

    dispatch = {
        "train"    : cmd_train,
        "predict"  : cmd_predict,
        "evaluate" : cmd_evaluate,
        "visualize": cmd_visualize,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
