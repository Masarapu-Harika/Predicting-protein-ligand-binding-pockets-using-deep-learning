"""
train.py
--------
Training loop for the FA-GAT (and optionally GCN baseline) pocket prediction
model.

Usage (from project root):
    python training/train.py --data_dir <path_to_scpdb> --model fagat --epochs 100

Checkpoints are saved to: checkpoints/best_<model>.pt
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

# Make project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import ScPDBDataset
from models.fagat import FAGAT
from models.gcn_baseline import GCNBaseline


def get_model(name: str, in_channels: int = 27) -> nn.Module:
    if name == "fagat":
        return FAGAT(in_channels=in_channels)
    elif name == "gcn":
        return GCNBaseline(in_channels=in_channels)
    else:
        raise ValueError(f"Unknown model: {name}. Choose 'fagat' or 'gcn'.")


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out  = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def eval_epoch(model, loader, device) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        batch  = batch.to(device)
        probs  = model(batch).flatten().cpu()
        preds  = (probs > 0.5).long().numpy()
        labels = batch.y.flatten().cpu().long().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    return {
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0)
    }


def main():
    parser = argparse.ArgumentParser(description="Train FA-GAT / GCN baseline")
    parser.add_argument("--data_dir", required=True,  help="Path to scPDB directory")
    parser.add_argument("--model",    default="fagat", choices=["fagat", "gcn"])
    parser.add_argument("--epochs",   type=int, default=100)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--batch",    type=int, default=4)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--alpha",    type=float, default=None,
                        help="Focal loss alpha (weight for positive class). "
                             "If None, calculated from class imbalance. "
                             "Lower values reduce overprediction.")
    parser.add_argument("--checkpoint_path", default=None,
                        help="Path to save the best model checkpoint. "
                             "If None, saved to checkpoints/best_<model>.pt")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from checkpoint if it exists.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Dataset ---
    print("Loading dataset …")
    dataset = ScPDBDataset(root=args.data_dir)
    print(f"Total graphs: {len(dataset)}")

    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=args.seed)
    train_data = [dataset[i] for i in train_idx]
    val_data   = [dataset[i] for i in val_idx]

    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=args.batch, shuffle=False)

    # --- Model ---
    model = get_model(args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    ckpt_path = args.checkpoint_path if args.checkpoint_path else f"checkpoints/best_{args.model}.pt"
    if args.resume and os.path.exists(ckpt_path):
        print(f"Resuming from checkpoint {ckpt_path} ...")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))


    # Focal loss — handles class imbalance far better than weighted BCE
    # gamma=2 down-weights easy negatives, alpha balances pos/neg
    if args.alpha is not None:
        alpha = args.alpha
        print(f"Using manual focal loss alpha: {alpha}")
    else:
        pos_w = dataset.pos_weight().to(device)
        alpha = float(pos_w / (1 + pos_w))   # fraction of negatives
        print(f"Using calculated focal loss alpha: {alpha:.4f}")

    def focal_loss(pred, target, gamma=2.0, alpha=alpha):
        eps = 1e-8
        # per-sample BCE
        bce = -(target * torch.log(pred + eps) + (1 - target) * torch.log(1 - pred + eps))
        # focal weight
        pt  = torch.where(target == 1, pred, 1 - pred)
        fw  = (1 - pt) ** gamma
        # class weight
        cw  = torch.where(target == 1, alpha, 1 - alpha)
        loss = cw * fw * bce
        return loss.sum() / torch.clamp(target.sum(), min=1.0)

    # --- Training ---
    os.makedirs("checkpoints", exist_ok=True)
    best_f1   = 0.0

    start_epoch = 1
    if args.resume:
        try:
            with open(f"metrics/{args.model}.json", "r") as f:
                history = json.load(f)
                if len(history) > 0:
                    start_epoch = history[-1]["epoch"] + 1
                    best_f1 = max(h["f1"] for h in history)
        except:
            pass

    for epoch in range(start_epoch, start_epoch + args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, focal_loss, device)
        val_metrics = eval_epoch(model, val_loader, device)
        val_f1 = val_metrics["f1"]
        scheduler.step(1 - val_f1)  # minimize (1 - F1)

        if val_f1 >= best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), ckpt_path)

        if epoch % 10 == 0 or epoch == start_epoch:
            print(f"Epoch {epoch:04d} | Loss: {train_loss:.4f} | "
                  f"F1: {val_metrics['f1']:.4f} | "
                  f"Prec: {val_metrics['precision']:.4f} | "
                  f"Recall: {val_metrics['recall']:.4f}")

        # --- Dashboard Logging ---
        metrics_dir = "metrics"
        os.makedirs(metrics_dir, exist_ok=True)
        log_file = os.path.join(metrics_dir, f"{args.model}.json")
        
        # Load or init
        if epoch == 1 and not args.resume:
            history = []
        else:
            try:
                with open(log_file, "r") as f:
                    history = json.load(f)
            except:
                history = []

        history.append({
            "epoch": epoch,
            "loss": round(train_loss, 4),
            "f1": round(val_metrics["f1"], 4),
            "precision": round(val_metrics["precision"], 4),
            "recall": round(val_metrics["recall"], 4),
        })
        
        with open(log_file, "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best Val F1: {best_f1:.4f}. Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
