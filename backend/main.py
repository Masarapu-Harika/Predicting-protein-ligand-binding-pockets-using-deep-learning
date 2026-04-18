"""
backend/main.py  —  FastAPI server for FA-GAT pocket prediction
"""
import os, sys, tempfile
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from preprocessing.pdb_parser import parse_pdb
from preprocessing.feature_extraction import extract_features
from preprocessing.graph_construction import build_graph
from postprocessing.pocket_detection import detect_pockets
from models.fagat import FAGAT
from models.gcn_baseline import GCNBaseline

app = FastAPI(title="FA-GAT Pocket Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── model cache ──────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_models: dict = {}

def get_model(name: str):
    if name not in _models:
        # Use the stable expanded model if it exists, else fallback to standard
        ckpt_name = f"best_{name}_expanded.pt" if name == "fagat" else f"best_{name}.pt"
        ckpt = os.path.join(ROOT, "checkpoints", ckpt_name)
        if not os.path.exists(ckpt):
            ckpt = os.path.join(ROOT, "checkpoints", f"best_{name}.pt")
        if not os.path.exists(ckpt):
            raise HTTPException(404, f"Checkpoint not found: {ckpt}")
        m = FAGAT() if name == "fagat" else GCNBaseline()
        m.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))
        m.to(DEVICE).eval()
        _models[name] = m
    return _models[name]


# ── routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE)}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model: str = "fagat",
    threshold: float = 0.5,
):
    if not file.filename.endswith(".pdb"):
        raise HTTPException(400, "Only .pdb files are accepted")

    # save upload to temp file
    content = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        residues = parse_pdb(tmp_path)
        if not residues:
            raise HTTPException(422, "No standard residues found in PDB file")

        features = extract_features(residues)
        coords   = np.array([r["coords"] for r in residues], dtype=np.float32)
        graph    = build_graph(features, coords).to(DEVICE)

        with torch.no_grad():
            probs = get_model(model)(graph).squeeze().cpu().numpy()
            if probs.ndim == 0:
                probs = probs.reshape(1)

        pockets = detect_pockets(probs, coords, threshold=threshold)

        residue_data = [
            {
                "chain":   r["chain_id"],
                "resSeq":  int(r["res_seq"]),
                "resName": r["res_name"],
                "coords":  list(r["coords"]),
                "prob":    float(probs[i]),
            }
            for i, r in enumerate(residues)
        ]

        pocket_data = [
            {
                "index":    int(p["index"]),
                "size":     int(p["size"]),
                "meanProb": round(float(p["mean_prob"]), 4),
                "center":   [round(float(c), 3) for c in p["center"]],
                "residues": [
                    {"chain": residues[j]["chain_id"], "resSeq": int(residues[j]["res_seq"])}
                    for j in np.where(p["residue_mask"])[0]
                ],
            }
            for p in pockets
        ]

        return {
            "filename":    file.filename,
            "numResidues": len(residues),
            "numPockets":  len(pockets),
            "pockets":     pocket_data,
            "residues":    residue_data,
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.get("/api/metrics/{model_name}")
def get_metrics(model_name: str):
    """Serve the training history JSON for a given model."""
    import json
    metrics_path = os.path.join(ROOT, "metrics", f"{model_name}.json")
    if not os.path.exists(metrics_path):
        return []
    try:
        with open(metrics_path, "r") as f:
            return json.load(f)
    except:
        return []


@app.get("/api/stats")
def get_stats():
    """Return general application and model stats."""
    ckpts = []
    ckpt_dir = os.path.join(ROOT, "checkpoints")
    if os.path.exists(ckpt_dir):
        ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    
    return {
        "status": "active",
        "checkpoints": ckpts,
        "device": str(DEVICE),
        "root": ROOT
    }
