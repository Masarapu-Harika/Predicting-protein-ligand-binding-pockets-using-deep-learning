# FA-GAT: Predicting Protein-Ligand Binding Pockets Using Deep Learning

Welcome to the **FA-GAT** project repository. This project aims to accurately predict protein-ligand binding pockets on 3D protein structures using Graph Search and Attention mechanisms (FA-GAT).

## Features
* **Graph Attention Architecture:** Extrapolates deep geometric and physicochemical features of proteins.
* **Binding Pocket Prediction:** Processes raw PDB files and returns precise spatial coordinates of high-probability binding regions.
* **Detailed Interactive 3D Viewer:** Visually highlights pockets, chains, and atoms directly in the browser.
* **Full-Stack Dashboard:** A React + Vite frontend seamlessly tied to a FastAPI backend prediction server.

## Installation & Setup

### Environment Requirements
* Python 3.10+
* Node.js v16+
* PyTorch & PyTorch Geometric

### Running locally
1. **Start the backend server:**
   ```bash
   python -m uvicorn backend.main:app --port 8000 --reload
   ```

2. **Start the frontend interface:**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

## Repository Structure
* `backend/` - The FastAPI prediction server.
* `frontend/` - The React Vite UI with NGL 3D Visualization capabilities.
* `models/` - PyTorch architectures (FA-GAT, GCN Baseline).
* `preprocessing/` & `postprocessing/` - PDB data parsing, graph construction, and pocket clustering.
* `training/` - Model training and active evaluation scripts based on scPDB dataset.
