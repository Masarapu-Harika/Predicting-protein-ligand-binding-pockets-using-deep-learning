"""
generate_labels_pdbbind.py
--------------------------
Converts PDBbind dataset into the format expected by the FA-GAT pipeline.

PDBbind directory layout:
    <pdbbind_dir>/
        index/
            INDEX_general_PL_data.2020   ← or any INDEX_*.txt file
        <pdb_id>/
            <pdb_id>_protein.pdb         ← protein structure
            <pdb_id>_ligand.mol2         ← bound ligand (preferred)
            <pdb_id>_ligand.sdf          ← fallback if mol2 absent

What this script does:
    1. Reads each complex from the index
    2. Parses the ligand file (mol2 or sdf) to get ligand atom coordinates
    3. Parses the protein PDB to get all residue atoms
    4. Finds residues within DISTANCE_CUTOFF Å of any ligand heavy atom
    5. Writes site_residues.txt (chain resseq) into each complex folder

Usage:
    python tools/generate_labels_pdbbind.py --pdbbind_dir D:\\PDBbind --cutoff 6.5

Requirements: biopython, numpy (already in requirements.txt)
"""

import os
import sys
import glob
import argparse
import re
import numpy as np
from tqdm import tqdm
from Bio.PDB import PDBParser

STANDARD_AA = {
    'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY',
    'HIS','ILE','LEU','LYS','MET','PHE','PRO','SER',
    'THR','TRP','TYR','VAL'
}


# ── Ligand parsers ────────────────────────────────────────────────────────────

def parse_ligand_mol2(mol2_path: str) -> np.ndarray:
    """Return (N, 3) array of ligand heavy-atom coordinates from a .mol2 file."""
    coords = []
    in_atom = False
    with open(mol2_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("@<TRIPOS>ATOM"):
                in_atom = True
                continue
            if line.startswith("@<TRIPOS>") and in_atom:
                break
            if not in_atom:
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            atom_type = parts[5] if len(parts) > 5 else ""
            # Skip hydrogens
            if atom_type.startswith("H") or parts[1].startswith("H"):
                continue
            try:
                coords.append([float(parts[2]), float(parts[3]), float(parts[4])])
            except ValueError:
                continue
    return np.array(coords, dtype=np.float32) if coords else np.zeros((0, 3))


def parse_ligand_sdf(sdf_path: str) -> np.ndarray:
    """Return (N, 3) array of ligand heavy-atom coordinates from a .sdf file."""
    coords = []
    with open(sdf_path, encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # Atom block starts at line 4 (0-indexed 3), count from the COUNTS line
    if len(lines) < 4:
        return np.zeros((0, 3))

    # Parse the counts line (line index 3)
    counts_line = lines[3]
    try:
        num_atoms = int(counts_line[:3].strip())
    except ValueError:
        return np.zeros((0, 3))

    for i in range(4, 4 + num_atoms):
        if i >= len(lines):
            break
        parts = lines[i].split()
        if len(parts) < 4:
            continue
        elem = parts[3] if len(parts) > 3 else ""
        if elem.startswith("H"):
            continue
        try:
            coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
        except ValueError:
            continue

    return np.array(coords, dtype=np.float32) if coords else np.zeros((0, 3))


# ── Protein residue atoms ─────────────────────────────────────────────────────

def get_protein_residue_atoms(pdb_path: str) -> list[dict]:
    """
    Return list of {chain, res_seq, coords (N,3)} per residue.
    Only standard amino acids.
    """
    bparser = PDBParser(QUIET=True)
    structure = bparser.get_structure("p", pdb_path)
    residue_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                res_name  = residue.get_resname().strip()
                if res_name not in STANDARD_AA:
                    continue
                atoms = []
                for atom in residue:
                    if atom.element and atom.element.strip().upper() == "H":
                        continue
                    atoms.append(atom.get_vector().get_array())
                if atoms:
                    residue_atoms.append({
                        "chain"  : chain.id,
                        "res_seq": residue.get_id()[1],
                        "coords" : np.array(atoms, dtype=np.float32),
                    })
        break  # first model only
    return residue_atoms


def find_pocket_residues(
    residue_atoms: list[dict],
    lig_coords: np.ndarray,
    cutoff: float,
) -> set[tuple[str, int]]:
    """Return (chain, res_seq) for any residue with atom within cutoff of ligand."""
    pocket = set()
    if len(lig_coords) == 0:
        return pocket
    for ra in residue_atoms:
        for ac in ra["coords"]:
            dists = np.linalg.norm(lig_coords - ac, axis=1)
            if dists.min() <= cutoff:
                pocket.add((ra["chain"], ra["res_seq"]))
                break
    return pocket


# ── Main per-complex processor ────────────────────────────────────────────────

def process_complex(pdb_id: str, complex_dir: str, cutoff: float) -> int:
    pdb_path   = os.path.join(complex_dir, f"{pdb_id}_protein.pdb")
    mol2_path  = os.path.join(complex_dir, f"{pdb_id}_ligand.mol2")
    sdf_path   = os.path.join(complex_dir, f"{pdb_id}_ligand.sdf")

    if not os.path.exists(pdb_path):
        return 0

    # Load ligand
    if os.path.exists(mol2_path):
        lig_coords = parse_ligand_mol2(mol2_path)
    elif os.path.exists(sdf_path):
        lig_coords = parse_ligand_sdf(sdf_path)
    else:
        return 0

    if len(lig_coords) == 0:
        return 0

    residue_atoms = get_protein_residue_atoms(pdb_path)
    pocket = find_pocket_residues(residue_atoms, lig_coords, cutoff)

    if not pocket:
        return 0

    out_path = os.path.join(complex_dir, "site_residues.txt")
    with open(out_path, "w") as f:
        f.write("# chain resseq\n")
        for chain, resseq in sorted(pocket):
            f.write(f"{chain} {resseq}\n")

    return len(pocket)


# ── Complex discovery ─────────────────────────────────────────────────────────

def discover_complexes(root: str) -> list[tuple[str, str]]:
    """Return list of (pdb_id, complex_dir) from a PDBbind root directory."""
    results = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(d):
            continue
        pdb_id = os.path.basename(d)
        if pdb_id == "index" or pdb_id.startswith("."):
            continue
        results.append((pdb_id, d))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate site_residues.txt for each PDBbind complex"
    )
    parser.add_argument("--pdbbind_dir", required=True,
                        help="Root directory of the PDBbind dataset")
    parser.add_argument("--cutoff", type=float, default=6.5,
                        help="Distance cutoff in Angstroms (default: 6.5)")
    args = parser.parse_args()

    complexes = discover_complexes(args.pdbbind_dir)
    if not complexes:
        print(f"No complexes found in {args.pdbbind_dir}")
        sys.exit(1)

    print(f"Found {len(complexes)} complexes. Processing with {args.cutoff}Å cutoff …")

    written = 0
    skipped = 0
    for pdb_id, cdir in tqdm(complexes, desc="PDBbind"):
        n = process_complex(pdb_id, cdir, args.cutoff)
        if n > 0:
            written += 1
        else:
            skipped += 1

    print(f"\nDone. site_residues.txt written: {written} | Skipped: {skipped}")
    print("You can now run: python main.py train --data_dir <pdbbind_dir>")


if __name__ == "__main__":
    main()
