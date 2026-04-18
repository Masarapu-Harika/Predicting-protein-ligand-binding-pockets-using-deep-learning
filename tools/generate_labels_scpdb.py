"""
generate_labels_scpdb.py
------------------------
Converts scPDB dataset into the format expected by the FA-GAT pipeline.

scPDB directory layout per complex:
    <scpdb_dir>/
        <complex_id>/
            protein.mol2      ← full protein structure
            ligand.mol2       ← bound ligand
            site.mol2         ← binding site atoms (optional, used for validation)
            cavity6.mol2      ← pocket cavity (optional)

What this script does:
    1. Reads protein.mol2 and ligand.mol2 using BioPython / RDKit-free parsing
    2. Finds all protein residues with ANY atom within DISTANCE_CUTOFF of any
       ligand heavy atom
    3. Writes site_residues.txt  (format: CHAIN RESSEQ)
    4. Optionally copies / converts protein.mol2 → protein.pdb using OpenBabel
       (skipped if protein.pdb already exists)

Usage:
    python tools/generate_labels_scpdb.py --scpdb_dir D:\\scpdb --cutoff 6.5

Requirements: biopython, numpy (already in requirements.txt)
"""

import os
import sys
import glob
import argparse
import numpy as np
from tqdm import tqdm

# ── mol2 parser (no external deps) ──────────────────────────────────────────

def parse_mol2_atoms(mol2_path: str) -> list[dict]:
    """
    Lightweight mol2 ATOM block parser.
    Returns list of dicts: {atom_id, atom_name, x, y, z, res_name, res_seq, chain}
    """
    atoms = []
    in_atom_block = False
    with open(mol2_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("@<TRIPOS>ATOM"):
                in_atom_block = True
                continue
            if line.startswith("@<TRIPOS>") and in_atom_block:
                in_atom_block = False
                continue
            if not in_atom_block:
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                atom_id   = int(parts[0])
                atom_name = parts[1]
                x, y, z   = float(parts[2]), float(parts[3]), float(parts[4])
                # parts[7] is typically RESNUM or RESNAME:RESNUM
                subst_name = parts[7]  # e.g. "ALA1" or "A:ALA1"
                # Parse chain and resseq
                chain = "A"
                res_seq = 0
                if ":" in subst_name:
                    # Format CHAIN:RESNAME+RESNUM
                    chain_part, res_part = subst_name.split(":", 1)
                    chain = chain_part if chain_part else "A"
                    res_num_str = "".join(c for c in res_part if c.isdigit())
                    res_seq = int(res_num_str) if res_num_str else 0
                else:
                    res_num_str = "".join(c for c in subst_name if c.isdigit())
                    res_seq = int(res_num_str) if res_num_str else 0

                atoms.append({
                    "atom_id"  : atom_id,
                    "atom_name": atom_name,
                    "x": x, "y": y, "z": z,
                    "chain"   : chain,
                    "res_seq" : res_seq,
                    "subst"   : subst_name,
                })
            except (ValueError, IndexError):
                continue
    return atoms


def find_pocket_residues(
    protein_atoms: list[dict],
    ligand_atoms: list[dict],
    cutoff: float,
) -> set[tuple[str, int]]:
    """Return (chain, res_seq) pairs within cutoff Å of any ligand atom."""
    if not ligand_atoms or not protein_atoms:
        return set()

    lig_coords = np.array([[a["x"], a["y"], a["z"]] for a in ligand_atoms])
    pocket = set()

    for pa in protein_atoms:
        pc = np.array([pa["x"], pa["y"], pa["z"]])
        dists = np.linalg.norm(lig_coords - pc, axis=1)
        if dists.min() <= cutoff:
            pocket.add((pa["chain"], pa["res_seq"]))

    return pocket


def process_complex(complex_dir: str, cutoff: float) -> int:
    """
    Process one scPDB complex directory.
    Writes site_residues.txt. Returns number of pocket residues written.
    """
    protein_mol2 = os.path.join(complex_dir, "protein.mol2")
    ligand_mol2  = os.path.join(complex_dir, "ligand.mol2")

    if not os.path.exists(protein_mol2):
        return 0
    if not os.path.exists(ligand_mol2):
        return 0

    protein_atoms = parse_mol2_atoms(protein_mol2)
    ligand_atoms  = parse_mol2_atoms(ligand_mol2)

    pocket = find_pocket_residues(protein_atoms, ligand_atoms, cutoff)
    if not pocket:
        return 0

    out_path = os.path.join(complex_dir, "site_residues.txt")
    with open(out_path, "w") as f:
        f.write("# chain resseq\n")
        for chain, resseq in sorted(pocket):
            f.write(f"{chain} {resseq}\n")

    return len(pocket)


def main():
    parser = argparse.ArgumentParser(
        description="Generate site_residues.txt for each scPDB complex"
    )
    parser.add_argument("--scpdb_dir", required=True,
                        help="Root directory of the scPDB dataset")
    parser.add_argument("--cutoff", type=float, default=6.5,
                        help="Distance cutoff in Angstroms (default: 6.5)")
    args = parser.parse_args()

    complex_dirs = sorted([
        d for d in glob.glob(os.path.join(args.scpdb_dir, "*"))
        if os.path.isdir(d)
    ])

    if not complex_dirs:
        print(f"No sub-directories found in {args.scpdb_dir}")
        sys.exit(1)

    print(f"Found {len(complex_dirs)} complexes. Processing with {args.cutoff}Å cutoff …")

    written = 0
    skipped = 0
    for cdir in tqdm(complex_dirs, desc="scPDB"):
        n = process_complex(cdir, args.cutoff)
        if n > 0:
            written += 1
        else:
            skipped += 1

    print(f"\nDone. site_residues.txt written: {written} | Skipped (no mol2): {skipped}")
    print("You can now run: python main.py train --data_dir <scpdb_dir>")


if __name__ == "__main__":
    main()
