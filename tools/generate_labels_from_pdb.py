"""
generate_labels_from_pdb.py
---------------------------
Generates site_residues.txt for each protein in a directory by extracting
ligand coordinates directly from HETATM records in the PDB file itself.

Steps per complex:
  1. Parse ATOM records → protein residue C-alpha coords
  2. Parse HETATM records → ligand heavy atom coords (skip water/metals)
  3. Find protein residues within CUTOFF Å of any ligand atom
  4. Write site_residues.txt

Usage:
    python tools/generate_labels_from_pdb.py --data_dir my_data --cutoff 6.5
"""

import os
import glob
import argparse
import numpy as np
from tqdm import tqdm

# Small molecules / solvents to skip as "ligand"
SKIP_RESIDUES = {
    'HOH', 'WAT', 'H2O',           # water
    'SO4', 'PO4', 'GOL', 'EDO',    # common crystallography additives
    'ACT', 'ACE', 'FMT', 'DMS',
    'CL', 'NA', 'MG', 'ZN', 'CA',
    'FE', 'MN', 'CU', 'NI', 'CO',
    'K',  'BR', 'IOD', 'ION',
}

STANDARD_AA = {
    'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY',
    'HIS','ILE','LEU','LYS','MET','PHE','PRO','SER',
    'THR','TRP','TYR','VAL'
}


def parse_pdb(pdb_path):
    """Return (residues, ligand_coords) from a PDB file."""
    residues = {}      # (chain, resseq) -> list of atom coords
    lig_coords = []

    with open(pdb_path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            rec = line[:6].strip()

            if rec == 'ATOM':
                res_name = line[17:20].strip()
                if res_name not in STANDARD_AA:
                    continue
                chain   = line[21].strip()
                try:
                    resseq  = int(line[22:26].strip())
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                except ValueError:
                    continue
                key = (chain, resseq, res_name)
                residues.setdefault(key, []).append([x, y, z])

            elif rec == 'HETATM':
                res_name = line[17:20].strip()
                if res_name in SKIP_RESIDUES:
                    continue
                atom_name = line[12:16].strip()
                # skip hydrogens
                if atom_name.startswith('H') or (len(atom_name) > 1 and atom_name[1] == 'H'):
                    continue
                try:
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    lig_coords.append([x, y, z])
                except ValueError:
                    continue

    return residues, np.array(lig_coords, dtype=np.float32) if lig_coords else np.zeros((0, 3))


def find_pocket(residues, lig_coords, cutoff):
    if len(lig_coords) == 0:
        return set()
    pocket = set()
    for (chain, resseq, res_name), atoms in residues.items():
        atom_arr = np.array(atoms, dtype=np.float32)
        for ac in atom_arr:
            dists = np.linalg.norm(lig_coords - ac, axis=1)
            if dists.min() <= cutoff:
                pocket.add((chain, resseq))
                break
    return pocket


def process_dir(data_dir, cutoff):
    complex_dirs = sorted([
        d for d in glob.glob(os.path.join(data_dir, '*'))
        if os.path.isdir(d)
    ])

    written = skipped = 0
    for cdir in tqdm(complex_dirs, desc='Generating labels'):
        pdb_files = glob.glob(os.path.join(cdir, '*.pdb'))
        if not pdb_files:
            skipped += 1
            continue

        pdb_path = pdb_files[0]
        residues, lig_coords = parse_pdb(pdb_path)

        if len(lig_coords) == 0:
            skipped += 1
            continue

        pocket = find_pocket(residues, lig_coords, cutoff)
        if not pocket:
            skipped += 1
            continue

        out_path = os.path.join(cdir, 'site_residues.txt')
        with open(out_path, 'w') as f:
            f.write('# chain resseq\n')
            for chain, resseq in sorted(pocket):
                f.write(f'{chain} {resseq}\n')
        written += 1

    print(f'\nDone. Labels written: {written} | Skipped (no ligand): {skipped}')
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Directory with protein subdirs')
    parser.add_argument('--cutoff', type=float, default=6.5, help='Distance cutoff in Angstroms')
    args = parser.parse_args()
    process_dir(args.data_dir, args.cutoff)


if __name__ == '__main__':
    main()
