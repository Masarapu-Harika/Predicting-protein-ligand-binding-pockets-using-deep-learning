"""
pdb_parser.py
-------------
Parses PDB files using BioPython and extracts standard amino acid residues
with their C-alpha coordinates.

Returns a list of dicts with keys:
    chain_id  : str
    res_seq   : int
    res_name  : str (3-letter code)
    coords    : tuple (x, y, z) of the C-alpha atom
"""

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

# Standard 3-letter amino acid codes
STANDARD_AA = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
    'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
    'THR', 'TRP', 'TYR', 'VAL'
}


def parse_pdb(pdb_path: str) -> list[dict]:
    """
    Parse a PDB file and extract residues with C-alpha coordinates.

    Parameters
    ----------
    pdb_path : str
        Absolute path to the PDB file.

    Returns
    -------
    list of dict
        Each dict contains: chain_id, res_seq, res_name, coords (x, y, z).
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)

    residues = []
    for model in structure:
        for chain in model:
            for residue in chain:
                res_name = residue.get_resname().strip()
                if res_name not in STANDARD_AA:
                    continue
                if "CA" not in residue:
                    continue
                ca = residue["CA"]
                x, y, z = ca.get_vector()
                residues.append({
                    "chain_id": chain.id,
                    "res_seq" : residue.get_id()[1],
                    "res_name": res_name,
                    "coords"  : (float(x), float(y), float(z)),
                })
        break  # Use only the first model

    return residues


if __name__ == "__main__":
    import sys
    res = parse_pdb(sys.argv[1])
    print(f"Parsed {len(res)} residues from {sys.argv[1]}")
    for r in res[:5]:
        print(r)
