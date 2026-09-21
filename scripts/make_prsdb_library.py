"""
Supplementary Data D8 - structure-to-protein mapping for the reverse-docking target library.

Methods S2 asks for the provenance of all 618 structures. This exports the part available in the
working directory: the structure identifier, UniProt accession, gene symbol and protein name. The
PDB or AlphaFold accession, chain, resolution and pocket coordinates require a separate export
from PRSDB and are still marked [TO BE COMPLETED] in the Supplementary Information.

Output: Kinase_paper/analysis_output/prsdb_kinase_library.csv
"""
from __future__ import annotations

import pandas as pd

KP = "/mnt/d/Deepmolscan/Kinase_paper"
SRC = f"{KP}/result_87/redock_zscore_long.csv"
OUT = f"{KP}/analysis_output/prsdb_kinase_library.csv"

df = pd.read_csv(SRC, usecols=["MD5", "UNIPROT_ID", "GENE_NAME", "PROT_NAME"])
lib = (df.drop_duplicates(subset=["MD5"])
         .rename(columns={"MD5": "structure_id", "UNIPROT_ID": "uniprot_id",
                          "GENE_NAME": "gene_symbol", "PROT_NAME": "protein_name"})
         .sort_values("gene_symbol")
         .reset_index(drop=True))
for col in ("pdb_or_alphafold_accession", "model_version", "chain", "resolution_A",
            "consensus_pocket_residues"):
    lib[col] = ""          # PRSDB export 대기 — SI Methods S2 참조

lib.to_csv(OUT, index=False)
print(f"[D8] 구조 {len(lib)}개 · UniProt {lib.uniprot_id.nunique()}개 · "
      f"gene {lib.gene_symbol.nunique()}개")
assert len(lib) == 618, f"618 이어야 함 (실제 {len(lib)})"
print(f"✅ 저장 → {OUT}")
