# A fully measured reverse-docking benchmark for kinases

Data, figures, tables and analysis code accompanying the manuscript
*"A fully measured reverse-docking benchmark for kinases and evaluation of AK-Score2 as a
target-prioritization tool"* (Arontier).

> **Status.** Manuscript under review. Author list, funding, CRediT roles and the archival DOI
> are not yet final; this repository will be tagged and deposited when they are.
>
> Repository: <https://github.com/arontier/kinase-reverse-docking-benchmark>

Reverse docking ranks many protein structures against a single ligand. Benchmarks for it are
usually built on sparse activity matrices where missing values must be imputed, which makes the
reported numbers hard to trust. This benchmark avoids that: both primary panels are **complete at
their assay concentration**, so every compound–kinase pair carries a real measurement and no
imputation is involved.

---

## What is in here

```
.
├── tables/        23 tables as printed in the paper, extracted from the manuscript itself
│   ├── main/      Table 1-6
│   └── si/        Table S1-S13 (incl. S5b, S12b) + the Supplementary Data index
├── figures/       16 figures as published, each next to the script that produced it
│   ├── main/      Figure 1-9
│   ├── si/        Figure S1-S7
│   └── _lib/      shared modules the figure scripts import
├── data/          Supplementary Data D1-D8
│   ├── raw/       the experimental matrices and SMILES the analysis reads as input
│   ├── D6_predictions/   raw per-pair predictions, four scoring functions x three panels
│   └── FILE_MAP.csv      maps the paths printed in the SI to the paths used here
├── scripts/       the analysis pipeline (Supplementary Data D7)
└── MANIFEST.csv   every file with its size, SHA-256, original path and a one-line
                    description — also the integrity record for the release
```

Files larger than 2 MB are gzipped. `pandas.read_csv` opens `.gz` directly — no manual step.
Total size is about 64 MB, of which ~52 MB is `data/D6_predictions/`.

The tables are extracted from the manuscript and Supplementary Information documents, not
recomputed, so they cannot drift from the printed values. One of the 24 CSVs in `tables/si/` is
the SI's own index of D1-D8 rather than a results table; because it is extracted verbatim it
lists the authoring paths, and `data/FILE_MAP.csv` translates those into the paths used here.

---

## Quick start

```bash
git clone https://github.com/arontier/kinase-reverse-docking-benchmark.git
cd kinase-reverse-docking-benchmark
```

Reproducing the paper's headline number — AK-Score2 per-drug ROC on KIR — needs nothing
outside this repository:

```python
import pandas as pd
from sklearn.metrics import roc_auc_score

# D1 is the primary reproduction file: one row per compound-kinase pair.
ks = pd.read_csv("data/D1_merged_labeled_kir.csv")
print(ks.shape, ks.UNIPROT_ID.nunique())        # (33141, 7)  369 kinases = the ranking pool

# Restrict to the leakage-free cohort. Table S2 lists the compounds that match a
# PDBbind kinase ligand by InChIKey14.
leaked = set(pd.read_csv("tables/si/Table_S2_training_set_matches.csv")["KIR"].dropna())
novel = ks[~ks.DRUG_NAME.isin(leaked)]
print(novel.DRUG_NAME.nunique())                # 61 of 92 compounds remain

# Per-drug ROC is defined only where a compound has both an active and an inactive measurement.
roc = [roc_auc_score(g.label, g.pred_score)
       for _, g in novel.groupby("DRUG_NAME") if g.label.nunique() == 2]
print(len(roc), round(sum(roc) / len(roc), 3))  # 53  0.700  — Table 4
```

Dropping the leakage filter gives 0.687 over 84 compounds, the full-cohort contrast reported in
Table S12.

---

## Where the data comes from

| Panel | Source |
|---|---|
| **KIR** — Kinase Inhibitor Resource | Saifudeen M, Zhu S, Liang S, Eason M, Goupil A, Mische DF, Loch CM, Ma H, Chan M, Gujral TS. *Comprehensive profiling of clinically approved kinase inhibitors reveals mutation-specific inhibitors and opportunities for drug repurposing.* Nat Biotechnol. 2026. doi:10.1038/s41587-026-03090-8. Supplementary Table S1: 92 clinically advanced inhibitors profiled against 409 wild-type kinases (of 758 assays including 349 oncogenic variants) in a **biochemical kinase assay**. Variants are excluded here because PRSDB carries one wild-type structure per kinase gene. |
| **PKIS2** — GSK Published Kinase Inhibitor Set 2 | Drewry et al., *Curr Top Med Chem* 2014; Drewry et al., *PLoS One* 2017; Elkins et al., *Nat Biotechnol* 2016. %inhibition at 1 µM. |
| **Kinase_ref** | Values transcribed from eleven medicinal-chemistry papers, listed with their citations in `tables/main/Table_2_kinase_ref_provenance.csv`. These series were profiled on Eurofins DiscoverX KINOMEscan and other commercial platforms; that column records the platform of each series. |

KIR reports the percentage of kinase **activity** remaining, not competition binding. Both panels
are converted to %inhibition (100 − %remaining for KIR) and a single threshold of %inhibition ≥ 50
defines an active interaction throughout.

---

## Supplementary Data

| File | Contents |
|---|---|
| **D1** | `data/D1_merged_labeled_kir.csv`, `data/D1_merged_labeled_pkis2.csv.gz` — compound × kinase table with the experimental value, the binary label and the direction-corrected prediction. Every per-drug, per-target and global metric in the paper is computed from this. |
| **D2** | `data/D2_kinase_ref_panels.tsv.gz` — the Kinase_ref compound × kinase listing (19,159 rows over 403 kinases) with source series, panel size, platform, citation, unit, value, label, pool membership, and a `reported` flag separating values the source paper released from panel entries reconstructed as described in Methods S3. |
| **D3** | `data/D3_per_drug_roc_ci.csv` — per-compound ROC with Hanley–McNeil 95% intervals for both panels. Source of Tables S1 and S11. |
| **D4** | `data/D4_per_target_metrics.csv`, `data/D4_target_preference_vs_pdbbind.csv` — per-target metrics and the ligand-independent target preference against PDBbind training frequency, for all 618 library entries. Source of Table S4. |
| **D5** | `data/D5_protein_coverage.csv` — every UniProt entry reachable from the experimental panels, flagged for presence in the experiment, in the prediction pool and in the merged analysis, with the reason for any exclusion. |
| **D6** | `data/D6_predictions/{panel}__{method}.csv.gz` — raw per-pair predictions before merging, for AK-Score2, RTMScore, GenScore and the AD4 energy. |
| **D7** | `scripts/` — the analysis code. |
| **D8** | `data/D8_prsdb_kinase_library.csv` — structure-to-protein mapping for the 618-structure target library, so any ranking position can be traced to a named protein. |

Beyond D1–D8, `data/raw/` holds the inputs the analysis scripts actually read: the KIR
%remaining matrix and PKIS2 %inhibition matrix, the assay-to-UniProt mapping, the compound
SMILES for both panels, and the PDBbind-derived identifiers used for leakage detection. They
total under 1 MB and are included so the pipeline can be re-run without assembling them again.

---

## Three definitions worth knowing before you use the data

**Ranking pool.** The 618 PRSDB kinase structures are the *docking target space*, not the ranking
pool. Every ranking metric in the paper is computed **within the kinases the corresponding panel
actually measured** — 369 for KIR, 387 for PKIS2, and the panel each source paper profiled
for Kinase_ref (median 380). Kinases a panel never assayed are not used as decoys: a never-measured
structure appearing high in a ranking can be neither confirmed as a false positive nor ruled out as
an unknown true positive, so admitting them would bias every recovery metric downward by an amount
the data cannot bound. The blind kinome-wide alternative is reported as a sensitivity analysis in
`tables/si/Table_S12b_blind_kinome_wide.csv` and Figure S7.

**Leakage-free cohort.** AK-Score2, GenScore and RTMScore were all trained on PDBbind-derived data.
Headline numbers are therefore reported on compounds whose InChIKey14 does not match any PDBbind
kinase ligand. `scripts/novel_cohort.py` defines it; setting `NOVEL_ONLY=0` reproduces the full
cohort for contrast.

**Reconstructed panels in Kinase_ref.** Nine compounds come from papers that published only the
kinases at or below a 35% remaining cut-off. Their panels were reconstructed from other papers
using the same assay platform, and the papers' own reporting cut-off was adopted as the activity
threshold for that series — which makes "unreported implies inactive" exact by construction rather
than an assumption. The `reported` column in D2 marks which rows are released values and which are
reconstructed. Methods S3 gives the procedure and its limits.

---

## Figures

Each PNG sits next to the script that produced it. Twelve of the sixteen carry their generating
code; see *Reproducing the analysis* for what each still needs as input.

| | Main text | | Supplementary |
|---|---|---|---|
| 1 | Overview of the PRSDB structure and binding-site database † | S1 | Agreement between scoring functions after direction correction |
| 2 | Protein-level training-data leakage | S2 | Why the two panels differ ‡ |
| 3 | Ligand-level leakage rates across benchmark datasets | S3 | Per-target ROC by Manning kinase group ‡ |
| 4 | Leave-leaked-out analysis on KIR | S4 | Why some compounds score below chance ‡ |
| 5 | Comparison of scoring functions on KIR | S5 | Ligand-independent top-1 target collapse |
| 6 | Primary-target recovery across benchmark panels | S6 | Statistical status of per-drug ROC below 0.5 |
| 7 | Target-identification success rates | S7 | Kinase_ref recovery under the two ranking-pool definitions |
| 8 | Kinase_ref target recovery by single scoring functions | | |
| 9 | PKIS2: AK-Score2 versus RTMScore versus consensus | | |

† Figure 1 is a schematic, not a script output. No editable source exists outside the manuscript;
the published bitmap is included with a note.
‡ Figures S2, S3 and S4 were produced in an earlier round and their generating code is no longer in
the project. The PNGs and the underlying numbers are here, but these three cannot currently be
regenerated. Each carries a `.SOURCE.txt` saying so.

---

## Reproducing the analysis

```bash
python -m pip install pandas numpy scipy scikit-learn matplotlib rdkit python-docx
```

Developed on Python 3.11. Exact docking and rescoring parameters are in Supplementary Methods S1.

**What reproduces from this repository alone.** Every number in the paper's tables. The metrics are
computed from D1, D2, D3 and D8, all of which are here, and the Quick start above reproduces the
headline per-drug ROC exactly. `data/raw/` additionally carries the experimental matrices and
SMILES that the analysis scripts read, so the leakage, recovery and consensus analyses can be
re-run end to end.

**What does not.** The docking itself. Poses come from AutoDock-GPU against PRSDB structures and
are rescored by AK-Score2; neither the structure database nor the model is released here, so the
chain starts from the predictions in D6 rather than from the structures. D6 is what the paper
reports, so starting there reproduces every published number.

Three practical notes before running `scripts/`:

- **Paths are absolute.** The scripts are released exactly as they were run and reference the
  authoring environment (`/mnt/d/Deepmolscan/...`). Repoint the `KP`, `PK` and `KR` constants at
  your copy, and the PDBbind lookup at `data/raw/pdbbind_uniprot_complex_counts.csv`.
- **Inline comments are in Korean.** Every module docstring - the "what does this file do"
  header - is English, as are identifiers, printed output and this README. The inline comments
  inside the functions are Korean, which is how the analysis was written.
- **Docking is not reproducible bit-for-bit.** AutoDock-GPU seeds from the system clock, so
  individual runs differ. Aggregate statistics are stable to repeated runs; Table S7 quantifies it.

## What is not included

PRSDB and AK-Score2 are proprietary assets of Arontier. This repository releases their **outputs**
— the predictions, the merged tables and the analysis code — not the database itself or the model
weights. RTMScore, GenScore and AutoDock-GPU are third-party tools; see Methods S1 for versions and
their own licences.

PDBbind v2020 is redistributed only as derived identifiers: `data/raw/pdbbind_kinase_ligands.csv`
(PDB ID, UniProt, InChIKey14 of the kinase ligands) and
`data/raw/pdbbind_uniprot_complex_counts.csv` (complexes per UniProt entry). These are what the
leakage analysis needs; the PDBbind index and structures themselves must be obtained from
PDBbind under their own terms.

`data/D8_prsdb_kinase_library.csv` carries the structure identifier, UniProt accession, gene symbol
and protein name for all 618 structures. The remaining provenance columns (PDB or AlphaFold
accession and model version, chain, resolution, consensus pocket residues) are present but empty,
pending an export from PRSDB before submission.

---

## Licence

Code in `scripts/` and `figures/`: MIT.
Data in `data/` and `tables/`: CC BY 4.0.

Derived from third-party sources under their own terms: the Kinase Inhibitor Resource
(Saifudeen et al., Nat Biotechnol 2026), GSK PKIS2, PDBbind
v2020, RCSB PDB, AlphaFold DB, UniProt, and the eleven medicinal-chemistry papers listed in
`tables/main/Table_2_kinase_ref_provenance.csv`.

## Citation

```bibtex
@article{arontier_kinase_reverse_docking,
  title   = {A fully measured reverse-docking benchmark for kinases and evaluation of
             AK-Score2 as a target-prioritization tool},
  author  = {[author list pending]},
  journal = {[under review]},
  year    = {2026},
  note    = {Data and code: https://github.com/arontier/kinase-reverse-docking-benchmark}
}
```
