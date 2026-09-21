"""
Definition of the leakage-free (novel) cohort - the single source for the paper's main results.

The paper establishes the leakage landscape first, then reports every performance figure on
compounds that were never part of the PDBbind v2020 general set. A compound counts as leaked when
the first 14 characters of its InChIKey - the 2D skeleton - exactly match a PDBbind kinase ligand.
The list comes from pdbbind_analysis/dataset_leaked_compounds.csv, which
dataset_leakage_compare.py generates.

Usage:
    import novel_cohort as nc
    M = nc.drop_leaked(M, "KIR")     # a no-op when NOVEL_ONLY=0

Running with NOVEL_ONLY=0 reproduces the full-cohort results, which the leakage analysis and the
supplementary contrast tables need.
"""
from __future__ import annotations

import os
import sys

import pandas as pd

KP = "/mnt/d/Deepmolscan/Kinase_paper"
LEAKED_CSV = f"{KP}/pdbbind_analysis/dataset_leaked_compounds.csv"

NOVEL_ONLY = os.environ.get("NOVEL_ONLY", "1") not in ("0", "false", "False")

_ALIAS = {"Barcitinib": "Baricitinib", "Vandetanib-Final": "Vandetanib"}


def _load() -> dict[str, set[str]]:
    d = pd.read_csv(LEAKED_CSV)
    out = {}
    for col in d.columns:
        names = {str(x).strip() for x in d[col].dropna()}
        out[col] = {_ALIAS.get(n, n) for n in names}
    return out


LEAKED = _load()


def drop_leaked(df: pd.DataFrame, dataset: str, col: str = "DRUG_NAME") -> pd.DataFrame:
    """dataset의 학습 누출 화합물을 제거. NOVEL_ONLY=0 이면 원본 반환."""
    if not NOVEL_ONLY:
        return df
    if dataset not in LEAKED:
        sys.exit(f"novel_cohort: 알 수 없는 데이터셋 {dataset!r} (가능: {sorted(LEAKED)})")
    return df[~df[col].astype(str).str.strip().isin(LEAKED[dataset])].copy()


def tag() -> str:
    return "leakage-free (novel)" if NOVEL_ONLY else "full cohort"


if __name__ == "__main__":
    import kinome_merge as km

    print(f"코호트 모드: {tag()}\n")
    for ds, s in LEAKED.items():
        print(f"  {ds}: 누출 {len(s)}개")
    print()
    e = km.experiment()
    m = km.merge_pred(e, f"{KP}/result_87/redock_zscore_long.csv")
    for label, mm in (("전체", m), ("novel", drop_leaked(m, "KIR"))):
        nd = mm.DRUG_NAME.nunique()
        pd_ok = sum(1 for _, g in mm.groupby("DRUG_NAME")
                    if g.label.sum() >= 1 and (g.label == 0).sum() >= 1)
        print(f"  KIR {label:5s}: 화합물 {nd}  per-drug 정의가능 {pd_ok}  쌍 {len(mm):,}")
    p2 = pd.read_csv("/mnt/d/Deepmolscan/PKIS2/PKIS2_result/analysis_merged_labeled.csv")
    for label, mm in (("전체", p2), ("novel", drop_leaked(p2, "PKIS2"))):
        nd = mm.DRUG_NAME.nunique()
        pd_ok = sum(1 for _, g in mm.groupby("DRUG_NAME")
                    if g.label.sum() >= 1 and (g.label == 0).sum() >= 1)
        print(f"  PKIS2      {label:5s}: 화합물 {nd}  per-drug 정의가능 {pd_ok}  쌍 {len(mm):,}")
