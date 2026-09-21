"""
Label-based per-compound ROC on Kinase_ref, over the measured-panel pool, for all four scoring
functions.

The value quoted earlier in the manuscript covered only AK-Score2 and RTMScore and predated the
panel reconstruction. Reconstruction raised the label-usable cohort from 33 to 42 compounds, so
all four functions are recomputed here.

The pool is the panel each source paper measured, intersected with the prediction pool. Score
direction is corrected as in ref_panel_eval.py: lower is stronger for AK-Score2 and the AD4
energy, higher is stronger for RTMScore and GenScore.

Output: Kinase_paper/kinase_ref_check/ref_labeled_roc.csv, ref_labeled_roc_per_compound.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/mnt/d/Deepmolscan/Kinase_paper")

KR = "/mnt/d/Deepmolscan/Kinase_ref"
OUT = "/mnt/d/Deepmolscan/Kinase_paper/kinase_ref_check"


def load_long(path: str) -> dict[str, dict[str, float]]:
    d = (pd.read_csv(path).groupby(["DRUG_NAME", "UNIPROT_ID"], as_index=False)
                          .agg(z=("zscore_mean", "min")))
    return {c: dict(zip(g.UNIPROT_ID.astype(str), g.z)) for c, g in d.groupby("DRUG_NAME")}


PRED = {"AK": load_long(f"{KR}/AK/redock_zscore_long.csv"),
        "RTM": load_long(f"{KR}/RTM/redock_zscore_long.csv"),
        "DOCK": load_long(f"{KR}/DOCK/redock_zscore_long.csv"),
        "GEN": load_long("/mnt/d/Deepmolscan/Kinase_paper/Gen2/redock_zscore_long.csv")}
SIGN = {"AK": -1.0, "RTM": 1.0, "DOCK": -1.0, "GEN": 1.0}   # 곱한 뒤 높을수록 강한 결합
DISP = {"AK": "AK-Score2", "RTM": "RTMScore", "DOCK": "AD4 energy", "GEN": "GenScore"}

P = pd.read_csv(f"{OUT}/KINASE_REF_PANELS.tsv", sep="\t")
P = P[(P.in_prediction_pool == 1) & P.label_active.notna()].copy()
P["label_active"] = P.label_active.astype(int)

rows = []
for cpd, g in P.groupby("compound"):
    g = g.drop_duplicates(subset=["uniprot"])
    y = g.label_active.values
    if y.sum() < 1 or (1 - y).sum() < 1:
        continue
    rec = dict(compound=cpd, n_pool=len(g), n_active=int(y.sum()))
    for m, tab in PRED.items():
        s = g.uniprot.map(tab.get(cpd, {}))
        if s.isna().any():
            rec[f"{m}_roc"] = np.nan
            continue
        rec[f"{m}_roc"] = roc_auc_score(y, SIGN[m] * s.values)
    rows.append(rec)

R = pd.DataFrame(rows)
R.to_csv(f"{OUT}/ref_labeled_roc_per_compound.csv", index=False)

summ = pd.DataFrame([dict(
    method=DISP[m], n=int(R[f"{m}_roc"].notna().sum()),
    mean_roc=R[f"{m}_roc"].mean(), sd=R[f"{m}_roc"].std(ddof=1),
    median_roc=R[f"{m}_roc"].median()) for m in PRED])
summ.to_csv(f"{OUT}/ref_labeled_roc.csv", index=False)

print(f"[A5] 라벨 사용가능 화합물 {len(R)}개 · 측정 패널 pool 중앙값 {R.n_pool.median():.0f}")
print(summ.round(3).to_string(index=False))
print(f"\n범위: {summ.mean_roc.min():.2f}–{summ.mean_roc.max():.2f}")
print(f"\n✅ 저장 → {OUT}/ref_labeled_roc.csv")
