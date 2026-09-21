"""
Table S10 - every metric Methods 4.6 defines, including those not quoted in the Results.

Partial ROC AUC, PR AUC, EF5%, NEF, BEDROC and Recall@10 are defined in the Methods but appear
nowhere else, which review flagged. analyze_pkis.compute_metrics already computes all of them, so
this script only has to dump them for both panels x four scoring functions.

Output: Kinase_paper/analysis_output/full_metrics.csv, FULL_METRICS.md
"""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd
KP="/mnt/d/Deepmolscan/Kinase_paper"; PK="/mnt/d/Deepmolscan/PKIS2"
sys.path.insert(0,PK); sys.path.insert(0,KP)
import analyze_pkis as ap, kinome_merge as km, novel_cohort as nc
OUT=f"{KP}/analysis_output"; os.makedirs(OUT,exist_ok=True)

SRC={"KIR":{"AK-Score2":f"{KP}/result_87/redock_zscore_long.csv",
                   "AD4 energy":f"{KP}/DOCK/redock_zscore_long.csv",
                   "GenScore":f"{KP}/Gen/redock_zscore_long.csv",
                   "RTMScore":f"{KP}/RTM/redock_zscore_long.csv"},
     "PKIS2":{"AK-Score2":f"{PK}/PKIS2_result/redock_zscore_long.csv",
              "RTMScore":f"{PK}/RTM/redock_zscore_long.csv",
              "GenScore":f"{PK}/Gen2/redock_zscore_long.csv"}}

def load(f):
    d=pd.read_csv(f); d["DRUG_NAME"]=d["DRUG_NAME"].astype(str).str.strip().replace(ap.DRUG_ALIAS)
    d["UNIPROT_ID"]=km.norm_uni(d["UNIPROT_ID"])
    g=d.groupby(["DRUG_NAME","UNIPROT_ID"],as_index=False).agg(z=("zscore_mean","min"),r=("rank_mean","min"))
    s=1.0 if np.corrcoef(g.z,g.r)[0,1]>0 else -1.0
    g["pred"]=-s*g.z
    return g[["DRUG_NAME","UNIPROT_ID","pred"]]

def experiment(panel):
    if panel=="KIR":
        return nc.drop_leaked(km.experiment(),"KIR")[["DRUG_NAME","UNIPROT_ID","label"]]
    e=ap.load_experiment(ap.CONFIGS["PKIS2"]); e["label"]=(e["exp_inh"]>=50).astype(int)
    return nc.drop_leaked(e,"PKIS2")[["DRUG_NAME","UNIPROT_ID","label"]]

rows=[]
for panel,meths in SRC.items():
    E=experiment(panel)
    for meth,f in meths.items():
        M=E.merge(load(f),on=["DRUG_NAME","UNIPROT_ID"])
        g=ap.compute_metrics(M.label.values,M.pred.values)
        per=[ap.compute_metrics(x.label.values,x.pred.values) for _,x in M.groupby("DRUG_NAME")
             if x.label.sum()>=ap.MIN_ACTIVES and (x.label==0).sum()>=1]
        P=pd.DataFrame(per)
        rows.append(dict(panel=panel,method=meth,n_drug=len(P),
            G_ROC=g["roc_auc"],G_pAUC10=g["pauc10"],G_PR=g["pr_auc"],
            G_EF1=g["ef1"],G_EF5=g["ef5"],G_NEF1=g["nef1"],G_NEF5=g["nef5"],G_BEDROC=g["bedroc"],
            PD_ROC=P.roc_auc.mean(),PD_pAUC10=P.pauc10.mean(),PD_PR=P.pr_auc.mean(),
            PD_EF1=P.ef1.mean(),PD_EF5=P.ef5.mean(),PD_NEF1=P.nef1.mean(),PD_NEF5=P.nef5.mean(),
            PD_BEDROC=P.bedroc.mean(),PD_Rprec=P.r_precision.mean(),PD_Recall10=P.recall10.mean()))
        print(f"  {panel:11s} {meth:11s} n={len(P):4d}")
T=pd.DataFrame(rows); T.to_csv(f"{OUT}/full_metrics.csv",index=False)
print("\n"+T.round(3).to_string(index=False))

md=["# Table S10 — 전체 지표 (Methods 4.6 정의 전부)\n",
    f"학습 미사용 코호트. BEDROC alpha=20, pAUC 는 FPR<=0.1 의 McClish 표준화 값, "
    "EF/NEF 는 상위 1%/5%. Global = 패널 전체 쌍, Per-drug = 화합물별 평균.\n",
    "| Panel | Method | n | ROC | pAUC10 | PR-AUC | EF1% | EF5% | NEF1 | NEF5 | BEDROC | R-prec | Recall@10 |",
    "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
for _,r in T.iterrows():
    md.append(f"| {r.panel} | {r.method} | {r.n_drug} | {r.PD_ROC:.3f} | {r.PD_pAUC10:.3f} | "
              f"{r.PD_PR:.3f} | {r.PD_EF1:.2f} | {r.PD_EF5:.2f} | {r.PD_NEF1:.3f} | {r.PD_NEF5:.3f} | "
              f"{r.PD_BEDROC:.3f} | {r.PD_Rprec:.3f} | {r.PD_Recall10:.3f} |")
open(f"{OUT}/FULL_METRICS.md","w").write("\n".join(md))
print(f"\n✅ 저장 → {OUT}/full_metrics.csv, FULL_METRICS.md")
