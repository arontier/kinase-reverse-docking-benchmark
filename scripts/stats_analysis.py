"""
Statistics for the manuscript tables: bootstrap 95% confidence intervals for the AK-Score2
headline metrics (global ROC and the per-compound metrics), and per-method per-drug ROC intervals
on KIR with paired Wilcoxon signed-rank p-values against AK-Score2.

Resampling is at the compound level, which is the unit of replication.

Output: Kinase_paper/analysis_output/STATS.md, stats_panel_ci.csv, stats_method.csv
"""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata, wilcoxon
sys.path.insert(0,"/mnt/d/Deepmolscan/PKIS2"); import analyze_pkis as ap
import sys; sys.path.insert(0,"/mnt/d/Deepmolscan/Kinase_paper"); import kinome_merge as km
import sys; sys.path.insert(0,"/mnt/d/Deepmolscan/Kinase_paper"); import novel_cohort as nc
KP="/mnt/d/Deepmolscan/Kinase_paper"; OUT=f"{KP}/analysis_output"; os.makedirs(OUT,exist_ok=True)
rng=np.random.default_rng(0); B=2000

def fast_auc(y,s):
    y=np.asarray(y); na=int(y.sum()); nn=len(y)-na
    if na<1 or nn<1: return np.nan
    r=rankdata(s); return (r[y==1].sum()-na*(na+1)/2)/(na*nn)

def load_pred(f):
    d=pd.read_csv(f); d["DRUG_NAME"]=d["DRUG_NAME"].replace(ap.DRUG_ALIAS)
    d["UNIPROT_ID"]=km.norm_uni(d["UNIPROT_ID"])   # 복합 accession 정규화(표 1 pool과 일치)
    d=d.groupby(["DRUG_NAME","UNIPROT_ID"],as_index=False).agg(z=("zscore_mean","min"),r=("rank_mean","min"))
    s=1.0 if np.corrcoef(d.z,d.r)[0,1]>0 else -1.0; d["pred_score"]=-s*d.z
    return d.rename(columns={"DRUG_NAME":"DRUG","UNIPROT_ID":"UNIPROT"})[["DRUG","UNIPROT","pred_score","z"]],s

def kinome():
    exp=pd.read_csv(f"{KP}/kinase_exp_uniprot_mapped_matrix.csv",index_col=0)
    c2u={c:c.split("|")[2] for c in exp.columns}
    el=exp.stack().reset_index(); el.columns=["DRUG","col","rem"]; el["UNIPROT"]=el["col"].map(c2u)
    el["rem"]=pd.to_numeric(el["rem"],errors="coerce"); el=el.dropna(subset=["rem"])
    ea=el.groupby(["DRUG","UNIPROT"],as_index=False)["rem"].min(); ea["label"]=(ea.rem<50).astype(int)
    pr,_=load_pred(f"{KP}/result_87/redock_zscore_long.csv")
    return ea.merge(pr[["DRUG","UNIPROT","pred_score"]],on=["DRUG","UNIPROT"])

MERGED={"KIR":kinome(),
        "PKIS2":pd.read_csv(f"{KP}/../PKIS2/PKIS2_result/analysis_merged_labeled.csv").rename(columns={"DRUG_NAME":"DRUG","UNIPROT_ID":"UNIPROT"})}
MERGED={k:nc.drop_leaked(v,k,col="DRUG") for k,v in MERGED.items()}   # 학습 미사용 코호트
print(f"코호트: {nc.tag()}  " + "  ".join(f"{k}={v.DRUG.nunique()}" for k,v in MERGED.items()))

def perdrug_df(M):
    rows=[]
    for d,g in M.groupby("DRUG"):
        if g.label.sum()>=ap.MIN_ACTIVES and (g.label==0).sum()>=1:
            r=ap.compute_metrics(g.label.values,g.pred_score.values); r["DRUG"]=d; rows.append(r)
    return pd.DataFrame(rows).set_index("DRUG")

def hodges_lehmann(d, alpha=0.05):
    """짝지은 차이의 Hodges-Lehmann 추정치 + Wilcoxon signed-rank 기반 CI.

    Wilcoxon 검정과 논리적으로 일치한다(CI가 0을 제외 <=> p<alpha). 부트스트랩 평균 CI를
    쓰면 순위 기반 검정과 결론이 어긋날 수 있어(예: AutoDock-GPU) 표 안에서 모순이 생긴다.
    """
    d = np.asarray(d, dtype=float); n = len(d)
    w = np.array([(d[i]+d[j])/2 for i in range(n) for j in range(i, n)])   # Walsh averages
    w.sort(); m = len(w)
    mu = n*(n+1)/4.0; sd = np.sqrt(n*(n+1)*(2*n+1)/24.0)
    from scipy.stats import norm as _norm
    k = max(int(np.floor(mu - _norm.ppf(1-alpha/2)*sd)), 0)
    return float(np.median(w)), float(w[k]), float(w[m-k-1])

def boot_mean_ci(vals):
    vals=np.asarray(vals); idx=rng.integers(0,len(vals),(B,len(vals)))
    bs=vals[idx].mean(1); return vals.mean(), np.percentile(bs,2.5), np.percentile(bs,97.5)

def boot_global_ci(M):
    drugs=M.DRUG.unique(); by={d:g for d,g in M.groupby("DRUG")}
    pt=fast_auc(M.label.values,M.pred_score.values); bs=[]
    for _ in range(1000):
        s=rng.choice(drugs,len(drugs),replace=True)
        cat=pd.concat([by[d] for d in s])
        bs.append(fast_auc(cat.label.values,cat.pred_score.values))
    return pt,np.percentile(bs,2.5),np.percentile(bs,97.5)

# ── (1) 패널별 AK-Score2 CI ──
rows=[]
PDS={}
for panel,M in MERGED.items():
    pdd=perdrug_df(M); PDS[panel]=pdd
    gm,glo,ghi=boot_global_ci(M)
    rows.append(dict(panel=panel,metric="Global ROC",mean=gm,lo=glo,hi=ghi))
    for met,col in [("Per-Drug ROC","roc_auc"),("Hit@1","hit1"),("Hit@5","hit5"),("Hit@10","hit10"),("R-precision","r_precision")]:
        m,lo,hi=boot_mean_ci(pdd[col].values); rows.append(dict(panel=panel,metric=met,mean=m,lo=lo,hi=hi))
PC=pd.DataFrame(rows); PC.to_csv(f"{OUT}/stats_panel_ci.csv",index=False)
print("[패널별 AK-Score2 95% CI]")
for panel in MERGED:
    print(f"\n{panel}:")
    for _,r in PC[PC.panel==panel].iterrows(): print(f"  {r.metric:14s} {r['mean']:.3f} [{r.lo:.3f}, {r.hi:.3f}]")

# ── (2) KIR 방법 비교 ──
METH={"AK_E":f"{KP}/result_87/redock_zscore_long.csv","BINDING_E":f"{KP}/DOCK/redock_zscore_long.csv",
      "Gen":f"{KP}/Gen/redock_zscore_long.csv","RTM":f"{KP}/RTM/redock_zscore_long.csv"}
ea=MERGED["KIR"][["DRUG","UNIPROT","label"]]
P={}; sign={}
for m,f in METH.items(): P[m],sign[m]=load_pred(f)
common=set(ea.DRUG)
for m in METH: common&=set(P[m].DRUG)
# consensus = AK_E·RTMScore의 동일가중 z평균 (§2.8의 정의와 일치)
ak=P["AK_E"].rename(columns={"pred_score":"sAK"}); rt=P["RTM"].rename(columns={"pred_score":"sRTM"})
mc=ak[["DRUG","UNIPROT","sAK"]].merge(rt[["DRUG","UNIPROT","sRTM"]],on=["DRUG","UNIPROT"])
def _znorm(g):
    sd=g.std(ddof=0)
    return (g-g.mean())/sd if sd>0 else g*0
for c in ("sAK","sRTM"):
    mc[f"z{c}"]=mc.groupby("DRUG")[c].transform(_znorm)
mc["pred_score"]=(mc["zsAK"]+mc["zsRTM"])/2.0; P["consensus"]=mc[["DRUG","UNIPROT","pred_score"]]
def perdrug_roc(pred):
    mg=ea.merge(pred[["DRUG","UNIPROT","pred_score"]],on=["DRUG","UNIPROT"]); r={}
    for d,g in mg.groupby("DRUG"):
        if g.label.sum()>=ap.MIN_ACTIVES and (g.label==0).sum()>=1:
            r[d]=ap.compute_metrics(g.label.values,g.pred_score.values)["roc_auc"]
    return pd.Series(r)
ROC={m:perdrug_roc(P[m]) for m in ["AK_E","BINDING_E","Gen","RTM","consensus"]}
akv=ROC["AK_E"]; rows2=[]
for m in ["AK_E","BINDING_E","Gen","RTM","consensus"]:
    v=ROC[m]; mn,lo,hi=boot_mean_ci(v.values)
    # 짝지은 차이(Δ vs AK)의 부트스트랩 CI — 표 안에서 평균/CI와 Wilcoxon p의 pool을 일치시킨다.
    # 독립 CI끼리의 겹침으로 짝지은 유의성을 판단하면 안 되므로 Δ의 CI를 함께 보고한다.
    if m=="AK_E": p=np.nan; dm=dlo=dhi=np.nan
    else:
        idx=akv.index.intersection(v.index); p=wilcoxon(akv.loc[idx],v.loc[idx]).pvalue
        dm,dlo,dhi=hodges_lehmann((v.loc[idx]-akv.loc[idx]).values)
    rows2.append(dict(method=m,perdrug_roc=mn,lo=lo,hi=hi,n=len(v),
                      delta_vs_AK=dm,delta_lo=dlo,delta_hi=dhi,wilcoxon_p_vs_AK=p))
MC=pd.DataFrame(rows2); MC.to_csv(f"{OUT}/stats_method.csv",index=False)
print("\n[KIR 방법별 Per-Drug ROC 95% CI + Δ vs AK 95% CI + Wilcoxon p]")
for _,r in MC.iterrows():
    pp="—" if np.isnan(r.wilcoxon_p_vs_AK) else f"{r.wilcoxon_p_vs_AK:.3f}"
    dd="reference" if np.isnan(r.delta_vs_AK) else f"{r.delta_vs_AK:+.3f} [{r.delta_lo:+.3f}, {r.delta_hi:+.3f}]"
    print(f"  {r.method:10s} {r.perdrug_roc:.3f} [{r.lo:.3f}, {r.hi:.3f}]  Δ={dd:26s} p={pp}")

md=["# 통계 요약 (부트스트랩 95% CI · Wilcoxon)\n","**분석일**: 2026-07-21\n","## AK-Score2 패널별 95% CI\n",
    "| 패널 | 지표 | 값 [95% CI] |","|---|---|---|"]
for _,r in PC.iterrows(): md.append(f"| {r.panel} | {r.metric} | {r['mean']:.3f} [{r.lo:.3f}, {r.hi:.3f}] |")
md+=["\n## KIR 방법 비교 (Per-Drug ROC)\n",
     "> Δ vs AK는 짝지은 차이의 Hodges-Lehmann 추정치와 Wilcoxon 기반 95% CI. 검정과 논리적으로 일치하므로\n> 독립 CI의 겹침이 아니라 이 열로 유의성을 읽는다.\n",
     "| 방법 | Per-Drug ROC [95% CI] | Δ vs AK [95% CI] | AK 대비 Wilcoxon p |","|---|---|---|---|"]
for _,r in MC.iterrows():
    pp="—(기준)" if np.isnan(r.wilcoxon_p_vs_AK) else f"{r.wilcoxon_p_vs_AK:.3f}"
    dd="—(기준)" if np.isnan(r.delta_vs_AK) else f"{r.delta_vs_AK:+.3f} [{r.delta_lo:+.3f}, {r.delta_hi:+.3f}]"
    md.append(f"| {r.method} | {r.perdrug_roc:.3f} [{r.lo:.3f}, {r.hi:.3f}] | {dd} | {pp} |")
open(f"{OUT}/STATS.md","w").write("\n".join(md))
print(f"\n✅ 저장 → {OUT}/ (STATS.md, stats_panel_ci.csv, stats_method.csv)")
