"""
Benchmark of four scoring functions (AK-Score2, the AD4 energy, GenScore, RTMScore) across the
KIR compounds.

One pipeline throughout: a single 50% activity threshold, the same cohort, ranking within the
kinases the panel measures, eleven metrics plus Hit@k. Each score's z-score direction is
determined automatically from its correlation with rank_mean, so an inverted sign convention
cannot slip through unnoticed.

Output: Kinase_paper/energy_benchmark/
"""
from __future__ import annotations
import sys as _s; _s.path.insert(0,"/mnt/d/Deepmolscan/Kinase_paper"); import figlabels as _fl
import os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/mnt/d/Deepmolscan/PKIS2")
import analyze_pkis as ap   # compute_metrics, ranking_metrics, MIN_ACTIVES
sys.path.insert(0,"/mnt/d/Deepmolscan/Kinase_paper"); import kinome_merge as km
import sys; sys.path.insert(0,"/mnt/d/Deepmolscan/Kinase_paper"); import novel_cohort as nc

KP="/mnt/d/Deepmolscan/Kinase_paper"; OUT=f"{KP}/energy_benchmark"; os.makedirs(OUT,exist_ok=True)
METH={"AK_E":f"{KP}/result_87/redock_zscore_long.csv",
      "BINDING_E":f"{KP}/DOCK/redock_zscore_long.csv",
      "Gen":f"{KP}/Gen/redock_zscore_long.csv",
      "RTM":f"{KP}/RTM/redock_zscore_long.csv"}
COL={"AK_E":"#2166AC","BINDING_E":"#4393C3","Gen":"#F4A582","RTM":"#5AAE61"}
# 그림 표시용 라벨 — 내부 키(BINDING_E 등)는 CSV 열명과 맞춰 두고 그림에서만 읽기 쉬운 이름을 쓴다.
DISP = {"AK_E": "AK-Score2", "BINDING_E": "AD4 energy", "Gen": "GenScore",
        "RTM": "RTMScore", "DOCK": "AD4 energy", "consensus": "consensus"}
def dsp(m): return DISP.get(str(m), str(m))


# ── 실험(KIR): %remaining → %inhibition, active if inh>=50 (remaining<50) ──
exp=pd.read_csv(f"{KP}/kinase_exp_uniprot_mapped_matrix.csv",index_col=0)
col2uni={c:c.split("|")[2] for c in exp.columns}
el=exp.stack().reset_index(); el.columns=["DRUG_NAME","col","rem"]
el["UNIPROT_ID"]=el["col"].map(col2uni); el["rem"]=pd.to_numeric(el["rem"],errors="coerce")
el=el.dropna(subset=["rem"]); el["exp_inh"]=100-el["rem"]
exp_agg=el.groupby(["DRUG_NAME","UNIPROT_ID"],as_index=False)["exp_inh"].max()
exp_agg["label"]=(exp_agg["exp_inh"]>=50).astype(int)
exp_agg=nc.drop_leaked(exp_agg,"KIR")   # 학습 미사용 코호트

# ── 각 스코어 로드 + 방향 자동 판정 ──
def load_pred(f):
    df=pd.read_csv(f)
    df["DRUG_NAME"]=df["DRUG_NAME"].astype(str).str.strip().replace(ap.DRUG_ALIAS)
    df["UNIPROT_ID"]=km.norm_uni(df["UNIPROT_ID"])   # 복합 accession 정규화(표 1 pool과 일치)
    agg=df.groupby(["DRUG_NAME","UNIPROT_ID"],as_index=False).agg(
        zscore_mean=("zscore_mean","min"), rank_mean=("rank_mean","min"))
    # rank_mean(1=best)과 zscore의 상관으로 방향 판정
    corr=np.corrcoef(agg.zscore_mean,agg.rank_mean)[0,1]
    sign=1.0 if corr>0 else -1.0   # corr>0: 낮은 zscore=낮은 rank=best → pred=-zscore
    agg["pred_score"]=-sign*agg["zscore_mean"]
    return agg,corr,sign

common=None; preds={}
for m,f in METH.items():
    agg,corr,sign=load_pred(f)
    preds[m]=agg
    cset=set(agg.DRUG_NAME)
    common=cset if common is None else common&cset
    print(f"[{m}] zscore~rank corr={corr:+.2f} → pred=-({sign:+.0f})*zscore  약물{agg.DRUG_NAME.nunique()}")
common&=set(exp_agg.DRUG_NAME)
common=sorted(common)
print(f"\n3종∩실험 공통 화합물: {len(common)}")

# ── 벤치마크 ──
def bench(m):
    p=preds[m]
    merged=exp_agg[exp_agg.DRUG_NAME.isin(common)].merge(
        p[["DRUG_NAME","UNIPROT_ID","pred_score"]],on=["DRUG_NAME","UNIPROT_ID"])
    g=ap.compute_metrics(merged.label.values,merged.pred_score.values)
    # per-drug (Active>=5), 순위지표는 전체 패널(=merged; 단일50이라 동일)
    rows=[]
    for d,gg in merged.groupby("DRUG_NAME"):
        if gg.label.sum()>=ap.MIN_ACTIVES and (gg.label==0).sum()>=1:
            rows.append(ap.compute_metrics(gg.label.values,gg.pred_score.values))
    pd_=pd.DataFrame(rows)
    return g,pd_,merged

res={}
print(f"\n{'method':10s}{'G_ROC':>7s}{'G_pAUC':>8s}{'G_EF1':>7s}{'G_NEF1':>8s}{'G_BED':>7s}"
      f"{'PD_ROC':>8s}{'PD_n':>6s}{'Hit@1':>7s}{'Hit@5':>7s}{'Hit@10':>8s}{'Rprec':>7s}")
for m in METH:
    g,pdd,mg=bench(m); res[m]=(g,pdd,mg)
    print(f"{m:10s}{g['roc_auc']:>7.3f}{g['pauc10']:>8.3f}{g['ef1']:>7.2f}{g['nef1']:>8.3f}{g['bedroc']:>7.3f}"
          f"{pdd.roc_auc.mean():>8.3f}{len(pdd):>6d}{pdd.hit1.mean():>7.2f}{pdd.hit5.mean():>7.2f}"
          f"{pdd.hit10.mean():>8.2f}{pdd.r_precision.mean():>7.3f}")

# ── 요약 CSV ──
rowsC=[]
for m in METH:
    g,pdd,_=res[m]
    rowsC.append(dict(method=m,n_drug=len(pdd),
        G_ROC=g["roc_auc"],G_pAUC=g["pauc10"],G_PR=g["pr_auc"],G_EF1=g["ef1"],G_EF5=g["ef5"],
        G_NEF1=g["nef1"],G_BEDROC=g["bedroc"],
        PD_ROC=pdd.roc_auc.mean(),PD_ROC_sd=pdd.roc_auc.std(),PD_pAUC=pdd.pauc10.mean(),
        PD_NEF1=pdd.nef1.mean(),PD_BEDROC=pdd.bedroc.mean(),PD_Rprec=pdd.r_precision.mean(),
        PD_Hit1=pdd.hit1.mean(),PD_Hit5=pdd.hit5.mean(),PD_Hit10=pdd.hit10.mean()))
C=pd.DataFrame(rowsC); C.to_csv(f"{OUT}/energy_benchmark_summary.csv",index=False)

# ── 그림 ──
ms=list(METH); nm=len(ms); w=0.8/nm
def off(i): return (i-(nm-1)/2)*w
fig,ax=plt.subplots(1,3,figsize=(19,6))
def grp(a,labels,valfn,fmt,title,ylab,ylim=None,hline=None):
    xx=np.arange(len(labels))
    for i,m in enumerate(ms):
        vals=valfn(m)
        b=a.bar(xx+off(i),vals,w,label=dsp(m),color=COL[m],edgecolor="white")
        for bi,v in zip(b,vals): a.text(bi.get_x()+bi.get_width()/2,v,fmt.format(v),ha="center",va="bottom",fontsize=7)
    if hline is not None: a.axhline(hline,color="red",ls="--")
    a.set_xticks(xx);a.set_xticklabels(labels);a.set_ylabel(ylab);a.set_title(title)
    if ylim: a.set_ylim(*ylim)
    a.legend(fontsize=8)
grp(ax[0],["Global\nROC","Per-Drug\nROC","Global\npAUC"],
    lambda m:[res[m][0]["roc_auc"],res[m][1].roc_auc.mean(),res[m][0]["pauc10"]],
    "{:.3f}","Ranking quality","AUC",(0.45,0.78),0.5)
grp(ax[1],["Hit@1","Hit@5","Hit@10","R-prec"],
    lambda m:[res[m][1].hit1.mean(),res[m][1].hit5.mean(),res[m][1].hit10.mean(),res[m][1].r_precision.mean()],
    "{:.2f}","Target-ID (Hit@k)","per-drug mean")
grp(ax[2],["EF1%","NEF1","BEDROC"],
    lambda m:[res[m][0]["ef1"],res[m][0]["nef1"],res[m][0]["bedroc"]],
    "{:.2f}","Early enrichment","Global")
# 랭킹은 618개 구조 전체에서 하지만, 지표는 KIR 이 실제로 측정한 타겟에서만 정의된다.
# 두 수를 함께 밝혀야 표·본문과 어긋나지 않는다.
n_tgt_eval = res[list(METH)[0]][2].UNIPROT_ID.nunique()
fig.suptitle(f"KIR: scoring-function benchmark  |  {nc.tag()} cohort "
             f"(n={len(common)} compounds ranked within the {n_tgt_eval} kinases the panel measures)",
             fontsize=13.5,fontweight="bold")
_fl.stamp(ax)   # 패널 문자 (a)(b)(c)
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(f"{OUT}/energy_benchmark.png",dpi=300,bbox_inches="tight"); plt.close()

# ── MD ──
best=C.sort_values("PD_ROC",ascending=False).method.tolist()
def mrow(label,col,fmt):
    return f"| {label} | "+" | ".join(fmt.format(C[C.method==m][col].iloc[0]) for m in ms)+" |"
md=[f"# 스코어링 함수 벤치마크 ({' / '.join(ms)})\n","**분석일**: 2026-07-21 | KIR\n",
    f"공통 {len(common)}개 화합물 × {n_tgt_eval}개 측정 타겟 안에서 랭킹, 단일 임계값 50%.\n",
    "> 각 스코어 zscore 방향은 rank_mean 상관으로 자동 판정 (부호 반대 자동 교정).\n",
    "![bench](energy_benchmark.png)\n","## 요약\n",
    "| 지표 | "+" | ".join(ms)+" |","|------|"+":---:|"*nm,
    mrow("Global ROC","G_ROC","{:.3f}"), mrow("Global pAUC","G_pAUC","{:.3f}"),
    mrow("Global EF1%","G_EF1","{:.2f}"), mrow("Global BEDROC","G_BEDROC","{:.3f}"),
    mrow("**Per-Drug ROC**","PD_ROC","{:.3f}"), mrow("Per-Drug Hit@1","PD_Hit1","{:.2f}"),
    mrow("Per-Drug Hit@5","PD_Hit5","{:.2f}"), mrow("Per-Drug Hit@10","PD_Hit10","{:.2f}"),
    mrow("Per-Drug R-prec","PD_Rprec","{:.3f}"),
    f"\n**Per-Drug ROC 순위**: {' > '.join(best)}\n",
    "상세 수치: `energy_benchmark_summary.csv`"]
open(f"{OUT}/ENERGY_BENCHMARK.md","w").write("\n".join(md))
print(f"\n✅ 저장 → {OUT}/ (ENERGY_BENCHMARK.md, energy_benchmark_summary.csv, energy_benchmark.png)")
