"""
Leave-leaked-out analysis on KIR: compare each method on compounds that exactly match a
PDBbind training ligand against those that do not, testing whether AK-Score2 is more robust to
leakage than RTMScore.

Leakage status comes from pdbbind_analysis/ligand_leakage_per_drug.csv, where an exact match is an
identical InChIKey14.

Output: Kinase_paper/pdbbind_analysis/ (LEAVE_LEAKED_OUT.md, leave_leaked_out.png, csv)
"""
from __future__ import annotations
import sys as _s; _s.path.insert(0,"/mnt/d/Deepmolscan/Kinase_paper"); import figlabels as _fl
import os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, wilcoxon
sys.path.insert(0, "/mnt/d/Deepmolscan/PKIS2"); import analyze_pkis as ap
import sys; sys.path.insert(0,"/mnt/d/Deepmolscan/Kinase_paper"); import kinome_merge as km

KP="/mnt/d/Deepmolscan/Kinase_paper"; OUT=f"{KP}/pdbbind_analysis"
METH={"AK_E":f"{KP}/result_87/redock_zscore_long.csv","BINDING_E":f"{KP}/DOCK/redock_zscore_long.csv",
      "Gen":f"{KP}/Gen/redock_zscore_long.csv","RTM":f"{KP}/RTM/redock_zscore_long.csv"}
RRF_K=60
COL={"AK_E":"#2166AC","BINDING_E":"#4393C3","Gen":"#F4A582","RTM":"#5AAE61","consensus":"#762A83"}

# 실험
exp=pd.read_csv(f"{KP}/kinase_exp_uniprot_mapped_matrix.csv",index_col=0)
c2u={c:c.split("|")[2] for c in exp.columns}
el=exp.stack().reset_index(); el.columns=["DRUG","col","rem"]; el["UNIPROT"]=el["col"].map(c2u)
el["rem"]=pd.to_numeric(el["rem"],errors="coerce"); el=el.dropna(subset=["rem"])
ea=el.groupby(["DRUG","UNIPROT"],as_index=False)["rem"].min(); ea["label"]=(ea.rem<50).astype(int)

def load_pred(f):
    d=pd.read_csv(f); d["DRUG_NAME"]=d["DRUG_NAME"].replace(ap.DRUG_ALIAS)
    d["UNIPROT_ID"]=km.norm_uni(d["UNIPROT_ID"])   # 복합 accession 정규화(표 1 pool과 일치)
    d=d.groupby(["DRUG_NAME","UNIPROT_ID"],as_index=False).agg(z=("zscore_mean","min"),r=("rank_mean","min"))
    s=1.0 if np.corrcoef(d.z,d.r)[0,1]>0 else -1.0
    d["pred"]=-s*d.z; return d.rename(columns={"DRUG_NAME":"DRUG","UNIPROT_ID":"UNIPROT"})[["DRUG","UNIPROT","pred","z"]],s
P={}; sign={}
for m,f in METH.items(): P[m],sign[m]=load_pred(f)
common=set(ea.DRUG)
for m in METH: common&=set(P[m].DRUG)
common=sorted(common)

# consensus(RRF) on AK+RTM (동일 화합물 내 순위)
ak=P["AK_E"].rename(columns={"z":"zAK"})[["DRUG","UNIPROT","zAK"]]
rt=P["RTM"].rename(columns={"z":"zRTM"})[["DRUG","UNIPROT","zRTM"]]
mc=ak.merge(rt,on=["DRUG","UNIPROT"])
mc["rAK"]=mc.groupby("DRUG").zAK.rank(method="min",ascending=(sign["AK_E"]>0))
mc["rRTM"]=mc.groupby("DRUG").zRTM.rank(method="min",ascending=(sign["RTM"]>0))
mc["pred"]=1.0/(RRF_K+mc.rAK)+1.0/(RRF_K+mc.rRTM)
P["consensus"]=mc[["DRUG","UNIPROT","pred"]]
METHODS=["AK_E","BINDING_E","Gen","RTM","consensus"]
# 본문 중간부 그림은 단일 스코어링 함수만 보여준다. consensus 는 뒤쪽 전용 절에서 다룬다.
FIGMETHODS=[m for m in METHODS if m!="consensus"]
# 그림 표시용 라벨 — 내부 키(BINDING_E 등)는 CSV 열명과 맞춰 두고 그림에서만 읽기 쉬운 이름을 쓴다.
DISP = {"AK_E": "AK-Score2", "BINDING_E": "AD4 energy", "Gen": "GenScore",
        "RTM": "RTMScore", "DOCK": "AD4 energy", "consensus": "consensus"}
def dsp(m): return DISP.get(str(m), str(m))


# per-drug 지표
def perdrug(m):
    mg=ea[ea.DRUG.isin(common)].merge(P[m][["DRUG","UNIPROT","pred"]],on=["DRUG","UNIPROT"])
    rows=[]
    for d,g in mg.groupby("DRUG"):
        if g.label.sum()>=ap.MIN_ACTIVES and (g.label==0).sum()>=1:
            r=ap.compute_metrics(g.label.values,g.pred.values); r["DRUG"]=d; rows.append(r)
    return pd.DataFrame(rows).set_index("DRUG")
PD={m:perdrug(m) for m in METHODS}

# 누출 상태 병합
lk=pd.read_csv(f"{OUT}/ligand_leakage_per_drug.csv")[["name","exact"]].rename(columns={"name":"DRUG"})
base=PD["AK_E"].reset_index()[["DRUG"]].merge(lk,on="DRUG",how="left")
base["exact"]=base["exact"].fillna(0).astype(int)
leaked=set(base[base.exact==1].DRUG); novel=set(base[base.exact==0].DRUG)
print(f"per-drug 화합물 {len(base)} = leaked {len(leaked)} + novel {len(novel)}")

# 요약: 방법×부분군 평균 + leaked vs novel Mann-Whitney(per-drug ROC)
rows=[]
for m in METHODS:
    d=PD[m]
    for sub,name in [(leaked,"leaked"),(novel,"novel"),(set(d.index),"all")]:
        s=d[d.index.isin(sub)]
        rows.append(dict(method=m,subset=name,n=len(s),roc=s.roc_auc.mean(),
                         hit1=s.hit1.mean(),hit10=s.hit10.mean(),ef1=s.ef1.mean()))
    L=d[d.index.isin(leaked)].roc_auc; N=d[d.index.isin(novel)].roc_auc
    mw=mannwhitneyu(L,N,alternative="two-sided") if len(L)>0 and len(N)>0 else None
    print(f"[{m:10s}] ROC leaked {L.mean():.3f}(n{len(L)}) vs novel {N.mean():.3f}(n{len(N)})  "
          f"drop {L.mean()-N.mean():+.3f}  MWU p={mw.pvalue:.3f}" if mw else f"[{m}] n/a")
S=pd.DataFrame(rows); S.to_csv(f"{OUT}/leave_leaked_out_summary.csv",index=False)

# novel-only 방법 비교: AK vs RTM (paired Wilcoxon, 공통 novel 화합물)
nv=sorted(novel & set(PD["AK_E"].index) & set(PD["RTM"].index))
wa=wilcoxon(PD["AK_E"].loc[nv].roc_auc, PD["RTM"].loc[nv].roc_auc)
print(f"\nnovel-only(n={len(nv)}) AK {PD['AK_E'].loc[nv].roc_auc.mean():.3f} vs RTM {PD['RTM'].loc[nv].roc_auc.mean():.3f} "
      f"Wilcoxon p={wa.pvalue:.3f}")

# 그림: 방법×부분군 (Per-Drug ROC, Hit@1)
fig,ax=plt.subplots(1,2,figsize=(15,5.6)); x=np.arange(len(FIGMETHODS)); w=0.35
for a,met,ttl in [(ax[0],"roc","Per-Drug ROC"),(ax[1],"hit1","Hit@1")]:
    lv=[S[(S.method==m)&(S.subset=="leaked")][met].iloc[0] for m in FIGMETHODS]
    nvv=[S[(S.method==m)&(S.subset=="novel")][met].iloc[0] for m in FIGMETHODS]
    a.bar(x-w/2,lv,w,label="leaked (in training)",color="#B2182B")
    a.bar(x+w/2,nvv,w,label="novel (not in training)",color="#4393C3")
    for i,(l,n) in enumerate(zip(lv,nvv)):
        a.text(i-w/2,l,f"{l:.2f}",ha="center",va="bottom",fontsize=8); a.text(i+w/2,n,f"{n:.2f}",ha="center",va="bottom",fontsize=8)
    a.set_xticks(x); a.set_xticklabels([dsp(m) for m in FIGMETHODS],rotation=12); a.set_ylabel(ttl); a.set_title(f"{ttl}: leaked vs novel")
    if met=="roc": a.axhline(0.5,color="gray",ls="--",lw=1); a.set_ylim(0.45,None)
    # 값 라벨을 가리지 않도록 위쪽에 여백을 두고 범례를 그 여백에 가로로 놓는다.
    lo,hi=a.get_ylim(); a.set_ylim(lo, hi+(hi-lo)*0.20)
    a.legend(fontsize=8, loc="upper center", ncol=2, framealpha=0.95)
# 지표가 정의되는 pool = 실험 라벨과 예측이 모두 있는 타겟
n_tgt = ea[ea.DRUG.isin(common)].merge(
    P["AK_E"][["DRUG","UNIPROT"]], on=["DRUG","UNIPROT"]).UNIPROT.nunique()
fig.suptitle(f"KIR leave-leaked-out: leaked({len(leaked)}) vs novel({len(novel)}) compounds\n"
             f"ranked within the {n_tgt} kinases KIR actually measures",
             fontsize=12.5,fontweight="bold")
_fl.stamp(ax)   # 패널 문자 (a)(b)(c)
fig.tight_layout(rect=[0,0,1,0.94]); fig.savefig(f"{OUT}/leave_leaked_out.png",dpi=300,bbox_inches="tight"); plt.close()

# MD
def row(m):
    g=lambda sub,c: S[(S.method==m)&(S.subset==sub)][c].iloc[0]
    return (f"| {m} | {g('leaked','roc'):.3f} | {g('novel','roc'):.3f} | {g('leaked','roc')-g('novel','roc'):+.3f} "
            f"| {g('leaked','hit1'):.2f} | {g('novel','hit1'):.2f} |")
md=[f"# Leave-leaked-out (KIR)\n","**분석일**: 2026-07-21\n",
 f"리간드 InChIKey14 완전일치(leaked) {len(leaked)} vs 신규(novel) {len(novel)} 화합물. 동일 200-포즈 재점수.\n",
 "![llo](leave_leaked_out.png)\n","## Per-Drug ROC · Hit@1 (leaked vs novel)\n",
 "| 방법 | ROC leaked | ROC novel | Δ(leaked−novel) | Hit@1 leaked | Hit@1 novel |","|---|---:|---:|---:|---:|---:|"]
md+=[row(m) for m in METHODS]
md+=["\n## 해석\n",
 "- Δ(leaked−novel)가 작을수록 학습 누출에 강건(신규 화합물에도 유지).",
 f"- novel-only에서 AK vs RTM: AK {PD['AK_E'].loc[nv].roc_auc.mean():.3f} vs RTM {PD['RTM'].loc[nv].roc_auc.mean():.3f} (Wilcoxon p={wa.pvalue:.3f}).",
 "- RTM의 leaked-novel 격차가 AK보다 크면 = RTM이 암기 의존, AK가 일반화 우위."]
open(f"{OUT}/LEAVE_LEAKED_OUT.md","w").write("\n".join(md))
print(f"\n✅ 저장 → {OUT}/ (LEAVE_LEAKED_OUT.md, leave_leaked_out.png, csv)")
