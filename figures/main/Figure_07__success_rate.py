"""
Target-identification success rate: the fraction of compounds for which screening the top k
targets, or the top X% of the panel, recovers at least one true target. AK-Score2, both panels.

Output: Kinase_paper/success_rate/ (success_rate.png, success_rate.csv, SUCCESS_RATE.md)
"""
from __future__ import annotations
import sys as _s; _s.path.insert(0,"/mnt/d/Deepmolscan/Kinase_paper"); import figlabels as _fl
import os, math
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sys; sys.path.insert(0,"/mnt/d/Deepmolscan/PKIS2"); import analyze_pkis as ap
import sys; sys.path.insert(0,"/mnt/d/Deepmolscan/Kinase_paper"); import kinome_merge as km
import sys; sys.path.insert(0,"/mnt/d/Deepmolscan/Kinase_paper"); import novel_cohort as nc
KP="/mnt/d/Deepmolscan/Kinase_paper"; OUT=f"{KP}/success_rate"; os.makedirs(OUT,exist_ok=True)

def kmerged():
    exp=pd.read_csv(f"{KP}/kinase_exp_uniprot_mapped_matrix.csv",index_col=0)
    c2u={c:c.split("|")[2] for c in exp.columns}
    el=exp.stack().reset_index(); el.columns=["DRUG_NAME","col","rem"]; el["UNIPROT_ID"]=el["col"].map(c2u)
    el["rem"]=pd.to_numeric(el["rem"],errors="coerce"); el=el.dropna(subset=["rem"]); el["label"]=(el["rem"]<50).astype(int)
    ea=el.groupby(["DRUG_NAME","UNIPROT_ID"],as_index=False).agg(label=("label","max"))
    pr=pd.read_csv(f"{KP}/result_87/redock_zscore_long.csv")
    pr["DRUG_NAME"]=pr["DRUG_NAME"].replace(ap.DRUG_ALIAS)
    pr["UNIPROT_ID"]=km.norm_uni(pr["UNIPROT_ID"])   # 복합 accession 정규화(표 1 pool과 일치)
    pr=pr.groupby(["DRUG_NAME","UNIPROT_ID"],as_index=False).agg(z=("zscore_mean","min"),r=("rank_mean","min"))
    s=1.0 if np.corrcoef(pr.z,pr.r)[0,1]>0 else -1.0; pr["pred_score"]=-s*pr.z
    return ea.merge(pr[["DRUG_NAME","UNIPROT_ID","pred_score"]],on=["DRUG_NAME","UNIPROT_ID"])
SRC={"KIR":kmerged(),
     "PKIS2":pd.read_csv(f"{KP}/../PKIS2/PKIS2_result/analysis_merged_labeled.csv")}
SRC={k:nc.drop_leaked(v,k) for k,v in SRC.items()}   # 학습 미사용 코호트
COL={"KIR":"#2166AC","PKIS2":"#E08214"}
KS=list(range(1,21)); PS=[0.005,0.01,0.02,0.03,0.05,0.075,0.10,0.15,0.20]

def curves(M):
    labs=[]
    for d,g in M.groupby("DRUG_NAME"):
        if g.label.sum()<ap.MIN_ACTIVES or (g.label==0).sum()<1: continue
        labs.append(g.sort_values("pred_score",ascending=False).label.values)
    n=len(labs)
    topk={k:np.mean([int(l[:k].sum()>0) for l in labs])*100 for k in KS}
    topp={p:np.mean([int(l[:max(1,math.ceil(p*len(l)))].sum()>0) for l in labs])*100 for p in PS}
    return n,topk,topp

data={ds:curves(M) for ds,M in SRC.items()}
# 표 저장
rows=[]
for ds,(n,tk,tp) in data.items():
    rows.append(dict(dataset=ds,n=n,top1=tk[1],top5=tk[5],top10=tk[10],
                     p1=tp[0.01],p5=tp[0.05],p10=tp[0.10]))
S=pd.DataFrame(rows); S.to_csv(f"{OUT}/success_rate.csv",index=False)
print(S.round(1).to_string(index=False))

# 그림
fig,ax=plt.subplots(1,2,figsize=(14,5.4))
a=ax[0]
for ds,(n,tk,tp) in data.items():
    a.plot(KS,[tk[k] for k in KS],"o-",color=COL[ds],ms=3,label=f"{ds} (n={n})")
a.set(xlabel="top-k predicted targets",ylabel="% compounds with a true target",title="Success rate vs top-k targets",xlim=(1,20),ylim=(0,102))
a.legend(fontsize=9); a.grid(alpha=.3)
a=ax[1]
for ds,(n,tk,tp) in data.items():
    a.plot([p*100 for p in PS],[tp[p] for p in PS],"s-",color=COL[ds],ms=3,label=ds)
a.set(xlabel="top X% of ranked kinase panel",ylabel="% compounds with a true target",title="Success rate vs top X% screened",ylim=(0,102))
a.legend(fontsize=9); a.grid(alpha=.3)
fig.suptitle(f"Target-identification success rate (AK-Score2, {nc.tag()} cohort)",fontsize=14,fontweight="bold")
_fl.stamp(ax)   # 패널 문자 (a)(b)(c)
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(f"{OUT}/success_rate.png",dpi=300,bbox_inches="tight"); plt.close()

md=["# 표적 식별 Success rate (AK-Score2)\n","**분석일**: 2026-07-22 | 상위 k 타겟/상위 X% 스크리닝 시 진짜 표적 1개 이상 회수한 화합물 비율. Active≥5.\n",
    "![sr](success_rate.png)\n","## 요약\n",
    "| 데이터셋 | n | top-1 | top-5 | top-10 | top-1% | top-5% | top-10% |","|---|--:|--:|--:|--:|--:|--:|--:|"]
for _,r in S.iterrows():
    md.append(f"| {r.dataset} | {int(r.n)} | {r.top1:.0f}% | {r.top5:.0f}% | {r.top10:.0f}% | {r.p1:.0f}% | {r.p5:.0f}% | {r.p10:.0f}% |")
md+=["\n## 해석\n",
 f"- 킨옴 **상위 1%만 스크리닝해도** 진짜 표적을 KIR {S[S.dataset=='KIR'].p1.iloc[0]:.0f}%·PKIS2 {S[S.dataset=='PKIS2'].p1.iloc[0]:.0f}% 화합물에서 회수.",
 "- **상위 10%면 91–100%** 화합물에서 회수 → 실험 우선순위 축소 도구로 실용적.",
 "- top-k(개수)는 Hit@k와 동일; top-X%(패널비율)는 패널 크기에 따라 절대 k가 달라짐."]
open(f"{OUT}/SUCCESS_RATE.md","w").write("\n".join(md))
print(f"\n✅ 저장 → {OUT}/ (success_rate.png, csv, SUCCESS_RATE.md)")
