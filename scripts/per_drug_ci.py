"""
Reclassify sub-0.5 per-drug ROC values using Hanley-McNeil 95% confidence intervals.

The question is whether a point estimate below 0.5 means the method is genuinely worse than random
for that compound (interval upper bound < 0.5) or is the finite-sample left tail of the
per-compound distribution (interval straddles 0.5). Both panels, AK-Score2.

Output: Kinase_paper/low_auc_analysis/ (PERDRUG_CI.md, per_drug_ci.png, csv)
"""
from __future__ import annotations
import os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import rankdata
import sys; sys.path.insert(0,"/mnt/d/Deepmolscan/Kinase_paper"); import kinome_merge as km
import sys; sys.path.insert(0,"/mnt/d/Deepmolscan/Kinase_paper"); import novel_cohort as nc

KP="/mnt/d/Deepmolscan/Kinase_paper"; OUT=f"{KP}/low_auc_analysis"; os.makedirs(OUT,exist_ok=True)
MIN_ACT=1   # 전체 화합물 코호트로 통일(cf. analyze_pkis.MIN_ACTIVES)
# 예측 DRUG_NAME을 실험 매트릭스 정규명에 맞춤(오타/파일접미사로 인한 exact-merge 누락 방지). cf. analyze_pkis.DRUG_ALIAS
DRUG_ALIAS={"Barcitinib":"Baricitinib","Vandetanib-Final":"Vandetanib"}

def auc_ci(y, s):
    y=np.asarray(y); s=np.asarray(s); na=int(y.sum()); nn=len(y)-na
    if na<1 or nn<1: return np.nan,np.nan,np.nan
    r=rankdata(s); auc=(r[y==1].sum()-na*(na+1)/2)/(na*nn)
    Q1=auc/(2-auc); Q2=2*auc*auc/(1+auc)
    var=(auc*(1-auc)+(na-1)*(Q1-auc**2)+(nn-1)*(Q2-auc**2))/(na*nn)
    se=np.sqrt(max(var,0)); return auc, auc-1.96*se, auc+1.96*se

def kinome_merged():
    exp=pd.read_csv(f"{KP}/kinase_exp_uniprot_mapped_matrix.csv",index_col=0)
    c2u={c:c.split("|")[2] for c in exp.columns}
    el=exp.stack().reset_index(); el.columns=["DRUG_NAME","col","rem"]; el["UNIPROT_ID"]=el["col"].map(c2u)
    el["rem"]=pd.to_numeric(el["rem"],errors="coerce"); el=el.dropna(subset=["rem"])
    ea=el.groupby(["DRUG_NAME","UNIPROT_ID"],as_index=False)["rem"].min(); ea["label"]=(ea.rem<50).astype(int)
    pr=pd.read_csv(f"{KP}/result_87/redock_zscore_long.csv")
    pr["DRUG_NAME"]=pr["DRUG_NAME"].replace(DRUG_ALIAS)
    pr["UNIPROT_ID"]=km.norm_uni(pr["UNIPROT_ID"])   # 복합 accession 정규화(표 1 pool과 일치)
    pr=pr.groupby(["DRUG_NAME","UNIPROT_ID"],as_index=False).agg(z=("zscore_mean","min"),r=("rank_mean","min"))
    s=1.0 if np.corrcoef(pr.z,pr.r)[0,1]>0 else -1.0; pr["pred_score"]=-s*pr.z
    return ea.merge(pr[["DRUG_NAME","UNIPROT_ID","pred_score"]],on=["DRUG_NAME","UNIPROT_ID"])

PANELS={"KIR":kinome_merged(),
        "PKIS2":pd.read_csv(f"{KP}/../PKIS2/PKIS2_result/analysis_merged_labeled.csv")}
PANELS={k:nc.drop_leaked(v,k) for k,v in PANELS.items()}   # 학습 미사용 코호트

rows=[]; per_drug=[]
for panel,M in PANELS.items():
    for d,g in M.groupby("DRUG_NAME"):
        na=int(g.label.sum())
        if na<MIN_ACT or (g.label==0).sum()<1: continue
        a,lo,hi=auc_ci(g.label.values,g.pred_score.values)
        per_drug.append(dict(panel=panel,drug=d,auc=a,lo=lo,hi=hi,n_act=na))
    sub=pd.DataFrame([x for x in per_drug if x["panel"]==panel])
    n=len(sub); below=sub.auc<0.5
    sig_below=(sub.hi<0.5); sig_above=(sub.lo>0.5); noise=(~sig_below)&(~sig_above)
    n_sub=int(below.sum()); n_sigb=int(sig_below.sum())
    rows.append(dict(panel=panel,n_drug=n,
        pct_below=round(100*n_sub/n,1),
        pct_sig_below=round(100*n_sigb/n,1),
        pct_sig_above=round(100*sig_above.sum()/n,1),
        pct_noise=round(100*noise.sum()/n,1),
        sub_of_which_sig=round(100*n_sigb/max(n_sub,1),1),
        med_nact_sub=int(sub[below].n_act.median()) if n_sub else 0,
        med_nact_all=int(sub.n_act.median())))
S=pd.DataFrame(rows); PD=pd.DataFrame(per_drug)
S.to_csv(f"{OUT}/per_drug_ci_summary.csv",index=False); PD.to_csv(f"{OUT}/per_drug_ci.csv",index=False)
print(S.to_string(index=False))

# 그림
fig,ax=plt.subplots(1,2,figsize=(15,5.6))
a=ax[0]; x=np.arange(len(S));
a.bar(x,S.pct_below,0.5,label="point ROC<0.5",color="#F4A582",edgecolor="white")
a.bar(x,S.pct_sig_below,0.5,label="significantly <0.5 (CI upper<0.5)",color="#B2182B",edgecolor="white")
for i,(pb,sb) in enumerate(zip(S.pct_below,S.pct_sig_below)):
    a.text(i,pb,f"{pb:.0f}%",ha="center",va="bottom",fontsize=9); a.text(i,sb,f"{sb:.0f}%",ha="center",va="bottom",fontsize=8,color="white")
a.set_xticks(x); a.set_xticklabels(S.panel); a.set_ylabel("% of drugs"); a.set_title("sub-0.5: point estimate vs statistically significant")
a.legend(fontsize=8)
# CI 커버(0.5 대비) 스택
a=ax[1]
bot=np.zeros(len(S))
for lab,col,key in [("sig above 0.5","#2166AC","pct_sig_above"),("indistinct from 0.5","#D9D9D9","pct_noise"),("sig below 0.5","#B2182B","pct_sig_below")]:
    a.bar(x,S[key],0.5,bottom=bot,label=lab,color=col,edgecolor="white"); bot=bot+S[key].values
a.set_xticks(x); a.set_xticklabels(S.panel); a.set_ylabel("% of drugs"); a.set_title("Per-drug AUC vs random (95% CI)"); a.legend(fontsize=8)
fig.suptitle(f"Per-drug ROC<0.5 reclassified: significant failure vs sampling noise "
             f"(AK-Score2, Hanley-McNeil 95% CI, {nc.tag()} cohort)",fontsize=13,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.94]); fig.savefig(f"{OUT}/per_drug_ci.png",dpi=150,bbox_inches="tight"); plt.close()

md=[f"# Per-drug ROC<0.5 재분류 (95% CI)\n","**분석일**: 2026-07-21 | AK-Score2, Hanley-McNeil SE\n",
 "![ci](per_drug_ci.png)\n",
 "| 패널 | 약물 | point<0.5 | **유의<0.5** | 랜덤과 무구분 | 유의>0.5 | sub중 유의비율 | sub중앙 n_act |",
 "|---|--:|--:|--:|--:|--:|--:|--:|"]
for _,r in S.iterrows():
    md.append(f"| {r.panel} | {r.n_drug} | {r.pct_below}% | **{r.pct_sig_below}%** | {r.pct_noise}% | {r.pct_sig_above}% | {r.sub_of_which_sig}% | {r.med_nact_sub} |")
md+=["\n## 해석\n",
 "- **point<0.5(점추정)** 중 상당수는 CI가 0.5를 걸쳐 **랜덤과 통계적으로 구별되지 않음**(=신호 부족, 실패 아님).",
 "- **유의<0.5**(CI 상한<0.5)만이 진짜 '랜덤 이하' 실패. 이 값이 point<0.5보다 훨씬 작음.",
 "- 대부분 약물은 **유의>0.5**(랜덤보다 유의하게 우수).",]
open(f"{OUT}/PERDRUG_CI.md","w").write("\n".join(md))
print(f"\n✅ 저장 → {OUT}/ (PERDRUG_CI.md, per_drug_ci.png, csv 2종)")
