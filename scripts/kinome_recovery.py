"""
KIR target recovery - the Kinase_ref panel analysis applied to the KIR matrix.

For each compound it records whether the experimentally strongest target (primary = min
%remaining) reaches the top of the prediction, and Hit@k over the strong actives
(%remaining <= 10). Compounds with no active kinase anywhere in their panel pose no ranking task
and are excluded, as they are from Table 4 and the success-rate figure.

The ranking pool defaults to the kinases the panel measured; POOL=full ranks against all 618
predicted structures instead, which is the blind kinome-wide sensitivity analysis. Methods:
AK-Score2 (pred = -z), RTMScore (pred = +z) and the consensus schemes (RRF, rank fusion,
z-average).

Output: Kinase_paper/energy_benchmark/ (KINOME_RECOVERY.md, kinome_recovery.png, csv)
"""
from __future__ import annotations
import os
import numpy as np, pandas as pd, matplotlib
import sys; sys.path.insert(0,"/mnt/d/Deepmolscan/Kinase_paper"); import kinome_merge as km
import sys; sys.path.insert(0,"/mnt/d/Deepmolscan/Kinase_paper"); import novel_cohort as nc
matplotlib.use("Agg"); import matplotlib.pyplot as plt

KP="/mnt/d/Deepmolscan/Kinase_paper"; OUT=f"{KP}/energy_benchmark"; os.makedirs(OUT,exist_ok=True)
ACT=10.0; RRF_K=60
# 랭킹 pool: 기본은 이 패널이 실제 측정한 kinase(369). POOL=full 이면 예측 pool 618 전체를 쓴다
# (SI Table S12b·Figure S7 의 blind kinome-wide 민감도 분석용). 측정되지 않은 kinase 를 decoy 로
# 넣으면 상위의 미측정 kinase 가 위양성인지 미지의 진양성인지 구분할 수 없기 때문에 기본값이 아니다.
POOL=os.environ.get("POOL","measured")
SUF="" if POOL=="measured" else "_full618"
# 예측 DRUG_NAME을 실험 매트릭스 정규명에 맞춤(오타/파일접미사로 인한 exact-merge 누락 방지). cf. analyze_pkis.DRUG_ALIAS
DRUG_ALIAS={"Barcitinib":"Baricitinib","Vandetanib-Final":"Vandetanib"}
METH={"AK":f"{KP}/result_87/redock_zscore_long.csv","RTM":f"{KP}/RTM/redock_zscore_long.csv"}
METHODS=["AK","RTM","RRF","RankFusion","Z-avg"]
COL={"AK":"#2166AC","RTM":"#F4A582","RRF":"#762A83","RankFusion":"#1B7837","Z-avg":"#E08214"}

# ── 실험: %remaining (낮을수록 강한 저해) ──
exp=pd.read_csv(f"{KP}/kinase_exp_uniprot_mapped_matrix.csv",index_col=0)
col2uni={c:c.split("|")[2] for c in exp.columns}
el=exp.stack().reset_index(); el.columns=["DRUG_NAME","col","rem"]
el["UNIPROT_ID"]=el["col"].map(col2uni); el["rem"]=pd.to_numeric(el["rem"],errors="coerce")
el=el.dropna(subset=["rem"])
exp_agg=el.groupby(["DRUG_NAME","UNIPROT_ID"],as_index=False)["rem"].min()
MEASURED=set(exp_agg.UNIPROT_ID)   # KIR 이 실제 측정한 kinase (369)

def load_raw(f):
    df=pd.read_csv(f); df["DRUG_NAME"]=df["DRUG_NAME"].replace(DRUG_ALIAS)
    df["UNIPROT_ID"]=km.norm_uni(df["UNIPROT_ID"])   # 복합 accession 정규화(표 1 pool과 일치)
    df=df.groupby(["DRUG_NAME","UNIPROT_ID"],as_index=False).agg(
        z=("zscore_mean","min"), r=("rank_mean","min"))
    sign=1.0 if np.corrcoef(df.z,df.r)[0,1]>0 else -1.0   # sign>0: 낮은 z=best
    return df[["DRUG_NAME","UNIPROT_ID","z"]], sign

AKp,aks=load_raw(METH["AK"]); RTp,rts=load_raw(METH["RTM"])
print(f"AK sign={aks:+.0f}(낮은z=best? {aks>0})  RTM sign={rts:+.0f}(낮은z=best? {rts>0})")
def zc(s):
    sd=s.std(ddof=0); return (s-s.mean())/sd if sd>0 else s*0.0

M=AKp.rename(columns={"z":"zAK"}).merge(RTp.rename(columns={"z":"zRTM"}),on=["DRUG_NAME","UNIPROT_ID"])
M=nc.drop_leaked(M,"KIR")   # 학습 미사용 코호트
drugs=sorted(set(M.DRUG_NAME)&set(exp_agg.DRUG_NAME))
print(f"공통 화합물(AK∩RTM∩실험): {len(drugs)}")

rows=[]
for dn in drugs:
    g=M[M.DRUG_NAME==dn].copy()
    if POOL=="measured": g=g[g.UNIPROT_ID.isin(MEASURED)]
    if len(g)<30: continue
    # 방향교정: AK sign>0이면 낮은 z=best → asc=True; RTM sign<0이면 높은 z=best → asc=False
    g["rAK"]=g.zAK.rank(method="min",ascending=(aks>0))
    g["rRTM"]=g.zRTM.rank(method="min",ascending=(rts>0))
    pAK=zc(g.zAK*(-1 if aks>0 else 1)); pRTM=zc(g.zRTM*(-1 if rts>0 else 1))  # higher=better
    sc={"AK":pAK,"RTM":pRTM,"RRF":1.0/(RRF_K+g.rAK)+1.0/(RRF_K+g.rRTM),
        "RankFusion":-(g.rAK+g.rRTM)/2.0,"Z-avg":(pAK+pRTM)/2.0}
    for m in METHODS: g[f"rk_{m}"]=sc[m].rank(method="min",ascending=False)
    rk={m:dict(zip(g.UNIPROT_ID,g[f"rk_{m}"])) for m in METHODS}
    N=len(g)
    e=exp_agg[exp_agg.DRUG_NAME==dn]; e=e[e.UNIPROT_ID.isin(set(g.UNIPROT_ID))]
    if len(e)==0: continue
    # 활성이 하나도 없는 화합물은 랭킹 과제가 성립하지 않는다(§2.2·Methods S3: "no ranking task
    # exists for them and they are excluded from per-compound metrics"). 표적이 패널 밖에 있어
    # Hit@k 가 구성상 0 이 되고 primary 도 실제 표적이 아니므로, Table 4·Figure 7 과 같은 기준으로 제외한다.
    if not (e.rem < 50.0).any(): continue
    prim=e.sort_values("rem").iloc[0]; acts=e[e.rem<=ACT]
    rr=dict(cpd=dn,N=N,n_meas=len(e),n_act=len(acts))
    for m in METHODS:
        pr=rk[m][prim.UNIPROT_ID]; rr[f"{m}_prank"]=int(pr); rr[f"{m}_pctl"]=round(pr/N,4)
        for k in (1,5,10):
            rr[f"{m}_hit{k}"]=int(any(rk[m][u]<=k for u in acts.UNIPROT_ID)) if len(acts) else 0
    rows.append(rr)
R=pd.DataFrame(rows); R.to_csv(f"{OUT}/kinome_recovery{SUF}_per_compound.csv",index=False)
Nmed=int(R.N.median())
def summ(m):
    p=R[f"{m}_prank"]
    return dict(method=m,top1=round((p<=1).mean()*100,1),top5=round((p<=5).mean()*100,1),
        top10=round((p<=10).mean()*100,1),top30=round((p<=30).mean()*100,1),
        medpctl=round(R[f"{m}_pctl"].median()*100,1),
        hit1=round(R[f"{m}_hit1"].mean()*100,1),hit5=round(R[f"{m}_hit5"].mean()*100,1),
        hit10=round(R[f"{m}_hit10"].mean()*100,1))
S=pd.DataFrame([summ(m) for m in METHODS]); S.to_csv(f"{OUT}/kinome_recovery{SUF}_summary.csv",index=False)
print(f"\n[KIR 회수] pool {Nmed}, n={len(R)}\n"+S.to_string(index=False))

# ── 그림 ──
fig,ax=plt.subplots(1,3,figsize=(19,6)); w=0.8/len(METHODS)
a=ax[0]; labs=["Top-1","Top-5","Top-10","Top-30"];x=np.arange(4)
for i,m in enumerate(METHODS):
    s=S[S.method==m].iloc[0];a.bar(x+(i-(len(METHODS)-1)/2)*w,[s.top1,s.top5,s.top10,s.top30],w,label=m,color=COL[m],edgecolor="white")
a.axhline(100*30/Nmed,color="red",ls="--",lw=1,label=f"random Top-30≈{100*30/Nmed:.0f}%")
a.set_xticks(x);a.set_xticklabels(labs);a.set_ylabel("% compounds primary in Top-k")
a.set_title(f"Primary-target recovery (n={len(R)})");a.legend(fontsize=8)
a=ax[1]; labs=["Hit@1","Hit@5","Hit@10"];x=np.arange(3)
for i,m in enumerate(METHODS):
    s=S[S.method==m].iloc[0];a.bar(x+(i-(len(METHODS)-1)/2)*w,[s.hit1,s.hit5,s.hit10],w,label=m,color=COL[m],edgecolor="white")
a.set_xticks(x);a.set_xticklabels(labs);a.set_ylabel("% (any strong active in top-k)");a.set_title("Hit@k (%rem<=10)");a.legend(fontsize=8)
a=ax[2]; mp=[S[S.method==m].iloc[0].medpctl for m in METHODS]
b=a.bar(range(len(METHODS)),mp,color=[COL[m] for m in METHODS],edgecolor="white")
for bi,v in zip(b,mp): a.text(bi.get_x()+bi.get_width()/2,v,f"{v:.1f}%",ha="center",va="bottom",fontsize=9)
a.set_xticks(range(len(METHODS)));a.set_xticklabels(METHODS,rotation=15);a.set_ylabel("median percentile (↓best)");a.set_title("Primary rank percentile")
_ptxt=("measured kinases" if POOL=="measured" else "all predicted structures")
fig.suptitle(f"KIR: primary-target recovery (ranking pool: {_ptxt}, {Nmed}; n={len(R)}) "
             f"| {nc.tag()} cohort — AK/RTM/consensus",fontsize=14,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(f"{OUT}/kinome_recovery{SUF}.png",dpi=150,bbox_inches="tight"); plt.close()

# ── MD ──
def rr(lab,key,fmt): return f"| {lab} | "+" | ".join(fmt.format(S[S.method==m].iloc[0][key]) for m in METHODS)+" |"
best=S[~S.method.isin(["AK","RTM"])].sort_values("top10",ascending=False).method.iloc[0]
md=[f"# KIR 표적 회수 (B) — AK/RTM/consensus\n","**분석일**: 2026-07-21\n",
 f"공통 {len(R)} 화합물, pool {Nmed}(AK∩RTM 예측). primary=실험 최강(min %rem), 강한 active=%rem≤10.\n",
 "> ref의 (B) 분석과 동일 방법. KIR은 완전 매트릭스라 음성 풍부.\n",
 "![kr](kinome_recovery.png)\n","## 요약\n",
 "| 지표 | "+" | ".join(METHODS)+" |","|---|"+":---:|"*len(METHODS),
 rr("primary Top-1","top1","{:.1f}%"),rr("primary Top-5","top5","{:.1f}%"),
 rr("**primary Top-10**","top10","{:.1f}%"),rr("primary Top-30","top30","{:.1f}%"),
 rr("percentile 중앙값(↓)","medpctl","{:.1f}%"),
 rr("Hit@1(any)","hit1","{:.1f}%"),rr("Hit@5(any)","hit5","{:.1f}%"),rr("Hit@10(any)","hit10","{:.1f}%"),
 "\n## 해석\n",
 f"- primary Top-10: AK {S[S.method=='AK'].iloc[0].top10}% / RTM {S[S.method=='RTM'].iloc[0].top10}% / 최고 consensus {best} {S[S.method==best].iloc[0].top10}%.",
 "- **ref(B)와 대조**: ref는 AK 우위, KIR은 RTM·consensus 우위 예상 → 데이터셋 특성 의존.",
 "- 상세: kinome_recovery_per_compound.csv, kinome_recovery_summary.csv.",]
open(f"{OUT}/KINOME_RECOVERY.md","w").write("\n".join(md))
print(f"\n✅ 저장 → {OUT}/ (KINOME_RECOVERY.md, kinome_recovery.png, csv 2종)")
