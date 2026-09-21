"""
Agreement between scoring functions after direction correction: how much their predictions
overlap, and how much they are complementary.

Uses the direction-corrected pred_score and reports both pooled agreement and within-compound rank
agreement, the latter being what matters for reverse docking.

Output: Kinase_paper/energy_benchmark/ (ENERGY_CORR.md, energy_corr.png)
"""
from __future__ import annotations
import sys as _s; _s.path.insert(0,"/mnt/d/Deepmolscan/Kinase_paper"); import figlabels as _fl
import sys; sys.path.insert(0,'/mnt/d/Deepmolscan/PKIS2'); import analyze_pkis as ap
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import sys; sys.path.insert(0,"/mnt/d/Deepmolscan/Kinase_paper"); import kinome_merge as km
import sys; sys.path.insert(0,"/mnt/d/Deepmolscan/Kinase_paper"); import novel_cohort as nc

KP="/mnt/d/Deepmolscan/Kinase_paper"; OUT=f"{KP}/energy_benchmark"
METH={"AK_E":f"{KP}/result_87/redock_zscore_long.csv",
      "BINDING_E":f"{KP}/DOCK/redock_zscore_long.csv",
      "Gen":f"{KP}/Gen/redock_zscore_long.csv",
      "RTM":f"{KP}/RTM/redock_zscore_long.csv"}
MS=list(METH)
# 그림 표시용 라벨 — 내부 키(BINDING_E 등)는 CSV 열명과 맞춰 두고 그림에서만 읽기 쉬운 이름을 쓴다.
DISP = {"AK_E": "AK-Score2", "BINDING_E": "AD4 energy", "Gen": "GenScore",
        "RTM": "RTMScore", "DOCK": "AD4 energy", "consensus": "consensus"}
def dsp(m): return DISP.get(str(m), str(m))
MSD=[dsp(m) for m in MS]

def load(f):
    df=(lambda _d: (_d.assign(DRUG_NAME=_d.DRUG_NAME.replace(ap.DRUG_ALIAS), UNIPROT_ID=km.norm_uni(_d.UNIPROT_ID))))(pd.read_csv(f)).groupby(["DRUG_NAME","UNIPROT_ID"],as_index=False).agg(
        z=("zscore_mean","min"), r=("rank_mean","min"))
    sign=1.0 if np.corrcoef(df.z,df.r)[0,1]>0 else -1.0
    df["pred"]=-sign*df.z
    return df[["DRUG_NAME","UNIPROT_ID","pred"]]

P={m:load(f) for m,f in METH.items()}
common=set.intersection(*[set(P[m].DRUG_NAME) for m in MS])
# 공통 (drug,uniprot)로 merge (모든 방법)
M=P[MS[0]].rename(columns={"pred":MS[0]})
for m in MS[1:]:
    M=M.merge(P[m].rename(columns={"pred":m}),on=["DRUG_NAME","UNIPROT_ID"])
M=M[M.DRUG_NAME.isin(common)]
M=nc.drop_leaked(M,"KIR")   # 학습 미사용 코호트
print(f"공통 (drug,target) 쌍: {len(M)}  화합물 {M.DRUG_NAME.nunique()}")

# ── pooled 상관 ──
pear=M[MS].corr(method="pearson"); spear=M[MS].corr(method="spearman")
print("\n[pooled Pearson]\n",pear.round(3).to_string())
print("\n[pooled Spearman]\n",spear.round(3).to_string())

# ── per-compound Spearman(화합물 내 타겟 순위 일치도) 평균 ──
pc={(a,b):[] for i,a in enumerate(MS) for b in MS[i+1:]}
for d,g in M.groupby("DRUG_NAME"):
    if len(g)<5: continue
    for i,a in enumerate(MS):
        for b in MS[i+1:]:
            pc[(a,b)].append(spearmanr(g[a],g[b])[0])
pcm={k:np.nanmean(v) for k,v in pc.items()}
print("\n[per-compound 평균 Spearman]")
for k,v in pcm.items(): print(f"  {k[0]} ~ {k[1]}: {v:.3f}")
# 대칭 행렬
nm=len(MS)
PCM=pd.DataFrame(np.eye(nm),index=MS,columns=MS)
for (a,b),v in pcm.items(): PCM.loc[a,b]=v; PCM.loc[b,a]=v

# ── 그림 ──
fig,ax=plt.subplots(1,3,figsize=(19,5.8))
def heat(a,mat,title):
    im=a.imshow(mat.values,vmin=0,vmax=1,cmap="RdYlBu_r")
    a.set_xticks(range(nm));a.set_xticklabels(MSD,rotation=20);a.set_yticks(range(nm));a.set_yticklabels(MSD)
    for i in range(nm):
        for j in range(nm):
            a.text(j,i,f"{mat.values[i,j]:.2f}",ha="center",va="center",
                   color="white" if abs(mat.values[i,j])>0.6 else "black",fontsize=11,fontweight="bold")
    a.set_title(title); plt.colorbar(im,ax=a,fraction=0.046)
heat(ax[0],spear,"Pooled Spearman (all pairs)")
heat(ax[1],PCM,"Per-compound mean Spearman\n(target-ranking agreement)")
# scatter: AK vs 나머지 (샘플)
a=ax[2]; s=M.sample(min(4000,len(M)),random_state=0)
palette={"BINDING_E":"#4393C3","Gen":"#F4A582","RTM":"#5AAE61"}
for b in [x for x in MS if x!="AK_E"]:
    a.scatter(s["AK_E"],s[b],s=6,alpha=.22,color=palette.get(b,"#999"),label=f"AK-Score2 vs {dsp(b)} (ρ={spear.loc['AK_E',b]:.2f})")
a.set(xlabel="AK-Score2 pred_score",ylabel="other pred_score",title="Pairwise scatter (sample)"); a.legend(fontsize=8)
fig.suptitle(f"Scoring-function agreement ({' / '.join(dsp(m) for m in MS)}, direction-corrected)",fontsize=14,fontweight="bold")
_fl.stamp(ax)   # 패널 문자 (a)(b)(c)
fig.tight_layout(rect=[0,0,1,0.94]); fig.savefig(f"{OUT}/energy_corr.png",dpi=300,bbox_inches="tight"); plt.close()

hdr="| | "+" | ".join(MS)+" |"; sep="|---|"+":---:|"*nm
md=[f"# 스코어 간 상관 ({' / '.join(MS)})\n","**분석일**: 2026-07-21 | KIR, 방향 교정 pred_score\n",
    f"공통 {M.DRUG_NAME.nunique()} 화합물, {len(M):,} (drug,target) 쌍.\n","![corr](energy_corr.png)\n",
    "## Pooled Spearman (전체 쌍)\n",hdr,sep]
for a in MS: md.append(f"| {a} | "+" | ".join(f"{spear.loc[a,b]:.2f}" for b in MS)+" |")
md+=["\n## Per-compound 평균 Spearman (화합물 내 타겟 순위 일치)\n",hdr,sep]
for a in MS: md.append(f"| {a} | "+" | ".join(f"{PCM.loc[a,b]:.2f}" for b in MS)+" |")
md+=["\n## 해석\n"]
for i,a in enumerate(MS):
    for b in MS[i+1:]:
        md.append(f"- {a} ~ {b}: pooled ρ={spear.loc[a,b]:.2f}, per-cpd ρ={pcm[(a,b)]:.2f}")
md.append("- 상관이 낮을수록 상보적 → **consensus(앙상블)로 개선 여지**. 높으면 중복(정보 유사).")
open(f"{OUT}/ENERGY_CORR.md","w").write("\n".join(md))
print(f"\n✅ 저장 → {OUT}/ENERGY_CORR.md, energy_corr.png")
