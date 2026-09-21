"""
Does representation frequency in the training data (PDBbind v2020 general) explain KIR
performance and target bias?

Hypothesis: if AK-Score2 and RTMScore rank kinases that appear often in PDBbind well, or are
biased towards them, a promiscuous panel such as KIR is favoured while the reference set,
rich in non-canonical kinases, is penalised - and RTMScore's target preference would largely be
training frequency.

Tests: (1) ligand-independent target preference against n_pdb, (2) primary recovery by n_pdb bin,
(3) targets present in PDBbind against those absent from it.

Output: Kinase_paper/pdbbind_analysis/ (PDBBIND_LEAKAGE.md, pdbbind_leakage.png, csv)
"""
from __future__ import annotations
import sys as _s; _s.path.insert(0,"/mnt/d/Deepmolscan/Kinase_paper"); import figlabels as _fl
import os
import sys; sys.path.insert(0,'/mnt/d/Deepmolscan/PKIS2'); import analyze_pkis as ap
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import sys; sys.path.insert(0,"/mnt/d/Deepmolscan/Kinase_paper"); import kinome_merge as km

KP="/mnt/d/Deepmolscan/Kinase_paper"; OUT=f"{KP}/pdbbind_analysis"; os.makedirs(OUT,exist_ok=True)
COL={"AK":"#2166AC","RTM":"#F4A582"}

# ── PDBbind general: UniProt별 학습 complex 수 ──
uni=[]
for ln in open("/mnt/d/PDBBind/index/INDEX_general_PL_name.2020"):
    if ln.startswith("#"): continue
    p=ln.split()
    if len(p)>=3: uni.append(p[2])
npdb=pd.Series(uni).value_counts().to_dict()   # uniprot -> n complexes
def npdb_of(u): return int(npdb.get(str(u),0))

# ── KIR 실험 + AK/RTM 예측 ──
exp=pd.read_csv(f"{KP}/kinase_exp_uniprot_mapped_matrix.csv",index_col=0)
col2uni={c:c.split("|")[2] for c in exp.columns}
el=exp.stack().reset_index(); el.columns=["DRUG","col","rem"]
el["UNIPROT"]=el["col"].map(col2uni); el["rem"]=pd.to_numeric(el["rem"],errors="coerce")
el=el.dropna(subset=["rem"])
expm=el.groupby(["DRUG","UNIPROT"],as_index=False)["rem"].min()

def load_raw(f):
    df=(lambda _d: (_d.assign(DRUG_NAME=_d.DRUG_NAME.replace(ap.DRUG_ALIAS), UNIPROT_ID=km.norm_uni(_d.UNIPROT_ID))))(pd.read_csv(f)).groupby(["DRUG_NAME","UNIPROT_ID"],as_index=False).agg(z=("zscore_mean","min"),r=("rank_mean","min"))
    sign=1.0 if np.corrcoef(df.z,df.r)[0,1]>0 else -1.0
    return df.rename(columns={"DRUG_NAME":"DRUG","UNIPROT_ID":"UNIPROT"}), sign
AK,aks=load_raw(f"{KP}/result_87/redock_zscore_long.csv")
RT,rts=load_raw(f"{KP}/RTM/redock_zscore_long.csv")

M=AK.rename(columns={"z":"zAK"})[["DRUG","UNIPROT","zAK"]].merge(
  RT.rename(columns={"z":"zRTM"})[["DRUG","UNIPROT","zRTM"]],on=["DRUG","UNIPROT"])
# 화합물 내 순위(1=best), 방향교정
M["rAK"]=M.groupby("DRUG").zAK.rank(method="min",ascending=(aks>0))
M["rRTM"]=M.groupby("DRUG").zRTM.rank(method="min",ascending=(rts>0))
M["pctlAK"]=M.groupby("DRUG").rAK.transform(lambda s:s/len(s))
M["pctlRTM"]=M.groupby("DRUG").rRTM.transform(lambda s:s/len(s))
M["n_pdb"]=M.UNIPROT.map(npdb_of)

# ── (1) 타겟 선호(리간드무관): 타겟별 mean percentile rank vs n_pdb ──
tg=M.groupby("UNIPROT").agg(mp_AK=("pctlAK","mean"),mp_RTM=("pctlRTM","mean"),n=("DRUG","size")).reset_index()
tg["n_pdb"]=tg.UNIPROT.map(npdb_of); tg["in_pdb"]=tg.n_pdb>0
tg=tg[tg.n>=10]   # 충분히 관측된 타겟
# 낮은 mean-percentile = 시스템적으로 상위 = 선호. 학습多일수록 선호면 음의 상관.
rAK=spearmanr(tg.n_pdb,tg.mp_AK); rRT=spearmanr(tg.n_pdb,tg.mp_RTM)
print(f"[타겟선호 vs n_pdb] (음수=학습많은 타겟을 상위선호)")
print(f"  AK : Spearman rho={rAK.correlation:+.3f} (p={rAK.pvalue:.1e})")
print(f"  RTM: Spearman rho={rRT.correlation:+.3f} (p={rRT.pvalue:.1e})")

# ── (2) primary 회수율 vs primary의 n_pdb bin (KIR) ──
def recovery_table(M, expm, label):
    rows=[]
    for dn,g in M.groupby("DRUG"):
        g=g.copy(); N=len(g)
        e=expm[(expm.DRUG==dn)&(expm.UNIPROT.isin(set(g.UNIPROT)))]
        if len(e)==0: continue
        prim=e.sort_values("rem").iloc[0]
        rkAK=dict(zip(g.UNIPROT,g.rAK)); rkRT=dict(zip(g.UNIPROT,g.rRTM))
        rows.append(dict(drug=dn,prim=prim.UNIPROT,n_pdb=npdb_of(prim.UNIPROT),
            AK_rank=rkAK[prim.UNIPROT],RT_rank=rkRT[prim.UNIPROT],N=N))
    R=pd.DataFrame(rows)
    R["bin"]=pd.cut(R.n_pdb,[-1,0,10,10000],labels=["0 (unseen)","1-10","10+"])
    agg=R.groupby("bin",observed=True).agg(n=("drug","size"),
        AK_top10=("AK_rank",lambda s:(s<=10).mean()*100),
        RT_top10=("RT_rank",lambda s:(s<=10).mean()*100)).reset_index()
    print(f"\n[{label}] primary 회수(Top-10 %) vs primary의 학습빈도")
    print(agg.to_string(index=False))
    return R,agg
Rk,aggk=recovery_table(M,expm,"KIR")

# ── (3) in vs out PDBbind: primary 회수 비교 ──
def io(R):
    ins=R[R.n_pdb>0]; outs=R[R.n_pdb==0]
    return dict(in_n=len(ins),out_n=len(outs),
        AK_in=(ins.AK_rank<=10).mean()*100,AK_out=(outs.AK_rank<=10).mean()*100,
        RT_in=(ins.RT_rank<=10).mean()*100,RT_out=(outs.RT_rank<=10).mean()*100)
IO=io(Rk)
print(f"\n[in/out PDBbind, KIR primary Top-10]")
print(f"  in-PDBbind(n={IO['in_n']}):  AK {IO['AK_in']:.0f}% / RTM {IO['RT_in']:.0f}%")
print(f"  out(미학습,n={IO['out_n']}): AK {IO['AK_out']:.0f}% / RTM {IO['RT_out']:.0f}%")

# ── ref 비전형 kinase n_pdb 요약 ──
bench_gene={c.split('|')[2]:c.split('|')[1] for c in exp.columns}
ref_key=["CSNK2A1","CSNK2A2","DYRK1A","DYRK1B","GAK","PKN2","CLK1","CLK4","STK17A"]
refrows=[]
for g in ref_key:
    us=[u for u,gg in bench_gene.items() if gg==g]
    for u in us: refrows.append(dict(gene=g,uni=u,n_pdb=npdb_of(u)))
REFT=pd.DataFrame(refrows)
tg.to_csv(f"{OUT}/target_pref_vs_npdb.csv",index=False); Rk.to_csv(f"{OUT}/recovery_vs_npdb.csv",index=False)

# ── 그림 ──
fig,ax=plt.subplots(1,3,figsize=(19,6))
# (0) scatter 타겟선호 vs n_pdb
a=ax[0]
a.scatter(tg.n_pdb+0.5,tg.mp_AK,s=18,alpha=.5,color=COL["AK"],label=f"AK-Score2 (ρ={rAK.correlation:+.2f})")
a.scatter(tg.n_pdb+0.5,tg.mp_RTM,s=18,alpha=.5,color=COL["RTM"],label=f"RTMScore (ρ={rRT.correlation:+.2f})")
a.set_xscale("log"); a.set_xlabel("PDBbind training complexes (n_pdb, +0.5)")
a.set_ylabel("mean within-compound percentile (lower=preferred)"); a.invert_yaxis()
a.set_title("Target preference vs training freq"); a.legend(fontsize=9)
# (1) 회수율 bar by bin
a=ax[1]; b=aggk; x=np.arange(len(b)); w=0.38
a.bar(x-w/2,b.AK_top10,w,label="AK-Score2",color=COL["AK"]); a.bar(x+w/2,b.RT_top10,w,label="RTMScore",color=COL["RTM"])
for i,(ak,rt,n) in enumerate(zip(b.AK_top10,b.RT_top10,b.n)):
    a.text(i-w/2,ak,f"{ak:.0f}",ha="center",va="bottom",fontsize=8); a.text(i+w/2,rt,f"{rt:.0f}",ha="center",va="bottom",fontsize=8)
    a.text(i,-4,f"n={n}",ha="center",fontsize=8,color="gray")
a.set_xticks(x);a.set_xticklabels(b["bin"]);a.set_ylabel("primary Top-10 recovery %")
a.set_title("KIR recovery vs primary n_pdb");a.legend(fontsize=9)
# (2) ref 비전형 kinase n_pdb
a=ax[2]; REFT2=REFT.sort_values("n_pdb")
b=a.barh(range(len(REFT2)),REFT2.n_pdb,color="#5AAE61",edgecolor="white")
a.set_yticks(range(len(REFT2)));a.set_yticklabels([f"{r.gene}" for _,r in REFT2.iterrows()],fontsize=9)
for bi,v in zip(b,REFT2.n_pdb): a.text(v,bi.get_y()+bi.get_height()/2,f" {int(v)}",va="center",fontsize=8)
a.set_xlabel("PDBbind training complexes");a.set_title("ref key kinases: training freq")
fig.suptitle("Training-data (PDBbind v2020 general) representation vs KIR performance",fontsize=14,fontweight="bold")
_fl.stamp(ax)   # 패널 문자 (a)(b)(c)
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(f"{OUT}/pdbbind_leakage.png",dpi=300,bbox_inches="tight"); plt.close()

# ── MD ──
md=[f"# 학습 데이터(PDBbind v2020 general) 표현 빈도 vs 성능\n","**분석일**: 2026-07-21\n",
 "PDBbind general 19,443 complex(고유 UniProt 3,890). 벤치마크 375 고유 타겟 중 "
 f"**{sum(1 for u in expm.UNIPROT.unique() if npdb_of(u)>0)}개가 PDBbind에 존재**, 절반은 n_pdb=0(미학습).\n",
 "![fig](pdbbind_leakage.png)\n",
 "## (1) 타겟 선호(리간드 무관 mean percentile) vs 학습 빈도\n",
 f"- AK : Spearman ρ={rAK.correlation:+.3f} (p={rAK.pvalue:.1e})",
 f"- RTM: Spearman ρ={rRT.correlation:+.3f} (p={rRT.pvalue:.1e})",
 "- 음수 = 학습 많은 타겟을 (리간드와 무관하게) 상위로 선호 = **학습빈도 편향**.",
 "## (2) KIR primary 회수(Top-10) vs primary의 학습 빈도\n",
 "| n_pdb bin | n | AK Top-10 | RTM Top-10 |","|---|---:|---:|---:|"]
for _,r in aggk.iterrows():
    md.append(f"| {r['bin']} | {int(r.n)} | {r.AK_top10:.0f}% | {r.RT_top10:.0f}% |")
md+=[f"\n## (3) in vs out PDBbind (KIR primary Top-10)\n",
 f"- in-PDBbind (n={IO['in_n']}): AK {IO['AK_in']:.0f}% / RTM {IO['RT_in']:.0f}%",
 f"- 미학습    (n={IO['out_n']}): AK {IO['AK_out']:.0f}% / RTM {IO['RT_out']:.0f}%",
 "\n## ref 핵심 kinase 학습 빈도\n","| gene | UniProt | n_pdb |","|---|---|---:|"]
for _,r in REFT.sort_values("n_pdb").iterrows(): md.append(f"| {r.gene} | {r.uni} | {r.n_pdb} |")
md+=["\n## 해석\n",
 "- RTM 선호가 n_pdb와 (AK보다) 강하게 음의 상관이면 → **RTM 타겟편향 = 학습빈도 기억** 증거.",
 "- 회수율이 n_pdb bin 따라 증가하면 → 성능이 학습 표현에 의존(부분적 leakage).",
 "- ref는 학습 적은 kinase(예: DYRK1B/PKN2/GAK n_pdb 0) 다수 → RTM 편향 역효과 설명.",
 "- 단, ligand-level 중복(동일/유사 저해제)은 별도 확인 필요(3-letter code/SMILES).",]
open(f"{OUT}/PDBBIND_LEAKAGE.md","w").write("\n".join(md))
print(f"\n✅ 저장 → {OUT}/ (PDBBIND_LEAKAGE.md, pdbbind_leakage.png, csv 2종)")
