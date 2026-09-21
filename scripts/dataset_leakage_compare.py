"""
Ligand-level leakage across the three datasets: how many KIR, Kinase_ref and PKIS2
compounds appear among the PDBbind v2020 kinase-binding ligands, and which ones.

Leakage is an exact match of the InChIKey14, the 2D skeleton. Rates are computed over the
compounds that enter the analysis - those with both experimental values and complete predictions -
so that the leakage-free remainder equals the cohort reported in Table 1.

Output: Kinase_paper/pdbbind_analysis/ (DATASET_LEAKAGE.md, dataset_leakage.png, csv)
"""
from __future__ import annotations
import sys as _s; _s.path.insert(0,"/mnt/d/Deepmolscan/Kinase_paper"); import figlabels as _fl
import os, glob
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

KP="/mnt/d/Deepmolscan/Kinase_paper"; OUT=f"{KP}/pdbbind_analysis"
def ik14(m):
    try: return Chem.MolToInchiKey(m).split("-")[0]
    except: return None

# ── PDBbind kinase 리간드 ik14 (캐시) ──
PL=pd.read_csv(f"{OUT}/pdbbind_kinase_ligands.csv").dropna(subset=["ik14"])
train_ik=set(PL.ik14)
print(f"PDBbind kinase 학습 리간드 고유 InChIKey14: {len(train_ik)}")

# ── 데이터셋별 화합물 SMILES → ik14 ──
def from_smi(path,sep=None):
    out={}
    for ln in open(path):
        p=ln.split() if sep is None else ln.rstrip("\n").split(sep)
        if len(p)<2: continue
        m=Chem.MolFromSmiles(p[0])
        if m is not None: out[p[1]]=ik14(m)
    return out
def from_sdfs(folder):
    out={}
    for f in glob.glob(f"{folder}/*.sdf"):
        nm=os.path.basename(f)[:-4]
        m=Chem.MolFromMolFile(f)
        if m is None:
            supp=Chem.SDMolSupplier(f); m=next((x for x in supp if x is not None),None)
        if m is not None: out[nm]=ik14(m)
    return out
def from_kirhub():
    try: D=pd.read_csv(f"{KP}/kirhub_92_inhibitors_smiles.csv")
    except UnicodeDecodeError: D=pd.read_csv(f"{KP}/kirhub_92_inhibitors_smiles.csv",encoding="latin-1")
    sc="rdkit_canonical_smiles_from_input" if "rdkit_canonical_smiles_from_input" in D else "canonical_smiles"
    out={}
    for _,r in D.iterrows():
        m=Chem.MolFromSmiles(str(r[sc]))
        if m is not None: out[r["mapped_compound_name"]]=ik14(m)
    return out

def analyzable_ref() -> set[str]:
    """Kinase_ref 는 수집분(52) 중 실험값과 네 방법 예측이 모두 있는 48개만 분석에 들어간다.
    수집분 전체를 분모로 쓰면 leaked 를 뺀 나머지가 47 이 되어 본문 코호트 44 와 어긋난다
    (누출 5개 중 CK_17 은 실험값이 없어 이미 탈락하므로 이중으로 빠진다). 분모를 48 로 맞추면
    novel = 44 가 되어 Table 1·Methods S3 와 정확히 일치한다."""
    KR = f"{KP}/../Kinase_ref"
    res = {os.path.basename(os.path.dirname(f))
           for f in glob.glob(f"{KR}/Pred/*/*_result.tsv")}
    pred = None
    for f in (f"{KR}/AK/redock_zscore_long.csv", f"{KR}/RTM/redock_zscore_long.csv",
              f"{KR}/DOCK/redock_zscore_long.csv", f"{KP}/Gen2/redock_zscore_long.csv"):
        k = set(pd.read_csv(f, usecols=["DRUG_NAME"]).DRUG_NAME)
        pred = k if pred is None else (pred & k)
    return res & pred


def analyzable_pkis2() -> set[str]:
    """PKIS2 는 공개분 640 중 도킹 예측이 있는 636 만 분석에 들어간다(Table 1 의 full cohort).
    640 을 분모로 쓰면 leaked 를 뺀 나머지가 608 이 되어 본문 코호트 604 와 어긋난다."""
    return set(pd.read_csv(f"{KP}/../PKIS2/PKIS2_result/redock_zscore_long.csv",
                           usecols=["DRUG_NAME"]).DRUG_NAME)


_pkis2_ok = analyzable_pkis2()
_ref_all = from_sdfs(f"{KP}/../Kinase_ref/data")
_ref_ok = analyzable_ref()
print(f"Kinase_ref: SDF {len(_ref_all)}개 → 실험값·예측 완비 {len(_ref_all.keys() & _ref_ok)}개 "
      f"(제외 {sorted(set(_ref_all) - _ref_ok)})")

DATASETS={
 "KIR": from_kirhub(),
 "Kinase_ref": {k: v for k, v in _ref_all.items() if k in _ref_ok},
 "PKIS2": {k: v for k, v in from_smi(f"{KP}/../PKIS2/smiles_PKIS2.smi").items()
           if k in _pkis2_ok},
}

rows=[]; leaked_lists={}
for ds,mp in DATASETS.items():
    mp={k:v for k,v in mp.items() if v}
    leaked=sorted([n for n,k in mp.items() if k in train_ik])
    leaked_lists[ds]=leaked
    rows.append(dict(dataset=ds,n_parsed=len(mp),n_leaked=len(leaked),
        pct=round(100*len(leaked)/len(mp),1) if mp else 0))
S=pd.DataFrame(rows); S.to_csv(f"{OUT}/dataset_leakage_summary.csv",index=False)
print("\n[데이터셋별 ligand leakage (InChIKey14 완전일치)]")
print(S.to_string(index=False))
# leaked 화합물 목록 저장
maxn=max(len(v) for v in leaked_lists.values())
pd.DataFrame({ds:leaked_lists[ds]+[""]*(maxn-len(leaked_lists[ds])) for ds in DATASETS}).to_csv(f"{OUT}/dataset_leaked_compounds.csv",index=False)
for ds,v in leaked_lists.items():
    print(f"\n[{ds}] 학습셋에 존재하는 화합물 {len(v)}개:")
    print("  "+(", ".join(v) if v else "(없음)"))

# ── 그림 ──
fig,ax=plt.subplots(1,2,figsize=(14,5.5))
a=ax[0]; x=np.arange(len(S))
b=a.bar(x,S.pct,color=["#2166AC","#5AAE61","#E08214"],edgecolor="white")
for bi,(p,nl,nt) in zip(b,zip(S.pct,S.n_leaked,S.n_parsed)):
    a.text(bi.get_x()+bi.get_width()/2,p,f"{p:.1f}%\n({nl}/{nt})",ha="center",va="bottom",fontsize=9)
a.set_xticks(x);a.set_xticklabels(S.dataset);a.set_ylabel("% compounds in PDBbind (exact InChIKey14)")
a.set_ylim(0,max(S.pct)*1.25);a.set_title("Ligand leakage rate by dataset")
a=ax[1]  # 절대 개수 stacked
b1=a.bar(x,S.n_leaked,color="#B2182B",edgecolor="white",label="in training (leaked)")
b2=a.bar(x,S.n_parsed-S.n_leaked,bottom=S.n_leaked,color="#D9D9D9",edgecolor="white",label="not in training")
for xi,nl,nt in zip(x,S.n_leaked,S.n_parsed):
    a.text(xi,nt,f"{nt-nl} novel",ha="center",va="bottom",fontsize=9,fontweight="bold")
    a.text(xi,nl/2 if nl>40 else nl,f"{nl}",ha="center",
           va="center" if nl>40 else "bottom",fontsize=8.5,
           color="white" if nl>40 else "#B2182B")
a.set_ylim(0,max(S.n_parsed)*1.12)
a.set_xticks(x);a.set_xticklabels(S.dataset);a.set_ylabel("# compounds");a.set_title("Compound counts (leaked vs novel)");a.legend(fontsize=9,loc="upper left")
fig.suptitle("Training-set (PDBbind v2020 kinase ligands) overlap across benchmark datasets",fontsize=13,fontweight="bold")
_fl.stamp(ax)   # 패널 문자 (a)(b)(c)
fig.tight_layout(rect=[0,0,1,0.93]); fig.savefig(f"{OUT}/dataset_leakage.png",dpi=300,bbox_inches="tight"); plt.close()

# ── MD ──
md=[f"# 데이터셋 간 Ligand Leakage 비교 (vs PDBbind v2020 kinase)\n","**분석일**: 2026-07-21\n",
 f"누출 = 화합물 InChIKey14(2D 구조)가 PDBbind kinase 학습 리간드({len(train_ik)} 고유)와 완전일치.\n",
 "![fig](dataset_leakage.png)\n","## 요약\n",
 "| 데이터셋 | 파싱 화합물 | 학습셋 존재 | 비율 |","|---|---:|---:|---:|"]
for _,r in S.iterrows(): md.append(f"| {r.dataset} | {r.n_parsed} | {r.n_leaked} | **{r.pct}%** |")
md+=["\n## 학습셋에 존재하는 화합물\n"]
for ds,v in leaked_lists.items():
    md.append(f"**{ds}** ({len(v)}개): {', '.join(v) if v else '(없음)'}\n")
md+=["## 해석\n",
 "- KIR은 FDA 승인 저해제 위주 → PDBbind 중복↑. ref/PKIS2는 신규/탐색 화합물 비중↑ → 중복↓ 예상.",
 "- leakage 비율이 높을수록 target-ID 지표(Hit@1/10)가 부풀 여지 (complex_leakage 참조).",
 "- 상세: dataset_leaked_compounds.csv.",]
open(f"{OUT}/DATASET_LEAKAGE.md","w").write("\n".join(md))
print(f"\n✅ 저장 → {OUT}/ (DATASET_LEAKAGE.md, dataset_leakage.png, csv 2종)")
