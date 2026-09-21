"""
PKIS2 target recovery, AK-Score2 only.

For each compound it records whether the experimentally strongest target (primary = max
%inhibition) reaches the top of the prediction, and Hit@k over the strong actives
(%inhibition >= 90, the counterpart of %remaining <= 10 on KIR). Compounds with no active
kinase anywhere in their panel are excluded, as they are from Table 4.

Sign convention. PKIS2 reports %inhibition, where higher is stronger; KIR and Kinase_ref
report %remaining, where lower is stronger. Hence primary = max inhibition here against min
remaining there, and the strong-active threshold is inh >= 90 against rem <= 10.

Input:  PKIS2/PKIS2_exp.tsv, PKIS2_protein_mapped.tsv, PKIS2_result/redock_zscore_long.csv
Output: Kinase_paper/energy_benchmark/ (PKIS2_RECOVERY.md, pkis2_recovery.png, csv)
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys; sys.path.insert(0,"/mnt/d/Deepmolscan/Kinase_paper"); import novel_cohort as nc

KP = "/mnt/d/Deepmolscan/Kinase_paper"
PK = "/mnt/d/Deepmolscan/PKIS2"
OUT = f"{KP}/energy_benchmark"
os.makedirs(OUT, exist_ok=True)
STRONG = 90.0   # %inhibition >= 90 = strong active (⟺ KIR %remaining <= 10)
AKCOL = "#2166AC"

# ── 실험: %inhibition (높을수록 강한 저해) ──
exp = pd.read_csv(f"{PK}/PKIS2_exp.tsv", sep="\t", encoding="latin-1")
mp = pd.read_csv(f"{PK}/PKIS2_protein_mapped.tsv", sep="\t")
assay2uni = dict(zip(mp.paper_name, mp.UNIPROT_ID))
id_col = exp.columns[0]
el = exp.melt(id_vars=[id_col], var_name="assay", value_name="inh")
el.columns = ["DRUG", "assay", "inh"]
el["UNIPROT"] = el["assay"].map(assay2uni)
el["inh"] = pd.to_numeric(el["inh"], errors="coerce")
el = el.dropna(subset=["inh", "UNIPROT"])
# 여러 assay(변이체/도메인) → UNIPROT별 최대 억제(가장 강한 값)
exp_agg = el.groupby(["DRUG", "UNIPROT"], as_index=False)["inh"].max()
exp_agg = nc.drop_leaked(exp_agg, "PKIS2", col="DRUG")   # 학습 미사용 코호트

# ── 예측: AK z-score (낮을수록 강) ──
pred = pd.read_csv(f"{PK}/PKIS2_result/redock_zscore_long.csv")
AKp = pred.groupby(["DRUG_NAME", "UNIPROT_ID"], as_index=False).agg(
    z=("zscore_mean", "min"), r=("rank_mean", "min"))
aks = 1.0 if np.corrcoef(AKp.z, AKp.r)[0, 1] > 0 else -1.0   # >0: 낮은 z=best
print(f"AK sign={aks:+.0f} (낮은 z=best? {aks > 0})")
AKp = AKp.rename(columns={"DRUG_NAME": "DRUG", "UNIPROT_ID": "UNIPROT"})[["DRUG", "UNIPROT", "z"]]

drugs = sorted(set(AKp.DRUG) & set(exp_agg.DRUG))
print(f"공통 화합물(AK∩실험): {len(drugs)}")

rows = []
for dn in drugs:
    g = AKp[AKp.DRUG == dn].copy()
    if len(g) < 30:
        continue
    N = len(g)
    # 방향교정: aks>0이면 낮은 z=best → ascending=True (rank 1 = best)
    g["rAK"] = g.z.rank(method="min", ascending=(aks > 0))
    rk = dict(zip(g.UNIPROT, g.rAK))
    e = exp_agg[exp_agg.DRUG == dn]
    e = e[e.UNIPROT.isin(set(g.UNIPROT))]
    if len(e) == 0:
        continue
    prim = e.sort_values("inh", ascending=False).iloc[0]   # 최강 = MAX inh
    # 활성이 하나도 없는 화합물은 랭킹 과제가 성립하지 않는다(§2.2·Methods S3: "no ranking task
    # exists for them and they are excluded from per-compound metrics"). 표적이 패널 밖에 있어
    # Hit@k 가 구성상 0 이 되고 primary 도 실제 표적이 아니므로, Table 4·Figure 7 과 같은 기준으로 제외한다.
    if not (e.inh >= 50.0).any():
        continue
    acts = e[e.inh >= STRONG]
    pr = int(rk[prim.UNIPROT])
    rr = dict(cpd=dn, N=N, n_meas=len(e), n_act=len(acts),
              AK_prank=pr, AK_pctl=round(pr / N, 4))
    for k in (1, 5, 10):
        rr[f"AK_hit{k}"] = int(any(rk[u] <= k for u in acts.UNIPROT)) if len(acts) else 0
    rows.append(rr)

R = pd.DataFrame(rows)
R.to_csv(f"{OUT}/pkis2_recovery_per_compound.csv", index=False)
Nmed = int(R.N.median())
p = R.AK_prank
S = pd.DataFrame([dict(
    method="AK",
    top1=round((p <= 1).mean() * 100, 1), top5=round((p <= 5).mean() * 100, 1),
    top10=round((p <= 10).mean() * 100, 1), top30=round((p <= 30).mean() * 100, 1),
    medpctl=round(R.AK_pctl.median() * 100, 1),
    hit1=round(R.AK_hit1.mean() * 100, 1), hit5=round(R.AK_hit5.mean() * 100, 1),
    hit10=round(R.AK_hit10.mean() * 100, 1))])
S.to_csv(f"{OUT}/pkis2_recovery_summary.csv", index=False)
s = S.iloc[0]
rnd30 = 100 * 30 / Nmed
print(f"\n[PKIS2 회수] pool {Nmed}, n={len(R)}")
print(S.to_string(index=False))

# ── 그림 (영어) ──
fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))
a = ax[0]
labs = ["Top-1", "Top-5", "Top-10", "Top-30"]
vals = [s.top1, s.top5, s.top10, s.top30]
b = a.bar(range(4), vals, color=AKCOL, edgecolor="white", width=0.6)
for bi, v in zip(b, vals):
    a.text(bi.get_x() + bi.get_width() / 2, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=10)
a.axhline(rnd30, color="red", ls="--", lw=1, label=f"random Top-30 = {rnd30:.1f}%")
a.set_xticks(range(4)); a.set_xticklabels(labs)
a.set_ylabel("% compounds with primary target in Top-k")
a.set_title(f"PKIS2 primary-target recovery (AK-Score2, n={len(R)})")
a.legend(fontsize=9)
a = ax[1]
labs = ["Hit@1", "Hit@5", "Hit@10"]
vals = [s.hit1, s.hit5, s.hit10]
b = a.bar(range(3), vals, color=AKCOL, edgecolor="white", width=0.55)
for bi, v in zip(b, vals):
    a.text(bi.get_x() + bi.get_width() / 2, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=10)
a.set_xticks(range(3)); a.set_xticklabels(labs)
a.set_ylabel("% (any strong active in top-k)")
a.set_title(f"Hit@k (strong active: %inh>=90)")
fig.suptitle(f"PKIS2: primary-target recovery by AK-Score2 (pool {Nmed}, n={len(R)}) "
             f"| {nc.tag()} cohort",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f"{OUT}/pkis2_recovery.png", dpi=150, bbox_inches="tight")
plt.close()

# ── MD ──
md = [
    "# PKIS2 표적 회수 (B) — AK 단독", "", "**분석일**: 2026-07-21", "",
    f"공통 {len(R)} 화합물, pool {Nmed}(AK 예측 타겟). primary=실험 최강(MAX %inh), 강한 active=%inh>=90.", "",
    "> ⚠ PKIS2는 %inhibition(높을수록 강) — KIR/ref의 %remaining(낮을수록 강)과 방향 반대.",
    "> primary=MAX %inh, 강활성=%inh>=90 (KIR %rem<=10 대응).",
    "> ⚠ **PKIS2에는 RTM 예측이 없음** → AK 단독. RTM/consensus는 N/A(RTM 미산출).", "",
    "![kr](pkis2_recovery.png)", "", "## 요약", "",
    "| 지표 | AK | RTM | consensus |",
    "|---|:---:|:---:|:---:|",
    f"| primary Top-1 | {s.top1:.1f}% | N/A | N/A |",
    f"| primary Top-5 | {s.top5:.1f}% | N/A | N/A |",
    f"| **primary Top-10** | {s.top10:.1f}% | N/A | N/A |",
    f"| primary Top-30 | {s.top30:.1f}% | N/A | N/A |",
    f"| percentile 중앙값(↓) | {s.medpctl:.1f}% | N/A | N/A |",
    f"| Hit@1(any) | {s.hit1:.1f}% | N/A | N/A |",
    f"| Hit@5(any) | {s.hit5:.1f}% | N/A | N/A |",
    f"| Hit@10(any) | {s.hit10:.1f}% | N/A | N/A |",
    "", f"랜덤 Top-30 기대 ≈ {rnd30:.1f}% (pool {Nmed}).", "",
    "## 해석", "",
    f"- AK primary Top-10 {s.top10:.1f}% (랜덤 Top-30 {rnd30:.1f}% 대비 농축), Hit@10 {s.hit10:.1f}%.",
    "- PKIS2는 신규/탐색 화합물(누출 5.0%)·선택적(active_ratio 0.052)·대형 패널(387)이라 회수 난도 높음.",
    "- RTM 미산출로 KIR(⑤-b)·ref(⑤-b)와 달리 consensus 대비는 불가.",
    "- 상세: pkis2_recovery_per_compound.csv, pkis2_recovery_summary.csv.",
]
open(f"{OUT}/PKIS2_RECOVERY.md", "w").write("\n".join(md))
print(f"\n✅ 저장 → {OUT}/ (PKIS2_RECOVERY.md, pkis2_recovery.png, csv 2종)")
