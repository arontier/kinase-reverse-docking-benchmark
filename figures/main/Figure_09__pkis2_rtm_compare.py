"""
PKIS2: AK-Score2 against RTMScore and their consensus, compared with the experiment
(%inhibition at 1 uM, active >= 50).

Global and per-drug ROC, Hit@k and EF1% on the merged set common to the experiment, AK-Score2 and
RTMScore, with a paired Wilcoxon signed-rank test on per-drug ROC.

Output: Kinase_paper/pkis2_rtm_compare/ (PKIS2_RTM.md, pkis2_rtm_compare.png, csv)
"""
from __future__ import annotations
import sys as _s; _s.path.insert(0,"/mnt/d/Deepmolscan/Kinase_paper"); import figlabels as _fl
import os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
sys.path.insert(0, "/mnt/d/Deepmolscan/PKIS2"); import analyze_pkis as ap
import sys; sys.path.insert(0,"/mnt/d/Deepmolscan/Kinase_paper"); import novel_cohort as nc

BASE = "/mnt/d/Deepmolscan/PKIS2"; KP = "/mnt/d/Deepmolscan/Kinase_paper"
OUT = f"{KP}/pkis2_rtm_compare"; os.makedirs(OUT, exist_ok=True)
RRF_K = 60

# ── 실험 라벨(single_thresh=50) ──
exp = ap.load_experiment(ap.CONFIGS["PKIS2"])
exp["label"] = (exp["exp_inh"] >= 50).astype(int)
exp = nc.drop_leaked(exp, "PKIS2")   # 학습 미사용 코호트

# ── 방향교정 예측 로드 ──
def load(f, name):
    d = pd.read_csv(f); d["DRUG_NAME"] = d["DRUG_NAME"].astype(str).str.strip().replace(ap.DRUG_ALIAS)
    d = d.groupby(["DRUG_NAME", "UNIPROT_ID"], as_index=False).agg(z=("zscore_mean", "min"), r=("rank_mean", "min"))
    s = 1.0 if np.corrcoef(d.z, d.r)[0, 1] > 0 else -1.0     # s>0: 낮은 z=best(AK)  s<0: 높은 z=best(RTM)
    d[name] = -s * d.z
    print(f"  {name}: 화합물 {d.DRUG_NAME.nunique()}  sign={s:+.0f} → pred={'-' if s>0 else '+'}z")
    return d[["DRUG_NAME", "UNIPROT_ID", name]]

ak = load(f"{BASE}/PKIS2_result/redock_zscore_long.csv", "AK")
rtm = load(f"{BASE}/RTM/redock_zscore_long.csv", "RTM")

# ── 공통 병합(실험∩AK∩RTM) + consensus ──
M = exp.merge(ak, on=["DRUG_NAME", "UNIPROT_ID"]).merge(rtm, on=["DRUG_NAME", "UNIPROT_ID"])
for c in ["AK", "RTM"]:
    M[f"z_{c}"] = M.groupby("DRUG_NAME")[c].transform(lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0 else s * 0)
    M[f"r_{c}"] = M.groupby("DRUG_NAME")[c].rank(method="min", ascending=False)
M["CONS"] = 1.0 / (RRF_K + M["r_AK"]) + 1.0 / (RRF_K + M["r_RTM"])   # RRF(AK+RTM)
M["ZAVG"] = (M["z_AK"] + M["z_RTM"]) / 2.0        # 논문이 채택한 consensus(동일가중 z평균, §2.8)
print(f"공통 병합: 화합물 {M.DRUG_NAME.nunique()}  타겟 {M.UNIPROT_ID.nunique()}  쌍 {len(M):,}  Active {int(M.label.sum()):,} ({M.label.mean()*100:.1f}%)")

METHODS = ["AK", "RTM", "CONS", "ZAVG"]
LABEL = {"AK": "AK-Score2", "RTM": "RTMScore", "CONS": "consensus (RRF)",
         "ZAVG": "consensus (z-avg)*"}

# ── Global ──
glob = {m: ap.compute_metrics(M.label.values, M[m].values) for m in METHODS}

# ── Per-drug (active≥5) ──
def perdrug(col):
    rows = {}
    for d, g in M.groupby("DRUG_NAME"):
        if g.label.sum() >= ap.MIN_ACTIVES and (g.label == 0).sum() >= 1:
            rows[d] = ap.compute_metrics(g.label.values, g[col].values)
    return rows
PD = {m: perdrug(m) for m in METHODS}
drugs = sorted(PD["AK"].keys())     # 세 방법 동일 pool(같은 M)
def agg(rows, k): return float(np.mean([v[k] for v in rows.values()]))

# ── 짝지은 Wilcoxon: RTM vs AK, CONS vs AK (per-drug ROC) ──
akroc = np.array([PD["AK"][d]["roc_auc"] for d in drugs])
def wilcox_vs_ak(m):
    v = np.array([PD[m][d]["roc_auc"] for d in drugs])
    return np.nan if m == "AK" else wilcoxon(v, akroc).pvalue

# ── 요약 표 ──
rows = []
for m in METHODS:
    r = PD[m]
    rows.append(dict(method=LABEL[m],
                     global_roc=glob[m]["roc_auc"], pd_roc=agg(r, "roc_auc"),
                     hit1=agg(r, "hit1"), hit5=agg(r, "hit5"), hit10=agg(r, "hit10"),
                     ef1=agg(r, "ef1"), rprec=agg(r, "r_precision"),
                     p_vs_AK=wilcox_vs_ak(m), n_drug=len(r)))
S = pd.DataFrame(rows); S.to_csv(f"{OUT}/pkis2_rtm_summary.csv", index=False)
print("\n" + S.round(3).to_string(index=False))

# ── 그림: (좌) 방법별 Per-Drug ROC 막대  (우) per-drug ROC 산점 AK vs RTM ──
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
a = ax[0]; cols = {"AK": "#2166AC", "RTM": "#F4A582", "CONS": "#9970AB", "ZAVG": "#762A83"}
vals = [agg(PD[m], "roc_auc") for m in METHODS]
b = a.bar([LABEL[m] for m in METHODS], vals, color=[cols[m] for m in METHODS], edgecolor="white")
a.axhline(0.5, color="gray", ls="--", lw=1, label="random")
for bi, v in zip(b, vals): a.text(bi.get_x() + bi.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=10)
a.set_ylim(0.5, max(vals) + 0.045); a.set_ylabel("Per-Drug ROC (mean)")
a.set_title(f"PKIS2 target-ID (per-drug n={len(drugs)}; * = adopted consensus)")
a.tick_params(axis="x", labelsize=8.5); a.legend(fontsize=8)

a = ax[1]
rtmroc = np.array([PD["RTM"][d]["roc_auc"] for d in drugs])
a.scatter(akroc, rtmroc, s=14, alpha=0.5, color="#2166AC")
lim = [min(akroc.min(), rtmroc.min()) - 0.02, max(akroc.max(), rtmroc.max()) + 0.02]
a.plot(lim, lim, color="gray", ls="--", lw=1); a.axhline(0.5, color="#B2182B", lw=0.6); a.axvline(0.5, color="#B2182B", lw=0.6)
a.set(xlabel="AK-Score2 per-drug ROC", ylabel="RTMScore per-drug ROC", xlim=lim, ylim=lim,
      title=f"per-compound ROC (RTM>AK: {int((rtmroc>akroc).sum())}/{len(drugs)})")
fig.suptitle(f"PKIS2: AK-Score2 vs RTMScore vs consensus  |  {nc.tag()} cohort, experimental %inh>=50",
             fontsize=13, fontweight="bold")
_fl.stamp(ax)   # 패널 문자 (a)(b)(c)
fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(f"{OUT}/pkis2_rtm_compare.png", dpi=300, bbox_inches="tight"); plt.close()

# ── 리포트 ──
def f3(x): return "—" if (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"
md = ["# PKIS2 — RTMScore 추가 후 AK vs RTM vs consensus\n",
      f"**공통 병합**(실험∩AK∩RTM): 화합물 {M.DRUG_NAME.nunique()}, 타겟 {M.UNIPROT_ID.nunique()}, 쌍 {len(M):,}, Active {M.label.mean()*100:.1f}%. "
      f"per-drug n={len(drugs)}(active≥5). 라벨 = %inh≥50.\n",
      "![cmp](pkis2_rtm_compare.png)\n",
      "| 방법 | Global ROC | Per-Drug ROC | Hit@1 | Hit@5 | Hit@10 | EF1% | R-prec | p vs AK |",
      "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
for _, r in S.iterrows():
    md.append(f"| {r.method} | {r.global_roc:.3f} | {r.pd_roc:.3f} | {r.hit1:.2f} | {r.hit5:.2f} | "
              f"{r.hit10:.2f} | {r.ef1:.2f} | {r.rprec:.3f} | {f3(r.p_vs_AK)} |")
md += ["\n## 해석\n",
       f"- per-drug 공통 {len(drugs)}개 화합물에서 RTM이 AK보다 ROC 높은 화합물: {int((rtmroc>akroc).sum())}/{len(drugs)}.",
       "- consensus(RRF)가 단일 최고를 상회하면 KIR과 동일한 상보성 결론이 PKIS2에서도 재현됨을 뜻한다."]
open(f"{OUT}/PKIS2_RTM.md", "w").write("\n".join(md))
print(f"\n✅ 저장 → {OUT}/ (PKIS2_RTM.md, pkis2_rtm_compare.png, csv)")
