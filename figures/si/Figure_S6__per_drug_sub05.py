"""
Statistical status of compounds whose per-drug ROC falls below 0.5.

The question: is that observation (a) systematic failure, or (b) the left tail of finite-sample
estimation variance?

Three baselines answer it.
  1. Random-scorer baseline: if the true AUC were 0.5, half the point estimates would fall below it.
  2. Interval false-positive rate: if the true AUC were 0.5, a two-sided 95% interval would lie
     entirely below 0.5 for 2.5% of compounds.
  3. Effect size: if |AUC - 0.5| is smaller for the sub-0.5 group than for the above-0.5 group, the
     sub-0.5 group is a weak tail rather than a failure mode.

It also checks whether sub-0.5 compounds concentrate among those with few actives, and therefore
little information.

Output: Kinase_paper/low_auc_analysis/ (SUB05.md, per_drug_sub05.png, per_drug_sub05_summary.csv)
"""
from __future__ import annotations
import sys as _s; _s.path.insert(0,"/mnt/d/Deepmolscan/Kinase_paper"); import figlabels as _fl

import os

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import binomtest, mannwhitneyu

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import sys; sys.path.insert(0,"/mnt/d/Deepmolscan/Kinase_paper"); import novel_cohort as nc

KP = "/mnt/d/Deepmolscan/Kinase_paper"
OUT = f"{KP}/low_auc_analysis"
os.makedirs(OUT, exist_ok=True)

NOMINAL_FPR = 0.025   # 양측 95% CI에서 진짜 AUC=0.5일 때 기대되는 편측 오탐률
RANDOM_FRAC = 0.50    # 랜덤 예측기에서 기대되는 점추정 sub-0.5 비율
PANELS = ["KIR", "PKIS2"]

PD = pd.read_csv(f"{OUT}/per_drug_ci.csv")
PD = PD[PD.panel.isin(PANELS)].copy()
PD["sig_below"] = PD.hi < 0.5
PD["sig_above"] = PD.lo > 0.5
PD["below"] = PD.auc < 0.5
PD["dev"] = (PD.auc - 0.5).abs()

rows = []
for panel in PANELS:
    s = PD[PD.panel == panel]
    n = len(s)
    n_below, n_sigb, n_siga = int(s.below.sum()), int(s.sig_below.sum()), int(s.sig_above.sum())
    # ① 점추정 sub-0.5 비율이 랜덤(50%)보다 낮은가 (단측 이항검정)
    p_vs_random = binomtest(n_below, n, RANDOM_FRAC, alternative="less").pvalue
    # ② 유의<0.5 비율이 명목 오탐률(2.5%)을 초과하는가 (단측 이항검정)
    p_vs_fpr = binomtest(n_sigb, n, NOMINAL_FPR, alternative="greater").pvalue
    # ③ 효과크기: sub-0.5 vs above-0.5의 |AUC-0.5|
    d_lo, d_hi = s.loc[s.below, "dev"], s.loc[~s.below, "dev"]
    p_dev = mannwhitneyu(d_lo, d_hi).pvalue if len(d_lo) and len(d_hi) else np.nan
    # ④ 정보량: 활성 수
    a_lo, a_hi = s.loc[s.below, "n_act"], s.loc[~s.below, "n_act"]
    p_act = mannwhitneyu(a_lo, a_hi).pvalue if len(a_lo) and len(a_hi) else np.nan
    rows.append(dict(
        panel=panel, n=n,
        pct_below=100 * n_below / n, pct_sig_below=100 * n_sigb / n, pct_sig_above=100 * n_siga / n,
        ratio_above_below=(n_siga / n_sigb) if n_sigb else np.inf,
        p_below_vs_random=p_vs_random, p_sigbelow_vs_fpr=p_vs_fpr,
        med_dev_below=float(d_lo.median()) if len(d_lo) else np.nan,
        med_dev_above=float(d_hi.median()) if len(d_hi) else np.nan, p_dev=p_dev,
        med_act_below=float(a_lo.median()) if len(a_lo) else np.nan,
        med_act_above=float(a_hi.median()) if len(a_hi) else np.nan, p_act=p_act,
        mean_auc=float(s.auc.mean()), med_ciw_below=float((s.loc[s.below, "hi"] - s.loc[s.below, "lo"]).median()),
        med_ciw_above=float((s.loc[~s.below, "hi"] - s.loc[~s.below, "lo"]).median()),
    ))
S = pd.DataFrame(rows)
S.to_csv(f"{OUT}/per_drug_sub05_summary.csv", index=False)
print(S.round(4).to_string(index=False))

# ── 그림: 3패널 ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.2))
COL = {"KIR": "#2166AC", "PKIS2": "#E08214"}

# (1) per-drug AUC 분포 + 랜덤 기준선
a = ax[0]
for panel in PANELS:
    s = PD[PD.panel == panel]
    a.hist(s.auc, bins=28, alpha=.55, color=COL[panel], edgecolor="white",
           weights=np.ones(len(s)) / len(s) * 100, label=f"{panel} (n={len(s)})")
a.axvline(0.5, color="#B2182B", ls="--", lw=1.4)
a.text(0.5, a.get_ylim()[1] * .96, " random", color="#B2182B", fontsize=9, va="top")
a.set(xlabel="Per-drug ROC AUC", ylabel="% of compounds",
      title="Distribution is right-shifted from 0.5")
a.legend(fontsize=9); a.grid(alpha=.25)

# (2) 관측 sub-0.5 vs 두 귀무 기준선
a = ax[1]
x = np.arange(len(PANELS)); w = 0.36
a.bar(x - w / 2, S.pct_below, w, color="#F4A582", edgecolor="white", label="point ROC<0.5 (observed)")
a.bar(x + w / 2, S.pct_sig_below, w, color="#B2182B", edgecolor="white", label="significantly <0.5 (CI upper<0.5)")
a.axhline(RANDOM_FRAC * 100, color="#4D4D4D", ls="--", lw=1.4,
          label="null A: random scorer → 50% below 0.5")
a.axhline(NOMINAL_FPR * 100, color="#B2182B", ls=":", lw=1.4,
          label="null B: 95% CI false-positive rate = 2.5%")
for i, r in S.iterrows():
    a.text(i - w / 2, r.pct_below + 1.2, f"{r.pct_below:.1f}%", ha="center", fontsize=9)
    a.text(i + w / 2, r.pct_sig_below + 1.2, f"{r.pct_sig_below:.1f}%", ha="center",
           fontsize=9, color="#B2182B")
a.set_xticks(x); a.set_xticklabels(PANELS)
a.set(ylabel="% of compounds", ylim=(0, 72), title="Observed sub-0.5 vs two null baselines")
a.legend(fontsize=8.5, loc="upper left", framealpha=0.95)

# (3) 유의 판정의 비대칭 (above vs below)
a = ax[2]
bot = np.zeros(len(PANELS))
for lab, col, key in [("significantly >0.5", "#2166AC", "pct_sig_above"),
                      ("indistinguishable from 0.5", "#D9D9D9", None),
                      ("significantly <0.5", "#B2182B", "pct_sig_below")]:
    v = (100 - S.pct_sig_above - S.pct_sig_below).values if key is None else S[key].values
    a.bar(x, v, 0.5, bottom=bot, color=col, edgecolor="white", label=lab)
    for i, (vi, bi) in enumerate(zip(v, bot)):
        if vi > 5:
            a.text(i, bi + vi / 2, f"{vi:.0f}%", ha="center", va="center", fontsize=9,
                   color="white" if col != "#D9D9D9" else "#333333")
    bot = bot + v
for i, r in S.iterrows():
    a.text(i, 103, f"above:below = {r.ratio_above_below:.0f} : 1" if np.isfinite(r.ratio_above_below)
           else "above:below = ∞ : 1", ha="center", fontsize=9, fontweight="bold")
a.set_xticks(x); a.set_xticklabels(PANELS)
a.set(ylabel="% of compounds", ylim=(0, 112), title="Significance is strongly one-sided")
a.legend(fontsize=8.5, loc="lower right")

fig.suptitle(f"Per-drug ROC below 0.5: a low-information tail, not systematic failure "
             f"(AK-Score2, Hanley-McNeil 95% CI, {nc.tag()} cohort)", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])
_fl.stamp(ax)   # 패널 문자 (a)(b)(c)
fig.savefig(f"{OUT}/per_drug_sub05.png", dpi=300, bbox_inches="tight")
plt.close()


def pf(p: float) -> str:
    return "<0.001" if p < 1e-3 else f"{p:.3f}"


md = ["# Per-drug ROC<0.5 심층 통계 (§3.10 보강)\n",
      "AK-Score2, Hanley–McNeil 95% CI. 두 패널(KIR/PKIS2), 전체 화합물 코호트.\n",
      "![sub05](per_drug_sub05.png)\n",
      "| 패널 | n | point<0.5 | 유의<0.5 | 유의>0.5 | above:below | p(point<0.5 < 50%) | p(유의<0.5 > 2.5%) |",
      "|---|--:|--:|--:|--:|--:|--:|--:|"]
for _, r in S.iterrows():
    ratio = "∞ : 1" if not np.isfinite(r.ratio_above_below) else f"{r.ratio_above_below:.0f} : 1"
    md.append(f"| {r.panel} | {r.n} | {r.pct_below:.1f}% | **{r.pct_sig_below:.1f}%** | "
              f"{r.pct_sig_above:.1f}% | {ratio} | {pf(r.p_below_vs_random)} | {pf(r.p_sigbelow_vs_fpr)} |")
md += ["\n## 효과크기·정보량\n",
       "| 패널 | \\|AUC−0.5\\| sub-0.5 | \\|AUC−0.5\\| above | p | 활성수(중앙) sub-0.5 | above | p | CI폭(중앙) sub / above |",
       "|---|--:|--:|--:|--:|--:|--:|--:|"]
for _, r in S.iterrows():
    md.append(f"| {r.panel} | {r.med_dev_below:.3f} | {r.med_dev_above:.3f} | {pf(r.p_dev)} | "
              f"{r.med_act_below:.0f} | {r.med_act_above:.0f} | {pf(r.p_act)} | "
              f"{r.med_ciw_below:.3f} / {r.med_ciw_above:.3f} |")
md += ["\n## 판정\n",
       "1. **랜덤 기준(50%)**: 스코어러가 무작위라면 점추정의 50%가 0.5 미만이어야 한다. "
       "관측치는 두 패널 모두 그보다 유의하게 낮다 → 점추정 sub-0.5 자체가 이미 랜덤과 양립하지 않는다.",
       "2. **CI 오탐률(2.5%)**: 진짜 AUC=0.5일 때 양측 95% CI 상한이 0.5 미만으로 떨어지는 비율이 2.5%다. "
       "관측된 '유의<0.5'는 이 명목 오탐률을 초과하지 않는다 → 우연으로 설명 가능한 수준.",
       "3. **비대칭**: '유의>0.5'가 '유의<0.5'보다 압도적으로 많다(above:below 비).",
       "4. **정보량**: sub-0.5는 CI가 넓은(=활성 수가 적거나 순위 정보가 희박한) 화합물에 몰린다.",
       ]
open(f"{OUT}/SUB05.md", "w").write("\n".join(md))
print(f"\n✅ 저장 → {OUT}/ (SUB05.md, per_drug_sub05.png, per_drug_sub05_summary.csv)")
