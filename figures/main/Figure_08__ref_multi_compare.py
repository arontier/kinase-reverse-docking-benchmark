"""
Kinase_ref target recovery by the four single scoring functions.

The figure originally showed three single functions together with the consensus schemes. Consensus
moved to its own section of the paper, so this keeps the single functions only and adds GenScore.
Compounds are ranked within the panel each source paper profiled.

Reads ref_panel_eval_summary.csv, so ref_panel_eval.py must run first.

Output: Kinase_paper/kinase_ref_check/ref_multi_compare.png
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, "/mnt/d/Deepmolscan/Kinase_paper")
import figlabels as _fl  # noqa: E402

KP = "/mnt/d/Deepmolscan/Kinase_paper"
OUT = f"{KP}/kinase_ref_check"
os.makedirs(OUT, exist_ok=True)

SUMMARY = f"{OUT}/ref_panel_eval_summary.csv"
if not os.path.exists(SUMMARY):
    sys.exit(f"✗ {SUMMARY} 없음 — ref_panel_eval.py 를 먼저 실행할 것")
T = pd.read_csv(SUMMARY)

METHODS = ["AK", "RTM", "DOCK", "GEN"]          # 단일 스코어링 함수만
DISP = {"AK": "AK-Score2", "RTM": "RTMScore", "DOCK": "AD4 energy", "GEN": "GenScore"}
COL = {"AK": "#2166AC", "RTM": "#5AAE61", "DOCK": "#4393C3", "GEN": "#B2182B"}
PANEL, FULL = "원 논문 측정 패널", "예측 pool 618"
POOLS = [(PANEL, "measured panel"), (FULL, "prediction pool of 618")]

missing = [m for m in METHODS if m not in set(T.method)]
if missing:
    sys.exit(f"✗ summary 에 없는 방법: {missing}")


def val(pool: str, meth: str, col: str) -> float:
    return float(T[(T["pool"] == pool) & (T["method"] == meth)].iloc[0][col])


fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.9))

# ── (a) primary-target Top-k 프로파일 (측정 패널) ────────────────────────────
KS = [("top1", "Top-1"), ("top5", "Top-5"), ("top10", "Top-10"), ("top30", "Top-30")]
a = ax[0]
x = np.arange(len(KS))
w = 0.8 / len(METHODS)
for i, m in enumerate(METHODS):
    v = [val(PANEL, m, k) for k, _ in KS]
    b = a.bar(x + (i - (len(METHODS) - 1) / 2) * w, v, w, label=DISP[m],
              color=COL[m], edgecolor="white")
    for bi, vv in zip(b, v):
        a.text(bi.get_x() + bi.get_width() / 2, vv, f"{vv:.1f}", ha="center",
               va="bottom", fontsize=7)
a.set_xticks(x)
a.set_xticklabels([lab for _, lab in KS])
a.set_ylabel("% of compounds")
n_panel = int(T[T.pool == PANEL].n.iloc[0])
a.set_title(f"Primary target within Top-k — measured panel (n = {n_panel})",
            fontsize=11.5, fontweight="bold")
a.set_ylim(0, max(a.get_ylim()[1] * 1.30, 1))
a.legend(fontsize=8.5)
a.grid(alpha=.25, axis="y")

# ── (b) Hit@10 (측정 패널) ──────────────────────────────────────────────────
n_full = int(T[T.pool == FULL].n.iloc[0])   # SI 민감도 표에서만 사용
a = ax[1]
xx = np.arange(len(METHODS))
v = [val(PANEL, m, "hit10") for m in METHODS]
b = a.bar(xx, v, 0.6, color=[COL[m] for m in METHODS], edgecolor="white")
for bi, vv in zip(b, v):
    a.text(bi.get_x() + bi.get_width() / 2, vv, f"{vv:.1f}", ha="center",
           va="bottom", fontsize=8)
a.set_xticks(xx)
a.set_xticklabels([DISP[m] for m in METHODS], fontsize=9, rotation=12)
a.set_ylabel("% of compounds with a measured active in Top-10")
a.set_title(f"Hit@10 — measured panel (n = {n_panel})", fontsize=11.5, fontweight="bold")
a.set_ylim(0, max(a.get_ylim()[1] * 1.25, 1))
a.grid(alpha=.25, axis="y")

fig.suptitle("Kinase_ref: target recovery by single scoring functions, ranked within the "
             "panel each source paper measured (leakage-free cohort)",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])
_fl.stamp(ax)
fig.savefig(f"{OUT}/ref_multi_compare.png", dpi=300, bbox_inches="tight")
plt.close()

# ── 리포트 ──────────────────────────────────────────────────────────────────
md = ["# Kinase_ref — 단일 스코어링 함수 표적 회수 (Figure 8)\n",
      f"학습 미사용 코호트. 본문은 원 논문이 실제 측정한 패널만 사용한다(n={n_panel}). "
      "측정하지 않은 키나아제를 decoy 로 넣는 618 pool 값은 SI 민감도 표에만 남긴다"
      f"(n={n_full}).\n",
      "primary = 원 논문에서 측정값이 가장 강한 키나아제(동률이면 그 집합의 최선 순위). "
      "consensus 조합은 이 그림에서 제외하고 2.9절에서 따로 다룬다.\n",
      "![m](ref_multi_compare.png)\n",
      "| pool | 방법 | Top-1 | Top-5 | Top-10 | Top-30 | pctl중앙 | Hit@10 |",
      "|---|---|---:|---:|---:|---:|---:|---:|"]
for pool, lab in POOLS:
    for m in METHODS:
        md.append(f"| {lab} | {DISP[m]} | " + " | ".join(
            f"{val(pool, m, c):.1f}%" if c != "med_pctl" else f"{val(pool, m, c):.1f}%"
            for c in ("top1", "top5", "top10", "top30", "med_pctl", "hit10")) + " |")
open(f"{OUT}/REF_MULTI.md", "w").write("\n".join(md))
print("\n".join(md[4:]))
print(f"\n✅ 저장 → {OUT}/ref_multi_compare.png, REF_MULTI.md")
