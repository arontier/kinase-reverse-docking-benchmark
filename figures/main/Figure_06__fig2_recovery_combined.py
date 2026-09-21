"""
Primary-target recovery, KIR and PKIS2 in one figure.

Replaces two separate PNGs with a single panel (a)/(b) layout. The metrics match the Results text:
primary Top-k (the experimentally strongest target within the top k) and Hit@k (any measured
active within the top k). Each panel ranks within the kinases its own panel measures, so the
random-expectation line differs slightly between them.

Input:  energy_benchmark/{kinome,pkis2}_recovery_{summary,per_compound}.csv (AK-Score2)
Output: Kinase_paper/energy_benchmark/fig2_recovery_combined.png
"""
from __future__ import annotations
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sys; sys.path.insert(0,"/mnt/d/Deepmolscan/Kinase_paper"); import novel_cohort as nc

KP = "/mnt/d/Deepmolscan/Kinase_paper"; EB = f"{KP}/energy_benchmark"
KS = [1, 5, 10]
C_PRIM, C_HIT = "#2166AC", "#92C5DE"

def load(tag):
    s = pd.read_csv(f"{EB}/{tag}_recovery_summary.csv")
    s = s[s.method == "AK"].iloc[0]
    pc = pd.read_csv(f"{EB}/{tag}_recovery_per_compound.csv")
    n = len(pc)
    pool = int(pc["N"].median()) if "N" in pc.columns else np.nan
    return s, n, pool

panels = [("(a)", "KIR", *load("kinome")), ("(b)", "PKIS2", *load("pkis2"))]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
for ax, (tag, name, s, n, pool) in zip(axes, panels):
    x = np.arange(len(KS)); w = 0.38
    prim = [s[f"top{k}"] for k in KS]
    hit = [s[f"hit{k}"] for k in KS]
    b1 = ax.bar(x - w / 2, prim, w, label="Primary target in top-k", color=C_PRIM, edgecolor="white")
    b2 = ax.bar(x + w / 2, hit, w, label="Any measured active in top-k (Hit@k)", color=C_HIT, edgecolor="white")
    for bars in (b1, b2):
        for bi in bars:
            ax.text(bi.get_x() + bi.get_width() / 2, bi.get_height() + 1.0,
                    f"{bi.get_height():.1f}", ha="center", va="bottom", fontsize=8.5)
    # 랜덤 기대치(primary): k / pool
    if np.isfinite(pool):
        rnd = [100.0 * k / pool for k in KS]
        ax.plot(x - w / 2, rnd, "v--", color="#B2182B", ms=5, lw=1.1,
                label=f"Random expectation (primary, pool = {pool})")
    ax.set_xticks(x); ax.set_xticklabels([f"top-{k}" for k in KS])
    ax.set_ylim(0, max(max(prim), max(hit)) * 1.22)
    ax.set_xlabel("Number of top-ranked kinases inspected")
    ax.set_ylabel("% of compounds")
    ax.set_title(f"{name}  (n = {n} compounds)", fontsize=11.5)
    ax.grid(axis="y", alpha=.25)
    ax.legend(fontsize=8, loc="upper left")
    # 패널 라벨
    ax.text(-0.10, 1.06, tag, transform=ax.transAxes, fontsize=17, fontweight="bold", va="top", ha="left")

fig.suptitle(f"Primary-target recovery by AK-Score2 reverse docking  |  {nc.tag()} cohort",
             fontsize=13.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = f"{EB}/fig2_recovery_combined.png"
fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close()

print("panel  dataset       n   pool   primary top-1/5/10        Hit@1/5/10")
for tag, name, s, n, pool in panels:
    print(f"{tag:5s}  {name:12s} {n:3d}  {pool:5d}   "
          f"{s.top1:5.1f}/{s.top5:5.1f}/{s.top10:5.1f}   {s.hit1:5.1f}/{s.hit5:5.1f}/{s.hit10:5.1f}")
print(f"\n✅ 저장 → {out}")
