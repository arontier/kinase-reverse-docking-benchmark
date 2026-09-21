"""
Decide whether a scoring function's top-1 target collapse is an artefact or warranted signal.

If the target a function keeps putting first is genuinely active for those compounds, the
preference is warranted. This computes the measured active rate of the collapsed target and
compares it with the panel-wide active rate.

Output: Kinase_paper/top1_bias/ (validate csv, md, png)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

KP = "/mnt/d/Deepmolscan/Kinase_paper"
PK = "/mnt/d/Deepmolscan/PKIS2"
KR = "/mnt/d/Deepmolscan/Kinase_ref"
sys.path.insert(0, PK)
sys.path.insert(0, KP)
import analyze_pkis as ap
import kinome_merge as km
import novel_cohort as nc

OUT = f"{KP}/top1_bias"
os.makedirs(OUT, exist_ok=True)

# Table S8 앞 열(top1_bias_all.py)이 정한 최빈 top-1 타겟을 그대로 쓴다.
# 동률(예: PKIS2 AK-Score2의 KDR/LCK 5.5%)일 때 두 표가 다른 유전자를 보고하는 것을 막는다.
_A = pd.read_csv(f"{OUT}/top1_bias_all.csv")
FIXED = {(r.panel, r.method): r.top_uniprot for _, r in _A.iterrows()}

PANELS = {
    "KIR": {
        "AK-Score2":             f"{KP}/result_87/redock_zscore_long.csv",
        "AD4 energy":          f"{KP}/DOCK/redock_zscore_long.csv",
        "RTMScore":              f"{KP}/RTM/redock_zscore_long.csv",
        "GenScore":              f"{KP}/Gen/redock_zscore_long.csv",
    },
    "PKIS2": {
        "AK-Score2":           f"{PK}/PKIS2_result/redock_zscore_long.csv",
        "RTMScore":            f"{PK}/RTM/redock_zscore_long.csv",
        "GenScore":            f"{PK}/Gen2/redock_zscore_long.csv",
    },
}


def exp_kinome() -> pd.DataFrame:
    # 논문 본문과 동일한 leakage-free 코호트를 쓴다(Table S8과 숫자를 맞추기 위함).
    e = nc.drop_leaked(km.experiment(), "KIR")
    return e[["DRUG_NAME", "UNIPROT_ID", "label"]]


def exp_pkis2() -> pd.DataFrame:
    e = ap.load_experiment(ap.CONFIGS["PKIS2"])
    e["label"] = (e["exp_inh"] >= 50).astype(int)
    e = nc.drop_leaked(e, "PKIS2")
    return e[["DRUG_NAME", "UNIPROT_ID", "label"]]


EXP = {"KIR": exp_kinome(), "PKIS2": exp_pkis2()}


def load(path: str) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path, usecols=["DRUG_NAME", "UNIPROT_ID", "GENE_NAME",
                                    "zscore_mean", "rank_mean"])
    df["DRUG_NAME"] = df["DRUG_NAME"].astype(str).str.strip().replace(ap.DRUG_ALIAS)
    gmap = df.drop_duplicates("UNIPROT_ID").set_index("UNIPROT_ID").GENE_NAME.to_dict()
    df["UNIPROT_ID"] = km.norm_uni(df["UNIPROT_ID"])
    agg = df.groupby(["DRUG_NAME", "UNIPROT_ID"], as_index=False).agg(
        z=("zscore_mean", "min"), r=("rank_mean", "min"))
    sign = 1.0 if np.corrcoef(agg.z, agg.r)[0, 1] > 0 else -1.0
    agg["pred"] = -sign * agg.z
    return agg, gmap


rows = []
for panel, meths in PANELS.items():
    E = EXP[panel]
    base_rate = E.label.mean()
    for meth, path in meths.items():
        if not os.path.exists(path):
            continue
        agg, gmap = load(path)
        agg = agg[agg.DRUG_NAME.isin(set(E.DRUG_NAME))]           # 논문 코호트로 제한
        M = E.merge(agg, on=["DRUG_NAME", "UNIPROT_ID"])          # 실험과 병합된 pool

        # 최빈 top-1은 예측 pool 전체(618/387 타겟)에서 정한다. 실험 병합 pool에서 다시 뽑으면
        # 동률일 때 다른 유전자가 뽑혀 Table S8 앞 열과 어긋난다(PKIS2 AK: KDR vs LCK).
        t1_full = agg.sort_values("pred", ascending=False).groupby("DRUG_NAME").head(1)
        vc = t1_full.UNIPROT_ID.value_counts()
        tgt = FIXED.get((panel, meth), vc.index[0])
        share = float(vc.get(tgt, 0)) / t1_full.DRUG_NAME.nunique()

        # 그 타겟을 1위로 받은 화합물들에서, 그 타겟이 실제 active인 비율
        sel_drugs = set(t1_full[t1_full.UNIPROT_ID == tgt].DRUG_NAME)
        sel = M[(M.UNIPROT_ID == tgt) & (M.DRUG_NAME.isin(sel_drugs))]
        prec_collapsed = float(sel.label.mean()) if len(sel) else float("nan")
        tgt_rate = float(M[M.UNIPROT_ID == tgt].label.mean())
        # Hit@1은 실측 pair 기준(라벨이 있어야 정의된다)
        top1 = M.sort_values("pred", ascending=False).groupby("DRUG_NAME").head(1)
        hit1 = float(top1.label.mean())

        rows.append(dict(panel=panel, method=meth, n_drug=len(top1),
                         top_gene=gmap.get(tgt, tgt), top_share=share,
                         collapsed_precision=prec_collapsed,
                         collapsed_target_active_rate=tgt_rate,
                         hit1=hit1, panel_active_rate=base_rate,
                         lift=prec_collapsed / base_rate if base_rate else np.nan))
        print(f"  {panel:11s} {meth:22s} top1={gmap.get(tgt,tgt):8s} share={share:5.1%}  "
              f"그 타겟 active율={prec_collapsed:5.1%}  Hit@1={hit1:.2f}  패널 active율={base_rate:.1%}")

V = pd.DataFrame(rows)
V.to_csv(f"{OUT}/top1_bias_validate.csv", index=False)

# ── 그림 ──
S = pd.read_csv(f"{OUT}/top1_bias_all.csv")
COL = {"AK-Score2": "#2166AC", "AD4 energy": "#4393C3", "RTMScore": "#5AAE61",
       "GenScore": "#B2182B"}

fig = plt.figure(figsize=(19, 11))
gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.22)

def panel_bars(ax, col, title, ylab, fmt="{:.0%}", hline=None, hlab=None, annot_gene=False, ylim=None):
    panels = list(S.panel.unique())
    x0 = 0
    ticks, tlabs = [], []
    for pn in panels:
        sub = S[S.panel == pn]
        xs = np.arange(len(sub)) + x0
        b = ax.bar(xs, sub[col], 0.78, color=[COL[m] for m in sub.method], edgecolor="white")
        for bi, (_, r) in zip(b, sub.iterrows()):
            lab = fmt.format(r[col])
            if annot_gene:
                lab = f"{r.top_gene}\n{lab}"
            ax.text(bi.get_x() + bi.get_width() / 2, r[col], lab, ha="center", va="bottom", fontsize=7.5)
        ticks += list(xs)
        tlabs += [m.replace(" (", "\n(") for m in sub.method]
        x0 += len(sub) + 1.2
        ax.axvline(x0 - 1.1, color="#CCCCCC", lw=1)
    if hline is not None:
        ax.axhline(hline, color="red", ls="--", lw=1, label=hlab)
        ax.legend(fontsize=8)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tlabs, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=12, fontweight="bold")
    if ylim:
        ax.set_ylim(*ylim)
    # 패널명 — 축 좌표계 상단에 고정해 막대와 겹치지 않게
    from matplotlib.transforms import blended_transform_factory
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    x0 = 0
    for pn in panels:
        sub = S[S.panel == pn]
        ax.text(x0 + (len(sub) - 1) / 2, 0.95, pn, transform=tr,
                ha="center", va="top", fontsize=11, fontweight="bold", color="#333333")
        x0 += len(sub) + 1.2

ax = fig.add_subplot(gs[0, 0])
panel_bars(ax, "top_share", "(a) Top-1 target collapse — share of compounds with the SAME top-1",
           "share of compounds", annot_gene=True, ylim=(0, 0.42))

ax = fig.add_subplot(gs[0, 1])
panel_bars(ax, "norm_entropy", "(b) Diversity of top-1 predictions (normalized entropy)",
           "0 = total collapse, 1 = fully diverse", fmt="{:.2f}", ylim=(0, 1.08))

ax = fig.add_subplot(gs[1, 0])
panel_bars(ax, "top10_share", "(c) Ligand-independent preference at top-10\n(most frequent target's appearance rate)",
           "share of compounds", ylim=(0, 1.12))

# (d) 쏠림이 정당한가: collapsed precision vs 패널 active율
ax = fig.add_subplot(gs[1, 1])
panels = list(V.panel.unique())
x0 = 0
ticks, tlabs = [], []
for pn in panels:
    sub = V[V.panel == pn]
    xs = np.arange(len(sub)) + x0
    ax.bar(xs, sub.collapsed_precision, 0.78, color=[COL[m] for m in sub.method], edgecolor="white")
    for xi, (_, r) in zip(xs, sub.iterrows()):
        ax.text(xi, r.collapsed_precision, f"{r.top_gene}\n{r.collapsed_precision:.0%}",
                ha="center", va="bottom", fontsize=7.5)
    base = sub.panel_active_rate.iloc[0]
    ax.plot([xs[0] - 0.5, xs[-1] + 0.5], [base, base], color="red", ls="--", lw=1.4)
    ticks += list(xs)
    tlabs += [m.replace(" (", "\n(") for m in sub.method]
    x0 += len(sub) + 1.2
ax.set_xticks(ticks)
ax.set_xticklabels(tlabs, fontsize=7, rotation=30, ha="right")
ax.set_ylabel("active rate of the collapsed target")
ax.set_title("(d) Is the collapse justified?\nred dashed = panel-wide active rate (chance)",
             fontsize=12, fontweight="bold")
ax.set_ylim(0, max(0.55, V.collapsed_precision.max() * 1.35))
from matplotlib.transforms import blended_transform_factory as _btf
_tr = _btf(ax.transData, ax.transAxes)
_x0 = 0
for pn in panels:
    sub = V[V.panel == pn]
    ax.text(_x0 + (len(sub) - 1) / 2, 0.95, pn, transform=_tr, ha="center", va="top",
            fontsize=11, fontweight="bold", color="#333333")
    _x0 += len(sub) + 1.2

fig.suptitle("Ligand-independent top-1 target collapse across scoring functions and panels",
             fontsize=16, fontweight="bold")
fig.savefig(f"{OUT}/top1_bias_all.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"\n✅ 저장 → {OUT}/")
