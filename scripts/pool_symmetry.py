"""
Primary-target recovery with both panels' ranking pools brought to the same definition.

Background. Recovery on KIR was originally computed over the 618-structure prediction pool
while PKIS2 was computed over the 387 kinases that panel measures, because PKIS2 predictions were
only ever generated for those 387. The two fold-enrichments were therefore not comparable.
Restricting KIR to the 369 kinases it measures puts both panels on the same footing.

Note. The paper's headline metrics (Tables 3, 4, 6 and the p-values) are computed after an inner
merge with the experiment, so they never depended on the pool size. The pool definition only moves
the pool-wide recovery metrics this script computes.

Definitions follow kinome_recovery.py: primary = experimentally strongest (min %remaining), strong
active = %rem <= 10 (the counterpart of %inh >= 90 on PKIS2), leakage-free cohort, and compounds
without any measured active excluded.

Output: Kinase_paper/analysis_output/ (POOL_SYMMETRY.md, pool_symmetry.csv)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/d/Deepmolscan/Kinase_paper")
import kinome_merge as km  # noqa: E402
import novel_cohort as nc  # noqa: E402

KP = "/mnt/d/Deepmolscan/Kinase_paper"
OUT = f"{KP}/analysis_output"
EB = f"{KP}/energy_benchmark"
os.makedirs(OUT, exist_ok=True)

ACT = 10.0                       # %remaining <= 10 = 강한 active
DRUG_ALIAS = {"Barcitinib": "Baricitinib", "Vandetanib-Final": "Vandetanib"}
METH = {"AK": f"{KP}/result_87/redock_zscore_long.csv",
        "RTM": f"{KP}/RTM/redock_zscore_long.csv"}

# ── 실험: %remaining ──
exp = pd.read_csv(f"{KP}/kinase_exp_uniprot_mapped_matrix.csv", index_col=0)
col2uni = {c: c.split("|")[2] for c in exp.columns}
el = exp.stack().reset_index()
el.columns = ["DRUG_NAME", "col", "rem"]
el["UNIPROT_ID"] = el["col"].map(col2uni)
el["rem"] = pd.to_numeric(el["rem"], errors="coerce")
el = el.dropna(subset=["rem"])
exp_agg = el.groupby(["DRUG_NAME", "UNIPROT_ID"], as_index=False)["rem"].min()
MEASURED = set(exp_agg.UNIPROT_ID)
print(f"[실험] KIR 측정 kinase {len(MEASURED)}개")


def load_raw(f: str) -> tuple[pd.DataFrame, float]:
    df = pd.read_csv(f)
    df["DRUG_NAME"] = df["DRUG_NAME"].replace(DRUG_ALIAS)
    df["UNIPROT_ID"] = km.norm_uni(df["UNIPROT_ID"])
    df = df.groupby(["DRUG_NAME", "UNIPROT_ID"], as_index=False).agg(
        z=("zscore_mean", "min"), r=("rank_mean", "min"))
    sign = 1.0 if np.corrcoef(df.z, df.r)[0, 1] > 0 else -1.0
    return df[["DRUG_NAME", "UNIPROT_ID", "z"]], sign


AKp, aks = load_raw(METH["AK"])
RTp, rts = load_raw(METH["RTM"])
M = AKp.rename(columns={"z": "zAK"}).merge(
    RTp.rename(columns={"z": "zRTM"}), on=["DRUG_NAME", "UNIPROT_ID"])
M = nc.drop_leaked(M, "KIR")
drugs = sorted(set(M.DRUG_NAME) & set(exp_agg.DRUG_NAME))
print(f"[코호트] 학습 미사용 공통 화합물 {len(drugs)}개")


def evaluate(pool_mode: str) -> pd.DataFrame:
    """pool_mode: 'full'(예측 618) 또는 'measured'(그 패널이 측정한 369)."""
    rows = []
    for dn in drugs:
        g = M[M.DRUG_NAME == dn].copy()
        if pool_mode == "measured":
            g = g[g.UNIPROT_ID.isin(MEASURED)]
        if len(g) < 30:
            continue
        rk = {}
        for m, z, sgn in (("AK", g.zAK, aks), ("RTM", g.zRTM, rts)):
            # 방향교정 후 높은 점수 = 강한 결합
            p = z * (-1 if sgn > 0 else 1)
            rk[m] = dict(zip(g.UNIPROT_ID, p.rank(method="min", ascending=False)))
        N = len(g)
        e = exp_agg[(exp_agg.DRUG_NAME == dn) & (exp_agg.UNIPROT_ID.isin(set(g.UNIPROT_ID)))]
        if e.empty:
            continue
        # 활성이 하나도 없는 화합물은 랭킹 과제가 성립하지 않아 제외한다(§2.2·Methods S3).
        if not (e.rem < 50.0).any():
            continue
        prim = e.sort_values("rem").iloc[0]
        acts = e[e.rem <= ACT]
        rr = dict(cpd=dn, pool=pool_mode, N=N, n_meas=len(e), n_act=len(acts))
        for m in ("AK", "RTM"):
            pr = rk[m][prim.UNIPROT_ID]
            rr[f"{m}_prank"] = int(pr)
            rr[f"{m}_pctl"] = pr / N
            for k in (1, 5, 10):
                rr[f"{m}_hit{k}"] = (int(any(rk[m][u] <= k for u in acts.UNIPROT_ID))
                                     if len(acts) else 0)
        rows.append(rr)
    return pd.DataFrame(rows)


R = pd.concat([evaluate("full"), evaluate("measured")], ignore_index=True)
R.to_csv(f"{OUT}/pool_symmetry_per_compound.csv", index=False)


def summarize(sub: pd.DataFrame, label: str) -> list[dict]:
    out = []
    for m in ("AK", "RTM"):
        p = sub[f"{m}_prank"]
        out.append(dict(panel="KIR", pool=label, n_pool=int(sub.N.median()),
                        method=m, n=len(sub),
                        top1=100 * (p <= 1).mean(), top5=100 * (p <= 5).mean(),
                        top10=100 * (p <= 10).mean(), top30=100 * (p <= 30).mean(),
                        med_pctl=100 * sub[f"{m}_pctl"].median(),
                        hit10=100 * sub[f"{m}_hit10"].mean()))
    return out


tab = summarize(R[R.pool == "full"], "prediction pool (618)")
tab += summarize(R[R.pool == "measured"], "measured kinases (369)")

# ── PKIS2 는 예측 자체가 387(=측정 kinase)이라 이미 measured 정의와 동일 ──
pk = pd.read_csv(f"{EB}/pkis2_recovery_summary.csv")
pk_ak = pk[pk.method.astype(str).str.upper().str.startswith("AK")].iloc[0]
pk_n = len(pd.read_csv(f"{EB}/pkis2_recovery_per_compound.csv"))
tab.append(dict(panel="PKIS2", pool="measured kinases (387)", n_pool=387, method="AK", n=pk_n,
                top1=float(pk_ak.top1), top5=float(pk_ak.top5), top10=float(pk_ak.top10),
                top30=float(pk_ak.top30), med_pctl=float(pk_ak.medpctl),
                hit10=float(pk_ak.hit10)))

T = pd.DataFrame(tab)
T.to_csv(f"{OUT}/pool_symmetry.csv", index=False)
print("\n[A1] pool 정의별 primary-target 회수")
print(T.round(1).to_string(index=False))

ak_full = T[(T.panel == "KIR") & (T.pool.str.contains("618")) & (T.method == "AK")].iloc[0]
ak_meas = T[(T.panel == "KIR") & (T.pool.str.contains("369")) & (T.method == "AK")].iloc[0]
pk_row = T[T.panel == "PKIS2"].iloc[0]

md = [
    "# A1 — 랭킹 pool 정의를 맞춘 primary-target 회수\n",
    "본문 2.6절은 KIR 을 618 pool, PKIS2 를 387 pool 에서 랭킹해 두 값을 가로로 비교할 수 "
    "없었다. KIR 을 자신이 측정한 369 kinase 로 제한하면 PKIS2 와 같은 정의가 된다.\n",
    "> 주의: 본문의 주요 지표(Table 3·4·6)는 실험과 inner merge 후 계산되므로 애초에 pool 크기의 "
    "영향을 받지 않는다. pool 정의가 값을 바꾸는 곳은 이 표의 pool-wide 회수 지표뿐이다.\n",
    "정의: primary = 실험 최강(min %remaining), 강한 active = %rem ≤ 10 "
    "(PKIS2 %inh ≥ 90 대응), 학습 미사용 코호트.\n",
    "| 패널 | pool 정의 | pool 크기 | 방법 | n | Top-1 | Top-5 | Top-10 | Top-30 | pctl중앙 | Hit@10 |",
    "|---|---|--:|---|--:|--:|--:|--:|--:|--:|--:|"]
for _, r in T.iterrows():
    md.append(f"| {r.panel} | {r['pool']} | {r.n_pool} | {r.method} | {r.n} | {r.top1:.1f}% | "
              f"{r.top5:.1f}% | {r.top10:.1f}% | {r.top30:.1f}% | {r.med_pctl:.1f}% | {r.hit10:.1f}% |")
md += ["", "## 본문에 쓸 문장",
       f"- KIR 을 자신이 측정한 369 kinase 로 제한하면 AK-Score2 의 primary Top-10 은 "
       f"{ak_full.top10:.1f}% → {ak_meas.top10:.1f}%, Hit@10 은 {ak_full.hit10:.1f}% → "
       f"{ak_meas.hit10:.1f}% 가 된다.",
       f"- 같은 정의(측정 kinase 만)에서 PKIS2 는 primary Top-10 {pk_row.top10:.1f}%, "
       f"Hit@10 {pk_row.hit10:.1f}% 이므로 두 패널을 직접 비교할 수 있다.",
       f"- 무작위 기대(Top-10): KIR 369 pool 에서 {1000/369:.1f}%, "
       f"PKIS2 387 pool 에서 {1000/387:.1f}%."]
open(f"{OUT}/POOL_SYMMETRY.md", "w").write("\n".join(md))
print(f"\n✅ 저장 → {OUT}/pool_symmetry.csv, POOL_SYMMETRY.md")
