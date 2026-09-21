"""
Kinase_ref: (A) build the per-compound experimental panel listing, (B) evaluate against that
panel as the ranking pool.

Background. Kinase_ref carries the values each source paper reported for a broad kinase panel of
300-470 entries. Ranking a compound against all 618 predicted structures would admit kinases the
experiment never measured as decoys. Restricting the pool to the panel the source paper actually
profiled asks the sharper question: within the candidates the experiment confirmed, did the method
put the strongest target on top? Both pools are computed; the paper reports the measured panel and
keeps the 618-structure pool as a sensitivity analysis.

Units. The first column header of result.tsv carries the measurement unit.
  1uM / 10uM / 100nM  -> % remaining (% control); lower means stronger binding.
  nM                  -> IC50 in nM of the hit kinases; lower is still stronger, so the rank
                         direction is unchanged, but the scale differs from a percentage and these
                         compounds are excluded from label-based metrics (active = %rem < 50).

Hits-only panels. Nine compounds come from papers that reported only kinases at or below a 35%
remaining cut-off. Their panels are reconstructed from other papers on the same assay platform,
and the paper's own reporting cut-off is adopted as the activity threshold for that series, which
makes "unreported implies inactive" exact by construction rather than an assumption.

Output: Kinase_paper/kinase_ref_check/
  KINASE_REF_PANELS.tsv         (compound x kinase, long form - the full panel listing)
  KINASE_REF_PANELS_summary.csv (per-compound panel summary)
  REF_PANEL_EVAL.md, ref_panel_eval.png, ref_panel_eval_summary.csv
"""
from __future__ import annotations
import sys as _s; _s.path.insert(0,"/mnt/d/Deepmolscan/Kinase_paper"); import figlabels as _fl

import glob
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, "/mnt/d/Deepmolscan/Kinase_paper")
import novel_cohort as nc  # noqa: E402
import ref_families as rf  # noqa: E402

KR = "/mnt/d/Deepmolscan/Kinase_ref"
OUT = "/mnt/d/Deepmolscan/Kinase_paper/kinase_ref_check"
os.makedirs(OUT, exist_ok=True)
RRF_K = 60
ACT_REM = 50.0        # % remaining < 50 = active
PCT_UNITS = {"1uM", "10uM", "100nM", "1um", "10um"}


def parse_val(v) -> float:
    try:
        return float(str(v).replace(",", ".").replace("%", "").strip())
    except (TypeError, ValueError):
        return np.nan


def zc(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 0 else s * 0.0


def load_long(path: str) -> dict[str, dict[str, float]]:
    d = (pd.read_csv(path).groupby(["DRUG_NAME", "UNIPROT_ID"], as_index=False)
                          .agg(z=("zscore_mean", "min")))
    return {c: dict(zip(g.UNIPROT_ID.astype(str), g.z)) for c, g in d.groupby("DRUG_NAME")}


AK = load_long(f"{KR}/AK/redock_zscore_long.csv")
RTM = load_long(f"{KR}/RTM/redock_zscore_long.csv")
DOCK = load_long(f"{KR}/DOCK/redock_zscore_long.csv")
# GenScore(Kinase_ref 패널)는 Kinase_paper/Gen2 에 산출돼 있다.
GEN = load_long("/mnt/d/Deepmolscan/Kinase_paper/Gen2/redock_zscore_long.csv")

# ══════════════ 코호트 확정 ══════════════
# Kinase_ref는 아래 세 조건을 모두 만족하는 화합물만으로 정의한다(= 44개).
#   ① 원 논문의 실험 활성값이 수집돼 있음(result.tsv 존재)
#   ② AK/RTM/DOCK 세 방법의 예측이 모두 있음
#   ③ PDBbind 학습 리간드와 InChIKey14 완전일치가 아님(학습 미사용)
# 이 조건에서 탈락한 화합물은 이후 어떤 산출물에도 등장하지 않는다.
_all = sorted(os.path.basename(d.rstrip("/")) for d in glob.glob(f"{KR}/Pred/*/"))
_res = {c for c in _all if os.path.isfile(f"{KR}/Pred/{c}/{c}_result.tsv")}
_pred = set(AK) & set(RTM) & set(DOCK) & set(GEN)
_leak = nc.LEAKED["Kinase_ref"]
COHORT = sorted((_res & _pred) - _leak)
print(f"[코호트] 라이브러리 {len(_all)} → 실험값 {len(_res)} → 3방법 예측 {len(_res & _pred)} "
      f"→ 학습 미사용 {len(COHORT)}")
print(f"   실험값 없음 {sorted(_res ^ set(_all))} / 예측 없음 {sorted(_res - _pred)} / "
      f"누출 제거 {sorted((_res & _pred) & _leak)}")

# ══════════════ hits-only 패널 복원 ══════════════
# CK2 시리즈는 403개짜리 DiscoverX 패널을 스크리닝하고 보고 컷오프(%rem ≲ 35) 이하만 표기했다.
# 미보고 키나아제는 '측정되지 않음'이 아니라 '컷오프 위'이므로, 같은 플랫폼의 full-panel
# 화합물에서 패널 명단을 복원하고 미보고분을 inactive 로 채운다. 원 논문의 보고 컷오프를
# 그대로 active 정의로 삼으므로(= S(35)) '미보고 = inactive' 는 가정이 아니라 구성상 정확하다.
# 컷오프(35%)와 다른 시리즈의 임계값(50%) 사이 구간을 미보고분에 적용하면 실제 active 의
# 약 1/3 을 inactive 로 오분류하게 되므로 이 방식을 쓰지 않는다.
RESTORE_PANEL = 403      # 복원 대상 시리즈의 스크리닝 패널 크기
REF_PANEL_SIZE = 468     # 명단을 빌려오는 같은 플랫폼 full-panel 크기


def _reference_panel() -> pd.DataFrame:
    """같은 DiscoverX 플랫폼 full-panel 화합물들의 합집합으로 패널 명단을 만든다."""
    rows = []
    for cpd in COHORT:
        fam = rf.family_of(cpd)
        if "DiscoverX" not in str(fam["platform"]) or int(fam["panel"]) != REF_PANEL_SIZE:
            continue
        t = pd.read_csv(f"{KR}/Pred/{cpd}/{cpd}_result.tsv", sep="\t")
        rows.append(pd.DataFrame({
            "assay_name": t[t.columns[0]].astype(str).str.strip(),
            "UNIPROT_ID": t["UNIPROT_ID"].astype(str).str.strip()}))
    R0 = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["assay_name"])
    return R0[R0.UNIPROT_ID.str.lower().ne("nan")].reset_index(drop=True)


REF_PANEL = _reference_panel()
RESTORE = {c for c in COHORT
           if "DiscoverX" in str(rf.family_of(c)["platform"])
           and int(rf.family_of(c)["panel"]) == RESTORE_PANEL}
print(f"[복원] 참조 패널 {len(REF_PANEL)} entry / UniProt {REF_PANEL.UNIPROT_ID.nunique()}개 "
      f"· 복원 대상 {len(RESTORE)}개 {sorted(RESTORE)}")

# ══════════════ (A) 패널 목록 파일 ══════════════
panel_rows, summ_rows = [], []
for cpd in COHORT:
    res = f"{KR}/Pred/{cpd}/{cpd}_result.tsv"
    fam = rf.family_of(cpd)
    if False:
        continue
    t = pd.read_csv(res, sep="\t")
    gene_col, val_col = t.columns[0], t.columns[1]
    unit = str(gene_col).strip()
    is_pct = unit in PCT_UNITS
    t["value"] = t[val_col].map(parse_val)
    t = t.dropna(subset=["value"])
    t["assay_name"] = t[gene_col].astype(str).str.strip()
    t["UNIPROT_ID"] = t["UNIPROT_ID"].astype(str).str.strip()
    t["reported"] = 1
    if cpd in RESTORE:
        add = REF_PANEL[~REF_PANEL.UNIPROT_ID.isin(set(t["UNIPROT_ID"]))].copy()
        add["value"] = np.nan          # 좌측 절단: 보고 컷오프 위, 정확값 미상
        add["reported"] = 0
        t = pd.concat([t, add[["assay_name", "UNIPROT_ID", "value", "reported"]]],
                      ignore_index=True)
    in_ak = set(AK.get(cpd, {})); in_rtm = set(RTM.get(cpd, {})); in_dk = set(DOCK.get(cpd, {}))
    pred_pool = in_ak & in_rtm & in_dk
    for _, r in t.iterrows():
        raw = r["UNIPROT_ID"]
        # 예측 pool의 키는 단일 accession일 수도, ';'로 이어진 복합 문자열일 수도 있다.
        # 원 result.tsv에 UNIPROT_ID가 없는 행(문자열 'nan')은 매칭 불가로 두어 집계에서 빠진다.
        cand = ([] if raw.lower() == "nan" else [raw] + [u for u in raw.split(";") if u])
        matched = next((u for u in cand if u in pred_pool), cand[-1] if cand else "")
        panel_rows.append(dict(
            compound=cpd, family=fam["key"], intended_target=fam["target"],
            source_paper=fam["series"], platform=fam["platform"],
            panel_size=fam["panel"], reference=fam["reference"], pdf_refs=fam["pdf_refs"],
            assay_name=r["assay_name"], uniprot_raw=r["UNIPROT_ID"], uniprot=matched,
            unit=unit, value=r["value"], reported=int(r["reported"]),
            label_active=int(r["value"] < ACT_REM) if is_pct else pd.NA,
            in_prediction_pool=int(matched in pred_pool)))
    sub = pd.DataFrame([x for x in panel_rows if x["compound"] == cpd])
    n_act = int(sub.label_active.sum()) if is_pct else -1
    n_rep = int(sub.reported.sum())
    summ_rows.append(dict(
        compound=cpd, family=fam["key"], intended_target=fam["target"], source_paper=fam["series"],
        platform=fam["platform"], panel_size=fam["panel"], reference=fam["reference"],
        pdf_refs=fam["pdf_refs"], unit=unit,
        value_type="% remaining" if is_pct else "IC50 (nM)",
        n_panel=len(sub), n_released=n_rep, n_restored=len(sub) - n_rep,
        n_in_pool=int(sub.in_prediction_pool.sum()),
        n_active=n_act if is_pct else None,
        n_inactive=(len(sub) - n_act) if is_pct else None,
        min_value=round(sub.value.min(), 2), median_value=round(sub.value.median(), 2),
        strongest_assay=sub.loc[sub.value.idxmin(), "assay_name"],
        report=("hits only (panel restored)" if cpd in RESTORE
                else "full panel" if n_rep >= 200 else "hits only"),
        label_usable=int(is_pct and n_act >= 1 and (len(sub) - n_act) >= 1)))
P = pd.DataFrame(panel_rows)
S = pd.DataFrame(summ_rows)
COLS = ["compound", "family", "intended_target", "source_paper", "panel_size", "platform",
        "reference", "pdf_refs", "assay_name", "uniprot_raw", "uniprot", "unit", "value",
        "reported", "label_active", "in_prediction_pool"]
P = P[COLS]
P.to_csv(f"{OUT}/KINASE_REF_PANELS.tsv", sep="\t", index=False)
S.to_csv(f"{OUT}/KINASE_REF_PANELS_summary.csv", index=False)
n_uni = P.loc[P.uniprot.astype(str).str.strip().ne("") & P.uniprot.notna(), "uniprot"].nunique()
print(f"[A] 패널 목록: 화합물 {S.compound.nunique()}개 · (화합물,키나아제) {len(P):,}행 · "
      f"고유 키나아제 {n_uni}개 (예측 pool과 매칭된 accession 기준)")
print(f"    단위: {S.unit.value_counts().to_dict()}   라벨 사용가능 {int(S.label_usable.sum())}개")

# ══════════════ (B) 패널 pool vs 618 pool 성능 ══════════════
METHODS = ["AK", "RTM", "DOCK", "GEN", "RRF(AK+RTM)", "Z-avg(AK+RTM)"]
# 중간부 그림(Figure 10)은 단일 스코어링 함수만. consensus 는 뒤쪽 전용 절에서 다룬다.
FIGMETHODS = ["AK", "RTM", "DOCK", "GEN"]
# 그림 표시용 라벨
import re as _re
DISP = {"AK": "AK-Score2", "RTM": "RTMScore", "DOCK": "AD4 energy", "GEN": "GenScore"}
def dsp(m):
    s = str(m)
    for k, v in DISP.items():
        s = _re.sub(rf"\b{k}\b", v, s)
    return s



def evaluate(pool_mode: str) -> pd.DataFrame:
    """pool_mode: 'panel'(원 논문 측정 패널) 또는 'full'(예측 pool 618)."""
    rows = []
    for cpd, g in P.groupby("compound"):
        if cpd not in AK or cpd not in RTM or cpd not in DOCK or cpd not in GEN:
            continue
        ak, rtm, dk, gn = AK[cpd], RTM[cpd], DOCK[cpd], GEN[cpd]
        full_pool = set(ak) & set(rtm) & set(dk) & set(gn)
        meas = g[g.in_prediction_pool == 1]
        if meas.empty:
            continue
        pool = sorted(set(meas.uniprot)) if pool_mode == "panel" else sorted(full_pool)
        if len(pool) < 5:
            continue
        Q = pd.DataFrame({"uniprot": pool})
        Q["zAK"] = Q.uniprot.map(ak); Q["zRTM"] = Q.uniprot.map(rtm); Q["zDOCK"] = Q.uniprot.map(dk); Q["zGEN"] = Q.uniprot.map(gn)
        Q["pAK"] = zc(-Q.zAK); Q["pRTM"] = zc(Q.zRTM); Q["pDOCK"] = zc(-Q.zDOCK); Q["pGEN"] = zc(Q.zGEN)
        for m in ("AK", "RTM", "DOCK", "GEN"):
            Q[f"r{m}"] = (-Q[f"p{m}"]).rank(method="min")
        sc = {"AK": Q.pAK, "RTM": Q.pRTM, "DOCK": Q.pDOCK, "GEN": Q.pGEN,
              "RRF(AK+RTM)": 1.0 / (RRF_K + Q.rAK) + 1.0 / (RRF_K + Q.rRTM),
              "Z-avg(AK+RTM)": (Q.pAK + Q.pRTM) / 2}
        # primary = 측정값 최소(% remaining 또는 IC50). 최소값 동률이 흔하므로(전체의 27%)
        # 임의로 하나를 고르지 않고 동률 집합 전체를 primary로 보고, 그 중 최선 순위를 취한다.
        vmin = meas.value.min()
        prim_set = set(meas.loc[meas.value == vmin, "uniprot"])
        prim_assays = "|".join(sorted(set(meas.loc[meas.value == vmin, "assay_name"])))
        strong = (set(meas.loc[meas.value <= 10.0, "uniprot"])
                  if meas.unit.iloc[0] in PCT_UNITS else set(prim_set))
        rec = dict(compound=cpd, pool=pool_mode, n_pool=len(pool),
                   primary_assay=prim_assays, n_primary_tied=len(prim_set),
                   primary_value=vmin, n_strong=len(strong))
        for m in METHODS:
            Q["_rk"] = sc[m].rank(method="min", ascending=False).values
            pr = int(Q.loc[Q.uniprot.isin(prim_set), "_rk"].min())
            rec[f"{m}_prank"] = pr
            rec[f"{m}_pctl"] = pr / len(pool)
            rec[f"{m}_hit10"] = int((Q.loc[Q.uniprot.isin(strong), "_rk"] <= 10).any())
        rows.append(rec)
    return pd.DataFrame(rows)


R = pd.concat([evaluate("panel"), evaluate("full")], ignore_index=True)
# P가 이미 학습 미사용 코호트라 추가 필터는 불필요(방어적으로 유지)
R = nc.drop_leaked(R, "Kinase_ref", col="compound")
R.to_csv(f"{OUT}/ref_panel_eval_per_compound.csv", index=False)


def summarize(sub: pd.DataFrame) -> list[dict]:
    out = []
    for m in METHODS:
        pr = sub[f"{m}_prank"]
        out.append(dict(method=m, n=len(sub), n_pool_med=int(sub.n_pool.median()),
                        top1=100 * (pr <= 1).mean(), top5=100 * (pr <= 5).mean(),
                        top10=100 * (pr <= 10).mean(), top30=100 * (pr <= 30).mean(),
                        med_pctl=100 * sub[f"{m}_pctl"].median(),
                        hit10=100 * sub[f"{m}_hit10"].mean()))
    return out


tab = []
for mode, lab in (("panel", "원 논문 측정 패널"), ("full", "예측 pool 618")):
    for r in summarize(R[R.pool == mode]):
        r["pool"] = lab
        tab.append(r)
T = pd.DataFrame(tab)[["pool", "method", "n", "n_pool_med", "top1", "top5", "top10", "top30",
                       "med_pctl", "hit10"]]
T.to_csv(f"{OUT}/ref_panel_eval_summary.csv", index=False)
print("\n[B] 표적 회수 (학습 미사용 코호트)")
print(T.round(1).to_string(index=False))

# ── 그림 ──
fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
x = np.arange(len(FIGMETHODS)); w = 0.38
COL = {"원 논문 측정 패널": "#2166AC", "예측 pool 618": "#E08214"}
for k, (lab, key, ttl) in enumerate([("primary Top-10 (%)", "top10", "Primary-target recovery in Top-10"),
                                     ("median rank percentile (%)", "med_pctl", "Median rank percentile (lower = better)")]):
    a = ax[k]
    for i, (pool, col) in enumerate(COL.items()):
        v = [T[(T.pool == pool) & (T.method == m)][key].iloc[0] for m in FIGMETHODS]
        n = int(T[T.pool == pool].n_pool_med.iloc[0])
        b = a.bar(x + (i - 0.5) * w, v, w, color=col, edgecolor="white",
                  label=f"{'measured panel' if i == 0 else 'prediction pool'} (median {n} kinases)")
        for bi, vv in zip(b, v):
            a.text(bi.get_x() + bi.get_width() / 2, vv + 0.6, f"{vv:.0f}", ha="center", fontsize=8)
    a.set_xticks(x); a.set_xticklabels([dsp(m) for m in FIGMETHODS], fontsize=8.5, rotation=15)
    a.set_ylim(0, max(a.get_ylim()[1] * 1.22, 1))          # 값 라벨·범례 여유
    a.set_ylabel(lab); a.set_title(ttl)
    a.legend(fontsize=8.5, loc="upper center", ncol=2, framealpha=0.95)
    a.grid(alpha=.25, axis="y")
n_cpd = R[R.pool == "panel"].compound.nunique()
n_full = R[R.pool == "full"].compound.nunique()
fig.suptitle(f"Kinase_ref: ranking within the panel actually profiled by the source paper "
             f"(n={n_cpd}) vs the full prediction pool (n={n_full}), leakage-free cohort",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])
_fl.stamp(ax)   # 패널 문자 (a)(b)(c)
fig.savefig(f"{OUT}/ref_panel_eval.png", dpi=300, bbox_inches="tight")
plt.close()

# ── 보고서 ──
md = ["# Kinase_ref — 실험 패널 목록과 패널 기준 성능\n",
      f"화합물 {S.compound.nunique()}개, (화합물,키나아제) 측정 {len(P):,}행, 고유 키나아제 {P.uniprot.nunique()}개.\n",
      "패널 목록 상세: `KINASE_REF_PANELS.tsv` (화합물별 long), `KINASE_REF_PANELS_summary.csv`.\n",
      "## 화합물별 패널 요약\n",
      "| 화합물 | 의도 표적 | 출처 계열 | 단위 | 패널 | pool 내 | active | inactive | 최강 표적 | 라벨사용 |",
      "|---|---|---|---|--:|--:|--:|--:|---|:--:|"]
for _, r in S.sort_values(["family", "compound"]).iterrows():
    md.append(f"| {r.compound} | {r.intended_target} | {r.family} | {r.unit} | {r.n_panel} | "
              f"{r.get('n_in_pool', '—')} | {r.get('n_active', '—')} | {r.get('n_inactive', '—')} | "
              f"{r.get('strongest_assay', '—')} | {'○' if r.get('label_usable') else '×'} |")
md += ["\n## 표적 회수 — pool 정의에 따른 비교 (학습 미사용 코호트)\n",
       "| pool | 방법 | n | pool 크기(중앙) | Top-1 | Top-5 | Top-10 | Top-30 | 중앙 percentile | Hit@10 |",
       "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
for _, r in T.iterrows():
    md.append(f"| {r.pool} | {r.method} | {r.n} | {r.n_pool_med} | {r.top1:.1f}% | {r.top5:.1f}% | "
              f"{r.top10:.1f}% | {r.top30:.1f}% | {r.med_pctl:.1f}% | {r.hit10:.1f}% |")
md += ["\n![eval](ref_panel_eval.png)\n", "## 주의\n",
       "- 단위가 `nM`인 화합물(ETC-206, BAY-1143269)은 hit kinase의 IC50만 보고된 경우로, "
       "순위 방향은 % remaining과 같지만 라벨(active = %rem<50) 척도가 달라 라벨 기반 지표에서 제외한다.",
       "- Comp8(PERK 후보)의 패널에는 on-target PERK/EIF2AK3가 포함되어 있지 않다 — 원 논문이 "
       "off-target 선택성만 표로 제시했기 때문이며, 따라서 primary는 최강 off-target(MLK2/MAP3K10)이 된다.",
       "- UNC-CA2-103은 TLK2 저해제이지만 DiscoverX 결합 패널에서 TLK2가 91% remaining으로 약하게 "
       "나타난다(경쟁 결합 assay 포맷의 한계). 본 평가의 primary는 '의도 표적'이 아니라 "
       "'실험 최강 표적'으로 정의하므로 이 경우 MYLK4가 primary가 된다."]
open(f"{OUT}/REF_PANEL_EVAL.md", "w").write("\n".join(md))
print(f"\n✅ 저장 → {OUT}/ (KINASE_REF_PANELS.tsv, *_summary.csv, REF_PANEL_EVAL.md, ref_panel_eval.png)")
