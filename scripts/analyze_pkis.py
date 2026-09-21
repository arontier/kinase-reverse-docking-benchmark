"""
PKIS reverse docking (redock z-score) against experimental kinase inhibition
(%inhibition at 1 uM).

Same approach as the KIR analysis, with the labels and aggregation inverted: PKIS reports
%inhibition, where higher means more active, whereas KIR reports %remaining activity, where
lower means more active.

Metrics: ROC AUC, PR AUC, EF1%, EF5%, BEDROC (alpha = 20).
Levels:  global, per-drug, per-target.

Usage:
    python analyze_pkis.py PKIS1     # -> PKIS_result/
    python analyze_pkis.py PKIS2     # -> PKIS2_result/
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

warnings.filterwarnings("ignore")

BASE = "/mnt/d/Deepmolscan/PKIS2"
KBASE = "/mnt/d/Deepmolscan/Kinase_paper"

# 예측 파이프라인의 DRUG_NAME 표기를 실험 매트릭스의 정규 이름에 맞춤.
# (도킹 출력 오타/파일접미사로 exact-merge 시 강한 활성 화합물이 조용히 누락되던 문제.)
# KIR 전용 키이며 PKIS 화합물명과 충돌하지 않아 전 데이터셋에 안전하게 적용된다.
DRUG_ALIAS = {"Barcitinib": "Baricitinib", "Vandetanib-Final": "Vandetanib"}

# ── 데이터셋별 설정 ──────────────────────────────────────────────────────────────
CONFIGS = {
    "KINOME": dict(
        title="KIR (DiscoverX) @1uM",
        exp_csv_kinome=f"{KBASE}/kinase_exp_uniprot_mapped_matrix.csv",  # 컬럼=ASSAY|GENE|UNIPROT, 값=% remaining activity
        pred_long=f"{KBASE}/result_87/redock_zscore_long.csv",
        out_dir=f"{KBASE}/analysis_output",
        single_thresh=50,   # 1uM: % inhibition >= 50 -> Active, < 50 -> Inactive
    ),
    "PKIS1": dict(
        title="PKIS (GSK Nanosyn) @1uM",
        exp_xlsx=f"{BASE}/3. PKIS Nanosyn Assay Heatmaps.xlsx",
        exp_sheet="PKIS HeatMap 1uM",
        exp_header_row=2,   # 0-based; 실제 헤더가 있는 행
        exp_id_col=0,       # Compound ID 열
        exp_first_kin=4,    # 키나아제 값이 시작되는 열 index
        map_tsv=f"{BASE}/PKIS_protein_mapped.tsv",
        pred_long=f"{BASE}/PKIS_result/redock_zscore_long.csv",
        out_dir=f"{BASE}/PKIS_result",
        single_thresh=50,   # 1uM: % inhibition >= 50 -> Active, < 50 -> Inactive
    ),
    "PKIS2": dict(
        title="PKIS2 (%Inh) @1uM",
        exp_tsv=f"{BASE}/PKIS2_exp.tsv",   # Regno × 406 kinase, 1uM % inhibition
        exp_id_col=0,                      # Regno 열
        exp_first_kin=1,                   # Regno 다음부터 전부 키나아제
        map_tsv=f"{BASE}/PKIS2_protein_mapped.tsv",
        pred_long=f"{BASE}/PKIS2_result/redock_zscore_long.csv",
        out_dir=f"{BASE}/PKIS2_result",
        single_thresh=50,   # 1uM: % inhibition >= 50 -> Active, < 50 -> Inactive
        # dual 70/20 통일 (KIR·PKIS1과 동일)
    ),
}

# ── 라벨 임계값 (% inhibition 기준) ─────────────────────────────────────────────
# 기본(dual): KIR(30/80 % remaining) 을 inhibition 축으로 변환
#   remaining < 30  <=>  inhibition >= 70  -> Active
#   remaining >= 80 <=>  inhibition <= 20  -> Inactive  (20~70 애매구간 제외)
# single_thresh 가 설정되면 단일 임계값(>= t = Active, < t = Inactive)으로 전환.
ACTIVE_INH = 70    # % inhibition >= 70 -> Active   (dual 기본값)
INACTIVE_INH = 20  # % inhibition <= 20 -> Inactive (dual 기본값)
MIN_ACTIVES = 1    # per-drug / per-target 최소 Active 수.
                   # 전체 화합물 코호트(KIR 92 / PKIS1 367 / PKIS2 636)로 통일하기 위해 1로 설정.
                   # ROC는 Active·Inactive가 각각 최소 1개 있어야 정의되므로 1이 하한이다.


# ── 지표 함수 (이전 스크립트와 동일) ────────────────────────────────────────────
def bedroc(y_true: np.ndarray, scores: np.ndarray, alpha: float = 20.0) -> float:
    """Truchon & Bayly (2007) BEDROC. scores: 높을수록 활성 예측."""
    n = len(y_true)
    ra = y_true.sum() / n
    if ra == 0 or ra == 1:
        return np.nan
    n_a = int(y_true.sum())
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    ranks = np.where(y_sorted == 1)[0] + 1
    rie_num = np.sum(np.exp(-alpha * ranks / n)) / n_a
    rie_denom = (1.0 / n) * (1 - np.exp(-alpha)) / (np.exp(alpha / n) - 1)
    rie = rie_num / rie_denom
    rie_max = (1 - np.exp(-alpha * ra)) / (ra * (1 - np.exp(-alpha)))
    rie_min = (1 - np.exp(alpha * ra)) / (ra * (1 - np.exp(alpha)))
    return float(np.clip((rie - rie_min) / (rie_max - rie_min), 0, 1))


def enrichment_factor(y_true: np.ndarray, scores: np.ndarray, frac: float = 0.01) -> float:
    n = len(y_true)
    n_actives = y_true.sum()
    if n_actives == 0:
        return np.nan
    top_n = max(1, int(np.ceil(n * frac)))
    order = np.argsort(-scores)
    n_top = y_true[order[:top_n]].sum()
    return (n_top / top_n) / (n_actives / n)


def ef_max(y_true: np.ndarray, frac: float) -> float:
    """해당 active 비율에서 가능한 최대 EF (NEF 정규화용)."""
    n = len(y_true)
    na = y_true.sum()
    if na == 0:
        return np.nan
    top_n = max(1, int(np.ceil(n * frac)))
    max_hits = min(top_n, na)
    return (max_hits / top_n) / (na / n)


def ranking_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """역도킹 용도 지표: R-precision, Recall@10, Hit@k (점수 내림차순 랭킹).
    Hit@k = 상위 k 타겟 안에 active가 하나라도 있으면 1."""
    n = len(y_true)
    na = int(y_true.sum())
    if na == 0:
        return dict(r_precision=np.nan, recall10=np.nan,
                    hit1=np.nan, hit5=np.nan, hit10=np.nan)
    ys = y_true[np.argsort(-scores)]
    return dict(
        r_precision=ys[:na].sum() / na,           # 상위 n_actives 중 active 비율
        recall10=ys[:10].sum() / na,              # 상위 10 안에 든 active / 전체 active
        hit1=float(ys[:1].sum() > 0),             # 상위 1에 active 존재
        hit5=float(ys[:5].sum() > 0),             # 상위 5에 active 존재
        hit10=float(ys[:10].sum() > 0),           # 상위 10에 active 존재
    )


def compute_metrics(y_true, scores) -> dict:
    y_true = np.array(y_true, dtype=float)
    scores = np.array(scores, dtype=float)
    mask = np.isfinite(scores)
    y_true, scores = y_true[mask], scores[mask]
    n_pos = int(y_true.sum())
    n_neg = int((1 - y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return dict(roc_auc=np.nan, pr_auc=np.nan, pauc10=np.nan, ef1=np.nan, ef5=np.nan,
                    nef1=np.nan, nef5=np.nan, bedroc=np.nan, r_precision=np.nan,
                    recall10=np.nan, hit1=np.nan, hit5=np.nan, hit10=np.nan,
                    n_actives=n_pos, n_total=len(y_true))
    ef1 = enrichment_factor(y_true, scores, 0.01)
    ef5 = enrichment_factor(y_true, scores, 0.05)
    ef1_max = ef_max(y_true, 0.01)
    ef5_max = ef_max(y_true, 0.05)
    try:
        pauc10 = roc_auc_score(y_true, scores, max_fpr=0.1)  # McClish 표준화 pAUC(FPR≤0.1)
    except ValueError:
        pauc10 = np.nan
    return dict(
        roc_auc=roc_auc_score(y_true, scores),
        pr_auc=average_precision_score(y_true, scores),
        pauc10=pauc10,
        ef1=ef1, ef5=ef5,
        nef1=ef1 / ef1_max if ef1_max else np.nan,
        nef5=ef5 / ef5_max if ef5_max else np.nan,
        bedroc=bedroc(y_true, scores, alpha=20),
        **ranking_metrics(y_true, scores),
        n_actives=n_pos,
        n_total=len(y_true),
    )


# ── 실험 데이터 로드 → long (DRUG_NAME, UNIPROT_ID, exp_inh) ────────────────────
def load_experiment(cfg: dict) -> pd.DataFrame:
    # KIR: 컬럼명에 UNIPROT 내장(ASSAY|GENE|UNIPROT), 값=% remaining activity
    if "exp_csv_kinome" in cfg:
        raw = pd.read_csv(cfg["exp_csv_kinome"], index_col=0)
        long = (raw.stack().reset_index())
        long.columns = ["DRUG_NAME", "assay_name", "exp_val"]
        long["DRUG_NAME"] = long["DRUG_NAME"].astype(str).str.strip()
        long["UNIPROT_ID"] = long["assay_name"].astype(str).str.split("|").str[2]
        long["exp_val"] = pd.to_numeric(long["exp_val"], errors="coerce")
        long = long.dropna(subset=["exp_val", "UNIPROT_ID"])
        # % remaining activity -> % inhibition (파이프라인 방향 통일: 높을수록 active)
        long["exp_inh"] = 100.0 - long["exp_val"]
        # (약물,UNIPROT) 여러 assay -> 가장 강한 억제(MAX inhibition = MIN remaining)
        return long.groupby(["DRUG_NAME", "UNIPROT_ID"], as_index=False)["exp_inh"].max()

    if "exp_tsv" in cfg:
        raw = pd.read_csv(cfg["exp_tsv"], sep="\t")
    else:
        raw = pd.read_excel(cfg["exp_xlsx"], sheet_name=cfg["exp_sheet"],
                            header=cfg["exp_header_row"], engine="openpyxl")
    cols = list(raw.columns)
    id_col = cols[cfg["exp_id_col"]]
    kin_cols = cols[cfg["exp_first_kin"]:]

    long = raw.melt(id_vars=[id_col], value_vars=kin_cols,
                    var_name="assay_name", value_name="exp_inh")
    long = long.rename(columns={id_col: "DRUG_NAME"})
    long["DRUG_NAME"] = long["DRUG_NAME"].astype(str).str.strip()
    long["assay_name"] = long["assay_name"].astype(str).str.strip()
    long["exp_inh"] = pd.to_numeric(long["exp_inh"], errors="coerce")
    long = long.dropna(subset=["exp_inh"])

    # assay_name -> UNIPROT_ID
    mp = pd.read_csv(cfg["map_tsv"], sep="\t")
    pn2uni = dict(zip(mp["paper_name"].astype(str).str.strip(), mp["UNIPROT_ID"]))
    long["UNIPROT_ID"] = long["assay_name"].map(pn2uni)
    long = long.dropna(subset=["UNIPROT_ID"])

    # (약물, UNIPROT) 여러 assay(변이체/도메인) -> 가장 강한 억제(MAX inhibition)
    agg = (long.groupby(["DRUG_NAME", "UNIPROT_ID"], as_index=False)["exp_inh"].max())
    return agg


def uni_metadata(cfg: dict) -> tuple[dict, dict]:
    """UNIPROT -> (GENE_NAME, etc/note) 매핑. 소스별로 다르게 구성."""
    if "exp_csv_kinome" in cfg:
        hdr = pd.read_csv(cfg["exp_csv_kinome"], index_col=0, nrows=0)
        uni2gene = {}
        for c in hdr.columns:
            parts = str(c).split("|")
            if len(parts) >= 3:
                uni2gene[parts[2]] = parts[1]
        return uni2gene, {}
    mp = pd.read_csv(cfg["map_tsv"], sep="\t")
    return (dict(zip(mp["UNIPROT_ID"], mp["GENE_NAME"])),
            dict(zip(mp["UNIPROT_ID"], mp["etc"].fillna(""))))


def label_experiment(agg: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    agg = agg.copy()
    if "single_thresh" in cfg:
        # 단일 임계값: >= t -> Active, < t -> Inactive (애매구간 없음, 전부 사용)
        t = cfg["single_thresh"]
        agg["label"] = (agg["exp_inh"] >= t).astype(int)
        return agg
    # dual: Active/Inactive 사이 애매구간 제외
    a = cfg.get("active_inh", ACTIVE_INH)
    i = cfg.get("inactive_inh", INACTIVE_INH)
    agg["label"] = np.nan
    agg.loc[agg["exp_inh"] >= a, "label"] = 1
    agg.loc[agg["exp_inh"] <= i, "label"] = 0
    labeled = agg.dropna(subset=["label"]).copy()
    labeled["label"] = labeled["label"].astype(int)
    return labeled


def threshold_desc(cfg: dict) -> str:
    """라벨 규칙 설명 문자열."""
    if "single_thresh" in cfg:
        t = cfg["single_thresh"]
        return f"Active: %inh >= {t} | Inactive: %inh < {t} (단일 임계값)"
    a = cfg.get("active_inh", ACTIVE_INH)
    i = cfg.get("inactive_inh", INACTIVE_INH)
    return f"Active: %inh >= {a} | Inactive: %inh <= {i} | 애매({i}-{a}) 제외"


# ── Protein 커버리지 / drop 진단 ────────────────────────────────────────────────
def protein_coverage(exp_agg: pd.DataFrame, pred_agg: pd.DataFrame, merged: pd.DataFrame,
                     uni2gene: dict, uni2etc: dict) -> pd.DataFrame:
    """실험/예측/병합 단계별로 각 UNIPROT이 어디서 살아남고 drop되는지 표로 반환."""
    exp_uni = set(exp_agg["UNIPROT_ID"])
    pred_uni = set(pred_agg["UNIPROT_ID"])
    mstat = merged.groupby("UNIPROT_ID").agg(n_pairs=("label", "size"),
                                             n_actives=("label", "sum"))
    rows = []
    for u in sorted(exp_uni | pred_uni):
        in_exp, in_pred = u in exp_uni, u in pred_uni
        in_merged = u in mstat.index
        n_pairs = int(mstat.loc[u, "n_pairs"]) if in_merged else 0
        n_act = int(mstat.loc[u, "n_actives"]) if in_merged else 0
        if in_exp and not in_pred:
            status = "drop: no_prediction(도킹패널에없음)"
        elif in_pred and not in_exp:
            status = "drop: no_experiment(실험데이터없음)"
        #elif in_merged and n_act < MIN_ACTIVES:
        #    status = f"kept(global), drop_per-target: few_actives(<{MIN_ACTIVES})"
        elif in_merged:
            status = "kept"
        else:
            status = "drop: no_overlap"
        rows.append(dict(UNIPROT_ID=u, GENE_NAME=uni2gene.get(u, ""),
                         in_exp=in_exp, in_pred=in_pred, in_merged=in_merged,
                         n_pairs=n_pairs, n_actives=n_act,
                         status=status, note=uni2etc.get(u, "")))
    return pd.DataFrame(rows)


# ── 예측 데이터 로드 → (DRUG_NAME, UNIPROT_ID, pred_score) ──────────────────────
def load_prediction(cfg: dict, valid: set) -> pd.DataFrame:
    pred = pd.read_csv(cfg["pred_long"])
    pred["DRUG_NAME"] = pred["DRUG_NAME"].astype(str).str.strip().replace(DRUG_ALIAS)
    # 복합 UNIPROT ID(다종 ortholog를 ';'로 연결, 예: 'P04049;Q5R5M7') ->
    # 실험에 존재하는 사람 accession으로 정규화 (매칭 누락 방지)
    valid = set(str(v) for v in valid)

    def norm(u: object) -> str:
        u = str(u)
        if ";" not in u:
            return u
        for p in u.split(";"):
            if p in valid:
                return p
        return u.split(";")[0]

    pred["UNIPROT_ID"] = pred["UNIPROT_ID"].map(norm)
    agg = (pred.groupby(["DRUG_NAME", "UNIPROT_ID"], as_index=False)
           .agg(zscore_mean=("zscore_mean", "min"), rank_mean=("rank_mean", "min")))
    agg["pred_score"] = -agg["zscore_mean"]  # 높을수록 활성 예측
    return agg


# ── 메인 ────────────────────────────────────────────────────────────────────────
def run(dataset: str) -> None:
    cfg = CONFIGS[dataset]
    os.makedirs(cfg["out_dir"], exist_ok=True)
    print(f"\n{'='*66}\n▶ {dataset}: {cfg['title']}\n{'='*66}")

    exp_agg = load_experiment(cfg)
    uni2gene, uni2etc = uni_metadata(cfg)
    exp_lab = label_experiment(exp_agg, cfg)
    print(f"  라벨 규칙: {threshold_desc(cfg)}")
    print(f"  실험 라벨 쌍: {len(exp_lab):,}  "
          f"Active={int((exp_lab.label==1).sum()):,}  "
          f"Inactive={int((exp_lab.label==0).sum()):,}  "
          f"Active비율={(exp_lab.label==1).mean()*100:.1f}%")

    pred_agg = load_prediction(cfg, set(exp_agg["UNIPROT_ID"]))
    print(f"  예측 쌍: {len(pred_agg):,}  "
          f"약물={pred_agg.DRUG_NAME.nunique()}  타겟={pred_agg.UNIPROT_ID.nunique()}")

    merged = exp_lab.merge(
        pred_agg[["DRUG_NAME", "UNIPROT_ID", "pred_score", "zscore_mean", "rank_mean"]],
        on=["DRUG_NAME", "UNIPROT_ID"], how="inner")
    print(f"  매칭 쌍: {len(merged):,}  Active={int(merged.label.sum()):,}  "
          f"약물={merged.DRUG_NAME.nunique()}  타겟={merged.UNIPROT_ID.nunique()}")

    # ── 순위 지표 전용 pool: 라벨 임계값과 무관하게 '측정∩예측 전체 패널'로 고정 ──
    #    (Hit@k/Recall@k/R-precision은 화합물이 스크리닝된 전체 타겟에 대해 줄세워야 함;
    #     애매구간 제외 등 라벨링이 순위 pool을 바꾸면 안 됨. single_thresh 모드에선 merged와 동일.)
    active_thr = cfg.get("single_thresh", cfg.get("active_inh", ACTIVE_INH))
    merged_rank = exp_agg.merge(
        pred_agg[["DRUG_NAME", "UNIPROT_ID", "pred_score"]],
        on=["DRUG_NAME", "UNIPROT_ID"], how="inner")
    merged_rank["label"] = (merged_rank["exp_inh"] >= active_thr).astype(int)
    rank_by_drug = {k: v for k, v in merged_rank.groupby("DRUG_NAME")}
    rank_by_tgt = {k: v for k, v in merged_rank.groupby("UNIPROT_ID")}

    def rank_over_full(key: str, by: str) -> dict:
        fg = (rank_by_drug if by == "DRUG_NAME" else rank_by_tgt).get(key)
        if fg is None or len(fg) == 0:
            return dict(r_precision=np.nan, recall10=np.nan, hit1=np.nan, hit5=np.nan, hit10=np.nan)
        return ranking_metrics(fg["label"].values.astype(float), fg["pred_score"].values)

    # Global
    g = compute_metrics(merged.label.values, merged.pred_score.values)
    g.update(ranking_metrics(merged_rank["label"].values.astype(float),
                             merged_rank["pred_score"].values))
    print(f"\n  [Global] ROC AUC={g['roc_auc']:.4f}  pAUC(FPR≤.1)={g['pauc10']:.4f}  "
          f"PR AUC={g['pr_auc']:.4f}  EF1%={g['ef1']:.2f}  NEF1={g['nef1']:.3f}  "
          f"BEDROC={g['bedroc']:.4f}")
    print(f"           R-prec={g['r_precision']:.3f}  Recall@10={g['recall10']:.3f}  "
          f"Hit@1/5/10={g['hit1']:.2f}/{g['hit5']:.2f}/{g['hit10']:.2f}")

    # Per-drug
    rows = []
    for drug, grp in merged.groupby("DRUG_NAME"):
        # 전체 화합물 사용(Active 수 필터 없음). ROC는 양·음성이 모두 있어야 정의되므로
        # compute_metrics가 한쪽만 있는 경우 NaN을 반환하고 평균에서 자동 제외된다.
        m = compute_metrics(grp.label.values, grp.pred_score.values)
        m.update(rank_over_full(drug, "DRUG_NAME"))   # 순위 지표는 전체 패널로
        m["DRUG_NAME"] = drug
        rows.append(m)
    per_drug = pd.DataFrame(rows).set_index("DRUG_NAME") if rows else pd.DataFrame()
    if len(per_drug):
        per_drug["active_ratio"] = per_drug["n_actives"] / per_drug["n_total"]
        print(f"  [Per-Drug] n={len(per_drug)}  ROC AUC={per_drug.roc_auc.mean():.4f}"
              f"±{per_drug.roc_auc.std():.4f}  pAUC={per_drug.pauc10.mean():.4f}  "
              f"NEF1={per_drug.nef1.mean():.3f}  R-prec={per_drug.r_precision.mean():.3f}  "
              f"Hit@1/5/10={per_drug.hit1.mean():.2f}/{per_drug.hit5.mean():.2f}/{per_drug.hit10.mean():.2f}")

    # Per-target
    rows = []
    for uni, grp in merged.groupby("UNIPROT_ID"):
        # per-target도 전체 사용(필터 없음)
        m = compute_metrics(grp.label.values, grp.pred_score.values)
        m.update(rank_over_full(uni, "UNIPROT_ID"))   # 순위 지표는 전체 패널로
        m["UNIPROT_ID"] = uni
        rows.append(m)
    per_target = pd.DataFrame(rows).set_index("UNIPROT_ID") if rows else pd.DataFrame()
    if len(per_target):
        per_target["active_ratio"] = per_target["n_actives"] / per_target["n_total"]
        print(f"  [Per-Target] n={len(per_target)}  ROC AUC={per_target.roc_auc.mean():.4f}"
              f"±{per_target.roc_auc.std():.4f}  EF1%={per_target.ef1.mean():.2f}  "
              f"BEDROC={per_target.bedroc.mean():.4f}")

    # ── Protein 커버리지 / drop 진단 ──
    cov = protein_coverage(exp_agg, pred_agg, merged, uni2gene, uni2etc)
    print("\n  [Protein 커버리지]")
    print(f"    실험 타겟(UNIPROT): {int(cov.in_exp.sum())}  "
          f"예측 타겟: {int(cov.in_pred.sum())}  병합: {int(cov.in_merged.sum())}")
    for status, n in cov["status"].value_counts().items():
        print(f"    - {status}: {n}")

    # ── 저장 (파일 잠금 시 해당 파일만 건너뛰고 계속) ──
    od = cfg["out_dir"]

    def safe_save(fn, path):
        try:
            fn(path)
        except PermissionError:
            print(f"  ⚠ 쓰기 실패(잠김, 건너뜀): {os.path.basename(path)} — "
                  f"열려있는 프로그램(Excel 등)을 닫고 재실행하세요.")

    safe_save(lambda p: merged.to_csv(p, index=False), f"{od}/analysis_merged_labeled.csv")
    safe_save(lambda p: pd.DataFrame([g]).to_csv(p, index=False), f"{od}/analysis_global_metrics.csv")
    safe_save(lambda p: per_drug.to_csv(p), f"{od}/analysis_per_drug_metrics.csv")
    safe_save(lambda p: per_target.to_csv(p), f"{od}/analysis_per_target_metrics.csv")
    safe_save(lambda p: cov.to_csv(p, index=False), f"{od}/analysis_protein_coverage.csv")

    safe_save(lambda p: _make_figure(dataset, cfg, merged, g, per_drug, per_target, p),
              f"{od}/analysis_summary_figure.png")
    safe_save(lambda p: _make_perdrug_auc_figure(dataset, per_drug, p),
              f"{od}/per_drug_auc.png")
    _make_report(dataset, cfg, exp_lab, pred_agg, merged, g, per_drug, per_target, cov)
    print(f"  ✅ 저장 완료 → {od}/")


# ── per-drug AUC 전용 그림 (분포 + 정렬 워터폴) ────────────────────────────────
def _make_perdrug_auc_figure(dataset, per_drug, out_path):
    """per-drug ROC AUC의 분포와 화합물별 정렬 값을 한 그림에 (영문 텍스트)."""
    if not len(per_drug):
        return
    auc = per_drug["roc_auc"].dropna().values
    n = len(auc); mean = auc.mean(); sd = auc.std()
    n_below = int((auc < 0.5).sum()); pct_below = 100 * n_below / n
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))
    # (a) 히스토그램
    a = ax[0]
    a.hist(auc, bins=24, range=(0.2, 1.0), color="#4393C3", edgecolor="white")
    a.axvline(0.5, color="#B2182B", ls="--", lw=1.5, label="random (0.5)")
    a.axvline(mean, color="#08306B", lw=2, label=f"mean = {mean:.3f}")
    a.axvspan(0.2, 0.5, color="#B2182B", alpha=0.06)
    a.set(xlabel="Per-drug ROC AUC", ylabel="# compounds",
          title=f"{dataset}: per-drug ROC AUC distribution\n(n={n}, mean {mean:.3f}±{sd:.3f}, <0.5: {n_below} = {pct_below:.0f}%)")
    a.legend(fontsize=9)
    # (b) 정렬 워터폴
    a = ax[1]
    s = np.sort(auc); x = np.arange(1, n + 1)
    a.fill_between(x, 0.5, s, where=(s >= 0.5), color="#4393C3", alpha=0.7, label="≥0.5")
    a.fill_between(x, 0.5, s, where=(s < 0.5), color="#B2182B", alpha=0.7, label="<0.5")
    a.axhline(0.5, color="#B2182B", ls="--", lw=1.2)
    a.set(xlabel="compound rank (sorted by AUC)", ylabel="Per-drug ROC AUC",
          ylim=(0.2, 1.0), title=f"{dataset}: per-drug AUC, sorted")
    a.legend(fontsize=9, loc="upper left")
    fig.suptitle(f"Per-drug ROC AUC — {dataset}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── 그림 (영문 텍스트: 한글 폰트 깨짐 방지) ─────────────────────────────────────
def _make_figure(dataset, cfg, merged, g, per_drug, per_target, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.subplots_adjust(hspace=0.45, wspace=0.35)

    ax = axes[0, 0]
    fpr, tpr, _ = roc_curve(merged.label, merged.pred_score)
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {g['roc_auc']:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="Global ROC Curve", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower right")

    ax = axes[0, 1]
    labels = ["ROC AUC", "PR AUC", "BEDROC", "EF 1%", "EF 5%"]
    vals = [g["roc_auc"], g["pr_auc"], g["bedroc"], g["ef1"], g["ef5"]]
    colors = ["steelblue", "darkorange", "forestgreen", "crimson", "orchid"]
    bars = ax.bar(labels, vals, color=colors, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(vals) * 0.02,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set(title="Global Metrics", ylabel="Score")
    ax.tick_params(axis="x", rotation=20)

    ax = axes[0, 2]
    act = merged.label == 1
    ax.scatter(merged.loc[~act, "pred_score"], merged.loc[~act, "exp_inh"],
               c="lightgray", alpha=0.2, s=4, rasterized=True, label="Inactive")
    ax.scatter(merged.loc[act, "pred_score"], merged.loc[act, "exp_inh"],
               c="steelblue", alpha=0.5, s=8, rasterized=True, label="Active")
    if "single_thresh" in cfg:
        ax.axhline(cfg["single_thresh"], color="purple", ls="--", lw=1.2)
    else:
        ax.axhline(cfg.get("active_inh", ACTIVE_INH), color="blue", ls="--", lw=1)
        ax.axhline(cfg.get("inactive_inh", INACTIVE_INH), color="red", ls="--", lw=1)
    ax.set(xlabel="Prediction Score (-zscore_mean)", ylabel="Exp % Inhibition",
           title="Score vs Experiment")
    ax.legend(fontsize=8, markerscale=2)

    ax = axes[1, 0]
    if len(per_drug):
        per_drug.roc_auc.dropna().hist(bins=20, color="steelblue", edgecolor="white", ax=ax)
        ax.axvline(0.5, color="red", ls="--", lw=1.5, label="Random")
        ax.axvline(per_drug.roc_auc.mean(), color="navy", lw=1.5,
                   label=f"Mean={per_drug.roc_auc.mean():.3f}")
        ax.legend(fontsize=8)
    ax.set(xlabel="ROC AUC", ylabel="Drug Count",
           title=f"Per-Drug ROC AUC (n={len(per_drug)})")

    ax = axes[1, 1]
    if len(per_drug):
        ef1 = per_drug.ef1.dropna()
        ef1.clip(upper=np.percentile(ef1, 95)).hist(bins=20, color="darkorange",
                                                     edgecolor="white", ax=ax)
        ax.axvline(1.0, color="red", ls="--", lw=1.5, label="Random")
        ax.axvline(ef1.mean(), color="saddlebrown", lw=1.5, label=f"Mean={ef1.mean():.2f}")
        ax.legend(fontsize=8)
    ax.set(xlabel="EF 1%", ylabel="Drug Count",
           title=f"Per-Drug EF 1% (n={len(per_drug)}, 95th clip)")

    ax = axes[1, 2]
    if len(per_target):
        per_target.roc_auc.dropna().hist(bins=20, color="forestgreen", edgecolor="white", ax=ax)
        ax.axvline(0.5, color="red", ls="--", lw=1.5, label="Random")
        ax.axvline(per_target.roc_auc.mean(), color="darkgreen", lw=1.5,
                   label=f"Mean={per_target.roc_auc.mean():.3f}")
        ax.legend(fontsize=8)
    ax.set(xlabel="ROC AUC", ylabel="Target Count",
           title=f"Per-Target ROC AUC (n={len(per_target)})")

    if "single_thresh" in cfg:
        sub = f"Active >={cfg['single_thresh']}% | Inactive <{cfg['single_thresh']}% inhibition"
    else:
        sub = (f"Active >={cfg.get('active_inh', ACTIVE_INH)}% | "
               f"Inactive <={cfg.get('inactive_inh', INACTIVE_INH)}% inhibition")
    fig.suptitle(f"Reverse Dock vs Experiment  |  {cfg['title']}\n{sub}",
                 fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── 보고서 (한글) ────────────────────────────────────────────────────────────────
def _make_report(dataset, cfg, exp_lab, pred_agg, merged, g, per_drug, per_target, cov):
    def top_tbl(df, col, n, asc):
        if not len(df):
            return "  (해당 없음)\n"
        sub = df.sort_values(col, ascending=asc).head(n)
        lines = ["| ID | ROC AUC | EF1% | EF5% | BEDROC | active_ratio | n_act | n_tot |",
                 "|----|--------:|-----:|-----:|-------:|-------------:|------:|------:|"]
        for idx, r in sub.iterrows():
            ar = r.active_ratio if "active_ratio" in sub.columns else r.n_actives / r.n_total
            lines.append(f"| {idx} | {r.roc_auc:.3f} | {r.ef1:.2f} | {r.ef5:.2f} | "
                         f"{r.bedroc:.3f} | {ar:.3f} | {int(r.n_actives)} | {int(r.n_total)} |")
        return "\n".join(lines) + "\n"

    pd_mean = per_drug.roc_auc.mean() if len(per_drug) else float("nan")
    pt_mean = per_target.roc_auc.mean() if len(per_target) else float("nan")
    exp_src = os.path.basename(cfg.get("exp_tsv", cfg.get("exp_xlsx",
                               cfg.get("exp_csv_kinome", ""))))
    map_src = os.path.basename(cfg["map_tsv"]) if cfg.get("map_tsv") else "컬럼명 내장(ASSAY|GENE|UNIPROT)"
    is_kinome = "exp_csv_kinome" in cfg
    exp_kind = "% remaining activity → % inhibition(100−remaining)" if is_kinome else "% inhibition"
    if "single_thresh" in cfg:
        thresh_note = (f"1µM 스크리닝에서 % inhibition ≥ {cfg['single_thresh']}%를 active로 보는 "
                       f"단일 임계값 기준. 애매 구간을 두지 않아 라벨된 모든 쌍을 분석에 사용.")
    else:
        thresh_note = ("임계값은 KIR 기준(30/80% remaining)을 inhibition 축으로 변환한 값 "
                       "(remaining<30 ⟺ inh≥70, remaining≥80 ⟺ inh≤20).")

    # ── Protein 커버리지 섹션 ──
    n_exp, n_pred, n_merged = int(cov.in_exp.sum()), int(cov.in_pred.sum()), int(cov.in_merged.sum())
    no_pred = cov[cov.status.str.startswith("drop: no_prediction")]
    no_exp = cov[cov.status.str.startswith("drop: no_experiment")]
    few = cov[cov.status.str.contains("few_actives")]

    def _gene_list(df, limit=60):
        if not len(df):
            return "(없음)"
        items = [f"{r.GENE_NAME or r.UNIPROT_ID}" for _, r in df.iterrows()]
        s = ", ".join(items[:limit])
        if len(items) > limit:
            s += f" … (외 {len(items)-limit}개)"
        return s

    coverage_section = f"""## 5. Protein 커버리지 / drop 진단

각 UNIPROT 타겟이 실험·예측·병합 단계에서 어떻게 처리됐는지. 상세: `analysis_protein_coverage.csv`

| 단계 | 타겟 수 |
|------|--------|
| 실험 데이터 보유 (in_exp) | {n_exp} |
| 예측(도킹) 보유 (in_pred) | {n_pred} |
| **병합됨 (분석 사용)** | **{n_merged}** |
| Per-Target 지표 포함 (Active≥{MIN_ACTIVES}) | {len(per_target)} |

### drop된 타겟

| 사유 | 개수 | 설명 |
|------|-----:|------|
| `no_prediction` | {len(no_pred)} | 실험엔 있으나 도킹 패널에 없음 |
| `no_experiment` | {len(no_exp)} | 도킹 예측은 있으나 실험 데이터 없음 |
| `few_actives(<{MIN_ACTIVES})` | {len(few)} | 병합됐으나 Active 부족 → Per-Target만 제외(Global엔 포함) |

- **no_prediction** ({len(no_pred)}): {_gene_list(no_pred)}
- **no_experiment** ({len(no_exp)}): {_gene_list(no_exp)}

---

"""

    md = f"""# {dataset} Reverse Docking 벤치마크 분석 결과

**대상**: {cfg['title']}
**예측**: `{os.path.basename(cfg['pred_long'])}` (AK-Score reverse docking z-score)
**실험**: `{exp_src}` — {exp_kind} @1µM
**단백질 매핑**: {map_src}
**분석일**: 2026-06-30

---

## 1. 분석 방법

세 벤치마크(KIR / PKIS1 / PKIS2) 공통 파이프라인. 실험값은 **% inhibition**
(높을수록 강한 억제 = active) 축으로 통일 — KIR은 % remaining activity를
`100 − remaining`으로 변환.

| 항목 | 설정 |
|------|------|
| 예측 점수 | `pred_score = -zscore_mean` (높을수록 활성 예측) |
| 실험 활성 | {exp_kind} @1µM |
| 라벨 규칙 | **{threshold_desc(cfg)}** |
| 타겟 집계 | assay(변이체/도메인) 여러 개 → UNIPROT별 **최대 억제** |
| 예측 집계 | 구조 여러 개 → (약물,UNIPROT)별 **최소 zscore** |
| 지표 | ROC AUC, pAUC(FPR≤0.1), PR AUC, EF1/5%, NEF1/5, BEDROC(α=20), R-precision, Recall@10, Hit@1/5/10 |

> {thresh_note}

**지표 그룹**: ① 전체 순위 = ROC AUC · pAUC · PR AUC
② 조기인식 = EF1/5% · NEF1/5(active비율 보정) · BEDROC
③ 역도킹 실사용(타겟 top-k) = R-precision(top-n_act) · Recall@10 · Hit@1/5/10

> **순위 pool 고정**: ③ 지표(R-precision/Recall@k/Hit@k)는 라벨 임계값과 무관하게
> 화합물이 **측정·예측된 전체 타겟 패널**로 줄세워 계산(active만 임계값으로 정의).
> → 임계값을 바꿔도 순위 pool이 변하지 않아 Hit@k가 임계값에 대해 단조적으로 거동.

---

## 2. 데이터 규모

| 항목 | 값 |
|------|----|
| 실험 라벨된 (약물,타겟) 쌍 | {len(exp_lab):,} |
| — Active | {int((exp_lab.label==1).sum()):,} ({(exp_lab.label==1).mean()*100:.1f}%) |
| — Inactive | {int((exp_lab.label==0).sum()):,} |
| 예측 쌍 | {len(pred_agg):,} (약물 {pred_agg.DRUG_NAME.nunique()}, 타겟 {pred_agg.UNIPROT_ID.nunique()}) |
| **매칭된 쌍** | **{len(merged):,}** (약물 {merged.DRUG_NAME.nunique()}, 타겟 {merged.UNIPROT_ID.nunique()}) |
| — 매칭 Active | {int(merged.label.sum()):,} ({merged.label.mean()*100:.1f}%) |

---

## 3. Global 성능

| 지표 | 값 | 그룹 |
|------|----|------|
| ROC AUC | **{g['roc_auc']:.4f}** | 순위 |
| pAUC (FPR≤0.1) | {g['pauc10']:.4f} | 순위(조기) |
| PR AUC | {g['pr_auc']:.4f} | 순위 |
| EF 1% / 5% | {g['ef1']:.2f} / {g['ef5']:.2f} | 조기인식 |
| NEF 1% / 5% | {g['nef1']:.3f} / {g['nef5']:.3f} | 조기인식(보정) |
| BEDROC (α=20) | {g['bedroc']:.4f} | 조기인식 |
| Active / 전체 | {int(g['n_actives'])} / {int(g['n_total'])} | — |

> R-precision·Recall@10·Hit@k(역도킹 top-k 회수)은 **한 화합물 내 타겟 순위** 지표라
> per-drug 수준에서만 의미 있음(§4 참조). Global 통합 순위에서는 산출하지 않음.

> Active 비율 {merged.label.mean()*100:.1f}% — 가상 스크리닝 벤치마크로 현실적인 수준
> (Davis 58.6% 대비 라벨 편향 없음). NEF는 프로미스큐이티 보정,
> R-precision/Recall@10/Hit@k는 역도킹 실사용(타겟 top-k 회수)을 직접 반영.

---

## 4. Per-Drug / Per-Target 요약

(Active ≥ {MIN_ACTIVES} 인 항목만)

| 레벨 | 유효 개수 | 평균 ROC AUC | 평균 EF1% | 평균 EF5% | 평균 BEDROC | active_ratio(중앙) |
|------|----------|-------------|-----------|-----------|-------------|--------------------|
| Per-Drug | {len(per_drug)} | {pd_mean:.4f} | {per_drug.ef1.mean() if len(per_drug) else float('nan'):.2f} | {per_drug.ef5.mean() if len(per_drug) else float('nan'):.2f} | {per_drug.bedroc.mean() if len(per_drug) else float('nan'):.4f} | {per_drug.active_ratio.median() if len(per_drug) else float('nan'):.3f} |
| Per-Target | {len(per_target)} | {pt_mean:.4f} | {per_target.ef1.mean() if len(per_target) else float('nan'):.2f} | {per_target.ef5.mean() if len(per_target) else float('nan'):.2f} | {per_target.bedroc.mean() if len(per_target) else float('nan'):.4f} | {per_target.active_ratio.median() if len(per_target) else float('nan'):.3f} |

### 추가 지표 (보정 · 역도킹 top-k)

| 레벨 | 평균 pAUC | 평균 NEF1 | 평균 R-precision | 평균 Recall@10 | Hit@1 | Hit@5 | Hit@10 |
|------|----------|-----------|------------------|----------------|-------|-------|--------|
| Per-Drug | {per_drug.pauc10.mean() if len(per_drug) else float('nan'):.4f} | {per_drug.nef1.mean() if len(per_drug) else float('nan'):.3f} | {per_drug.r_precision.mean() if len(per_drug) else float('nan'):.3f} | {per_drug.recall10.mean() if len(per_drug) else float('nan'):.3f} | {per_drug.hit1.mean() if len(per_drug) else float('nan'):.2f} | {per_drug.hit5.mean() if len(per_drug) else float('nan'):.2f} | {per_drug.hit10.mean() if len(per_drug) else float('nan'):.2f} |
| Per-Target | {per_target.pauc10.mean() if len(per_target) else float('nan'):.4f} | {per_target.nef1.mean() if len(per_target) else float('nan'):.3f} | {per_target.r_precision.mean() if len(per_target) else float('nan'):.3f} | {per_target.recall10.mean() if len(per_target) else float('nan'):.3f} | {per_target.hit1.mean() if len(per_target) else float('nan'):.2f} | {per_target.hit5.mean() if len(per_target) else float('nan'):.2f} | {per_target.hit10.mean() if len(per_target) else float('nan'):.2f} |

> **조기 인식(early recognition)**: EF1%/EF5%·BEDROC·pAUC가 상위 농축을 반영.
> **NEF**는 active_ratio에 따른 EF 상한을 보정(범결합 vs 선택적 공정 비교).
> **R-precision/Recall@10/Hit@k**는 역도킹 실사용(한 화합물의 진짜 타겟을 top-k로 회수)을 직접 측정.
> Hit@k = 상위 k 예측 타겟 안에 실제 active가 하나라도 있을 확률.
> `active_ratio` 높은(범결합) 화합물은 ROC AUC가 낮게 나오는 경향(해석 주의).

### 4-1. Per-Drug 상위 10 (ROC AUC)
{top_tbl(per_drug, 'roc_auc', 10, asc=False)}
### 4-2. Per-Drug 하위 10 (ROC AUC)
{top_tbl(per_drug, 'roc_auc', 10, asc=True)}
### 4-3. Per-Target 상위 10 (ROC AUC)
{top_tbl(per_target, 'roc_auc', 10, asc=False)}
### 4-4. Per-Target 하위 10 (ROC AUC)
{top_tbl(per_target, 'roc_auc', 10, asc=True)}
---

{coverage_section}## 6. 산출 파일

| 파일 | 내용 |
|------|------|
| `analysis_summary.md` | 본 보고서 |
| `analysis_summary_figure.png` | 6-패널 요약 그림 |
| `analysis_global_metrics.csv` | Global 지표 |
| `analysis_per_drug_metrics.csv` | 약물별 지표 |
| `analysis_per_target_metrics.csv` | 타겟별 지표 |
| `analysis_merged_labeled.csv` | 병합·라벨된 원본 쌍 |
| `analysis_protein_coverage.csv` | 타겟별 커버리지/drop 진단 |
"""
    try:
        with open(f"{cfg['out_dir']}/analysis_summary.md", "w") as f:
            f.write(md)
    except PermissionError:
        print(f"  ⚠ 쓰기 실패(잠김, 건너뜀): analysis_summary.md")


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "PKIS1"
    run(ds)
