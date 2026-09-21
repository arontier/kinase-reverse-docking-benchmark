"""
Canonical source for the KIR experiment-prediction merge.

Composite UniProt accessions ('P04049;Q5R5M7' and the like) are normalized to the human
accession present in the experimental matrix, exactly as analyze_pkis.load_prediction does.
Omitting this step silently drops 9 targets and 828 pairs, which puts the pool out of step with
Table 1 (92 x 369, 33,141 pairs) and with every metric table.

Usage:
    import kinome_merge as km
    exp = km.experiment()                       # DRUG_NAME, UNIPROT_ID, exp_inh, label
    M   = km.merge_pred(exp, "result_87/redock_zscore_long.csv")
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/d/Deepmolscan/PKIS2")
import analyze_pkis as ap  # noqa: E402

KP = "/mnt/d/Deepmolscan/Kinase_paper"
EXP_CSV = f"{KP}/kinase_exp_uniprot_mapped_matrix.csv"


def experiment() -> pd.DataFrame:
    """KIR 실험 라벨 (% remaining -> % inhibition, active = inh>=50)."""
    raw = pd.read_csv(EXP_CSV, index_col=0)
    long = raw.stack().reset_index()
    long.columns = ["DRUG_NAME", "assay_name", "exp_val"]
    long["DRUG_NAME"] = long["DRUG_NAME"].astype(str).str.strip()
    long["UNIPROT_ID"] = long["assay_name"].astype(str).str.split("|").str[2]
    long["exp_val"] = pd.to_numeric(long["exp_val"], errors="coerce")
    long = long.dropna(subset=["exp_val", "UNIPROT_ID"])
    long["exp_inh"] = 100.0 - long["exp_val"]
    agg = long.groupby(["DRUG_NAME", "UNIPROT_ID"], as_index=False)["exp_inh"].max()
    agg["label"] = (agg["exp_inh"] >= 50).astype(int)
    return agg


_VALID: set | None = None


def valid_uniprots() -> set:
    """KIR 실험 매트릭스에 존재하는 UNIPROT accession 집합(캐시)."""
    global _VALID
    if _VALID is None:
        hdr = pd.read_csv(EXP_CSV, index_col=0, nrows=0)
        _VALID = {str(c).split("|")[2] for c in hdr.columns if len(str(c).split("|")) >= 3}
    return _VALID


def norm_uni(series: pd.Series, valid: set | None = None) -> pd.Series:
    """예측 측 복합 UNIPROT ID를 실험에 존재하는 accession으로 정규화.

    이 한 줄이 없으면 KIR 병합에서 타겟 9개(쌍 828개)가 조용히 탈락한다.
    """
    v = {str(x) for x in (valid if valid is not None else valid_uniprots())}

    def pick(u: object) -> str:
        u = str(u)
        if ";" not in u:
            return u
        for part in u.split(";"):
            if part in v:
                return part
        return u.split(";")[0]

    return series.map(pick)


def load_pred(path: str, valid: set, score_col: str = "pred_score") -> pd.DataFrame:
    """예측 long CSV -> (DRUG_NAME, UNIPROT_ID, score). 방향은 rank 상관으로 자동 교정."""
    pred = pd.read_csv(path)
    pred["DRUG_NAME"] = pred["DRUG_NAME"].astype(str).str.strip().replace(ap.DRUG_ALIAS)
    valid = {str(v) for v in valid}

    def norm(u: object) -> str:
        u = str(u)
        if ";" not in u:
            return u
        for part in u.split(";"):
            if part in valid:
                return part
        return u.split(";")[0]

    pred["UNIPROT_ID"] = pred["UNIPROT_ID"].map(norm)
    agg = (pred.groupby(["DRUG_NAME", "UNIPROT_ID"], as_index=False)
                .agg(z=("zscore_mean", "min"), r=("rank_mean", "min")))
    sign = 1.0 if np.corrcoef(agg.z, agg.r)[0, 1] > 0 else -1.0
    agg[score_col] = -sign * agg["z"]
    return agg[["DRUG_NAME", "UNIPROT_ID", score_col]]


def merge_pred(exp: pd.DataFrame, path: str, score_col: str = "pred_score") -> pd.DataFrame:
    pred = load_pred(path, set(exp["UNIPROT_ID"]), score_col)
    return exp.merge(pred, on=["DRUG_NAME", "UNIPROT_ID"])


if __name__ == "__main__":
    e = experiment()
    m = merge_pred(e, f"{KP}/result_87/redock_zscore_long.csv")
    print(f"실험 {len(e):,}쌍 / 병합 {len(m):,}쌍  "
          f"화합물 {m.DRUG_NAME.nunique()}  타겟 {m.UNIPROT_ID.nunique()}  "
          f"Active {m.label.mean()*100:.2f}%")
