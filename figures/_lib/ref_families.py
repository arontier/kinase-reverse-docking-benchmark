"""
Kinase_ref compound -> source-paper series mapping (single source of truth).

Provenance document: Deepmolscan/Kinase_ref_출처.pdf  ("출처" is Korean for "source")
  — a collection of medicinal-chemistry papers, each profiling a small number
    (typically 1-5) of selected lead/candidate compounds against a 300-470 kinase
    panel and reporting the resulting selectivity table or heat map.

Assignment evidence: compound-name series plus each compound's measured target count,
cross-checked against the panel size and compound count stated for each source paper.
Compound counts match exactly in several series (PKN2 "cpd. 3, 8" -> PKN2_3/PKN2_8;
GAK "probe 11 + control 14" -> GAK1/GAK1N), which validates the mapping.

All field values are English so that the exported panel table is publication-ready.
`reference` carries the full primary citation, verified against PubMed. The provenance
PDF identifies each paper but its own reference list is a set of web-search snippets
(several of them supplier catalogue pages), so it supplies no authors, volume, pages or
DOI, and four of its journal attributions are wrong. Those are corrected here:

  PERK/GSK2656157   PDF "J. Med. Chem. / ACS Chem. Biol. 2013" -> ACS Med Chem Lett 2013
  MNK1/2 BAY1143269 PDF "Eur. J. Med. Chem."                   -> Cancer Lett 2017
  DYRK1A/AnnH75     PDF "PLoS Genet. / Pharmacol. 2018"        -> PLoS One 2015
  CK2/SGC-CK2-1     PDF "J. Med. Chem. 2021"                   -> Cell Chem Biol 2021

`pdf_refs` still records the bracketed numbers of the provenance PDF so the mapping back
to that document stays auditable.
"""
from __future__ import annotations

import re

_FIELDS = ("key", "target", "series", "panel", "platform", "pdf_refs", "reference")

# key, matcher, intended target, compound series, panel size, platform, PDF refs, reference
_FAMILIES: list[tuple] = [
    ("PERK", lambda c: c == "Comp8",
     "PERK (EIF2AK3)", "GSK2656157 series, compound 8", "300",
     "Reaction Biology HotSpot", "1, 2",
     "Axten JM, Romeril SP, Shu A, Ralph J, Medina JR, Feng Y, et al. Discovery of "
     "GSK2656157: an optimized PERK inhibitor selected for preclinical development. "
     "ACS Med Chem Lett. 2013;4(10):964-968. doi:10.1021/ml400228e [PDF refs 1, 2]"),

    ("MNK_ETC", lambda c: c == "ETC-206",
     "MNK1/2", "ETC-206 (AUM001)", "414",
     "Life Technologies SelectScreen", "5, 6",
     "Yang H, Chennamaneni LR, Ho MWT, Ang SH, Tan ESW, Jeyaraj DA, et al. Optimization "
     "of selective mitogen-activated protein kinase interacting kinases 1 and 2 "
     "inhibitors for the treatment of blast crisis leukemia. J Med Chem. "
     "2018;61(10):4348-4369. doi:10.1021/acs.jmedchem.7b01714 [PDF refs 5, 6]"),

    ("MNK_BAY", lambda c: c == "BAY-1143269",
     "MNK1/2", "BAY 1143269", "395",
     "Miltenyi Biotec / Eurofins", "7, 8",
     "Santag S, Siegel F, Wengner AM, Lange C, Bomer U, Eis K, et al. BAY 1143269, a "
     "novel MNK1 inhibitor, targets oncogenic protein expression and shows potent "
     "anti-tumor activity. Cancer Lett. 2017;390:21-29. "
     "doi:10.1016/j.canlet.2016.12.029 [PDF refs 7, 8 — vendor pages listing this as "
     "Eur. J. Med. Chem.]"),

    ("DYRK_bc", lambda c: c == "AnnH75",
     "DYRK1A", "AnnH75 (beta-carboline)", "300",
     "functional assay", "9",
     "Ruben K, Wurzlbauer A, Walte A, Sippl W, Bracher F, Becker W. Selectivity "
     "profiling and biological activity of novel beta-carbolines as potent and selective "
     "DYRK1 kinase inhibitors. PLoS One. 2015;10(7):e0132453. "
     "doi:10.1371/journal.pone.0132453 [PDF ref 9 — listed there as PLoS Genet./"
     "Pharmacol. 2018]"),

    ("DYRK_pp", lambda c: c.startswith("DYRK_"),
     "DYRK1A/1B", "pyrazolo[1,5-b]pyridazine series", "468",
     "Eurofins DiscoverX KINOMEscan", "10, 11",
     "Henderson SH, Sorrell F, Bennett J, Fedorov O, Hanley MT, Godoi PH, et al. "
     "Discovery and characterization of selective and ligand-efficient DYRK inhibitors. "
     "J Med Chem. 2021;64(15):11709-11728. doi:10.1021/acs.jmedchem.1c01115 "
     "[PDF refs 10, 11]"),

    ("CK2", lambda c: c.startswith("CK_") or c == "Silmitasertib",
     "CK2 (CSNK2A1/A2)", "SGC-CK2-1 and CX-4945 (silmitasertib) analogue series", "403",
     "Eurofins DiscoverX KINOMEscan", "12",
     "Wells CI, Drewry DH, Pickett JE, Tjaden A, Kramer A, Muller S, et al. Development "
     "of a potent and selective chemical probe for the pleiotropic kinase CK2. Cell Chem "
     "Biol. 2021;28(4):546-558.e10. doi:10.1016/j.chembiol.2020.12.013 "
     "[PDF ref 12 — listed there as J. Med. Chem. 2021]"),

    ("GAK_probe", lambda c: c in ("GAK1", "GAK1N"),
     "GAK", "SGC-GAK-1 (probe) + SGC-GAK-1N (negative control)", "406",
     "Eurofins DiscoverX KINOMEscan", "14, 15",
     "Asquith CRM, Berger BT, Wan J, Bennett JM, Capuzzi SJ, Crona DJ, et al. SGC-GAK-1: "
     "a chemical probe for cyclin G associated kinase (GAK). J Med Chem. "
     "2019;62(5):2830-2836. doi:10.1021/acs.jmedchem.8b01213 [PDF refs 14, 15]"),

    ("GAK_quin", lambda c: c in ("GAK_1", "GAK_15"),
     "GAK", "quinoline series (comprehensive + mini-kinome panel)", "468",
     "Eurofins DiscoverX KINOMEscan", "16, 17",
     "Asquith CRM, Treiber DK, Zuercher WJ. Utilizing comprehensive and mini-kinome "
     "panels to optimize the selectivity of quinoline inhibitors for cyclin G associated "
     "kinase (GAK). Bioorg Med Chem Lett. 2019;29(14):1727-1731. "
     "doi:10.1016/j.bmcl.2019.05.025 [PDF refs 16, 17]"),

    ("SLK_STK10", lambda c: c.startswith("STK_"),
     "SLK / STK10", "3-anilino-4-arylmaleimide series", "468",
     "Eurofins DiscoverX KINOMEscan", "18, 19",
     "Serafim RAM, Sorrell FJ, Berger BT, Collins RJ, Vasconcelos SNS, Massirer KB, "
     "et al. Discovery of a potent dual SLK/STK10 inhibitor based on a maleimide "
     "scaffold. J Med Chem. 2021;64(18):13259-13278. doi:10.1021/acs.jmedchem.0c01579 "
     "[PDF refs 18, 19]"),

    ("PKN", lambda c: c.startswith("PKN2_"),
     "PKN1/2", "PKN2 inhibitor series (cpd. 3, 8)", "468",
     "Eurofins DiscoverX KINOMEscan", "25",
     "Scott F, Fala AM, Takarada JE, Ficu MP, Pennicott LE, Reuillon TD, et al. "
     "Development of dihydropyrrolopyridinone-based PKN2/PRK2 chemical tools to enable "
     "drug discovery. Bioorg Med Chem Lett. 2022;60:128588. "
     "doi:10.1016/j.bmcl.2022.128588 [PDF ref 25]"),

    # SU9516 is an oxindole (3-substituted indolinone) and is compound 1 — the starting
    # hit — of this oxindole series, confirmed by the authors.
    ("TLK2", lambda c: c == "SU9516" or re.match(r"^UNC-?CA2-\d+$", c) is not None,
     "TLK2", "oxindole series (compound 1 = SU9516; lead 128 = UNC-CA2-103)", "468",
     "Eurofins DiscoverX KINOMEscan", "20, 21, 22",
     "Asquith CRM, East MP, Laitinen T, Alamillo-Ferrer C, Hartikainen E, Wells CI, "
     "et al. Discovery and optimization of narrow spectrum inhibitors of Tousled like "
     "kinase 2 (TLK2) using quantitative structure activity relationships. Eur J Med "
     "Chem. 2024;271:116357. doi:10.1016/j.ejmech.2024.116357 [PDF refs 20, 21, 22]"),

    ("UNASSIGNED", lambda c: True,
     "[to be confirmed]", "[no matching source row in the provenance PDF]", "—",
     "—", "—", "[to be completed by the authors]"),
]


def family_of(compound: str) -> dict:
    """Compound name -> source-paper series record."""
    hit = next(f for f in _FAMILIES if f[1](compound))
    return dict(zip(_FIELDS, (hit[0],) + hit[2:]))


def all_families() -> list[dict]:
    return [dict(zip(_FIELDS, (f[0],) + f[2:])) for f in _FAMILIES]
