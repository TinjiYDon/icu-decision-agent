"""Coarse Elixhauser-style comorbidity flags from `mimic.mimiciv_hosp.diagnoses_icd`.

`paper_feat.sample_matrix`'s `elix_*` columns are all zero (the old ETL never filled
them), so the "comorbidity" static block is rebuilt here as coarse ICD-9/10 group
flags. NOTE: ICD codes are assigned at discharge, so these flags are *not* strictly
causal at hour h — this is reported as a limitation and isolated in the ablation.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import polars as pl
import psycopg

OUT = Path(__file__).resolve().parents[1]
CACHE = OUT / "cache"
DSN = __import__("os").environ["MIMIC_DSN"]

# (name, icd10 prefixes, icd9 prefixes) -- prefix matching on the raw code string
GROUPS = [
    ("chf", ["I50", "I110", "I130", "I132"], ["428", "40211", "40291", "40411", "40491"]),
    ("arrhythmia", ["I44", "I45", "I46", "I47", "I48", "I49"], ["426", "427"]),
    ("hypertension", ["I10", "I11", "I12", "I13", "I15"], ["401", "402", "403", "404", "405"]),
    ("diabetes", ["E10", "E11", "E12", "E13", "E14"], ["250"]),
    ("renal_failure", ["N17", "N18", "N19", "N99"], ["584", "585", "586"]),
    ("liver_disease", ["K70", "K71", "K72", "K73", "K74", "K76", "K77"], ["570", "571", "572", "573"]),
    ("cancer_solid", ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"], ["140", "150", "160", "170", "180", "190"]),
    ("metastatic", ["C77", "C78", "C79", "C80"], ["196", "197", "198", "199"]),
    ("lymphoma", ["C81", "C82", "C83", "C84", "C85", "C86", "C88", "C90", "C91", "C92", "C93", "C94", "C95", "C96"],
     ["200", "201", "202", "203", "204", "205", "206", "207", "208"]),
    ("copd", ["J40", "J41", "J42", "J43", "J44", "J45", "J46", "J47"], ["490", "491", "492", "493", "494", "495", "496"]),
    ("cerebrovascular", ["I60", "I61", "I62", "I63", "I64", "I65", "I66", "I67", "I68", "I69"], ["430", "431", "432", "433", "434", "435", "436", "437", "438"]),
    ("peripheral_vascular", ["I70", "I71", "I72", "I73"], ["440", "441", "442", "443", "444", "445", "446", "447", "448"]),
    ("obesity", ["E66"], ["2780"]),
    ("anemia_deficiency", ["D50", "D51", "D52", "D53"], ["280"]),
    ("blood_loss_anemia", ["D62"], ["2851"]),
    ("coagulopathy", ["D65", "D66", "D67", "D68", "D69"], ["286"]),
    ("fluid_electrolyte", ["E86", "E87"], ["276"]),
    ("depression", ["F32", "F33", "F34"], ["296", "311"]),
    ("psychoses", ["F20", "F21", "F22", "F23", "F24", "F25", "F28", "F29"], ["295", "297", "298", "299"]),
    ("valvular", ["I05", "I06", "I07", "I08", "I09", "I34", "I35", "I36", "I37"], ["394", "395", "396", "397"]),
    ("pulmonary_circ", ["I26", "I27", "I28"], ["415", "416", "417"]),
    ("alcohol", ["F10"], ["291", "303"]),
    ("drug_abuse", ["F11", "F12", "F13", "F14", "F15", "F16", "F17", "F18", "F19"], ["292", "304", "305"]),
    ("paralysis", ["G81", "G82", "G83"], ["342", "343", "344"]),
    ("neuro_other", ["G10", "G11", "G12", "G13", "G20", "G21", "G22", "G23", "G24", "G25", "G26",
                     "G35", "G36", "G37", "G40", "G41", "G43"], ["330", "331", "332", "333",
                     "334", "335", "340", "341", "345"]),
    ("hypothyroid", ["E03"], ["244"]),
    ("rheumatoid", ["M05", "M06", "M32", "M33", "M34"], ["714"]),
    ("peptic_ulcer", ["K25", "K26", "K27", "K28"], ["531", "532", "533", "534"]),
    ("hiv", ["B20", "B21", "B22", "B23", "B24"], ["042", "043", "044"]),
]


def main():
    t0 = time.time()
    sql = """
    SELECT s.stay_id, d.icd_code, d.icd_version
    FROM mimiciv_hosp.diagnoses_icd d
    JOIN (SELECT DISTINCT hadm_id FROM mimiciv_icu.icustays) h USING (hadm_id)
    JOIN (SELECT DISTINCT ON (stay_id) stay_id, hadm_id FROM mimiciv_icu.icustays
          ORDER BY stay_id, intime) s USING (hadm_id)
    """
    with psycopg.connect(DSN) as c:
        dx = pl.read_database(sql, c)
    print(f"diagnoses rows: {dx.height} ({time.time()-t0:.1f}s)", flush=True)
    dx = dx.with_columns(pl.col("icd_code").cast(pl.Utf8).str.replace_all(r"\.", "").str.to_uppercase())

    flags = []
    for name, p10, p9 in GROUPS:
        cond = pl.lit(False)
        for p in p10:
            cond = cond | ((pl.col("icd_version") == 10) & pl.col("icd_code").str.starts_with(p))
        for p in p9:
            cond = cond | ((pl.col("icd_version") == 9) & pl.col("icd_code").str.starts_with(p))
        f = (dx.filter(cond).select(pl.col("stay_id").unique().alias("stay_id"))
               .with_columns(pl.lit(1, dtype=pl.Int8).alias(f"hx_{name}")))
        flags.append(f)

    stays = pl.read_parquet(CACHE / "stays.parquet").select(["stay_id"])
    out = stays
    for f in flags:
        out = out.join(f, on="stay_id", how="left")
    namecol = [c for c in out.columns if c != "stay_id"]
    out = out.with_columns([pl.col(c).fill_null(0).cast(pl.Int8) for c in namecol])

    # a couple of extra static descriptors kept for parity with the old feature set
    out = out.with_columns(
        pl.sum_horizontal([pl.col(c) for c in namecol]).cast(pl.Int8).alias("hx_n_groups"))
    out.write_parquet(CACHE / "static_extra.parquet")
    print(f"[{time.strftime('%H:%M:%S')}] static_extra {out.shape} ({time.time()-t0:.1f}s)")
    print(json.dumps({c: round(float(out[c].mean()), 3) for c in namecol[:10]}, indent=1))


if __name__ == "__main__":
    main()
