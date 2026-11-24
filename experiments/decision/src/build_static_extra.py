"""Static comorbidity / history flags from `paper_feat.sample_matrix`.

The old pipeline's `elix_*` comorbidity flags are static per stay; pull them once
(hour_index = 0 row per stay) into a small table used as static features.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import polars as pl
import psycopg

OUT = Path(__file__).resolve().parents[1]
CACHE = OUT / "cache"
DSN = __import__("os").environ["ICU_DECISION_DSN"]

KEYS = None


def main():
    t0 = time.time()
    with psycopg.connect(DSN) as c:
        with c.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT key FROM paper_feat.sample_matrix s,
                     jsonb_each_text(s.feature_json) AS t(key, value)
                WHERE key LIKE 'elix%%' OR key LIKE 'has_%%'
            """)
            keys = [r[0] for r in cur.fetchall()]
    print("keys:", len(keys), keys[:5], flush=True)
    exprs = ", ".join(f"(feature_json->>'{k}')::double precision AS \"{k}\"" for k in keys)
    sql = f"""SELECT DISTINCT ON (stay_id) stay_id, {exprs}
              FROM paper_feat.sample_matrix WHERE hour_index = 0 ORDER BY stay_id"""
    with psycopg.connect(DSN) as c:
        df = pl.read_database(sql, c)
    # rename elix_* -> hx_* (comorbidity history) to keep the static prefix scheme
    ren = {k: ("hx_" + k[5:] if k.startswith("elix_") else "hx_" + k[4:] if k.startswith("has_") else k)
           for k in keys}
    df = df.rename(ren)
    df.write_parquet(CACHE / "static_extra.parquet")
    print(f"[{time.strftime('%H:%M:%S')}] static_extra {df.shape} ({time.time()-t0:.1f}s)", flush=True)
    print(json.dumps({c: int(df[c].null_count()) for c in df.columns[:6]}, indent=1))


if __name__ == "__main__":
    main()
