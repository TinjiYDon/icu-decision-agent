"""Verify risk distribution across many stays after train/inference alignment."""
import sys, os
sys.path.insert(0, r"c:\Users\lewis\Desktop\icu-decision-agent-main (5)\icu-decision-agent-main")
os.chdir(r"c:\Users\lewis\Desktop\icu-decision-agent-main (5)\icu-decision-agent-main")

from application.predict_grud import predict_grud
import psycopg

conn = psycopg.connect("host=localhost dbname=mimic_iv user=icu_dev password=lewis790919")
with conn.cursor() as cur:
    cur.execute("""
        SELECT i.stay_id FROM mimiciv_icu.icustays i
        JOIN mimiciv_icu.chartevents c ON i.stay_id = c.stay_id
        WHERE c.itemid IN (220045,220179,220210,220277,223761)
        GROUP BY i.stay_id ORDER BY RANDOM() LIMIT 15
    """)
    sids = [int(r[0]) for r in cur.fetchall()]
conn.close()

risks = []
for sid in sids:
    r = predict_grud(sid)
    if r.get("status") == "ok":
        risk = r["risk_score"]
        risks.append(risk)
        print(f"stay_id={sid}: risk={risk:.4f} ({risk*100:.1f}%)")
    else:
        print(f"stay_id={sid}: status={r['status']}")

if risks:
    import numpy as np
    print(f"\nn={len(risks)}, mean={np.mean(risks):.4f}, median={np.median(risks):.4f}, "
          f"min={min(risks):.4f}, max={max(risks):.4f}, >90%: {sum(1 for x in risks if x>0.9)}/{len(risks)}")
