"""Load filtered MIMIC-IV data into mimic_iv database using E: drive tablespace.

Only loads the 8 itemids needed by GRU-D to keep the import fast and small.
Uses COPY for bulk loading instead of individual INSERTs.
"""
import gzip, csv, os, sys
import psycopg

SRC = r"E:\mimic-iv-3.1"
DSN_APP = "host=localhost dbname=mimic_iv user=icu_dev password=lewis790919"
DSN_SUPER = "host=localhost dbname=mimic_iv user=postgres password=lewis790919"

# 8 itemids needed by predict_grud.py
TARGET_ITEMIDS = {220045, 220179, 50813, 50912, 220210, 223761, 220277, 51006}


def create_schemas_and_tables(super_conn):
    with super_conn.cursor() as cur:
        cur.execute("CREATE SCHEMA IF NOT EXISTS mimiciv_icu")
        cur.execute("CREATE SCHEMA IF NOT EXISTS mimiciv_hosp")
        cur.execute("GRANT ALL ON SCHEMA mimiciv_icu TO icu_dev")
        cur.execute("GRANT ALL ON SCHEMA mimiciv_hosp TO icu_dev")

        # icustays - small table, default tablespace
        cur.execute("""
            DROP TABLE IF EXISTS mimiciv_icu.icustays CASCADE;
            CREATE TABLE mimiciv_icu.icustays (
                subject_id     INT,
                hadm_id        INT,
                stay_id        BIGINT PRIMARY KEY,
                first_careunit TEXT,
                last_careunit  TEXT,
                intime         TIMESTAMPTZ,
                outtime        TIMESTAMPTZ,
                los            DOUBLE PRECISION
            );
        """)

        # chartevents - large table, use E: tablespace
        cur.execute("""
            DROP TABLE IF EXISTS mimiciv_icu.chartevents CASCADE;
            CREATE TABLE mimiciv_icu.chartevents (
                subject_id     INT,
                hadm_id        INT,
                stay_id        BIGINT,
                caregiver_id   BIGINT,
                charttime      TIMESTAMPTZ,
                storetime      TIMESTAMPTZ,
                itemid         INT,
                value          TEXT,
                valuenum       DOUBLE PRECISION,
                valueuom       TEXT,
                warning        INTEGER
            ) TABLESPACE ts_mimic_e;
            CREATE INDEX idx_chart_stay ON mimiciv_icu.chartevents(stay_id) TABLESPACE ts_mimic_e;
            CREATE INDEX idx_chart_item ON mimiciv_icu.chartevents(itemid) TABLESPACE ts_mimic_e;
        """)

        # labevents - large table, use E: tablespace
        cur.execute("""
            DROP TABLE IF EXISTS mimiciv_hosp.labevents CASCADE;
            CREATE TABLE mimiciv_hosp.labevents (
                labevent_id      BIGINT PRIMARY KEY,
                subject_id       INT,
                hadm_id          INT,
                specimen_id      INT,
                itemid           INT,
                order_provider_id TEXT,
                charttime        TIMESTAMPTZ,
                storetime        TIMESTAMPTZ,
                value            TEXT,
                valuenum         DOUBLE PRECISION,
                valueuom         TEXT,
                ref_range_lower  DOUBLE PRECISION,
                ref_range_upper  DOUBLE PRECISION,
                flag             TEXT,
                priority         TEXT,
                comments         TEXT
            ) TABLESPACE ts_mimic_e;
            CREATE INDEX idx_lab_hadm ON mimiciv_hosp.labevents(hadm_id) TABLESPACE ts_mimic_e;
            CREATE INDEX idx_lab_item ON mimiciv_hosp.labevents(itemid) TABLESPACE ts_mimic_e;
        """)

        cur.execute("GRANT ALL ON ALL TABLES IN SCHEMA mimiciv_icu TO icu_dev")
        cur.execute("GRANT ALL ON ALL TABLES IN SCHEMA mimiciv_hosp TO icu_dev")
        cur.execute("GRANT ALL ON ALL SEQUENCES IN SCHEMA mimiciv_icu TO icu_dev")
        cur.execute("GRANT ALL ON ALL SEQUENCES IN SCHEMA mimiciv_hosp TO icu_dev")
    super_conn.commit()
    print("Schema + tables created.")


def load_icustays(conn):
    path = os.path.join(SRC, "icu", "icustays.csv.gz")
    count = 0
    with conn.cursor() as cur:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cur.execute(
                    """INSERT INTO mimiciv_icu.icustays
                       (subject_id, hadm_id, stay_id, first_careunit, last_careunit, intime, outtime, los)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                    (
                        int(row["subject_id"]),
                        int(row["hadm_id"]),
                        int(row["stay_id"]),
                        row["first_careunit"],
                        row["last_careunit"],
                        row["intime"] if row.get("intime") else None,
                        row["outtime"] if row.get("outtime") else None,
                        float(row["los"]) if row.get("los") else None,
                    ),
                )
                count += 1
                if count % 100000 == 0:
                    conn.commit()
                    print(f"  icustays {count:,}")
        conn.commit()
    print(f"icustays loaded: {count:,}")


def load_filtered_csv(conn, table_schema_table, csv_path):
    """Load a gzipped CSV filtering by itemid set, using batched INSERTs."""
    parts = table_schema_table.split(".")
    schema, table = parts[0], parts[1]

    # Get table columns
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position
        """, (schema, table))
        cols = [r[0] for r in cur.fetchall()]
        placeholders = ",".join(["%s"] * len(cols))
        col_list = ",".join(cols)
        insert_sql = f"INSERT INTO {table_schema_table} ({col_list}) VALUES ({placeholders})"

    total_loaded = 0
    total_skipped = 0
    batch = []

    with gzip.open(csv_path, "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                itemid = int(row.get("itemid", ""))
            except (ValueError, TypeError):
                total_skipped += 1
                continue
            if itemid not in TARGET_ITEMIDS:
                total_skipped += 1
                continue

            vals = []
            for col in cols:
                v = row.get(col)
                if v is None or v == "":
                    vals.append(None)
                else:
                    vals.append(v)
            batch.append(tuple(vals))
            total_loaded += 1

            if len(batch) >= 5000:
                with conn.cursor() as cur:
                    cur.executemany(insert_sql, batch)
                conn.commit()
                batch = []
                if total_loaded % 50000 == 0:
                    print(f"  {table} {total_loaded:,} rows (skipped {total_skipped:,})")

    if batch:
        with conn.cursor() as cur:
            cur.executemany(insert_sql, batch)
        conn.commit()

    print(f"{table} loaded: {total_loaded:,} (skipped {total_skipped:,})")
    return total_loaded


def verify(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM mimiciv_icu.icustays")
        print(f"  icustays: {cur.fetchone()[0]:,}")
        cur.execute("SELECT COUNT(*) FROM mimiciv_icu.chartevents")
        print(f"  chartevents: {cur.fetchone()[0]:,}")
        cur.execute("SELECT COUNT(*) FROM mimiciv_hosp.labevents")
        print(f"  labevents: {cur.fetchone()[0]:,}")
        # How many stays have chart data
        cur.execute("SELECT COUNT(DISTINCT stay_id) FROM mimiciv_icu.chartevents")
        print(f"  stays with chartevents: {cur.fetchone()[0]:,}")
        # How many stays have lab data via hadm_id join
        cur.execute("""
            SELECT COUNT(DISTINCT i.stay_id)
            FROM mimiciv_icu.icustays i
            JOIN mimiciv_hosp.labevents l ON i.hadm_id = l.hadm_id
        """)
        print(f"  stays with labevents (via hadm): {cur.fetchone()[0]:,}")


if __name__ == "__main__":
    # Phase 1: superuser - create schemas and tables
    print("Phase 1: Creating schemas and tables as postgres...")
    super_conn = psycopg.connect(DSN_SUPER)
    try:
        create_schemas_and_tables(super_conn)
    finally:
        super_conn.close()

    # Phase 2: app user - load data
    print("\nPhase 2: Loading data as icu_dev...")
    conn = psycopg.connect(DSN_APP)
    try:
        print("Loading icustays...")
        load_icustays(conn)

        print("\nLoading chartevents (filtered)...")
        load_filtered_csv(
            conn,
            "mimiciv_icu.chartevents",
            os.path.join(SRC, "icu", "chartevents.csv.gz"),
        )

        print("\nLoading labevents (filtered)...")
        load_filtered_csv(
            conn,
            "mimiciv_hosp.labevents",
            os.path.join(SRC, "hosp", "labevents.csv.gz"),
        )

        print("\n=== Verification ===")
        verify(conn)
        print("\nDone!")
    finally:
        conn.close()
