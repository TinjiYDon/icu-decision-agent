-- P0 staging：从 Layer0 同步 icustays
CREATE TABLE IF NOT EXISTS staging.icustays (
    stay_id         BIGINT PRIMARY KEY,
    subject_id      BIGINT NOT NULL,
    hadm_id         BIGINT NOT NULL,
    first_careunit  TEXT,
    last_careunit   TEXT,
    intime          TIMESTAMPTZ,
    outtime         TIMESTAMPTZ,
    los_hours       REAL
);
