-- Mock schema for icu_decision (connect: icu_decision)
CREATE SCHEMA IF NOT EXISTS mock;

CREATE TABLE IF NOT EXISTS mock.icustays (
    stay_id    INTEGER PRIMARY KEY,
    subject_id INTEGER NOT NULL,
    intime     TIMESTAMP NOT NULL,
    outtime    TIMESTAMP,
    hospital_expire_flag SMALLINT DEFAULT 0
);

INSERT INTO mock.icustays (stay_id, subject_id, intime, outtime, hospital_expire_flag)
VALUES (1, 100, '2020-01-01 08:00', '2020-01-02 20:00', 1),
       (2, 101, '2020-01-01 12:00', '2020-01-05 08:00', 0)
ON CONFLICT (stay_id) DO NOTHING;

GRANT USAGE ON SCHEMA mock TO icu_dev;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA mock TO icu_dev;
ALTER DEFAULT PRIVILEGES IN SCHEMA mock GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO icu_dev;
