-- Mock schema for icu_scheduling (connect: icu_scheduling)
CREATE SCHEMA IF NOT EXISTS mock;

CREATE TABLE IF NOT EXISTS mock.icustays (
    stay_id   INTEGER PRIMARY KEY,
    subject_id INTEGER NOT NULL,
    intime    TIMESTAMP NOT NULL,
    outtime   TIMESTAMP
);

CREATE TABLE IF NOT EXISTS mock.transfers (
    stay_id   INTEGER NOT NULL,
    eventtype TEXT,
    careunit  TEXT,
    intime    TIMESTAMP,
    outtime   TIMESTAMP
);

INSERT INTO mock.icustays (stay_id, subject_id, intime, outtime)
VALUES (1, 100, '2020-01-01 08:00', '2020-01-03 10:00'),
       (2, 101, '2020-01-01 12:00', NULL)
ON CONFLICT (stay_id) DO NOTHING;

GRANT USAGE ON SCHEMA mock TO icu_dev;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA mock TO icu_dev;
ALTER DEFAULT PRIVILEGES IN SCHEMA mock GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO icu_dev;
