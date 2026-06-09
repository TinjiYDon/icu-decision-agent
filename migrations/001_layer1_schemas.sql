-- Layer 1 schema（P0）项目二 icu_decision
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS feat;
CREATE SCHEMA IF NOT EXISTS label;
CREATE SCHEMA IF NOT EXISTS model;
CREATE SCHEMA IF NOT EXISTS app;

GRANT USAGE ON SCHEMA staging, feat, label, model, app TO icu_dev;
GRANT ALL ON ALL TABLES IN SCHEMA staging, feat, label, model, app TO icu_dev;
ALTER DEFAULT PRIVILEGES IN SCHEMA staging GRANT ALL ON TABLES TO icu_dev;
ALTER DEFAULT PRIVILEGES IN SCHEMA feat GRANT ALL ON TABLES TO icu_dev;
ALTER DEFAULT PRIVILEGES IN SCHEMA label GRANT ALL ON TABLES TO icu_dev;
ALTER DEFAULT PRIVILEGES IN SCHEMA model GRANT ALL ON TABLES TO icu_dev;
ALTER DEFAULT PRIVILEGES IN SCHEMA app GRANT ALL ON TABLES TO icu_dev;

-- P0 占位：ETL 完成后填充
CREATE TABLE IF NOT EXISTS feat.sample_matrix (
    stay_id     BIGINT NOT NULL,
    hour_index  INT NOT NULL,
    feature_json JSONB,
    PRIMARY KEY (stay_id, hour_index)
);

CREATE TABLE IF NOT EXISTS label.mortality_12h (
    stay_id     BIGINT NOT NULL,
    hour_index  INT NOT NULL,
    label       SMALLINT NOT NULL,
    PRIMARY KEY (stay_id, hour_index)
);

CREATE TABLE IF NOT EXISTS model.registry (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    version     TEXT NOT NULL,
    path        TEXT NOT NULL,
    metrics     JSONB,
    created_at  TIMESTAMPTZ DEFAULT now()
);
