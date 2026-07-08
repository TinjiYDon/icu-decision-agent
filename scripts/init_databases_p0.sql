-- P0: icu_decision (Layer1) + Layer0 demo/full databases
-- Run: psql -U postgres -f scripts/init_databases_p0.sql

CREATE ROLE icu_dev LOGIN PASSWORD 'icu_dev';

CREATE DATABASE icu_decision OWNER icu_dev;
CREATE DATABASE mimic_iv_demo OWNER icu_dev;
CREATE DATABASE mimic_iv OWNER icu_dev;

GRANT ALL PRIVILEGES ON DATABASE icu_decision TO icu_dev;
GRANT ALL PRIVILEGES ON DATABASE mimic_iv_demo TO icu_dev;
GRANT ALL PRIVILEGES ON DATABASE mimic_iv TO icu_dev;
