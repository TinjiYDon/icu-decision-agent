-- P0：仅两个 Layer 0 库 + 两个 Layer 1 项目库
-- Docker 首次启动 / 手动：psql -U postgres -f scripts/init_databases_p0.sql

CREATE ROLE icu_dev LOGIN PASSWORD 'icu_dev';

CREATE DATABASE icu_scheduling OWNER icu_dev;
CREATE DATABASE icu_decision OWNER icu_dev;
CREATE DATABASE mimic_iv_demo OWNER icu_dev;
CREATE DATABASE mimic_iv OWNER icu_dev;

GRANT ALL PRIVILEGES ON DATABASE icu_scheduling TO icu_dev;
GRANT ALL PRIVILEGES ON DATABASE icu_decision TO icu_dev;
GRANT ALL PRIVILEGES ON DATABASE mimic_iv_demo TO icu_dev;
GRANT ALL PRIVILEGES ON DATABASE mimic_iv TO icu_dev;
