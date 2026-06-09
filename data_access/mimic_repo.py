"""Read ICU stays from mock or Layer0 MIMIC (P0 stub)."""

from __future__ import annotations

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from infra.config import get_data_source, get_layer0_dsn, get_settings


def _mock_engine() -> Engine:
    return create_engine(get_settings().database_url, pool_pre_ping=True)


def _layer0_engine() -> Engine | None:
    dsn = get_layer0_dsn()
    if not dsn:
        return None
    return create_engine(dsn, pool_pre_ping=True)


def count_icustays() -> int:
    source = get_data_source()
    if source == "mock":
        with _mock_engine().connect() as conn:
            return conn.execute(text("SELECT COUNT(*) FROM mock.icustays")).scalar_one()
    engine = _layer0_engine()
    if engine is None:
        raise RuntimeError("layer0 DSN not configured; set configs/database.yaml")
    with engine.connect() as conn:
        return conn.execute(text("SELECT COUNT(*) FROM icu.icustays")).scalar_one()
