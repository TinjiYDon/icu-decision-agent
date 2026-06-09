from __future__ import annotations

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from infra.config import get_settings


def get_engine() -> Engine:
    return create_engine(get_settings().database_url, pool_pre_ping=True)


def check_connection() -> bool:
    with get_engine().connect() as conn:
        conn.execute(text("SELECT 1"))
    return True
