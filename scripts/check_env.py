"""Environment check without requiring PostgreSQL or MIMIC."""

from __future__ import annotations

import sys


def main() -> int:
    ok = True
    print(f"Python: {sys.version.split()[0]}")

    try:
        import sqlalchemy

        print(f"SQLAlchemy: {sqlalchemy.__version__}")
    except ImportError:
        print("SQLAlchemy: NOT INSTALLED")
        ok = False

    try:
        import polars

        print(f"Polars: {polars.__version__}")
    except ImportError:
        print("Polars: NOT INSTALLED")
        ok = False

    try:
        from infra.config import get_settings, load_yaml

        s = get_settings()
        print(f"DB URL (default): {s.database_url}")
        labels = load_yaml("labels.yaml")
        if labels:
            print(f"Primary label: {labels.get('primary', {})}")
    except Exception as e:
        print(f"Project config: FAIL ({e})")
        ok = False

    try:
        from infra.db import check_connection

        check_connection()
        print("PostgreSQL: CONNECTED")
    except Exception as e:
        print(f"PostgreSQL: NOT READY ({e.__class__.__name__})")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
