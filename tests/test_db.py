def test_postgres_connection():
    from infra.db import check_connection

    assert check_connection() is True


def test_mock_icustays_readable():
    from sqlalchemy import text
    from infra.db import get_engine

    with get_engine().connect() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM mock.icustays")).scalar_one()
    assert n >= 1
