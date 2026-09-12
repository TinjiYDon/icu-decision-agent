"""Test L4 UI query APIs."""

from application.ui_queries import list_all_stay_ids, list_demo_stays, list_hours_for_stay


def test_list_demo_stays_returns_list():
    result = list_demo_stays(limit=10)
    assert isinstance(result, list)
    if result:
        assert "stay_id" in result[0]
        assert "los_hours" in result[0]
        assert "first_hour" in result[0]
        assert isinstance(result[0]["stay_id"], int)
        assert isinstance(result[0]["los_hours"], float)
        assert isinstance(result[0]["first_hour"], int)


def test_list_demo_stays_respects_limit():
    result = list_demo_stays(limit=5)
    assert len(result) <= 5


def test_list_hours_for_stay_returns_list():
    stays = list_demo_stays(limit=1)
    if stays:
        sid = stays[0]["stay_id"]
        hours = list_hours_for_stay(sid)
        assert isinstance(hours, list)
        if hours:
            assert all(isinstance(h, int) for h in hours)
            assert hours == sorted(hours)


def test_list_all_stay_ids_returns_list():
    result = list_all_stay_ids(limit=10)
    assert isinstance(result, list)
    if result:
        assert all(isinstance(sid, int) for sid in result)
        assert result == sorted(result)


def test_list_all_stay_ids_respects_limit():
    result = list_all_stay_ids(limit=3)
    assert len(result) <= 3
