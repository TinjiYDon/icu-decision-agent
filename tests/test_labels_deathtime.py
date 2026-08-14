"""Unit tests for deathtime-aware mortality window logic."""

from datetime import date, datetime

from domain.labels.mortality_12h import death_within_window, resolve_death_time


def test_prefer_deathtime_over_dod():
    row = {
        "deathtime": datetime(2020, 1, 1, 15, 0, 0),
        "dod": date(2020, 1, 1),
    }
    death, src = resolve_death_time(row)
    assert src == "deathtime"
    assert death == row["deathtime"]


def test_precise_deathtime_inside_12h_window():
    t0 = datetime(2020, 1, 1, 10, 0, 0)
    death = datetime(2020, 1, 1, 15, 0, 0)
    assert death_within_window(t0, t0.replace(hour=22), death) == 1


def test_precise_deathtime_outside_window():
    t0 = datetime(2020, 1, 1, 10, 0, 0)
    death = datetime(2020, 1, 2, 10, 0, 0)
    assert death_within_window(t0, datetime(2020, 1, 1, 22, 0, 0), death) == 0


def test_date_dod_full_day_overlap():
    t0 = datetime(2020, 1, 1, 20, 0, 0)
    # date-level dod on 2020-01-01 overlaps evening window
    assert death_within_window(t0, datetime(2020, 1, 2, 8, 0, 0), date(2020, 1, 1)) == 1
