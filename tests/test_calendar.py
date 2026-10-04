from datetime import date

from hlzf.calendar import DayRules, day_info, holiday_map


def test_weekend_is_off_peak():
    assert day_info(date(2026, 1, 10), DayRules(states=["BY"])).off_peak is True


def test_state_specific_holidays():
    epiphany = date(2026, 1, 6)
    assert day_info(epiphany, DayRules(states=["BY"])).kind == "holiday"
    assert day_info(epiphany, DayRules(states=["NW"])).kind == "workday"


def test_assumption_day_depends_on_municipality_in_bavaria():
    d = date(2025, 8, 15)  # a Friday; in 2026 it falls on a Saturday
    assert day_info(d, DayRules(states=["BY"])).off_peak is None
    # N-ERGIE style: only holidays valid in the whole BY+BW area, explicitly not Assumption
    rules = DayRules(states=["BY", "BW"], holiday_mode="intersection",
                     holiday_exclude=["Maria Himmelfahrt"])
    assert day_info(d, rules).off_peak is False
    # an explicit exclusion also settles it for a Bavaria-only operator
    rules = DayRules(states=["BY"], holiday_exclude=["Mariä Himmelfahrt"])
    assert day_info(d, rules).off_peak is False


def test_intersection_keeps_shared_holidays():
    hol, maybe = holiday_map(DayRules(states=["BY", "BW"], holiday_mode="intersection"), 2026)
    assert date(2026, 6, 4) in hol  # Fronleichnam: both states
    assert not maybe


def test_bridge_day_variants():
    monday = date(2026, 1, 5)  # between the weekend and Epiphany (Tue)
    assert day_info(monday, DayRules(states=["BY"], bridge="working_day")).off_peak is False
    assert day_info(monday, DayRules(states=["BY"], bridge="off_peak")).off_peak is True
    assert day_info(monday, DayRules(states=["BY"], bridge="off_peak_max_one")).off_peak \
        is None
    assert day_info(monday, DayRules(states=["BY"])).off_peak is None  # silent document
    assert day_info(monday, DayRules(states=["NW"])).kind == "workday"  # no holiday next day
    assert day_info(date(2026, 6, 5), DayRules(states=["BY"], bridge="off_peak")).kind \
        == "bridge"


def test_christmas_period():
    explicit = DayRules(states=["BY"], christmas=("12-24", "01-01"))
    assert day_info(date(2026, 12, 24), explicit).off_peak is True
    assert day_info(date(2026, 12, 31), explicit).off_peak is True
    vague = DayRules(states=["BY"], christmas=None)
    assert day_info(date(2026, 12, 28), vague).off_peak is True
    assert day_info(date(2026, 12, 24), vague).off_peak is None
    assert day_info(date(2026, 12, 31), vague).off_peak is None
    assert day_info(date(2026, 12, 23), vague).off_peak is False


def test_nationwide_only_holidays():
    # Schweinfurt: only "bundeseinheitliche Feiertage" are off-peak, so Epiphany is a workday
    rules = DayRules(states=["DE"])
    assert day_info(date(2026, 1, 6), rules).kind == "workday"
    assert day_info(date(2026, 5, 14), rules).kind == "holiday"  # Christi Himmelfahrt
