import pytest

from hlzf.models import Season
from hlzf.textnorm import map_level, map_season, match_key, parse_time, time_ranges_in


@pytest.mark.parametrize("label,expected", [
    ("Höchstspannung", "NE1"),
    ("HöS/HS", "NE2"),
    ("Umspannung in Hochspannung", "NE2"),
    ("Hochspannung HS", "NE3"),
    ("Hochspannung (HS)", "NE3"),
    ("Umspannung in Mittelspannung HS/MS", "NE4"),  # N-ERGIE
    ("Umspannung HS/MS", "NE4"),
    ("Umspannung Hoch-/Mittelspannung", "NE4"),
    ("Mittelspannung", "NE5"),
    ("MS", "NE5"),
    ("NETZEBENE 5", "NE5"),  # Wunsiedel
    ("Netzebene 6 (MS/NS)", "NE6"),
    ("Umspannung Mittel-/Niederspannung", "NE6"),
    ("Umspannung MS/NS", "NE6"),
    ("Niederspannung (NS)", "NE7"),
    ("Entnahme aus der Niederspannung", "NE7"),
    ("Level 4 (HS/MS)", "NE4"),
])
def test_map_level(label, expected):
    assert map_level(label) == expected


def test_map_level_unknown():
    assert map_level("Fernwärme") is None


@pytest.mark.parametrize("label,expected", [
    ("Winter", Season.winter),
    ("Winter Januar, Februar und Dezember", Season.winter),
    ("Dez. - Feb.", Season.winter),
    ("Frühling März bis Mai", Season.spring),
    ("Fruehjahr", Season.spring),
    ("Sommer Juni bis August", Season.summer),
    ("Herbst September bis November", Season.autumn),
])
def test_map_season(label, expected):
    assert map_season(label) is expected


def test_parse_time():
    assert parse_time("07:45") == 465
    assert parse_time("7.45") == 465
    assert parse_time("24:00") == 1440
    assert parse_time("08:00 Uhr") == 480
    assert parse_time("24:15") is None
    assert parse_time("7:61") is None
    assert parse_time("abc") is None


def test_time_ranges_in_quotes():
    assert time_ranges_in("07:45 – 09:00 Uhr") == [(465, 540)]
    assert time_ranges_in("12:15 – 13:15 Uhr; 15:00 – 18:15 Uhr") == [(735, 795), (900, 1095)]
    assert time_ranges_in("von 8.30 bis 13.45 Uhr") == [(510, 825)]
    assert time_ranges_in("Stand: 31.10.2025") == []


def test_match_key_unifies_dashes_and_clock_formats():
    assert match_key("07:45 – 09:00 Uhr") == match_key("7.45 - 09:00")


def test_dotenv_inline_comments(tmp_path, monkeypatch):
    import os

    from hlzf.config import load_settings

    monkeypatch.setattr(os, "environ", dict(os.environ))  # keep .env values out of other tests

    (tmp_path / ".env").write_text(
        "GLM_EXTRACT_MODEL=glm-5.3   # text-only\nLLM_BUDGET_USD='2.5'\nHLZF_OFFLINE=1\n")
    for k in ("GLM_EXTRACT_MODEL", "LLM_BUDGET_USD", "HLZF_OFFLINE", "ZAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    s = load_settings(root=tmp_path)
    assert s.extract_model == "glm-5.3" and s.budget_usd == 2.5 and s.offline and not s.live
