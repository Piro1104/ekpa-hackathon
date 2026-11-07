"""Settings file (config/settings.json) and its use for the extra route cost."""
import json

import pandas as pd

from rafi import settings
from rafi.suggestions import MARGINAL, URGENT, delivery_plan

TODAY = pd.Timestamp("2026-11-01")


def test_missing_file_is_created_with_defaults(tmp_path):
    path = tmp_path / "config" / "settings.json"
    values = settings.load_settings(path)
    assert values == settings.DEFAULTS
    assert json.loads(path.read_text(encoding="utf-8")) == settings.DEFAULTS
    assert values["extra_route_cost_eur"] == 60 and values["stock_cover_weeks"] == 4
    assert values["safety_weeks"] == 1 and values["after_approval_mode"] == settings.DRAFT_ONLY


def test_missing_keys_are_filled_and_others_kept(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text('{"extra_route_cost_eur": 85}', encoding="utf-8")
    values = settings.load_settings(path)
    assert values["extra_route_cost_eur"] == 85 and values["safety_weeks"] == 1
    assert json.loads(path.read_text(encoding="utf-8"))["stock_cover_weeks"] == 4


def test_broken_file_falls_back_to_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("not json", encoding="utf-8")
    assert settings.extra_route_cost_eur(path) == 60


def test_save_keeps_unknown_keys_out_of_the_way(tmp_path):
    path = tmp_path / "settings.json"
    settings.save_settings({"extra_route_cost_eur": 40,
                            "after_approval_mode": settings.DRAFT_AND_MESSAGE}, path)
    values = settings.load_settings(path)
    assert values["extra_route_cost_eur"] == 40
    assert values["after_approval_mode"] == settings.DRAFT_AND_MESSAGE
    assert values["stock_cover_weeks"] == 4


def test_extra_route_cost_comes_from_the_settings_file(monkeypatch, tmp_path):
    path = tmp_path / "settings.json"
    monkeypatch.setenv("RAFI_SETTINGS", str(path))
    wed, dep = [2], pd.Timestamp("2026-11-02")
    # 59 EUR lost: below the default 60 -> wait for the route (ΟΡΙΑΚΟ)
    _, _, _, flags = delivery_plan(dep, wed, TODAY, lost_eur=59)
    assert flags == [MARGINAL]
    # cost lowered to 50 in the settings -> the same loss now pays for an extra run
    settings.save_settings({"extra_route_cost_eur": 50}, path)
    _, _, _, flags = delivery_plan(dep, wed, TODAY, lost_eur=59)
    assert URGENT in flags and MARGINAL not in flags
    # an explicit cost wins over the file
    _, _, _, flags = delivery_plan(dep, wed, TODAY, lost_eur=59, route_cost=100)
    assert flags == [MARGINAL]

