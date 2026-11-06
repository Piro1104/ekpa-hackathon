"""Tests for rafi.features: working days, route days, seasonality, winter closures."""
from pathlib import Path

import pandas as pd
import pytest

from rafi.features import (likely_closer, next_working_day, route_days, seasonal_factors,
                           winter_closers, winter_history, working_day)
from rafi.ingest import build_deliveries, load_from_files

DATA = Path(__file__).resolve().parents[1] / "data"
CUTOFF = pd.Timestamp("2026-10-31")
D = pd.Timestamp


@pytest.fixture(scope="module")
def data():
    headers, lines, items, customers = load_from_files(DATA)
    return build_deliveries(headers, lines, items), customers, items


# --- working days ----------------------------------------------------------

def test_sunday_and_christmas_are_not_working_days():
    assert not working_day("2026-11-01")   # Sunday
    assert not working_day("2026-12-25")   # Christmas
    assert working_day("2026-11-02")       # ordinary Monday
    assert working_day("2026-11-07")       # Saturday is a delivery day


def test_next_working_day_skips_holidays_and_sunday():
    # 25/12 Friday holiday, 26/12 Saturday holiday, 27/12 Sunday -> Monday 28/12
    assert next_working_day("2026-12-24") == D("2026-12-28")
    assert next_working_day("2026-10-31") == D("2026-11-02")


# --- route days ------------------------------------------------------------

def test_route_days_keeps_weekdays_with_at_least_20_percent():
    # 4 Mondays, 4 Thursdays, 1 Wednesday (11%), all for customer 1
    dates = (list(pd.date_range("2026-09-07", periods=4, freq="7D"))      # Mondays
             + list(pd.date_range("2026-09-10", periods=4, freq="7D"))    # Thursdays
             + [D("2026-09-16")])                                         # one Wednesday
    deliveries = pd.DataFrame({"TRDR": 1, "MTRL": 1, "TRNDATE": dates, "QTY": 5,
                               "IS_CREDIT": False})
    future = pd.DataFrame({"TRDR": [1] * 5, "MTRL": 1, "TRNDATE": [D("2026-11-04")] * 5,
                           "QTY": 5, "IS_CREDIT": False})  # Wednesdays after the cutoff
    assert route_days(deliveries, CUTOFF) == {1: [0, 3]}
    assert route_days(pd.concat([deliveries, future]), CUTOFF) == {1: [0, 3]}


# --- seasonality -----------------------------------------------------------

def test_seasonal_factors_average_one_for_every_combination(data):
    deliveries, customers, items = data
    for cutoff in ["2025-10-31", "2026-06-30", "2026-10-31"]:
        f = seasonal_factors(deliveries, customers, items, cutoff)
        counts = f.groupby(["AREA", "MTRCATEGORY"]).size()
        assert (counts == 12).all()
        means = f.groupby(["AREA", "MTRCATEGORY"])["FACTOR"].mean()
        assert means.round(9).eq(1).all()


def test_seasonal_factors_crete_summer_much_higher_than_november(data):
    deliveries, customers, items = data
    f = seasonal_factors(deliveries, customers, items, CUTOFF)
    crete = f[(f["AREA"] == "Κρήτη") & (f["MTRCATEGORY"] == "Νερά")].set_index("MONTH")["FACTOR"]
    assert crete[7] > 4 * crete[11]


def test_seasonal_factors_ignore_the_future(data):
    deliveries, customers, items = data
    cutoff = D("2025-10-31")
    a = seasonal_factors(deliveries, customers, items, cutoff)
    b = seasonal_factors(deliveries[deliveries["TRNDATE"] <= cutoff], customers, items, cutoff)
    pd.testing.assert_frame_equal(a, b)


# --- winter closures -------------------------------------------------------

EXPECTED_CLOSERS = {1013, 1019, 1043, 1045, 1047, 1052, 1065, 1096, 1098, 1110,
                    1111, 1122, 1127, 1129, 1137}


def test_winter_closers_are_the_15_expected(data):
    deliveries, customers, _ = data
    w = winter_closers(deliveries, customers, CUTOFF)
    assert set(w["TRDR"]) == EXPECTED_CLOSERS
    counts = w.groupby(["AREA", "TRDCATEGORY"]).size().to_dict()
    assert counts == {("Κρήτη", "Καφέ-Μπαρ"): 7, ("Κρήτη", "Ξενοδοχείο"): 4,
                      ("Κυκλάδες", "Καφέ-Μπαρ"): 3, ("Κυκλάδες", "Ξενοδοχείο"): 1}
    # they stop around the end of October and come back in early April
    assert w["LAST_BEFORE_CLOSE"].between("10-15", "10-31").all()
    assert w["FIRST_AFTER_OPEN"].between("03-15", "04-30").all()


def test_winter_closers_ignore_the_future(data):
    deliveries, customers, _ = data
    cutoff = D("2025-10-31")
    a = winter_closers(deliveries, customers, cutoff)
    b = winter_closers(deliveries[deliveries["TRNDATE"] <= cutoff], customers, cutoff)
    pd.testing.assert_frame_equal(a, b)


def test_likely_closer_for_new_island_customers(data):
    deliveries, customers, _ = data
    history = winter_history(deliveries, customers, CUTOFF)
    # new café-bar in the Cyclades: its peers (3 of 3) close in winter
    is_closer, last, first = likely_closer(1101, customers, history)
    assert is_closer and last.startswith("10-") and first.startswith("04-")
    # new mini market in the Cyclades: no old peers -> not assumed closed
    assert likely_closer(1078, customers, history)[0] is False
    # new café-bar in Athens
    assert likely_closer(1020, customers, history)[0] is False
