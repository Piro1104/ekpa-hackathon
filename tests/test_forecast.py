"""Tests for rafi.forecast."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from rafi.evaluate import make_targets
from rafi.features import seasonal_factors, winter_closers
from rafi.forecast import K_DELIVERIES, REASONS, forecast
from rafi.ingest import build_deliveries, daily_qty, load_from_files

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CUTOFF = pd.Timestamp("2026-10-31")
D = pd.Timestamp


@pytest.fixture(scope="module")
def data():
    headers, lines, items, customers = load_from_files(DATA)
    return build_deliveries(headers, lines, items), customers, items


@pytest.fixture(scope="module")
def official(data):
    """Forecast for the real targets at the real cutoff."""
    deliveries, customers, items = data
    past = deliveries[deliveries["TRNDATE"] <= CUTOFF]
    targets = pd.read_csv(DATA / "prediction_targets.csv", encoding="utf-8-sig")
    return forecast(daily_qty(past), past, customers, items, targets, CUTOFF)


# --- hand-made example -------------------------------------------------------

CUSTOMERS = pd.DataFrame({"TRDR": [1], "AREA": ["Α"], "TRDCATEGORY": ["Mini Market"],
                          "INSDATE": ["2020-01-01"]})
ITEMS = pd.DataFrame({"MTRL": [10], "MTRCATEGORY": ["Νερά"]})
TARGETS = pd.DataFrame({"TRDR": [1], "MTRL": [10]})


def _flat_factors(november=1.0):
    months = range(1, 13)
    return pd.DataFrame({"AREA": "Α", "MTRCATEGORY": "Νερά", "MONTH": list(months),
                         "FACTOR": [november if m == 11 else 1.0 for m in months]})


def _run(rows, november=1.0):
    deliveries = pd.DataFrame(rows, columns=["TRNDATE", "QTY"])
    deliveries["TRNDATE"] = pd.to_datetime(deliveries["TRNDATE"])
    deliveries = deliveries.assign(TRDR=1, MTRL=10, FINDOC=range(len(deliveries)),
                                   IS_CREDIT=deliveries["QTY"] < 0, QTY_FREE=0)
    return forecast(daily_qty(deliveries), deliveries, CUSTOMERS, ITEMS, TARGETS, CUTOFF,
                    factors=_flat_factors(november)).iloc[0]


def test_hand_example_constant_rate():
    # 10 boxes every 10 days -> 1 box/day. Last delivery 21/10: 15 boxes.
    # 15 boxes last until 21/10 + 15 days = 5/11. Stock at the cutoff 15 - 10 = 5.
    r = _run([("2026-10-01", 10), ("2026-10-11", 10), ("2026-10-21", 15)])
    assert r["BASE_RATE"] == pytest.approx(1.0)
    assert r["PREDICTED_DEPLETION_DATE"] == D("2026-11-05")
    assert r["STOCK_AT_CUTOFF"] == pytest.approx(5.0)
    assert r["LAST_DELIVERY_DATE"] == D("2026-10-21") and r["LAST_DELIVERY_QTY"] == 15
    assert r["N_DELIVERIES"] == 3
    assert r["SUGGESTED_QTY"] == 10        # 1 box/day x factor 1 x 10 days between deliveries
    assert r["REASON_CODE"] == "ΚΑΝΟΝΙΚΗ"
    # identical intervals (CV = 0); history = 2 intervals out of k - 1
    assert r["CONFIDENCE"] == pytest.approx(round(0.95 * min(1, 2 / (K_DELIVERIES - 1)), 2))


def test_hand_example_with_november_factor_two():
    # Same pair, but November consumption is double: 5 boxes left at the cutoff,
    # 2 per day -> 1/11: 3, 2/11: 1, 3/11: -1 -> runs out on 3/11.
    r = _run([("2026-10-01", 10), ("2026-10-11", 10), ("2026-10-21", 15)], november=2.0)
    assert r["PREDICTED_DEPLETION_DATE"] == D("2026-11-03")
    assert r["SEASONAL_FACTOR_NOW"] == 2.0
    assert r["SUGGESTED_QTY"] == 20        # 1 x 2 x 10 days


def test_hand_example_credit_after_last_delivery_reduces_stock():
    # 3 boxes returned on 25/10: 15 - 3 = 12 boxes from 21/10 -> 2/11
    r = _run([("2026-10-01", 10), ("2026-10-11", 10), ("2026-10-21", 15), ("2026-10-25", -3)])
    assert r["PREDICTED_DEPLETION_DATE"] == D("2026-11-02")


def test_hand_example_already_empty_is_cutoff_plus_one():
    r = _run([("2026-09-01", 10), ("2026-09-11", 10), ("2026-09-21", 10)])
    assert r["PREDICTED_DEPLETION_DATE"] == CUTOFF + pd.Timedelta(days=1)
    assert r["STOCK_AT_CUTOFF"] == 0
    assert r["REASON_CODE"] == "ΗΔΗ_ΕΞΑΝΤΛΗΜΕΝΟ"


def test_hand_example_beyond_horizon_is_blank():
    r = _run([("2026-07-01", 10), ("2026-07-11", 10), ("2026-10-21", 200)])
    assert pd.isna(r["PREDICTED_DEPLETION_DATE"])
    assert r["REASON_CODE"] == "ΕΚΤΟΣ_ΟΡΙΖΟΝΤΑ"


# --- no peeking ----------------------------------------------------------

def test_forecast_ignores_deliveries_after_cutoff(data):
    deliveries, customers, items = data
    cutoff = D("2025-10-31")
    past = deliveries[deliveries["TRNDATE"] <= cutoff]
    targets = make_targets(past, customers, cutoff)
    fake = targets.assign(FINDOC=-1, TRNDATE=cutoff + pd.Timedelta(days=2), SALESMAN=0,
                          QTY=999, QTY_FREE=0, IS_CREDIT=False, LINEVAL=0.0, PROMO_DISC=False)
    with_future = pd.concat([deliveries, fake], ignore_index=True)

    a = forecast(daily_qty(past), past, customers, items, targets, cutoff)
    b = forecast(daily_qty(with_future), with_future, customers, items, targets, cutoff)
    pd.testing.assert_frame_equal(a, b)


# --- winter closures, new customers, whole output ------------------------------

def test_winter_closer_in_november_is_blank(data, official):
    deliveries, customers, _ = data
    closers = set(winter_closers(deliveries, customers, CUTOFF)["TRDR"])
    rows = official[official["TRDR"].isin(closers)]
    assert len(rows) == 189
    assert rows["PREDICTED_DEPLETION_DATE"].isna().all()
    assert (rows["REASON_CODE"] == "ΚΛΕΙΣΤΟ_ΧΕΙΜΩΝΑ").all()


def test_every_new_customer_pair_gets_a_forecast(data, official):
    _, customers, _ = data
    new = customers.loc[customers["INSDATE"].str.startswith("2026"), "TRDR"]
    rows = official[official["TRDR"].isin(new)]
    assert len(rows) == 137
    assert rows["REASON_CODE"].isin(REASONS).all()
    dated = rows["PREDICTED_DEPLETION_DATE"].notna()
    # a blank must have a reason other than "no rate"
    assert (rows.loc[~dated, "REASON_CODE"] != "ΧΩΡΙΣ_ΡΥΘΜΟ").all()
    assert (rows.loc[rows["REASON_CODE"] != "ΚΛΕΙΣΤΟ_ΧΕΙΜΩΝΑ", "BASE_RATE"] > 0).all()


def test_output_is_complete_and_sane(official):
    assert len(official) == 1975
    assert official["REASON_CODE"].isin(REASONS).all()
    assert official["CONFIDENCE"].between(0, 1).all()
    dates = official["PREDICTED_DEPLETION_DATE"].dropna()
    assert (dates > CUTOFF).all() and (dates <= CUTOFF + pd.Timedelta(days=90)).all()
    dated = official[official["PREDICTED_DEPLETION_DATE"].notna()]
    assert (dated["SUGGESTED_QTY"] >= 1).all()
    assert (official["STOCK_AT_CUTOFF"].dropna() >= 0).all()


@pytest.mark.parametrize("cutoff", ["2024-11-01", "2025-04-04"])
def test_forecast_runs_at_early_cutoffs(data, cutoff):
    # before any winter closer is known, and while a closer has not re-opened yet
    deliveries, customers, items = data
    past = deliveries[deliveries["TRNDATE"] <= cutoff]
    targets = make_targets(past, customers, cutoff)
    out = forecast(daily_qty(past), past, customers, items, targets, cutoff)
    assert len(out) == len(targets)


# --- submission ----------------------------------------------------------

def _load_script():
    spec = importlib.util.spec_from_file_location(
        "make_submission", ROOT / "scripts" / "make_submission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_submission_has_1975_rows_in_template_format(tmp_path):
    _load_script().main(tmp_path / "team_rafi.csv")
    sub = pd.read_csv(tmp_path / "team_rafi.csv", encoding="utf-8", dtype=str,
                      keep_default_na=False)
    template = pd.read_csv(DATA / "submission_template.csv", encoding="utf-8-sig",
                           dtype=str, keep_default_na=False)
    assert len(sub) == 1975
    assert list(sub.columns) == ["TRDR", "MTRL", "PREDICTED_DEPLETION_DATE", "SUGGESTED_QTY"]
    assert sub[["TRDR", "MTRL"]].equals(template[["TRDR", "MTRL"]])
    has_date = sub["PREDICTED_DEPLETION_DATE"] != ""
    assert sub.loc[has_date, "SUGGESTED_QTY"].str.fullmatch(r"[1-9]\d*").all()
    assert (sub.loc[~has_date, "SUGGESTED_QTY"] == "").all()

    details = pd.read_csv(tmp_path / "forecast_details.csv", encoding="utf-8")
    assert len(details) == 1975
    assert {"BASE_RATE", "SEASONAL_FACTOR_NOW", "STOCK_AT_CUTOFF", "REASON_CODE",
            "CONFIDENCE"} <= set(details.columns)


# --- separate seasonal factors for winter closers ------------------------------

def test_closer_and_open_customer_get_different_november_factor(data, official):
    deliveries, customers, items = data
    closers = set(winter_closers(deliveries, customers, CUTOFF)["TRDR"])
    df = official.merge(customers[["TRDR", "AREA"]], on="TRDR").merge(
        items[["MTRL", "MTRCATEGORY"]], on="MTRL")
    crete_water = df[(df["AREA"] == "Κρήτη") & (df["MTRCATEGORY"] == "Νερά")]
    closer_f = crete_water.loc[crete_water["TRDR"].isin(closers), "SEASONAL_FACTOR_NOW"]
    open_f = crete_water.loc[~crete_water["TRDR"].isin(closers), "SEASONAL_FACTOR_NOW"]
    assert len(closer_f) and len(open_f)
    assert closer_f.nunique() == 1 and open_f.nunique() == 1   # one factor per group
    assert open_f.iloc[0] > closer_f.iloc[0]                   # open shops still sell in November


def test_group_without_data_uses_the_common_factor(data, official):
    # No closers in Athens: Athens customers keep the common AREA x MTRCATEGORY factor.
    deliveries, customers, items = data
    common = seasonal_factors(deliveries, customers, items, CUTOFF)
    nov = common[common["MONTH"] == 11].set_index(["AREA", "MTRCATEGORY"])["FACTOR"]
    df = official.merge(customers[["TRDR", "AREA"]], on="TRDR").merge(
        items[["MTRL", "MTRCATEGORY"]], on="MTRL")
    athens = df[df["AREA"] == "Αθήνα"]
    expected = [round(max(nov[(a, c)], 0.05), 3) for a, c in zip(athens["AREA"], athens["MTRCATEGORY"])]
    assert list(athens["SEASONAL_FACTOR_NOW"]) == expected
