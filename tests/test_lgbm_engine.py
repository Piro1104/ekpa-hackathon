"""Tests for rafi.lgbm_engine (hybrid: rafi + LightGBM correction)."""
from pathlib import Path

import pandas as pd
import pytest

from rafi import lgbm_engine
from rafi.features import working_day
from rafi.forecast import HORIZON_DAYS, OUTPUT_COLUMNS, forecast
from rafi.ingest import build_deliveries, daily_qty, load_from_files
from rafi.lgbm_engine import (CENSORED_DAYS, days_to_next_delivery, forecast_hybrid,
                              training_set, virtual_cutoffs)

DATA = Path(__file__).resolve().parents[1] / "data"
CUTOFF = pd.Timestamp("2026-10-31")
HORIZON_END = CUTOFF + pd.Timedelta(days=HORIZON_DAYS)

needs_lgbm = pytest.mark.skipif(not lgbm_engine.HAS_LGBM, reason="lightgbm not installed")


@pytest.fixture(scope="module")
def data():
    headers, lines, items, customers = load_from_files(DATA)
    return build_deliveries(headers, lines, items), customers, items


@pytest.fixture(scope="module")
def official(data):
    deliveries, customers, items = data
    past = deliveries[deliveries["TRNDATE"] <= CUTOFF]
    targets = pd.read_csv(DATA / "prediction_targets.csv", encoding="utf-8-sig")
    return forecast_hybrid(daily_qty(past), past, customers, items, targets, CUTOFF)


# --- no look into the future -------------------------------------------------

def test_virtual_cutoffs_keep_their_whole_horizon_before_the_cutoff():
    cuts = virtual_cutoffs("2024-01-02", CUTOFF)
    assert 1 <= len(cuts) <= lgbm_engine.MAX_SNAPSHOTS
    assert all(c + pd.Timedelta(days=HORIZON_DAYS) <= CUTOFF for c in cuts)
    assert cuts == sorted(cuts)


def test_virtual_cutoffs_need_enough_history():
    assert virtual_cutoffs("2026-01-01", "2026-06-30") == []


def test_label_is_days_to_next_delivery_or_censored():
    deliveries = pd.DataFrame({
        "TRDR": [1, 1, 2], "MTRL": [10, 10, 10],
        "TRNDATE": pd.to_datetime(["2026-01-05", "2026-01-20", "2026-06-01"]),
        "QTY": [5.0, 5.0, 5.0], "IS_CREDIT": [False, False, False]})
    targets = pd.DataFrame({"TRDR": [1, 2], "MTRL": [10, 10]})
    days = days_to_next_delivery(deliveries, targets, "2026-01-01")
    assert list(days) == [4.0, float(CENSORED_DAYS)]  # pair 2: first delivery after 90 days


def test_training_ignores_data_after_the_cutoff(data):
    deliveries, customers, items = data
    cutoff = pd.Timestamp("2025-03-31")  # few snapshots, keeps the test fast
    X1, y1, w1 = training_set(deliveries, customers, items, cutoff)
    X2, y2, w2 = training_set(deliveries[deliveries["TRNDATE"] <= cutoff], customers, items, cutoff)
    pd.testing.assert_frame_equal(X1, X2)
    assert (y1 == y2).all() and (w1 == w2).all()


# --- the official forecast -----------------------------------------------------

@needs_lgbm
def test_hybrid_has_all_pairs_and_the_explanation_columns(official):
    assert len(official) == 1975
    assert set(OUTPUT_COLUMNS) <= set(official.columns)
    assert {"RAFI_DEPLETION_DATE", "MODEL_CORRECTION_DAYS"} <= set(official.columns)


@needs_lgbm
def test_hybrid_dates_are_working_days_inside_the_horizon(official):
    dates = official["PREDICTED_DEPLETION_DATE"].dropna()
    assert ((dates > CUTOFF) & (dates < HORIZON_END)).all()
    assert all(working_day(d) for d in dates)


@needs_lgbm
def test_hybrid_keeps_closures_and_no_rate_blank(official):
    fixed = official["REASON_CODE"].isin(["ΚΛΕΙΣΤΟ_ΧΕΙΜΩΝΑ", "ΧΩΡΙΣ_ΡΥΘΜΟ"])
    assert official.loc[fixed, "PREDICTED_DEPLETION_DATE"].isna().all()
    assert (official["REASON_CODE"] == "ΚΛΕΙΣΤΟ_ΧΕΙΜΩΝΑ").sum() >= 150


@needs_lgbm
def test_hybrid_correction_is_explained(official):
    both = official["PREDICTED_DEPLETION_DATE"].notna() & official["RAFI_DEPLETION_DATE"].notna()
    diff = (official["PREDICTED_DEPLETION_DATE"] - official["RAFI_DEPLETION_DATE"]).dt.days
    assert (official.loc[both, "MODEL_CORRECTION_DAYS"] == diff[both]).all()
    # a correction, not a different forecast: most dates move only a few days
    assert diff[both].abs().median() <= 5


@needs_lgbm
def test_hybrid_reason_matches_the_date(official):
    blank = official["PREDICTED_DEPLETION_DATE"].isna()
    assert not official.loc[blank, "REASON_CODE"].isin(["ΚΑΝΟΝΙΚΗ", "ΗΔΗ_ΕΞΑΝΤΛΗΜΕΝΟ"]).any()
    assert (official.loc[~blank, "REASON_CODE"] != "ΕΚΤΟΣ_ΟΡΙΖΟΝΤΑ").all()


def test_without_lightgbm_falls_back_to_rafi(data, monkeypatch):
    deliveries, customers, items = data
    past = deliveries[deliveries["TRNDATE"] <= CUTOFF]
    targets = pd.read_csv(DATA / "prediction_targets.csv", encoding="utf-8-sig").head(50)
    monkeypatch.setattr(lgbm_engine, "HAS_LGBM", False)
    with pytest.warns(UserWarning):
        out = forecast_hybrid(daily_qty(past), past, customers, items, targets, CUTOFF)
    plain = forecast(daily_qty(past), past, customers, items, targets, CUTOFF)
    pd.testing.assert_frame_equal(out[OUTPUT_COLUMNS], plain)
    assert (out["MODEL_CORRECTION_DAYS"] == 0).all()
