"""Tests for the official score, the target rule, the baselines and the submission."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from rafi.baseline import plus30, rolling180
from rafi.evaluate import make_targets, proxy_truth, score
from rafi.ingest import build_deliveries, daily_qty, load_from_files

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CUTOFF = pd.Timestamp("2026-10-31")  # horizon_end = 2027-01-29, At Risk up to 2026-11-14
D = pd.Timestamp


def _one(pred, true, cutoff=CUTOFF):
    """Score a single pair and return its error."""
    p = pd.DataFrame({"TRDR": [1], "MTRL": [1], "PREDICTED_DEPLETION_DATE": [pred]})
    t = pd.DataFrame({"TRDR": [1], "MTRL": [1], "TRUE_DEPLETION_DATE": [true]})
    table, _ = score(p, t, cutoff)
    return table["ERROR"].iloc[0]


# --- score: one test per rule ------------------------------------------

def test_both_dates_is_absolute_difference():
    assert _one("2026-11-10", D("2026-11-05")) == 5
    assert _one("2026-11-05", D("2026-11-10")) == 5


def test_true_blank_and_pred_blank_is_zero():
    assert _one("", pd.NaT) == 0
    assert _one(None, pd.NaT) == 0


def test_true_blank_and_pred_date_is_horizon_minus_pred():
    # 2027-01-29 - 2026-12-30 = 30 days
    assert _one("2026-12-30", pd.NaT) == 30


def test_true_date_and_pred_blank_is_horizon_minus_true():
    # 2027-01-29 - 2027-01-09 = 20 days
    assert _one("", D("2027-01-09")) == 20


def test_pred_on_or_after_horizon_end_counts_as_blank():
    assert _one("2027-01-29", pd.NaT) == 0
    assert _one("2027-03-15", pd.NaT) == 0
    assert _one("2027-01-29", D("2027-01-09")) == 20


def test_missing_pair_is_60():
    p = pd.DataFrame({"TRDR": [1], "MTRL": [1], "PREDICTED_DEPLETION_DATE": ["2026-11-05"]})
    t = pd.DataFrame({"TRDR": [1, 2], "MTRL": [1, 2],
                      "TRUE_DEPLETION_DATE": [D("2026-11-05"), pd.NaT]})
    table, metrics = score(p, t, CUTOFF)
    assert list(table["ERROR"]) == [0, 60]
    assert metrics["MAE"] == 30


@pytest.mark.parametrize("bad", ["31/12/2026", "abc", "2026-13-01", "20261105"])
def test_unreadable_date_is_60(bad):
    assert _one(bad, D("2026-11-05")) == 60
    assert _one(bad, pd.NaT) == 60


def test_horizon_end_is_cutoff_plus_90():
    # other cutoff: 2025-10-31 -> horizon_end 2026-01-29
    assert _one("", D("2026-01-19"), cutoff=D("2025-10-31")) == 10


# --- precision / recall -------------------------------------------------

def test_precision_recall_and_within():
    p = pd.DataFrame({"TRDR": [1, 2, 3, 4, 5], "MTRL": [1] * 5,
                      "PREDICTED_DEPLETION_DATE": ["2026-11-05", "2026-11-12", "2026-11-25", "",
                                                   "2026-11-14"]})
    t = pd.DataFrame({"TRDR": [1, 2, 3, 4, 5], "MTRL": [1] * 5,
                      "TRUE_DEPLETION_DATE": [D("2026-11-10"), D("2026-11-20"), D("2026-11-08"),
                                              pd.NaT, D("2026-11-14")]})
    # 1: TP (both <= 11-14), 2: FP, 3: FN, 4: neither, 5: TP on the boundary
    table, m = score(p, t, CUTOFF)
    assert list(table["ERROR"]) == [5, 8, 17, 0, 0]
    assert m["n"] == 5
    assert m["MAE"] == pytest.approx(30 / 5)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["n_both_dates"] == 4
    assert m["within_3"] == pytest.approx(1 / 4)  # errors 5, 8, 17, 0
    assert m["within_7"] == pytest.approx(2 / 4)


# --- real data -----------------------------------------------------------

@pytest.fixture(scope="module")
def data():
    headers, lines, items, customers = load_from_files(DATA)
    return build_deliveries(headers, lines, items), customers


def _pairs(df):
    return set(zip(df["TRDR"], df["MTRL"]))


def test_make_targets_matches_prediction_targets(data):
    deliveries, customers = data
    ours = _pairs(make_targets(deliveries, customers, CUTOFF))
    official = _pairs(pd.read_csv(DATA / "prediction_targets.csv", encoding="utf-8-sig"))
    missing, extra = sorted(official - ours), sorted(ours - official)
    assert not missing and not extra, (
        f"same {len(ours & official)}, missing {len(missing)} e.g. {missing[:5]}, "
        f"extra {len(extra)} e.g. {extra[:5]}")


def test_make_targets_ignores_the_future(data):
    deliveries, customers = data
    cutoff = D("2025-10-31")
    full = make_targets(deliveries, customers, cutoff)
    cut = make_targets(deliveries[deliveries["TRNDATE"] <= cutoff], customers, cutoff)
    assert full.equals(cut)


def test_proxy_truth_is_first_delivery_after_cutoff():
    deliveries = pd.DataFrame({
        "TRDR": [1, 1, 1, 2, 3], "MTRL": [1, 1, 1, 1, 1],
        "TRNDATE": [D("2026-10-30"), D("2026-11-20"), D("2026-11-10"), D("2027-02-10"),
                    D("2026-11-03")],
        "QTY": [5, 5, 5, 5, -2], "IS_CREDIT": [False, False, False, False, True],
    })
    targets = pd.DataFrame({"TRDR": [1, 2, 3], "MTRL": [1, 1, 1]})
    truth = proxy_truth(deliveries, targets, CUTOFF)
    assert truth["TRUE_DEPLETION_DATE"].iloc[0] == D("2026-11-10")
    assert truth["TRUE_DEPLETION_DATE"].iloc[1:].isna().all()  # beyond horizon / credit only


# --- baselines ----------------------------------------------------------

def test_rolling180_hand_example():
    daily = pd.DataFrame({
        "TRDR": [1, 1, 2, 3], "MTRL": [1, 1, 1, 1],
        "TRNDATE": [D("2026-07-23"), D("2026-10-21"), D("2026-10-30"), D("2026-10-20")],
        "QTY": [18, 18, 180, -4],
    })
    targets = pd.DataFrame({"TRDR": [1, 2, 3], "MTRL": [1, 1, 1]})
    pred = rolling180(daily, targets, CUTOFF)["PREDICTED_DEPLETION_DATE"]
    # pair 1: rate 36/180 = 0.2, stock 18 - 0.2*10 = 16, 16/0.2 = 80 days
    assert pred.iloc[0] == CUTOFF + pd.Timedelta(days=80)
    # pair 2: rate 1, stock 180 - 1 = 179 days -> after horizon_end -> blank
    assert pd.isna(pred.iloc[1])
    # pair 3: only a credit -> rate below 0 -> blank
    assert pd.isna(pred.iloc[2])


def test_rolling180_never_before_cutoff_plus_1():
    daily = pd.DataFrame({"TRDR": [1, 1], "MTRL": [1, 1],
                          "TRNDATE": [D("2026-06-01"), D("2026-08-01")], "QTY": [90, 1]})
    targets = pd.DataFrame({"TRDR": [1], "MTRL": [1]})
    pred = rolling180(daily, targets, CUTOFF)["PREDICTED_DEPLETION_DATE"]
    assert pred.iloc[0] == CUTOFF + pd.Timedelta(days=1)


@pytest.mark.parametrize("method", ["plus30", "rolling180"])
def test_baselines_ignore_deliveries_after_cutoff(data, method):
    deliveries, customers = data
    cutoff = D("2025-10-31")
    targets = make_targets(deliveries[deliveries["TRNDATE"] <= cutoff], customers, cutoff)
    past = daily_qty(deliveries[deliveries["TRNDATE"] <= cutoff])
    fake = targets.assign(TRNDATE=cutoff + pd.Timedelta(days=3), QTY=999, QTY_FREE=0)
    with_future = pd.concat([daily_qty(deliveries), fake], ignore_index=True)

    run = {"plus30": lambda d: plus30(targets, cutoff),
           "rolling180": lambda d: rolling180(d, targets, cutoff)}[method]
    pd.testing.assert_frame_equal(run(past), run(with_future))


# --- submission ---------------------------------------------------------

def _load_script():
    spec = importlib.util.spec_from_file_location(
        "make_submission", ROOT / "scripts" / "make_submission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_make_submission_writes_1975_rows_in_template_format(tmp_path):
    script = _load_script()
    out = tmp_path / "team_rafi.csv"
    script.main(out)

    sub = pd.read_csv(out, encoding="utf-8", dtype=str, keep_default_na=False)
    template = pd.read_csv(DATA / "submission_template.csv", encoding="utf-8-sig",
                           dtype=str, keep_default_na=False)
    assert len(sub) == 1975
    assert list(sub.columns) == list(template.columns)
    assert sub[["TRDR", "MTRL"]].equals(template[["TRDR", "MTRL"]])
    dates = sub["PREDICTED_DEPLETION_DATE"]
    filled = dates[dates != ""]
    assert filled.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all()
    assert (pd.to_datetime(filled) > CUTOFF).all()
