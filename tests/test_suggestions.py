"""Suggestions built from the real forecast (cutoff 2026-10-31, today 2026-11-01)."""
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from rafi.features import route_days, winter_history, working_day
from rafi.forecast import forecast
from rafi.ingest import build_deliveries, daily_qty, load_from_files
from rafi.suggestions import (EXTRA_ROUTE, EXTRA_ROUTE_COST_EUR, LOSS, MARGINAL, OFFER, ORDER,
                              URGENT, build_suggestions, delivery_plan, load_promos)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TODAY = pd.Timestamp("2026-11-01")
CUTOFF = pd.Timestamp("2026-10-31")
FIRST_DAY = pd.Timestamp("2026-11-02")


@pytest.fixture(scope="module")
def data():
    headers, lines, items, customers = load_from_files(DATA)
    deliveries = build_deliveries(headers, lines, items)
    past = deliveries[deliveries["TRNDATE"] <= CUTOFF]
    targets = pd.read_csv(DATA / "prediction_targets.csv", encoding="utf-8-sig")
    details = forecast(daily_qty(past), past, customers, items, targets, CUTOFF)
    return {"details": details, "customers": customers, "items": items, "past": past,
            "promos": load_promos(DATA), "routes": route_days(past, CUTOFF)}


@pytest.fixture(scope="module")
def suggestions(data):
    d = data
    return build_suggestions(d["details"], d["customers"], d["items"], d["past"],
                             d["promos"], TODAY)


def _of(suggestions, kind):
    return [s for s in suggestions if s["ACTION_TYPE"] == kind]


def _dep(line):
    return pd.Timestamp(line["PREDICTED_DEPLETION_DATE"])


# --- delivery date ---------------------------------------------------------

def test_delivery_is_a_working_day_from_tomorrow(suggestions):
    assert suggestions
    for s in suggestions:
        d = pd.Timestamp(s["DELIVERY_DATE"])
        assert working_day(d), s["DELIVERY_DATE"]
        assert d >= FIRST_DAY
        assert s["payload"]["SUGGESTED_DELIVERY_DATE"] == s["DELIVERY_DATE"]


def test_order_delivery_before_every_depletion_unless_urgent_or_marginal(suggestions):
    orders = _of(suggestions, ORDER)
    assert orders
    for s in orders:
        d = pd.Timestamp(s["DELIVERY_DATE"])
        if URGENT in s["FLAGS"]:
            assert d == FIRST_DAY
        elif MARGINAL in s["FLAGS"]:
            assert d == pd.Timestamp(s["FACTS"]["usual_visit"])  # waits for the usual visit
        elif not s["FLAGS"] and d == min(_dep(l) for l in s["payload"]["LINES"]):
            # runs out on the usual visit day: morning delivery on the normal route
            assert d == pd.Timestamp(s["FACTS"]["usual_visit"]) and s["FACTS"]["lost_eur"] == 0
            assert all(d <= _dep(l) for l in s["payload"]["LINES"])
        else:
            assert all(d < _dep(l) for l in s["payload"]["LINES"]), s["TRDR"]


def test_order_on_route_day_when_one_exists_before_depletion(suggestions, data):
    for s in _of(suggestions, ORDER):
        days = data["routes"][s["TRDR"]]
        earliest = min(_dep(l) for l in s["payload"]["LINES"])
        candidates = pd.date_range(FIRST_DAY, earliest - pd.Timedelta(days=1))
        has_route_day = any(working_day(c) and c.weekday() in days for c in candidates)
        if has_route_day:
            assert pd.Timestamp(s["DELIVERY_DATE"]).weekday() in days, s["TRDR"]
            assert EXTRA_ROUTE not in s["FLAGS"] and URGENT not in s["FLAGS"]
        else:
            assert (EXTRA_ROUTE in s["FLAGS"]
                    or pd.Timestamp(s["DELIVERY_DATE"]).weekday() in days)


def test_small_lost_sales_wait_for_the_route_large_ones_get_an_extra_run():
    wed, dep = [2], pd.Timestamp("2026-11-02")
    d, d1, _, flags = delivery_plan(dep, wed, TODAY, lost_eur=EXTRA_ROUTE_COST_EUR - 1)
    assert d == d1 == pd.Timestamp("2026-11-04") and flags == [MARGINAL]
    d, _, _, flags = delivery_plan(dep, wed, TODAY, lost_eur=EXTRA_ROUTE_COST_EUR)
    assert d == FIRST_DAY and URGENT in flags and MARGINAL not in flags
    d, _, _, flags = delivery_plan(pd.Timestamp("2026-11-04"), [3], TODAY, lost_eur=500)
    assert d == pd.Timestamp("2026-11-03") and flags == [EXTRA_ROUTE]
    # nothing lost (runs out on D1) -> D1 without ΟΡΙΑΚΟ, whatever the threshold
    d, d1, _, flags = delivery_plan(pd.Timestamp("2026-11-04"), wed, TODAY, lost_eur=0)
    assert d == d1 and flags == []


def test_lost_sales_decide_between_route_and_extra_run(suggestions):
    for s in _of(suggestions, ORDER):
        lost, p = s["FACTS"]["lost_eur"], s["payload"]
        assert p["LOST_EUR"] == lost and lost >= 0
        usual = pd.Timestamp(s["FACTS"]["usual_visit"])
        earliest = min(_dep(l) for l in p["LINES"])
        if earliest >= usual:
            assert not s["FLAGS"] and lost == 0
        elif lost < EXTRA_ROUTE_COST_EUR:
            assert lost > 0
            assert s["FLAGS"] == [MARGINAL]
            assert "λιγότερες από το κόστος έκτακτου δρομολογίου" in p["REASON"]
            assert "επιβεβαιώσετε" in p["SALES_NOTE"]
        else:
            assert MARGINAL not in s["FLAGS"]
            assert URGENT in s["FLAGS"] or EXTRA_ROUTE in s["FLAGS"]
            assert "περισσότερες από το κόστος έκτακτου δρομολογίου" in p["REASON"]
    flags = {f for s in suggestions for f in s["FLAGS"]}
    assert {MARGINAL, URGENT, EXTRA_ROUTE} <= flags


def test_order_lines_run_out_before_the_next_visit(suggestions):
    for s in _of(suggestions, ORDER):
        next_visit = pd.Timestamp(s["FACTS"]["next_visit"])
        assert all(_dep(l) < next_visit for l in s["payload"]["LINES"])


def test_delivery_plan_rules():
    wed = [2]
    # depletion after the first route day -> usual route
    assert delivery_plan(pd.Timestamp("2026-11-10"), wed, TODAY)[0] == pd.Timestamp("2026-11-04")
    assert delivery_plan(pd.Timestamp("2026-11-10"), wed, TODAY)[3] == []
    # depletion on the route day -> that route day, no flag (morning delivery)
    d, d1, d2, flags = delivery_plan(pd.Timestamp("2026-11-04"), wed, TODAY)
    assert (d, d1, d2) == (pd.Timestamp("2026-11-04"), pd.Timestamp("2026-11-04"),
                           pd.Timestamp("2026-11-11"))
    assert flags == []
    # depletion the day before the route day -> the working day before it, extra route
    d, _, _, flags = delivery_plan(pd.Timestamp("2026-11-04"), [3], TODAY)
    assert d == pd.Timestamp("2026-11-03") and flags == [EXTRA_ROUTE]
    # already out -> first working day, urgent
    d, _, _, flags = delivery_plan(pd.Timestamp("2026-11-01"), wed, TODAY)
    assert d == FIRST_DAY and URGENT in flags
    # holidays are skipped: Fridays 25/12 and 1/1 are holidays -> 8/1
    d, d1, _, _ = delivery_plan(pd.Timestamp("2027-01-20"), [4], pd.Timestamp("2026-12-23"))
    assert d == d1 == pd.Timestamp("2027-01-08")


# --- drafts and quantities -------------------------------------------------

def test_one_draft_per_customer_and_date(suggestions):
    keys = [(s["TRDR"], s["DELIVERY_DATE"]) for s in suggestions]
    assert len(keys) == len(set(keys))
    for s in suggestions:
        mtrls = [l["MTRL"] for l in s["payload"]["LINES"]]
        assert len(mtrls) == len(set(mtrls))


def test_quantities_are_integers_above_zero(suggestions):
    for s in suggestions:
        assert s["payload"]["LINES"], s["TRDR"]
        for l in s["payload"]["LINES"]:
            assert isinstance(l["QTY1"], int) and not isinstance(l["QTY1"], bool)
            assert l["QTY1"] > 0


def test_payload_matches_the_erp_format(suggestions):
    for s in suggestions:
        p = s["payload"]
        for key in ("TRDR", "CUSTOMER_CODE", "SALESMAN", "SUGGESTED_DELIVERY_DATE",
                    "ACTION_TYPE", "REASON", "SALES_NOTE", "RAFI_SUGGESTION_ID", "LINES"):
            assert key in p
        assert isinstance(p["TRDR"], int) and isinstance(p["SALESMAN"], int)
        assert p["REASON"] and p["SALES_NOTE"]
        for l in p["LINES"]:
            assert isinstance(l["MTRL"], int)
            for key in ("ITEM_CODE", "QTY1", "PREDICTED_DEPLETION_DATE", "CONFIDENCE"):
                assert key in l
        json.dumps(p, ensure_ascii=False)  # must be serialisable as is
        assert s["VALUE_EUR"] == round(sum(l["QTY1"] * l["PRICER"] for l in p["LINES"]), 2)


def test_priority_urgent_then_loss_then_orders_then_offers(suggestions):
    assert [s["PRIORITY"] for s in suggestions] == list(range(1, len(suggestions) + 1))

    def group(s):
        if URGENT in s["FLAGS"]:
            return 0
        return {LOSS: 1, ORDER: 2, OFFER: 3}[s["ACTION_TYPE"]]
    groups = [group(s) for s in suggestions]
    assert groups == sorted(groups)
    assert set(groups) == {0, 1, 2, 3}
    loss_values = [s["FACTS"]["purchases_12m_eur"] for s in _of(suggestions, LOSS)]
    assert loss_values == sorted(loss_values, reverse=True)
    orders = [s for s in suggestions if group(s) == 2]
    deps = [pd.Timestamp(s["EARLIEST_DEPLETION"]) for s in orders]
    assert deps == sorted(deps)


# --- winter closures and loss risk -----------------------------------------

def test_no_closed_customer_gets_an_order(suggestions, data):
    closed = set(data["details"].loc[data["details"]["REASON_CODE"] == "ΚΛΕΙΣΤΟ_ΧΕΙΜΩΝΑ", "TRDR"])
    assert closed
    assert not closed & {s["TRDR"] for s in _of(suggestions, ORDER)}


def test_1127_is_a_loss_risk(suggestions, data):
    loss = {s["TRDR"]: s for s in _of(suggestions, LOSS)}
    assert 1127 in loss
    assert loss[1127]["FACTS"]["last_delivery"] == "2026-08-17"
    assert loss[1127]["FACTS"]["days_early"] > 30
    # closed for the season now: the order is for the re-opening in April 2027
    s = loss[1127]
    d = pd.Timestamp(s["DELIVERY_DATE"])
    assert s["FACTS"]["closed_now"]
    assert (d.year, d.month) == (2027, 4)
    assert d >= pd.Timestamp(s["FACTS"]["usual_reopen"])
    assert working_day(d) and d.weekday() in data["routes"][1127] and not s["FLAGS"]
    assert "κλειστό για τη σεζόν" in s["payload"]["REASON"]
    assert "πώς πήγε η σεζόν" in s["payload"]["SALES_NOTE"]
    for fl, line in zip(s["FACTS"]["lines"], s["payload"]["LINES"]):
        assert line["QTY1"] == max(1, math.ceil(fl["median_qty"] * fl["season_scale"] - 1e-9))
    # April quantities are larger than a November delivery would get
    from rafi.suggestions import _Context, usual_basket
    d0 = data
    ctx = _Context(d0["details"], d0["customers"], d0["items"], d0["past"], d0["promos"], TODAY)
    november = {m: q for m, q, *_ in usual_basket(ctx, 1127, FIRST_DAY)}
    april = {l["MTRL"]: l["QTY1"] for l in s["payload"]["LINES"]}
    assert april.keys() == november.keys()
    assert all(april[m] >= november[m] for m in april)
    assert sum(april.values()) > sum(november.values())
    assert 1127 not in {s["TRDR"] for s in suggestions if s["ACTION_TYPE"] != LOSS}


def test_usual_winter_closure_is_not_a_loss_risk(suggestions, data):
    h = winter_history(data["past"], data["customers"], CUTOFF)
    closers = set(h.loc[h["IS_CLOSER"], "TRDR"]) - {1127}
    assert len(closers) >= 10
    assert not closers & {s["TRDR"] for s in _of(suggestions, LOSS)}


# --- offers ----------------------------------------------------------------

def test_seasonal_offers(suggestions, data):
    offers = _of(suggestions, OFFER)
    assert offers
    seasonal = set(data["items"].loc[data["items"]["MTRCATEGORY"] == "Εποχικά", "MTRL"])
    closed = set(data["details"].loc[data["details"]["REASON_CODE"] == "ΚΛΕΙΣΤΟ_ΧΕΙΜΩΝΑ", "TRDR"])
    p = data["past"]
    for s in offers:
        assert s["TRDR"] not in closed
        assert pd.Timestamp(s["DELIVERY_DATE"]) >= pd.Timestamp("2026-11-20")
        for l, fl in zip(s["payload"]["LINES"], s["FACTS"]["lines"]):
            assert l["MTRL"] in seasonal
            bought = p[(p["TRDR"] == s["TRDR"]) & (p["MTRL"] == l["MTRL"])
                       & (p["TRNDATE"].dt.year == 2025)]
            assert bought["QTY"].sum() == fl["last_year_total"]  # net of returns, in REASON
            first = bought[bought["QTY"] > 0].sort_values("TRNDATE")
            first = first[first["TRNDATE"] == first["TRNDATE"].iloc[0]]
            assert l["QTY1"] == first["QTY"].sum()  # quantity of last year's first delivery
            assert str(fl["last_year_total"]) in s["payload"]["REASON"]


def test_promo_on_delivery_day_is_marked(data):
    d = data
    promos = pd.DataFrame([{"PROMO": 99, "NAME": "test", "MTRL": None, "MTRCATEGORY": "Νερά",
                            "FROMDATE": pd.Timestamp("2026-11-02"),
                            "TODATE": pd.Timestamp("2026-11-30"), "DISC_PRC": 11}])
    built = build_suggestions(d["details"], d["customers"], d["items"], d["past"], promos, TODAY)
    water = set(d["items"].loc[d["items"]["MTRCATEGORY"] == "Νερά", "MTRL"])
    marked = [s for s in _of(built, ORDER)
              if any(l.get("PROMO_DISC_PRC") == 11 for l in s["payload"]["LINES"])]
    assert marked
    for s in marked:
        assert "προσφορά" in s["payload"]["REASON"]
        for l in s["payload"]["LINES"]:
            assert (l.get("PROMO_DISC_PRC") == 11) == (l["MTRL"] in water)


def test_text_writer_hook(data):
    d = data
    args = (d["details"], d["customers"], d["items"], d["past"], d["promos"], TODAY)
    ai = build_suggestions(*args, text_writer=lambda facts: {"REASON": "AI κείμενο"})
    assert all(s["payload"]["REASON"] == "AI κείμενο" for s in ai)
    assert all(s["payload"]["TEXT_SOURCE"] == "AI" for s in ai)

    def broken(facts):
        raise RuntimeError("LLM down")
    kept = build_suggestions(*args, text_writer=broken)
    assert all(s["payload"]["TEXT_SOURCE"] == "ΠΡΟΤΥΠΟ" for s in kept)
    assert all(s["payload"]["REASON"] != "AI κείμενο" for s in kept)
