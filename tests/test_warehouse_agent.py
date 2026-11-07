"""Warehouse agent: settings, suppliers, stock, demand, simulation, texts and a full run."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rafi import store
from rafi.agents import warehouse_agent as w

ROOT = Path(__file__).resolve().parents[1]
T = pd.Timestamp
TODAY = T("2026-11-01")  # a Sunday


def items_df():
    return pd.DataFrame({
        "MTRL": [5001, 5002, 5021, 5023, 5077, 5081],
        "CODE": ["10.01.001", "10.01.002", "10.03.001", "10.03.003", "10.09.001", "99.00.001"],
        "NAME": ["Νερό 0,5L", "Νερό 1,5L", "Μπύρα lager", "Μπύρα pils", "Μελομακάρονα",
                 "Μεταφορικά"],
        "MTRCATEGORY": ["Νερά", "Νερά", "Μπύρες", "Μπύρες", "Εποχικά", "Υπηρεσίες"],
        "MTRTYPE": [0, 0, 0, 0, 0, 1],
        "MTRUNIT1": ["ΚΙΒ", "ΚΙΒ", "ΚΙΒ", "ΚΙΒ", "ΚΙΒ", "ΤΕΜ"],
        "PRICER": [4.2, 2.6, 17.5, 19.0, 34.0, 15.0],
    })


def deliv(rows):
    """rows: (date, TRDR, MTRL, QTY) with credits as negative QTY, like build_deliveries."""
    d = pd.DataFrame(rows, columns=["TRNDATE", "TRDR", "MTRL", "QTY"])
    d["TRNDATE"] = pd.to_datetime(d["TRNDATE"])
    return d


def details_df(rows):
    """rows: (TRDR, MTRL, depletion date or None, SUGGESTED_QTY or None, REASON_CODE)."""
    d = pd.DataFrame(rows, columns=["TRDR", "MTRL", "PREDICTED_DEPLETION_DATE",
                                    "SUGGESTED_QTY", "REASON_CODE"])
    d["PREDICTED_DEPLETION_DATE"] = pd.to_datetime(d["PREDICTED_DEPLETION_DATE"])
    return d


def suggestion(sid, trdr, date, lines, status=store.PROPOSED):
    return {"id": sid, "status": status,
            "payload": {"TRDR": trdr, "SUGGESTED_DELIVERY_DATE": date,
                        "LINES": [{"MTRL": m, "QTY1": q} for m, q in lines]}}


# --- settings, suppliers, stock --------------------------------------------

def test_settings_defaults_when_file_missing(tmp_path):
    assert w.load_settings(tmp_path / "settings.json") == {"stock_cover_weeks": 4,
                                                           "safety_weeks": 1}


def test_settings_read_from_file(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"stock_cover_weeks": 6, "safety_weeks": 2, "other": 1}), encoding="utf-8")
    assert w.load_settings(p) == {"stock_cover_weeks": 6, "safety_weeks": 2}


def test_suppliers_one_per_goods_category():
    items = pd.read_csv(ROOT / "data" / "MTRL.csv", encoding="utf-8-sig")
    sup = w.load_suppliers(items)
    goods = set(items.loc[items["MTRTYPE"] == 0, "MTRCATEGORY"])
    assert set(sup["MTRCATEGORY"]) == goods
    assert len(sup) == len(goods) and sup["SUPPLIER_ID"].is_unique
    assert sup["LEAD_TIME_DAYS"].between(3, 7).all()
    assert (sup["MIN_ORDER_QTY"] > 0).all()
    assert sup["CONTACT"].isna().all()
    assert list(sup.columns) == ["SUPPLIER_ID", "NAME", "MTRCATEGORY", "LEAD_TIME_DAYS",
                                 "MIN_ORDER_QTY", "CONTACT"]


def test_suppliers_missing_category_is_an_error(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("SUPPLIER_ID,NAME,MTRCATEGORY,LEAD_TIME_DAYS,MIN_ORDER_QTY,CONTACT\n"
                 "S1,Α,Νερά,3,10,\n", encoding="utf-8")
    with pytest.raises(ValueError):
        w.load_suppliers(items_df(), p)


def test_estimated_stock_is_net_sales_same_weeks_last_year():
    d = deliv([("2025-11-01", 1, 5001, 10),     # first day of the window
               ("2025-11-28", 1, 5001, 5),      # last day of 4 weeks
               ("2025-11-29", 1, 5001, 100),    # outside
               ("2025-10-31", 1, 5001, 100),    # outside
               ("2025-11-10", 2, 5001, -3),     # credit note
               ("2025-11-10", 2, 5002, -4)])    # only credits -> 0, never negative
    s = w.estimate_stock(d, items_df(), TODAY, weeks=4).set_index("MTRL")
    assert s.at[5001, "QTY"] == 12
    assert s.at[5002, "QTY"] == 0
    assert s.at[5021, "QTY"] == 0
    assert 5081 not in s.index  # shipping is not stock
    assert (s["STOCK_SOURCE"] == w.ESTIMATE).all()


def test_estimated_stock_follows_cover_weeks():
    d = deliv([("2025-11-01", 1, 5001, 10), ("2025-12-05", 1, 5001, 7)])
    assert w.estimate_stock(d, items_df(), TODAY, 4).set_index("MTRL").at[5001, "QTY"] == 10
    assert w.estimate_stock(d, items_df(), TODAY, 6).set_index("MTRL").at[5001, "QTY"] == 17


def test_stock_file_wins_and_missing_items_keep_the_estimate(tmp_path):
    d = deliv([("2025-11-02", 1, 5001, 10), ("2025-11-02", 1, 5002, 8)])
    p = tmp_path / "stock.csv"
    p.write_text("MTRL,QTY\n5001,250\n", encoding="utf-8")
    s = w.current_stock(d, items_df(), TODAY, 4, p).set_index("MTRL")
    assert s.at[5001, "QTY"] == 250 and s.at[5001, "STOCK_SOURCE"] == w.FILE
    assert s.at[5002, "QTY"] == 8 and s.at[5002, "STOCK_SOURCE"] == w.ESTIMATE


def test_no_stock_file_means_estimate(tmp_path):
    d = deliv([("2025-11-02", 1, 5001, 10)])
    s = w.current_stock(d, items_df(), TODAY, 4, tmp_path / "none.csv")
    assert (s["STOCK_SOURCE"] == w.ESTIMATE).all()


# --- working days -----------------------------------------------------------

def test_add_working_days_skips_sundays_and_holidays():
    assert w.add_working_days("2026-11-06", 1) == T("2026-11-07")   # Fri -> Sat
    assert w.add_working_days("2026-11-07", 1) == T("2026-11-09")   # Sat -> Mon
    assert w.add_working_days("2026-12-24", 1) == T("2026-12-28")   # 25, 26 holidays, 27 Sunday
    assert w.add_working_days("2026-12-31", 3) == T("2027-01-05")   # 1/1 holiday, 3/1 Sunday


def test_latest_order_day_arrives_in_time():
    assert w.latest_order_day("2026-12-28", 1) == T("2026-12-24")
    assert w.latest_order_day("2026-12-27", 2) == T("2026-12-22")   # Sunday -> from 24/12
    for needed in pd.date_range("2026-11-01", "2027-01-31"):
        for lead in (3, 5, 7):
            d = w.latest_order_day(needed, lead)
            assert w.add_working_days(d, lead) <= needed
            nxt = w.add_working_days(d, 1)
            assert w.add_working_days(nxt, lead) > needed


def test_delivery_day_is_a_working_day_not_before_today():
    assert w.delivery_day("2026-11-08", TODAY) == T("2026-11-07")    # Sunday -> Saturday
    assert w.delivery_day("2026-10-20", TODAY) == T("2026-11-02")    # past -> first working day
    assert w.delivery_day("2026-12-25", TODAY) == T("2026-12-24")


# --- demand -----------------------------------------------------------------

def test_store_suggestion_replaces_the_forecast_first_delivery():
    det = details_df([(1, 5001, "2026-11-10", 6, "ΚΑΝΟΝΙΚΗ"),
                      (1, 5002, "2026-11-12", 4, "ΚΑΝΟΝΙΚΗ")])
    sugg = [suggestion(7, 1, "2026-11-05", [(5001, 9), (5021, 3)])]  # 5021 is no target
    first = w.first_deliveries(det, sugg, TODAY)
    assert first[(1, 5001)] == (T("2026-11-05"), 9.0, 7)
    assert first[(1, 5002)] == (T("2026-11-12"), 4.0, None)
    assert (1, 5021) not in first


def test_rejected_suggestions_are_ignored_and_earliest_wins():
    det = details_df([(1, 5001, "2026-11-10", 6, "ΚΑΝΟΝΙΚΗ")])
    sugg = [suggestion(1, 1, "2026-11-03", [(5001, 2)], status=store.REJECTED),
            suggestion(2, 1, "2026-11-09", [(5001, 5)], status=store.SENT),
            suggestion(3, 1, "2026-11-06", [(5001, 4)])]
    assert w.first_deliveries(det, sugg, TODAY)[(1, 5001)] == (T("2026-11-06"), 4.0, 3)


def test_each_delivery_counted_once_then_repeats_every_interval():
    det = details_df([(1, 5001, "2026-11-10", 6, "ΚΑΝΟΝΙΚΗ")])
    sugg = [suggestion(7, 1, "2026-11-05", [(5001, 9)])]
    rows = w.target_deliveries(det, sugg, {(1, 5001): 14}, {}, TODAY, T("2026-12-31"))
    # the store's 5/11 replaces the forecast's 10/11: no delivery on 10/11
    assert T("2026-11-10") not in set(rows["DATE"])
    assert list(rows["QTY"]) == [9, 6, 6, 6, 6]
    assert list(rows["DATE"]) == [T("2026-11-05"), T("2026-11-19"), T("2026-12-03"),
                                  T("2026-12-17"), T("2026-12-31")]
    assert rows["SUGGESTION_ID"].iloc[0] == 7 and rows["SUGGESTION_ID"].iloc[1:].isna().all()
    assert not rows.duplicated(["TRDR", "MTRL", "DATE"]).any()


def test_default_interval_and_no_forecast_no_delivery():
    det = details_df([(1, 5001, "2026-11-10", 6, "ΚΑΝΟΝΙΚΗ"),
                      (2, 5001, None, None, "ΕΚΤΟΣ_ΟΡΙΖΟΝΤΑ")])
    rows = w.target_deliveries(det, [], {}, {}, TODAY, T("2026-11-30"))
    assert set(rows["TRDR"]) == {1}
    assert list(rows["DATE"]) == [T("2026-11-10"), T("2026-11-24")]  # 14 days by default


def test_closed_customers_get_no_deliveries():
    det = details_df([(1, 5001, "2026-11-10", 6, "ΚΑΝΟΝΙΚΗ")])
    sugg = [suggestion(7, 1, "2026-11-05", [(5001, 9)])]
    windows = {1: ("10-20", "04-10")}
    rows = w.target_deliveries(det, sugg, {(1, 5001): 7}, windows, TODAY, T("2027-01-29"))
    assert rows.empty


def test_customer_reopening_in_the_horizon_gets_deliveries_after_it():
    det = details_df([(1, 5001, "2026-11-10", 6, "ΚΑΝΟΝΙΚΗ")])
    rows = w.target_deliveries(det, [], {(1, 5001): 7}, {1: ("10-20", "01-10")},
                               TODAY, T("2027-01-29"))
    assert len(rows) and rows["DATE"].min() >= T("2027-01-10")


def test_usual_intervals_median_of_last_deliveries():
    daily = deliv([("2026-09-01", 1, 5001, 5), ("2026-09-08", 1, 5001, 5),
                   ("2026-09-15", 1, 5001, 5), ("2026-09-29", 1, 5001, 5),
                   ("2026-10-01", 1, 5001, -2),            # credit: not a delivery
                   ("2026-10-01", 2, 5001, 3)])            # one delivery: no interval
    assert w.usual_intervals(daily) == {(1, 5001): 7}


def test_background_demand_skips_target_pairs():
    d = deliv([("2026-10-01", 1, 5001, 180),   # target pair: its demand comes from deliveries
               ("2026-10-01", 2, 5001, 90),    # other pair: 90 / 180 = 0.5 a day
               ("2026-10-02", 2, 5001, -18),   # credit: 72 / 180 = 0.4 a day
               ("2026-04-01", 2, 5002, 999)])  # older than 180 days
    targets = pd.DataFrame({"TRDR": [1], "MTRL": [5001]})
    bg = w.background_demand(d, items_df(), targets, {}, TODAY, days=10)
    assert set(bg["MTRL"]) == {5001}
    assert len(bg) == 10 and np.allclose(bg["QTY"], 0.4)


def test_background_demand_closed_customer_and_seasonal_items():
    d = deliv([("2026-10-01", 3, 5001, 180),
               ("2025-11-20", 2, 5077, 12)])      # seasonal: same weekday 52 weeks later
    bg = w.background_demand(d, items_df(), pd.DataFrame({"TRDR": [], "MTRL": []}),
                             {3: ("10-20", "04-10")}, TODAY, days=30)
    assert set(bg["MTRL"]) == {5077}
    assert bg.set_index("DATE").at[T("2026-11-19"), "QTY"] == 12


def promos_df(rows):
    p = pd.DataFrame(rows, columns=["PROMO", "NAME", "MTRL", "MTRCATEGORY", "FROMDATE",
                                    "TODATE", "DISC_PRC"])
    p["FROMDATE"] = pd.to_datetime(p["FROMDATE"])
    p["TODATE"] = pd.to_datetime(p["TODATE"])
    return p


def _daily_sales(mtrl, start, end, qty):
    return [(d, 1, mtrl, qty) for d in pd.date_range(start, end)]


def test_promo_lift_from_the_items_past_promotions():
    d = deliv(_daily_sales(5021, "2026-05-04", "2026-05-31", 1)     # 4 weeks before: 1/day
              + _daily_sales(5021, "2026-06-01", "2026-06-10", 2))  # in the promo: 2/day
    p = promos_df([(1, "Μπύρες", None, "Μπύρες", "2026-06-01", "2026-06-10", 10),
                   (2, "Μπύρες", None, "Μπύρες", "2026-11-09", "2026-11-27", 10)])
    assert w.promo_lift(d, items_df(), p, 5021, TODAY) == 2.0


def test_promo_lift_falls_back_to_category_then_one():
    d = deliv(_daily_sales(5023, "2026-05-04", "2026-05-31", 2)
              + _daily_sales(5023, "2026-06-01", "2026-06-10", 3))
    p = promos_df([(1, "pils", 5023, None, "2026-06-01", "2026-06-10", 10)])
    assert w.promo_lift(d, items_df(), p, 5021, TODAY) == 1.5  # 5021 no own promo -> category
    assert w.promo_lift(d, items_df(), p, 5001, TODAY) == 1.0  # nothing for waters


def test_promo_factor_applied_once_inside_the_window():
    demand = pd.DataFrame({"DATE": pd.to_datetime(["2026-11-08", "2026-11-09", "2026-11-27",
                                                   "2026-11-28", "2026-11-10"]),
                           "MTRL": [5021, 5021, 5021, 5021, 5001], "QTY": [10.0] * 5})
    d = deliv(_daily_sales(5021, "2026-05-04", "2026-05-31", 1)
              + _daily_sales(5021, "2026-06-01", "2026-06-10", 2))
    p = promos_df([(1, "Μπύρες", None, "Μπύρες", "2026-06-01", "2026-06-10", 10),
                   (2, "Μπύρες", None, "Μπύρες", "2026-11-09", "2026-11-27", 10),
                   (3, "lager", 5021, None, "2026-11-09", "2026-11-15", 5)])  # overlaps
    out = w.apply_promos(demand, d, items_df(), p, TODAY)
    assert list(out["QTY"]) == [10, 20, 20, 10, 10]


def test_daily_demand_adds_targets_and_background_once():
    targets = pd.DataFrame({"TRDR": [1], "MTRL": [5001], "DATE": [T("2026-11-03")],
                            "QTY": [6.0], "SUGGESTION_ID": [None], "SOURCE": ["ΠΡΟΒΛΕΨΗ"]})
    bg = pd.DataFrame({"DATE": pd.date_range(TODAY, periods=5), "MTRL": 5001, "QTY": 0.5})
    table = w.daily_demand(targets, bg, deliv([]), items_df(), promos_df([]), TODAY, days=5)
    assert table[5001].tolist() == [0.5, 0.5, 6.5, 0.5, 0.5]
    assert table[5002].sum() == 0 and 5081 not in table.columns


# --- simulation -------------------------------------------------------------

def flat(qty, days=130):
    return pd.Series(float(qty), index=pd.date_range(TODAY, periods=days))


def test_no_order_when_stock_lasts():
    orders, level = w.simulate_item(flat(1), 1000, 3, 10, today=TODAY)
    assert orders == []
    assert level[0] == 999


def test_order_timing_and_quantity():
    # 10 a day, safety 1 week = 70. End of day i: 300 - 10(i+1) -> under 70 on day 23 (24/11).
    orders, level = w.simulate_item(flat(10), 300, 3, 1, safety_weeks=1, today=TODAY)
    o = orders[0]
    assert o["BREACH_DATE"] == T("2026-11-24")
    assert o["ORDER_BY"] == T("2026-11-20")   # Fri; +3 working days: Sat, Mon, Tue
    assert o["ARRIVAL"] == T("2026-11-24")
    assert o["FLAG"] == ""
    assert o["SAFETY_STOCK"] == 70
    # end of 24/11 without the order: 60; 2 more weeks: 140; QTY = 140 + 70 - 60
    assert o["QTY"] == 150
    assert o["STOCKOUT_DATE"] == T("2026-11-30")
    assert level[(o["ARRIVAL"] - TODAY).days] == 210
    assert level[: (o["ARRIVAL"] - TODAY).days].min() >= 70


def test_simulation_goes_on_after_the_arrival():
    orders, level = w.simulate_item(flat(10), 300, 3, 1, today=TODAY)
    assert len(orders) > 2
    for a, b in zip(orders, orders[1:]):
        assert b["ORDER_BY"] >= a["ARRIVAL"]
    horizon = level[:90]
    assert horizon.min() >= 70 - 1e-9 or orders[-1]["ARRIVAL"] > TODAY + pd.Timedelta(days=89)
    for o in orders:
        assert w.add_working_days(o["ORDER_BY"], 3) == o["ARRIVAL"]
        assert w.working_day(o["ORDER_BY"]) and w.working_day(o["ARRIVAL"])


def test_min_order_qty():
    orders, _ = w.simulate_item(flat(10), 300, 3, 500, today=TODAY)
    assert orders[0]["QTY"] == 500 and orders[0]["NEEDED_QTY"] == 150


def test_quantity_is_rounded_up():
    orders, _ = w.simulate_item(flat(1.05), 30, 3, 1, today=TODAY)
    assert isinstance(orders[0]["QTY"], int)
    assert orders[0]["QTY"] == int(np.ceil(orders[0]["NEEDED_QTY"]))


def test_too_late_when_order_by_is_before_today():
    # 100 - 10(i+1) < 70 already on 4/11: should have ordered on Sat 31/10
    orders, _ = w.simulate_item(flat(10), 100, 3, 1, today=TODAY)
    o = orders[0]
    assert o["FLAG"] == w.TOO_LATE
    assert o["IDEAL_ORDER_BY"] == T("2026-10-31")
    assert o["ORDER_BY"] == T("2026-11-02")       # first working day from today (Sunday)
    assert o["ARRIVAL"] == T("2026-11-05")
    assert all(x["FLAG"] == "" for x in orders[1:])


def test_approved_orders_on_the_way_are_counted():
    orders, _ = w.simulate_item(flat(10), 100, 3, 1, today=TODAY,
                                incoming={T("2026-11-03"): 1000})
    assert orders == [] or orders[0]["FLAG"] == ""


def test_affected_deliveries_before_arrival():
    targets = pd.DataFrame({
        "TRDR": [1, 2, 3, 4], "MTRL": [5001, 5001, 5001, 5002],
        "DATE": pd.to_datetime(["2026-11-02", "2026-11-04", "2026-11-05", "2026-11-03"]),
        "QTY": [5.0, 6.0, 7.0, 8.0], "SUGGESTION_ID": [11, None, 13, 14],
        "SOURCE": ["ΠΡΟΤΑΣΗ_STORE", "ΠΡΟΒΛΕΨΗ", "ΠΡΟΤΑΣΗ_STORE", "ΠΡΟΤΑΣΗ_STORE"]})
    level = np.array([5, 1, -1, -6, 0], dtype=float)
    out = w.affected_deliveries(targets, 5001, TODAY, T("2026-11-05"), level)
    assert [a["TRDR"] for a in out] == [1, 2]
    assert out[0] == {"TRDR": 1, "MTRL": 5001, "DELIVERY_DATE": "2026-11-02", "QTY": 5.0,
                      "SUGGESTION_ID": 11, "SHORT": False}
    assert out[1]["SUGGESTION_ID"] is None and out[1]["SHORT"] is True


# --- plan and texts ---------------------------------------------------------

def _suppliers():
    return pd.DataFrame({"SUPPLIER_ID": ["S1", "S2", "S3"], "NAME": ["Νερά Α.Ε.", "Μπύρα Α.Ε.", "Γλυκά Ο.Ε."],
                         "MTRCATEGORY": ["Νερά", "Μπύρες", "Εποχικά"], "LEAD_TIME_DAYS": [3, 5, 7],
                         "MIN_ORDER_QTY": [10, 10, 10], "CONTACT": [None] * 3})


def _customers():
    return pd.DataFrame({"TRDR": [1, 2], "CODE": ["a", "b"], "NAME": ["Α", "Β"],
                         "TRDCATEGORY": ["Mini Market"] * 2, "AREA": ["Αθήνα"] * 2,
                         "CITY": ["Αθήνα"] * 2, "SALESMAN": [101] * 2,
                         "INSDATE": ["2020-01-01"] * 2, "ISACTIVE": [1, 1]})


def test_plan_too_late_lists_the_customer_deliveries(tmp_path):
    # Stock 0 (no sales last November), a target delivery in 3 days: too late.
    d = deliv([(dd, 1, 5001, 20) for dd in pd.date_range("2026-07-01", "2026-10-31", freq="7D")])
    det = details_df([(1, 5001, "2026-11-03", 20, "ΚΑΝΟΝΙΚΗ")])
    sugg = [suggestion(42, 1, "2026-11-03", [(5001, 25)])]
    orders, stock = w.plan(d, items_df(), _customers(), det, sugg, promos_df([]), _suppliers(),
                           TODAY, stock_path=tmp_path / "none.csv")
    first = orders[orders["MTRL"] == 5001].iloc[0]
    assert first["FLAG"] == w.TOO_LATE
    assert first["AFFECTED"][0]["SUGGESTION_ID"] == 42
    assert first["AFFECTED"][0]["QTY"] == 25
    assert first["QTY"] >= 10
    assert "έπρεπε να είχε παραγγελθεί" in first["REASON"]
    assert "επηρεάζονται 1 παραδόσεις" in first["REASON"]
    assert first["VALUE_EUR"] == round(first["QTY"] * 4.2, 2)


def test_reason_template_has_the_numbers():
    o = {"ITEM_NAME": "Νερό 0,5L", "UNIT": "κιβ.", "STOCKOUT_DATE": T("2026-11-20"),
         "BREACH_DATE": T("2026-11-16"), "SAFETY_STOCK": 70, "LEAD_TIME_DAYS": 3,
         "QTY": 150, "NEEDED_QTY": 150, "MIN_ORDER_QTY": 50, "ORDER_BY": T("2026-11-12"),
         "ARRIVAL": T("2026-11-16"), "IDEAL_ORDER_BY": T("2026-11-12"), "FLAG": "",
         "AFFECTED": []}
    assert w.reason_text(o) == ("Το «Νερό 0,5L» τελειώνει περίπου 20/11 με τη ζήτηση που "
                                "προβλέπουμε. Ο προμηθευτής θέλει 3 εργάσιμες· παράγγειλε "
                                "150 κιβ. έως 12/11.")
    o.update(QTY=50, NEEDED_QTY=12.3)
    assert "Χρειάζονται 13 κιβ., αλλά η ελάχιστη ποσότητα του προμηθευτή είναι 50 κιβ." \
        in w.reason_text(o)


# --- full run on the real data ----------------------------------------------

def test_full_run_saves_once_and_cli_works(tmp_path, monkeypatch, capsys):
    det = details_df([(1001, 5003, "2026-11-06", 3, "ΚΑΝΟΝΙΚΗ"),
                      (1001, 5013, "2026-11-19", 5, "ΚΑΝΟΝΙΚΗ")])
    det_path = tmp_path / "details.csv"
    det.to_csv(det_path, index=False, date_format="%Y-%m-%d", encoding="utf-8")
    db = tmp_path / "t.db"
    kw = dict(db=db, details_path=det_path, settings_path=tmp_path / "no.json",
              stock_path=tmp_path / "no.csv")
    orders, result, stock = w.run(TODAY, **kw)
    assert len(orders) > 0 and len(result["saved"]) == len(orders)
    assert (orders["ORDER_BY"] >= T("2026-11-02")).all()
    assert orders["ORDER_BY"].is_monotonic_increasing
    assert (stock["STOCK_SOURCE"] == w.ESTIMATE).all() and len(stock) == 80

    _, again, _ = w.run(TODAY, **kw)
    assert again == {"saved": [], "skipped": len(orders)}

    pid = result["saved"][0]
    monkeypatch.setattr(store, "PURCHASE_DIR", tmp_path / "po")
    assert w.main(["--db", str(db), "approve", str(pid)]) == 0
    assert w.main(["--db", str(db), "approve", str(pid)]) == 1
    assert w.main(["--db", str(db), "reject", str(result["saved"][1]), "έχουμε απόθεμα"]) == 0
    assert w.main(["--db", str(db), "list"]) == 0
    out = capsys.readouterr().out
    assert "ΕΓΚΡΙΘΗΚΕ" in out and "ΑΠΟΡΡΙΦΘΗΚΕ" in out
    assert (tmp_path / "po" / f"{pid}.json").exists()

    # an approved order on the way is counted on the next run
    approved = store.list_purchase_suggestions(store.APPROVED, db=db)[0]
    incoming = w.incoming_from_store(db, TODAY)
    assert incoming[approved["mtrl"]][T(approved["arrival"])] == approved["qty"]
