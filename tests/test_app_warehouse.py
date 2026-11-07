"""Helpers of the Αποθήκη page (app/warehouse.py), with fake purchase orders."""
import json
import sys
import types

import pandas as pd

from app import warehouse as wh

SHORT = {
    "supplier_id": "S07", "supplier_name": "Προμηθευτής Απορρυπαντικών Α.Ε.",
    "order_by": "2026-11-02", "arrival": pd.Timestamp("2026-11-09"), "status": "ΕΛΛΕΙΨΗ",
    "short": True, "late": False, "deliveries_before_arrival": 11,
    "total_value_eur": 460.8, "reason": "Το «Λαμπρυντικό» τελειώνει περίπου 17/11.",
    "note": "Καλημέρα σας.",
    "lines": [{"purchase_id": 1, "mtrl": 5075, "item_name": "Λαμπρυντικό πλυντηρίου 5L",
               "qty": 36, "unit": "τεμ.", "value_eur": 460.8, "flags": ["ΕΛΛΕΙΨΗ"]}],
    "affected_deliveries": [
        {"trdr": 1036, "customer_name": "Mini Market Ήλιος", "date": "2026-11-03", "qty": 3,
         "suggestion_id": 1},
        {"trdr": 1123, "customer_name": "Μπαρ Πέτρα", "date": "2026-11-02", "qty": 2,
         "suggestion_id": None},
        {"trdr": 1010, "customer_name": "Mini Market Οικονομικό", "date": "2026-11-05", "qty": 1,
         "suggestion_id": None}],
}
PAPER = {
    "supplier_id": "S05", "supplier_name": "Προμηθευτής Χαρτικών Α.Ε.", "order_by": "2026-11-06",
    "arrival": "2026-11-14", "status": "", "short": False, "late": False,
    "total_value_eur": 1551.5, "reason": "", "note": "",
    "lines": [{"purchase_id": 2, "mtrl": 5059, "item_name": "Χαρτί υγείας 48 ρολά", "qty": 91,
               "unit": "κιβ.", "value_eur": 1319.5, "flags": []},
              {"purchase_id": 3, "mtrl": 5064, "item_name": "Μεμβράνη τροφίμων", "qty": 40,
               "unit": "τεμ.", "value_eur": 232.0, "flags": []}],
    "affected_deliveries": [],
}
COFFEE = {**PAPER, "supplier_id": "S02", "supplier_name": "Προμηθευτής Καφέ Ο.Ε.",
          "order_by": "2026-11-02", "total_value_eur": 4526.0,
          "lines": [{"purchase_id": 4, "mtrl": 5033, "item_name": "Στιγμιαίος καφές 200g",
                     "qty": 73, "unit": "τεμ.", "value_eur": 4526.0, "flags": []}]}


# the real case of 1/11: 44 units in stock, 27 for the 11 deliveries until 9/11
LATE = {**SHORT, "status": "ΑΡΓΗΣΑΜΕ", "short": False, "late": True, "supplier_id": "S08",
        "lines": [{**SHORT["lines"][0], "flags": ["ΑΡΓΗΣΑΜΕ"]}], "affected_deliveries": []}


def test_short_orders_come_first_then_late_then_by_day():
    assert [o["supplier_id"] for o in wh.sort_orders([PAPER, COFFEE, LATE, SHORT])] == \
        ["S07", "S08", "S02", "S05"]


def test_metrics():
    m = wh.metrics([SHORT, PAPER, COFFEE])
    assert m == {"orders": 3, "short_items": 1, "short_deliveries": 3, "late_items": 0,
                 "value_eur": round(460.8 + 1551.5 + 4526.0, 2)}
    assert wh.metrics([]) == {"orders": 0, "short_items": 0, "short_deliveries": 0,
                              "late_items": 0, "value_eur": 0}
    m = wh.metrics([LATE, PAPER])
    assert m["short_items"] == 0 and m["short_deliveries"] == 0 and m["late_items"] == 1


def test_status_and_tags():
    assert wh.status(SHORT) == "ΕΛΛΕΙΨΗ" and wh.tag_of(SHORT) == ("Έλλειψη", "short")
    assert wh.status(LATE) == "ΑΡΓΗΣΑΜΕ" and wh.tag_of(LATE) == ("Αργήσαμε", "late")
    assert wh.status(PAPER) == "" and wh.tag_of(PAPER) == ("Αγορά", "buy")
    old = {k: v for k, v in SHORT.items() if k != "status"}  # only "short", as before
    assert wh.status(old) == "ΕΛΛΕΙΨΗ"
    assert wh.short_lines(LATE) == [] and wh.late_lines(LATE) == LATE["lines"]


def test_late_text_says_customers_are_covered():
    assert wh.late_text(LATE) == (
        "Οι 11 παραδόσεις σε πελάτες μέχρι την άφιξη καλύπτονται από το απόθεμα, αλλά θα "
        "πέσουμε κάτω από το απόθεμα ασφαλείας. Παράγγειλε σήμερα.")
    assert wh.affected_phrase(3) == "Επηρεάζονται 3 παραδόσεις σε πελάτες"
    assert wh.affected_phrase(1) == "Επηρεάζεται 1 παράδοση σε πελάτη"


def test_card_lines():
    assert wh.card_line(SHORT) == \
        "Λαμπρυντικό πλυντηρίου 5L · 36 τεμ. · έως Δευτ. 2/11 · 460,80 €"
    assert wh.card_line(PAPER) == \
        "2 είδη (Χαρτί υγείας 48 ρολά, Μεμβράνη τροφίμων) · έως Παρ. 6/11 · 1.551,50 €"
    many = {**PAPER, "lines": PAPER["lines"] * 3}
    assert wh.card_line(many).startswith("6 είδη (") and "και 3 ακόμη" in wh.card_line(many)


def test_keys_are_supplier_and_day():
    assert wh.order_key(SHORT) == "S07|2026-11-02"
    assert wh.order_key({**SHORT, "order_by": pd.Timestamp("2026-11-02")}) == "S07|2026-11-02"
    assert wh.later_count([SHORT, PAPER, COFFEE], [SHORT]) == 2


def test_lines_table_keeps_numbers():
    df = wh.lines_frame(PAPER)
    assert list(df.columns) == ["Είδος", "Ποσότητα", "Μονάδα", "Αξία"]
    assert str(df["Ποσότητα"].dtype) == "int64" and df["Αξία"].dtype == float
    assert list(df.sort_values("Ποσότητα")["Ποσότητα"]) == [40, 91]


def test_affected_deliveries_text_in_date_order():
    text = wh.affected_text(SHORT)
    assert text == "Μπαρ Πέτρα · 2/11 · 2 — Mini Market Ήλιος · 3/11 · 3 — + 1 ακόμη"


def test_home_sentences():
    s = wh.home_sentence(SHORT)
    assert "Λαμπρυντικό πλυντηρίου 5L" in s and "36 τεμ." in s
    assert "τη Δευτέρα 2/11" in s and "9/11" in s and "επηρεάζονται 3 παραδόσεις" in s
    s = wh.home_sentence(LATE)
    assert s.startswith("Αργήσαμε για: <b>Λαμπρυντικό πλυντηρίου 5L</b>")
    assert "Οι πελάτες καλύπτονται" in s and "επηρεάζ" not in s
    s = wh.home_sentence(COFFEE)
    assert s.startswith("<b>Προμηθευτής Καφέ Ο.Ε.:</b> 73 τεμ. Στιγμιαίος καφές 200g")
    assert "έως Δευτέρα 2/11" in s


def test_contexts_are_json_ready():
    ctx = wh.order_context(SHORT)
    json.dumps(ctx, ensure_ascii=False)
    assert ctx["purchase_order"]["arrival"] == "2026-11-09"
    assert wh.week_context(None) == wh.SOON
    assert len(wh.week_context([PAPER] * 20)) == 15


def test_agent_missing_or_present(monkeypatch):
    empty = types.ModuleType("rafi.agents.warehouse_agent")
    monkeypatch.setitem(sys.modules, "rafi.agents.warehouse_agent", empty)
    assert wh.agent() is None
    monkeypatch.setitem(sys.modules, "rafi.agents.warehouse_agent", None)  # import fails
    assert wh.agent() is None
    full = types.ModuleType("rafi.agents.warehouse_agent")
    full.open_purchase_orders = lambda days=7, db=None: [SHORT]
    full.approve_purchase_order = lambda supplier_id, order_by, user: True
    full.reject_purchase_order = lambda supplier_id, order_by, reason, user: True
    monkeypatch.setitem(sys.modules, "rafi.agents.warehouse_agent", full)
    funcs = wh.agent()
    assert funcs["open_purchase_orders"](days=7) == [SHORT]
