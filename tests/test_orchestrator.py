"""Orchestrator, customer agent, supplier orders per group, shortages and the morning summary."""
import json
from pathlib import Path

import pandas as pd
import pytest

from rafi import llm, store
from rafi.agents import customer_agent as ca
from rafi.agents import orchestrator as orch
from rafi.agents import warehouse_agent as wh
from rafi.suggestions import EXTRA_ROUTE, LOSS, OFFER, ORDER, URGENT

ROOT = Path(__file__).resolve().parents[1]


# --- customer agent on the real data ------------------------------------------

@pytest.fixture(scope="module")
def data():
    return ca.load_data()


@pytest.fixture(scope="module")
def details_file(data, tmp_path_factory):
    """The hybrid forecast, computed once and written like scripts/make_submission.py does."""
    path = tmp_path_factory.mktemp("details") / "forecast_details.csv"
    details, source = ca.forecast_details(data, details_path=path)
    assert source == "υπολογισμός" and path.exists()
    return path


def _at_risk(suggestions):
    return round(sum(s["VALUE_EUR"] for s in suggestions if s["ACTION_TYPE"] in (ORDER, LOSS)), 2)


def test_customer_agent_gives_the_same_results(data, details_file, tmp_path):
    out = ca.run(db=tmp_path / "a.db", data=data, details_path=details_file, ai=False,
                 route_cost=60)
    assert out["details_source"] == "αρχείο"
    assert len(out["suggestions"]) == 179 and len(out["result"]["saved"]) == 179
    assert _at_risk(out["suggestions"]) == 42767.0
    assert {s["payload"]["TEXT_SOURCE"] for s in out["suggestions"]} == {"ΠΡΟΤΥΠΟ"}
    again = ca.run(db=tmp_path / "a.db", data=data, details_path=details_file, ai=False)
    assert again["result"] == {"saved": [], "skipped": 179}


def test_ai_texts_only_for_needs_you_and_the_top_n(data, details_file, tmp_path):
    seen = []

    def writer(facts):
        seen.append(facts["trdr"])
        return {"REASON": "AI αιτιολόγηση", "SALES_NOTE": "AI σημείωμα"}

    out = ca.run(db=tmp_path / "b.db", data=data, details_path=details_file, top_n=5,
                 text_writer=writer, route_cost=60)
    sugg = out["suggestions"]
    urgent = [s for s in sugg if ca.needs_you(s)]
    rest = [s for s in sugg if not ca.needs_you(s)]
    assert len(urgent) == 7 and out["ai_chosen"] == out["ai_written"] == 7 + 5
    assert all(s["payload"]["TEXT_SOURCE"] == "AI" for s in urgent + rest[:5])
    assert all(s["payload"]["TEXT_SOURCE"] == "ΠΡΟΤΥΠΟ" for s in rest[5:])
    assert rest[5]["payload"]["REASON"] != "AI αιτιολόγηση"
    # the texts change, the numbers do not
    assert len(sugg) == 179 and _at_risk(sugg) == 42767.0
    assert [s["PRIORITY"] for s in sugg] == list(range(1, 180))
    saved = store.list_suggestions(db=tmp_path / "b.db")
    assert sum(r["payload"]["TEXT_SOURCE"] == "AI" for r in saved) == 12


def test_top_n_comes_from_the_settings(data, details_file, tmp_path):
    from rafi.settings import DEFAULTS, ai_texts_top_n
    assert DEFAULTS["ai_texts_top_n"] == 20 and ai_texts_top_n() == 20
    out = ca.run(db=tmp_path / "c.db", data=data, details_path=details_file,
                 text_writer=lambda f: {"REASON": "x", "SALES_NOTE": "y"})
    assert out["ai_chosen"] == 7 + 20


def test_failing_or_missing_llm_keeps_the_templates(data, details_file, tmp_path, monkeypatch):
    def broken(facts):
        raise RuntimeError("model down")

    out = ca.run(db=tmp_path / "d.db", data=data, details_path=details_file, text_writer=broken)
    assert out["ai_written"] == 0 and len(out["result"]["saved"]) == 179
    assert {s["payload"]["TEXT_SOURCE"] for s in out["suggestions"]} == {"ΠΡΟΤΥΠΟ"}

    monkeypatch.setattr(llm, "available", lambda *a, **k: False)
    out = ca.run(db=tmp_path / "e.db", data=data, details_path=details_file)
    assert out["ai_chosen"] == 0 and "δεν απαντά" in out["ai_status"]
    assert {s["payload"]["TEXT_SOURCE"] for s in out["suggestions"]} == {"ΠΡΟΤΥΠΟ"}


def test_mock_llm_writes_marked_texts(data, details_file, tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_MODE", "mock")
    out = ca.run(db=tmp_path / "f.db", data=data, details_path=details_file, top_n=2)
    ai = [s for s in out["suggestions"] if s["payload"]["TEXT_SOURCE"] == "AI"]
    assert len(ai) == 9
    assert all(s["payload"]["REASON"].startswith(llm.MOCK_TAG) for s in ai)


def test_no_ai_texts_for_suggestions_that_will_be_skipped(data, details_file, tmp_path):
    db = tmp_path / "g.db"
    ca.run(db=db, data=data, details_path=details_file, ai=False)
    calls = []
    out = ca.run(db=db, data=data, details_path=details_file,
                 text_writer=lambda f: calls.append(1) or {"REASON": "x", "SALES_NOTE": "y"})
    assert calls == [] and out["result"]["saved"] == []


# --- small stores for the warehouse and the coordination ---------------------

def sugg(sid, kind=ORDER, flags=(), value=100.0, trdr=None, mtrl=5001, qty=3):
    trdr = trdr or 1000 + sid
    return {"ACTION_TYPE": kind, "VALUE_EUR": value, "PRIORITY": sid, "FLAGS": list(flags),
            "payload": {"TRDR": trdr, "CUSTOMER_CODE": "x", "SALESMAN": 101,
                        "SUGGESTED_DELIVERY_DATE": "2026-11-03", "ACTION_TYPE": kind,
                        "FLAGS": list(flags), "REASON": f"Λόγος {sid}. Δεύτερη πρόταση.",
                        "SALES_NOTE": "σ", "TEXT_SOURCE": "ΠΡΟΤΥΠΟ",
                        "LINES": [{"MTRL": mtrl, "QTY1": qty, "PRICER": 1.0}]}}


def purchase(mtrl, supplier="S1", order_by="2026-11-02", arrival="2026-11-09", qty=10,
             flag="", stock=0.0, affected=(), name=None):
    return {"MTRL": mtrl, "SUPPLIER_ID": supplier, "SUPPLIER_NAME": f"Προμηθευτής {supplier}",
            "ORDER_BY": order_by, "ARRIVAL": arrival, "QTY": qty, "VALUE_EUR": qty * 2.5,
            "FLAG": flag, "ITEM_NAME": name or f"Είδος {mtrl}", "UNIT": "κιβ.",
            "STOCK_NOW": stock, "STOCKOUT_DATE": "2026-11-07", "BREACH_DATE": "2026-11-05",
            "SAFETY_STOCK": 4, "AFFECTED": list(affected),
            "REASON": f"Το «{name or f'Είδος {mtrl}'}» τελειώνει περίπου 7/11."}


def affected(trdr, sid, qty=3.0, date="2026-11-03", mtrl=5001, missing=None):
    a = {"TRDR": trdr, "MTRL": mtrl, "DELIVERY_DATE": date, "QTY": qty,
         "SUGGESTION_ID": sid, "SHORT": None}
    if missing is not None:  # as the warehouse agent writes it (cover_deliveries)
        a["MISSING"] = missing
    return a


# --- add_flag ----------------------------------------------------------------

def test_add_flag(tmp_path):
    db = tmp_path / "t.db"
    store.save_new([sugg(1), sugg(2)], db=db)
    assert store.add_flag(1, store.WAREHOUSE_SHORT, "Λείπουν 2 κιβ.", "orchestrator", db=db)
    p = store.get(1, db=db)["payload"]
    assert p["FLAGS"] == [store.WAREHOUSE_SHORT]
    assert p["FLAG_NOTES"] == {store.WAREHOUSE_SHORT: "Λείπουν 2 κιβ."}
    h = store.history(suggestion_id=1, db=db)[-1]
    assert h["action"] == store.COORDINATE == "ΣΥΝΤΟΝΙΣΜΟΣ" and h["user"] == "orchestrator"
    assert h["details"] == {"flag": store.WAREHOUSE_SHORT, "note": "Λείπουν 2 κιβ."}
    # the same note again: nothing changes, nothing logged
    assert not store.add_flag(1, store.WAREHOUSE_SHORT, "Λείπουν 2 κιβ.", "orchestrator", db=db)
    assert len(store.history(suggestion_id=1, db=db)) == 2
    # another item short: the note grows, the flag stays once
    assert store.add_flag(1, store.WAREHOUSE_SHORT, "Και 1 τεμ.", "orchestrator", db=db)
    p = store.get(1, db=db)["payload"]
    assert p["FLAGS"] == [store.WAREHOUSE_SHORT]
    assert p["FLAG_NOTES"][store.WAREHOUSE_SHORT] == "Λείπουν 2 κιβ. Και 1 τεμ."
    # only while ΠΡΟΤΑΣΗ
    store.reject(2, "όχι", "101", db=db)
    with pytest.raises(store.StoreError):
        store.add_flag(2, store.WAREHOUSE_SHORT, "x", "orchestrator", db=db)
    with pytest.raises(store.StoreError):
        store.add_flag(99, store.WAREHOUSE_SHORT, "x", "orchestrator", db=db)


def test_active_pairs(tmp_path):
    db = tmp_path / "t.db"
    store.save_new([sugg(1, mtrl=5001), sugg(2, mtrl=5002)], db=db)
    store.reject(2, "όχι", "101", db=db)
    assert store.active_pairs(db=db) == {(1001, 5001)}


# --- purchase orders per supplier ----------------------------------------------

@pytest.fixture
def pdb(tmp_path):
    db = tmp_path / "p.db"
    store.save_new([sugg(1, LOSS, value=10.0), sugg(2, value=500.0), sugg(3, value=50.0)], db=db)
    store.save_purchase_new([
        # stock 5 for 11 units: the loss risk (1) is covered, three deliveries are not
        purchase(5001, "S1", flag=wh.SHORT, stock=5,
                 affected=[affected(1003, 3, missing=3.0), affected(1002, 2, missing=1.0),
                           affected(1001, 1, missing=0.0),
                           affected(1099, None, qty=2, date="2026-11-04", missing=2.0)]),
        purchase(5002, "S1"),
        purchase(5003, "S2", order_by="2026-11-05", arrival="2026-11-10"),
        purchase(5004, "S2", order_by="2026-11-20", arrival="2026-11-25"),   # after 7 days
        purchase(5005, "S3", order_by="2026-11-03"),
    ], db=db)
    store.approve_purchase(5, "αποθήκη", db=db, out_dir=tmp_path / "po")    # not open any more
    return db


def test_open_purchase_orders_are_grouped(pdb):
    orders = wh.open_purchase_orders(7, db=pdb)
    assert [(o["supplier_id"], o["order_by"]) for o in orders] == [("S1", "2026-11-02"),
                                                                   ("S2", "2026-11-05")]
    s1 = orders[0]
    assert set(s1) >= {"supplier_id", "supplier_name", "order_by", "arrival", "status", "short",
                       "late", "total_value_eur", "reason", "note", "lines",
                       "affected_deliveries", "deliveries_before_arrival"}
    assert s1["status"] == wh.SHORT and s1["short"] and not s1["late"]
    assert orders[1]["status"] == "" and not orders[1]["short"]
    assert s1["supplier_name"] == "Προμηθευτής S1" and s1["arrival"] == "2026-11-09"
    assert s1["total_value_eur"] == 50.0
    assert [l["mtrl"] for l in s1["lines"]] == [5001, 5002]
    assert set(s1["lines"][0]) == {"purchase_id", "mtrl", "item_name", "qty", "unit",
                                   "value_eur", "flags"}
    assert s1["lines"][0]["flags"] == [wh.SHORT] and s1["lines"][1]["flags"] == []
    # only the deliveries that are not covered; all 4 are before the arrival
    a = s1["affected_deliveries"]
    assert [x["trdr"] for x in a] == [1003, 1002, 1099] and s1["deliveries_before_arrival"] == 4
    assert {"trdr", "customer_name", "date", "qty", "missing", "suggestion_id"} <= set(a[0])
    assert a[0]["customer_name"] and a[1]["missing"] == 1.0  # TRDR 1003 is in TRDR.csv
    # one reason for the whole order, in full sentences
    assert "2 είδη" in s1["reason"] and "«Είδος 5001»" in s1["reason"]
    assert "Δεν φτάνει το απόθεμα για το «Είδος 5001»: επηρεάζονται 3 παραδόσεις" in s1["reason"]
    assert s1["note"].startswith(llm.HELLO) and "10 κιβ. «Είδος 5001»" in s1["note"]
    assert s1["note_source"] == llm.TEMPLATE
    # a one-line order keeps the line's own reason
    assert orders[1]["reason"] == "Το «Είδος 5003» τελειώνει περίπου 7/11."
    assert len(wh.open_purchase_orders(30, db=pdb)) == 3


def test_open_purchase_orders_late_is_not_a_shortage(tmp_path):
    db = tmp_path / "t.db"
    covered = [affected(1000 + i, None, qty=2.0, missing=0.0) for i in range(11)]
    store.save_purchase_new([purchase(5075, "S7", flag=wh.LATE, stock=44, affected=covered,
                                      name="Λαμπρυντικό πλυντηρίου 5L"),
                             purchase(5076, "S7", name="Μαλακτικό 4L")], db=db)
    o = orch.morning_summary(db=db)["warehouse"]
    assert o["short_count"] == 0 and o["late_count"] == 1
    g = o["orders"][0]
    assert g["status"] == wh.LATE and g["late"] and not g["short"]
    assert g["affected_deliveries"] == [] and g["deliveries_before_arrival"] == 11
    assert g["lines"][0]["flags"] == [wh.LATE]
    assert "Αργήσαμε για το «Λαμπρυντικό πλυντηρίου 5L»" in g["reason"]
    assert "επηρεάζ" not in g["reason"]
    assert orch.coordinate_shortages(db=db) == []


def test_old_store_rows_read_as_shortage(tmp_path):
    db = tmp_path / "t.db"
    store.save_purchase_new([purchase(5001, flag=wh.OLD_TOO_LATE, stock=0,
                                      affected=[affected(1001, None), affected(1002, None)])],
                            db=db)
    g = wh.open_purchase_orders(7, db=db)[0]
    assert g["status"] == wh.SHORT and len(g["affected_deliveries"]) == 2


def test_approve_and_reject_a_whole_order(pdb, tmp_path):
    files = wh.approve_purchase_order("S1", "2026-11-02", "αποθήκη", db=pdb,
                                      out_dir=tmp_path / "po")
    assert len(files) == 2 and all(Path(f).exists() for f in files)
    rows = {r["id"]: r for r in store.list_purchase_suggestions(db=pdb)}
    assert rows[1]["status"] == rows[2]["status"] == store.APPROVED
    with pytest.raises(store.StoreError):  # never twice
        wh.approve_purchase_order("S1", "2026-11-02", "αποθήκη", db=pdb)

    with pytest.raises(store.StoreError):
        wh.reject_purchase_order("S2", "2026-11-05", " ", "αποθήκη", db=pdb)
    assert wh.reject_purchase_order("S2", "2026-11-05", "έχουμε απόθεμα", "αποθήκη",
                                    db=pdb) == [3]
    rows = {r["id"]: r for r in store.list_purchase_suggestions(db=pdb)}
    assert rows[3]["status"] == store.REJECTED
    assert [o["supplier_id"] for o in wh.open_purchase_orders(30, db=pdb)] == ["S2"]  # 5004


def test_supplier_notes_ai_and_fallback(pdb, monkeypatch):
    monkeypatch.setenv("LLM_MODE", "mock")
    assert wh.write_group_notes(7, db=pdb) == {"orders": 2, "ai": 2}
    orders = wh.open_purchase_orders(7, db=pdb)
    assert all(o["note_source"] == llm.AI and o["note"].startswith(llm.MOCK_TAG) for o in orders)
    assert wh.open_purchase_orders(30, db=pdb)[2]["note_source"] == llm.TEMPLATE  # not written
    # a new line in the same order: the old note no longer covers it, back to the template
    store.save_purchase_new([purchase(5006, "S1")], db=pdb)
    assert wh.open_purchase_orders(7, db=pdb)[0]["note_source"] == llm.TEMPLATE

    monkeypatch.setattr(llm, "available", lambda *a, **k: False)
    assert wh.write_group_notes(7, db=pdb)["ai"] == 1  # S2 keeps its AI note, S1 template
    assert wh.open_purchase_orders(7, db=pdb)[0]["note_source"] == llm.TEMPLATE


def test_supplier_payload_matches_write_texts():
    group = {"supplier_id": "S1", "supplier_name": "Α", "order_by": "2026-11-02",
             "arrival": "2026-11-09", "reason": "Λόγος.",
             "lines": [{"item_name": "Νερό", "qty": 5, "unit": "κιβ."}]}
    t = llm.template_for(wh.supplier_payload(group), "supplier")
    assert t["reason"] == "Λόγος." and "5 κιβ. «Νερό», με παράδοση έως 9/11" in t["note"]


# --- coordination of shortages -------------------------------------------------

def test_shortage_goes_to_priorities_first(pdb):
    out = orch.coordinate_shortages(db=pdb)
    # stock 5: the loss risk (1) gets 3, the 500 € order (2) gets 2 and misses 1,
    # the 50 € order (3) misses 3, the forecast delivery without a suggestion misses 2
    got = {(e["suggestion_id"], e["missing"], e["flagged"]) for e in out}
    assert got == {(2, 1.0, True), (3, 3.0, True), (None, 2.0, False)}
    s1 = store.get(1, db=pdb)["payload"]
    assert store.WAREHOUSE_SHORT not in s1["FLAGS"]
    s3 = store.get(3, db=pdb)["payload"]
    assert store.WAREHOUSE_SHORT in s3["FLAGS"]
    note = s3["FLAG_NOTES"][store.WAREHOUSE_SHORT]
    assert "«Είδος 5001»" in note and "λείπουν 3 κιβ." in note and "φτάνει 9/11" in note
    assert store.history(suggestion_id=3, db=pdb)[-1]["action"] == store.COORDINATE
    # again: no new history entries
    n = len(store.history(db=pdb))
    orch.coordinate_shortages(db=pdb)
    assert len(store.history(db=pdb)) == n


def test_no_shortage_when_the_stock_is_enough(tmp_path):
    db = tmp_path / "t.db"
    store.save_new([sugg(1)], db=db)
    store.save_purchase_new([purchase(5001, flag=wh.LATE, stock=50,
                                      affected=[affected(1001, 1, missing=0.0)])], db=db)
    assert orch.coordinate_shortages(db=db) == []
    assert store.get(1, db=db)["payload"]["FLAGS"] == []


# --- morning summary ---------------------------------------------------------------

def test_morning_summary(pdb):
    store.save_new([sugg(4, ORDER, [URGENT, EXTRA_ROUTE], value=70.0, mtrl=5010),
                    sugg(5, ORDER, ["ΟΡΙΑΚΟ"], value=30.0, mtrl=5010),
                    sugg(6, OFFER, value=40.0, mtrl=5010)], db=pdb)
    m = orch.morning_summary(db=pdb)
    c = m["customers"]
    assert c["counts"] == {"pending": 6, "extra_routes": 1, "loss_risks": 1, "marginal": 1,
                           "orders": 4, "offers": 1}
    assert c["value_at_risk_eur"] == 10 + 500 + 50 + 70 + 30
    assert c["offers_value_eur"] == 40.0
    assert [(n["suggestion_id"], n["kind"]) for n in c["needs_you"]] == [(1, LOSS), (4, URGENT)]
    n = c["needs_you"][0]
    assert set(n) == {"suggestion_id", "trdr", "customer_name", "kind", "delivery_date",
                      "value_eur", "one_line"}
    assert n["one_line"] == "Λόγος 1" and n["customer_name"]
    w = m["warehouse"]
    assert w["orders_count"] == 2 and w["short_count"] == 1 and w["late_count"] == 0
    assert w["orders"] == wh.open_purchase_orders(7, db=pdb)
    json.dumps(m, ensure_ascii=False)  # ready for Telegram / the dashboard


def test_one_line():
    assert orch.one_line("Το «Κόλα κιβ. 6τεμ» τελειώνει 2/11. Άλλα 3 είδη.") == \
        "Το «Κόλα κιβ. 6τεμ» τελειώνει 2/11"


# --- the whole run ------------------------------------------------------------------

def test_orchestrator_run_no_ai(details_file, tmp_path, capsys, monkeypatch):
    db = tmp_path / "o.db"
    out = orch.run(db=db, ai=False, details_path=details_file)
    assert len(out["customer"]["result"]["saved"]) == 179
    c = out["summary"]["customers"]
    assert c["value_at_risk_eur"] == 42767.0 and c["counts"]["pending"] == 179
    assert len(c["needs_you"]) == 7
    w = out["summary"]["warehouse"]
    assert w["orders_count"] == len(wh.open_purchase_orders(7, db=db)) > 0
    assert all(o["note_source"] == llm.TEMPLATE for o in w["orders"])
    flagged = [s for s in store.list_suggestions(db=db)
               if store.WAREHOUSE_SHORT in s["payload"]["FLAGS"]]
    assert len(flagged) == sum(e["flagged"] for e in out["shortages"])
    again = orch.run(db=db, ai=False, details_path=details_file)
    assert again["customer"]["result"]["saved"] == [] and again["warehouse"]["result"]["saved"] == []
    text = orch.warehouse_briefing(again["summary"], again["shortages"])
    assert "Αποθήκη" in text and "Συντονισμός ελλείψεων" in text
