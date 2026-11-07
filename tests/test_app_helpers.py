"""Helpers of the dashboard (app/helpers.py)."""
import json
import re
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from app import helpers as h
from rafi import erp_client, store
from rafi.erp_client import ErpError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import daily_run  # noqa: E402


def make(sid=1, kind="ΠΑΡΑΓΓΕΛΙΑ", flags=(), value=10.0, status="ΠΡΟΤΑΣΗ", trdr=1001,
         reason="Το «Νερό» τελειώνει περίπου 6/11· ο πελάτης καταναλώνει ~2 κιβ./ημέρα."):
    lines = [{"MTRL": 5001, "ITEM_CODE": "10.01.001", "QTY1": 4, "PRICER": 4.2,
              "PREDICTED_DEPLETION_DATE": "2026-11-06", "CONFIDENCE": 0.8},
             {"MTRL": 5002, "ITEM_CODE": "10.01.002", "QTY1": 2, "PRICER": 2.6,
              "PREDICTED_DEPLETION_DATE": None, "CONFIDENCE": None}]
    payload = {"TRDR": trdr, "CUSTOMER_CODE": "30.00.0001", "SALESMAN": 103,
               "SUGGESTED_DELIVERY_DATE": "2026-11-04", "ACTION_TYPE": kind,
               "FLAGS": list(flags), "REASON": reason, "SALES_NOTE": "Καλημέρα σας.",
               "TEXT_SOURCE": "ΠΡΟΤΥΠΟ", "LINES": lines}
    return {"id": sid, "trdr": trdr, "salesman": 103, "action_type": kind,
            "delivery_date": "2026-11-04", "value_eur": value, "priority": sid,
            "status": status, "erp_id": None, "payload": payload}


SAMPLE = [
    make(1, "ΠΑΡΑΓΓΕΛΙΑ", ["ΕΠΕΙΓΟΝ", "ΕΚΤΑΚΤΟ_ΔΡΟΜΟΛΟΓΙΟ"], 1173.5),
    make(2, "ΚΙΝΔΥΝΟΣ_ΑΠΩΛΕΙΑΣ", [], 1247.1),
    make(3, "ΠΑΡΑΓΓΕΛΙΑ", ["ΟΡΙΑΚΟ"], 100.0),
    make(4, "ΠΑΡΑΓΓΕΛΙΑ", [], 50.25),
    make(5, "ΠΡΟΣΦΟΡΑ", ["ΕΚΤΑΚΤΟ_ΔΡΟΜΟΛΟΓΙΟ"], 300.0),
    make(6, "ΠΡΟΣΦΟΡΑ", [], 20.0),
    make(7, "ΠΑΡΑΓΓΕΛΙΑ", ["ΕΠΕΙΓΟΝ"], 999.0, status="ΣΤΑΛΘΗΚΕ_ΣΤΟ_ERP"),
    make(8, "ΚΙΝΔΥΝΟΣ_ΑΠΩΛΕΙΑΣ", [], 500.0, status="ΑΠΟΡΡΙΦΘΗΚΕ"),
    make(9, "ΠΑΡΑΓΓΕΛΙΑ", ["ΕΚΤΑΚΤΟ_ΔΡΟΜΟΛΟΓΙΟ"], 10.0),
]


# --- formatting --------------------------------------------------------------

def test_greek_numbers():
    assert h.fmt_eur(42589.2) == "42.589,20 €"
    assert h.fmt_eur(0) == "0,00 €"
    assert h.fmt_eur(1234567.891) == "1.234.567,89 €"
    assert h.fmt_num(1500) == "1.500"
    assert h.fmt_num(2.45, 1) in ("2,5", "2,4")
    assert h.fmt_qty(3.0) == "3" and h.fmt_qty(None) == "—"


def test_greek_dates():
    assert h.fmt_date("2026-11-02") == "Δευτ. 02/11/2026"
    assert h.fmt_date(pd.Timestamp("2026-11-07")) == "Σάβ. 07/11/2026"
    assert h.fmt_date(None) == "—" and h.fmt_date("") == "—"
    assert h.fmt_date_long(h.SCENARIO_TODAY) == "Κυριακή 01/11/2026"
    assert h.fmt_datetime("2026-11-03T09:05:40") == "Τρ. 03/11/2026 09:05"


def test_scenario_day_matches_daily_run():
    assert pd.Timestamp(h.SCENARIO_TODAY) == daily_run.TODAY


def test_codes_become_words_without_underscores():
    assert h.label("ΕΚΤΑΚΤΟ_ΔΡΟΜΟΛΟΓΙΟ") == "Έκτακτο δρομολόγιο"
    assert h.label("ΚΙΝΔΥΝΟΣ_ΑΠΩΛΕΙΑΣ") == "Κίνδυνος απώλειας"
    assert h.label("ΣΤΑΛΘΗΚΕ_ΣΤΟ_ERP") == "Στάλθηκε στο ERP"
    assert h.label("ΣΦΑΛΜΑ_ΑΠΟΣΤΟΛΗΣ") == "Σφάλμα αποστολής"
    assert "_" not in h.label("ΚΑΤΙ_ΑΓΝΩΣΤΟ")
    for code in h.LABELS:
        assert "_" not in h.LABELS[code]
    msg = h.humanize("Η πρόταση 3 είναι ΣΤΑΛΘΗΚΕ_ΣΤΟ_ERP· δεν στάλθηκε τίποτα.")
    assert "_" not in msg and "Στάλθηκε στο ERP" in msg


def test_flag_labels_mark_marginal_orders_for_a_call():
    assert h.flag_labels(SAMPLE[2]) == ["Οριακό · τηλεφώνημα πρώτα"]
    assert h.flag_labels(SAMPLE[0]) == ["Επείγον", "Έκτακτο δρομολόγιο"]


def test_users():
    assert h.store_user({"salesman": 103}) == "admin"
    assert h.user_label(h.store_user()) == "Διαχειριστής"
    assert h.user_label("system") == "Σύστημα"
    assert h.user_label("telegram") == "Διαχειριστής (Telegram)"
    assert h.user_label("103", {103: "Ελένη"}) == "Ελένη"


# --- tabs and home -----------------------------------------------------------

def test_tabs_split_proposals_without_overlap():
    tabs = h.split_tabs(SAMPLE)
    ids = {t: [s["id"] for s in v] for t, v in tabs.items()}
    assert ids[h.TAB_NEEDS_YOU] == [1, 2, 9]
    assert ids[h.TAB_ORDERS] == [3, 4]
    assert ids[h.TAB_OFFERS] == [5, 6]       # an offer with an extra route stays an offer
    all_ids = sum(ids.values(), [])
    assert len(all_ids) == len(set(all_ids))  # no overlap
    assert 7 not in all_ids and 8 not in all_ids  # only ΠΡΟΤΑΣΗ


def test_home_metrics():
    m = h.home_metrics(SAMPLE)
    assert m["pending"] == 7
    assert m["extra_routes"] == 2  # orders 1 and 9 (not the offer, not the sent one)
    assert m["loss_risks"] == 1 and m["offers"] == 2
    assert m["at_risk_eur"] == pytest.approx(1173.5 + 1247.1 + 100.0 + 50.25 + 10.0)
    assert m["offers_eur"] == pytest.approx(320.0)


def test_home_values_match_the_daily_run_briefing(tmp_path):
    """The dashboard and the terminal briefing print the same two values."""
    db = tmp_path / "t.db"
    rows = []
    for i, (kind, value) in enumerate([("ΠΑΡΑΓΓΕΛΙΑ", 1173.5), ("ΚΙΝΔΥΝΟΣ_ΑΠΩΛΕΙΑΣ", 41415.7),
                                       ("ΠΡΟΣΦΟΡΑ", 26520.0), ("ΠΑΡΑΓΓΕΛΙΑ", 0.33)]):
        s = make(i + 1, kind, value=value, trdr=1001 + i)
        rows.append({"ACTION_TYPE": kind, "VALUE_EUR": value, "PRIORITY": i + 1,
                     "FLAGS": [], "payload": s["payload"]})
    store.save_new(rows, db=db)
    store.reject(4, "Ο πελάτης έκλεισε", "admin", db=db)  # not pending any more

    m = h.home_metrics(store.list_suggestions(db=db))
    customers = pd.read_csv(ROOT / "data" / "TRDR.csv", encoding="utf-8-sig")
    salesmen = pd.read_csv(ROOT / "data" / "PRSN.csv", encoding="utf-8-sig")
    text = daily_run.briefing({"saved": [], "skipped": 0}, customers, salesmen, db=db)
    risk = re.search(r"Αξία σε κίνδυνο.*?([\d,]+\.\d\d) €", text).group(1)
    offers = re.search(r"Αξία προτεινόμενων προσφορών.*?([\d,]+\.\d\d) €", text).group(1)
    assert float(risk.replace(",", "")) == pytest.approx(m["at_risk_eur"]) == 42589.2
    assert float(offers.replace(",", "")) == pytest.approx(m["offers_eur"]) == 26520.0
    assert h.fmt_eur(m["at_risk_eur"]) == "42.589,20 €"


def test_summary_is_one_short_line():
    line = h.summary(SAMPLE[0])
    assert line == "Το «Νερό» τελειώνει περίπου 6/11"
    long = make(reason="α" * 300)
    assert len(h.summary(long)) <= 120 and "\n" not in h.summary(long)


# --- lines, corrections, rejection --------------------------------------------

ITEMS = pd.DataFrame({"MTRL": [5001, 5002], "NAME": ["Νερό 0,5L", "Νερό 1,5L"],
                      "MTRUNIT1": ["ΚΙΒ", "ΤΕΜ"]})


def test_lines_rows_use_mtrl_names_and_units():
    rows = h.lines_rows(SAMPLE[0], h.item_lookup(ITEMS))
    assert rows[0] == {"Είδος": "Νερό 0,5L", "Ποσότητα": 4, "Μονάδα": "κιβ.",
                       "Τελειώνει": pd.Timestamp("2026-11-06")}
    assert rows[1]["Μονάδα"] == "τεμ." and rows[1]["Τελειώνει"] is None


def test_only_changed_quantities_are_returned():
    lines = SAMPLE[0]["payload"]["LINES"]
    assert h.changed_quantities(lines, {5001: 4, 5002: 2}) == []
    assert h.changed_quantities(lines, {5001: 6, 5002: 2}) == [(5001, 6)]
    assert h.changed_quantities(lines, {5001: 6.0, 5002: 3}) == [(5001, 6), (5002, 3)]
    for bad in (0, -1, 2.5):
        with pytest.raises(ValueError):
            h.changed_quantities(lines, {5001: bad})


def test_rejection_text():
    assert h.rejection_text("Ο πελάτης έκλεισε", "  έκλεισε για ανακαίνιση ") == \
        "Ο πελάτης έκλεισε — έκλεισε για ανακαίνιση"
    assert h.rejection_text("Άλλο", "") == "Άλλο"
    with pytest.raises(ValueError):
        h.rejection_text(None, "σχόλιο")


def test_erp_errors_are_told_apart():
    with pytest.raises(ErpError) as down:
        erp_client.post_draft(make()["payload"], base_url="http://127.0.0.1:9", timeout=2)
    assert h.erp_error_kind(down.value) == "down"
    rejected = ErpError("Το ERP απέρριψε το draft: x", ["LINES[0]: MTRL 9999 not found"])
    assert h.erp_error_kind(rejected) == "rejected"
    assert h.erp_rejection_message(rejected) == "LINES[0]: MTRL 9999 not found"
    assert h.erp_error_kind(ErpError("Το ERP απάντησε με σφάλμα HTTP 500")) == "other"


# --- history -----------------------------------------------------------------

def test_history_details_are_plain_text():
    items = h.item_lookup(ITEMS)
    cases = [
        ({"action": "ΔΗΜΙΟΥΡΓΙΑ", "details": {"flags": ["ΕΚΤΑΚΤΟ_ΔΡΟΜΟΛΟΓΙΟ"], "value_eur": 1173.5,
                                              "priority": 1, "facts": {"x": 1}}},
         "Αξία 1.173,50 € · προτεραιότητα 1 · σημάνσεις: Έκτακτο δρομολόγιο"),
        ({"action": "ΔΙΟΡΘΩΣΗ", "details": {"MTRL": 5001, "old_qty": 4, "new_qty": 6}},
         "Νερό 0,5L: ποσότητα 4 → 6"),
        ({"action": "ΑΠΟΡΡΙΨΗ", "details": {"reason": "Ο πελάτης έκλεισε"}},
         "Λόγος: Ο πελάτης έκλεισε"),
        ({"action": "ΑΠΟΣΤΟΛΗ", "details": None, "erp_id": "DRAFT-1"},
         "Πρόχειρη στο ERP με αριθμό DRAFT-1"),
    ]
    for entry, expected in cases:
        text = h.history_details(entry, items)
        assert text == expected
        assert "{" not in text and "_" not in text


def test_history_rows_from_the_store(tmp_path):
    db = tmp_path / "t.db"
    s = make()
    store.save_new([{"ACTION_TYPE": s["action_type"], "VALUE_EUR": 10.0, "PRIORITY": 1,
                     "FLAGS": [], "payload": s["payload"]}], db=db)
    store.correct(1, 5001, 6, "103", db=db)
    rows = h.history_rows(store.history(db=db), {1001: ("Hotel Ιόνιο", "30.00.0001")},
                          h.item_lookup(ITEMS), {103: "Ελένη"})
    assert [r["Κίνηση"] for r in rows] == ["Διόρθωση", "Δημιουργία"]
    assert rows[0]["Πελάτης"] == "Hotel Ιόνιο" and rows[0]["Χρήστης"] == "Ελένη"
    assert rows[0]["Τύπος"] == "Παραγγελία"
    assert isinstance(rows[0]["Ημερομηνία"], pd.Timestamp)
    assert re.match(r"^\S+ \d\d/\d\d/\d{4} \d\d:\d\d$", h.fmt_datetime(rows[0]["Ημερομηνία"]))


# --- assistant ---------------------------------------------------------------

def test_assistant_missing_or_failing_gives_the_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "rafi.llm", types.ModuleType("rafi.llm"))  # no answer()
    assert h.ask_assistant("Με ποιον ξεκινάω;", {}) == {"text": h.ASSISTANT_UNAVAILABLE,
                                                         "source": None}
    broken = types.ModuleType("rafi.llm")
    broken.answer = lambda q, c: 1 / 0
    monkeypatch.setitem(sys.modules, "rafi.llm", broken)
    assert h.ask_assistant("x", {})["text"] == h.ASSISTANT_UNAVAILABLE


def test_assistant_answer_is_passed_through(monkeypatch):
    fake = types.ModuleType("rafi.llm")
    fake.answer = lambda q, c: {"text": f"Ξεκίνα από {c['top']}", "source": "local"}
    monkeypatch.setitem(sys.modules, "rafi.llm", fake)
    import rafi
    monkeypatch.setattr(rafi, "llm", fake, raising=False)
    assert h.ask_assistant("x", {"top": 1036}) == {"text": "Ξεκίνα από 1036", "source": "local"}


def test_chat_contexts_are_json_ready():
    ctx = h.general_context(SAMPLE, {1001: ("Hotel Ιόνιο", "30.00.0001")})
    assert ctx["metrics"]["pending"] == 7 and len(ctx["top_suggestions"]) == 7
    json.dumps(ctx, ensure_ascii=False)
    many = [make(i) for i in range(1, 15)]
    assert len(h.general_context(many)["top_suggestions"]) == 10
    one = h.suggestion_context(SAMPLE[0], {}, h.item_lookup(ITEMS))
    assert one["id"] == 1 and one["payload"]["LINES"][0]["UNIT"] == "κιβ."
    json.dumps(one, ensure_ascii=False)


# --- settings ----------------------------------------------------------------

def test_settings_round_trip(tmp_path):
    path = tmp_path / "config" / "settings.json"
    values = h.read_settings(path)
    assert values["extra_route_cost_eur"] == 60
    assert values["after_approval_mode"] in h.AFTER_APPROVAL_OPTIONS
    h.write_settings(75, "draft_and_message", path)
    values = h.read_settings(path)
    assert values["extra_route_cost_eur"] == 75
    assert values["after_approval_mode"] == "draft_and_message"
    assert values["stock_cover_weeks"] == 4 and values["safety_weeks"] == 1
    with pytest.raises(ValueError):
        h.write_settings(-1, "draft_only", path)
    with pytest.raises(ValueError):
        h.write_settings(60, "κάτι άλλο", path)


def test_summary_does_not_stop_inside_an_item_name():
    s = make(reason="Το «Νερό 0,75L γυάλινο κιβ. 12τεμ» τελειώνει περίπου 3/11· άλλα.")
    assert h.summary(s) == "Το «Νερό 0,75L γυάλινο κιβ. 12τεμ» τελειώνει περίπου 3/11"
    s = make(reason="Ο πελάτης παρήγγελνε κάθε ~5 ημέρες. Αγορές 12 μηνών: 3.000 €.")
    assert h.summary(s) == "Ο πελάτης παρήγγελνε κάθε ~5 ημέρες"
