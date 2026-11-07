"""Card lines, marks and home sentences of the dashboard, with fake data."""
import sys
import types

from app import helpers as h
from tests.test_app_helpers import ITEMS, SAMPLE, make


def test_card_badges():
    assert h.card_badge(SAMPLE[0]) == ("Επείγον · Έκτακτο", "urgent")
    assert h.card_badge(SAMPLE[1]) == ("Κίνδυνος απώλειας", "loss")
    assert h.card_badge(SAMPLE[2]) == ("Οριακό · τηλεφώνημα", "call")
    assert h.card_badge(SAMPLE[3]) == ("Παραγγελία", "order")
    assert h.card_badge(SAMPLE[4]) == ("Προσφορά", "offer")
    assert h.card_badge(SAMPLE[8]) == ("Έκτακτο", "urgent")
    assert h.caps("Κίνδυνος απώλειας") == "ΚΙΝΔΥΝΟΣ ΑΠΩΛΕΙΑΣ"
    assert h.caps("Επείγον · Έκτακτο") == "ΕΠΕΙΓΟΝ · ΕΚΤΑΚΤΟ"


def test_card_line_extra_route_like_the_picture():
    s = make(1, "ΠΑΡΑΓΓΕΛΙΑ", ["ΕΠΕΙΓΟΝ", "ΕΚΤΑΚΤΟ_ΔΡΟΜΟΛΟΓΙΟ"], 1173.5)
    s["delivery_date"] = "2026-11-02"
    s["payload"]["LINES"] = s["payload"]["LINES"] * 6 + s["payload"]["LINES"][:1]  # 13 lines
    assert h.card_line(s) == "Έκτακτο δρομολόγιο τη Δευτέρα 2/11 · 13 είδη · 1.173,50 €"


def test_card_lines_for_every_kind():
    assert h.card_line(SAMPLE[3]) == "Παράδοση την Τετάρτη 4/11 · 2 είδη · 50,25 €"
    assert h.card_line(SAMPLE[2]).startswith("Κανονική επίσκεψη την Τετάρτη 4/11 · "
                                              "τηλεφώνημα πρώτα")
    assert h.card_line(SAMPLE[4]).startswith("Εποχικά είδη · παράδοση την Τετάρτη 4/11")
    assert h.card_line(SAMPLE[1], {"days_early": 72}) == \
        "Σταμάτησε 72 ημέρες νωρίς · τηλεφώνημα σήμερα"
    assert "45 ημέρες" in h.card_line(SAMPLE[1], {"days_silent": 45})
    assert h.card_line(SAMPLE[1]) == "Σιωπή πέρα από το συνηθισμένο · τηλεφώνημα"
    one = make()
    one["payload"]["LINES"] = one["payload"]["LINES"][:1]
    assert "· 1 είδος ·" in h.card_line(one)


def test_card_lines_are_short_plain_text():
    for s in SAMPLE:
        line = h.card_line(s, {"days_early": 3})
        assert "\n" not in line and "_" not in line and len(line) <= 90


def test_day_phrases():
    assert h.day_phrase("2026-11-02") == "τη Δευτέρα 2/11"
    assert h.day_phrase("2026-11-05") == "την Πέμπτη 5/11"
    assert h.day_phrase("2026-11-07") == "το Σάββατο 7/11"
    assert h.fmt_day("2026-11-02") == "Δευτ. 02/11"
    assert h.fmt_date_words("2026-11-01") == "Κυριακή 1 Νοεμβρίου 2026"


def test_facts_come_from_the_creation_entry():
    entries = [{"suggestion_id": 2, "action": "ΔΗΜΙΟΥΡΓΙΑ", "details": {"facts": {"days_early": 72}}},
               {"suggestion_id": 2, "action": "ΕΓΚΡΙΣΗ", "details": None},
               {"suggestion_id": 3, "action": "ΔΗΜΙΟΥΡΓΙΑ", "details": {"flags": []}}]
    assert h.facts_by_suggestion(entries) == {2: {"days_early": 72}}


def test_home_sentences():
    text = h.home_sentence(SAMPLE[0], "Mini Market Ήλιος Α.Ε.", 1036, route_cost=60)
    assert "έκτακτο δρομολόγιο την Τετάρτη 4/11" in text
    assert "Mini Market Ήλιος Α.Ε. (1036)" in text and "60 €" in text
    loss = make(2, "ΚΙΝΔΥΝΟΣ_ΑΠΩΛΕΙΑΣ")
    loss["delivery_date"] = "2027-04-05"
    text = h.home_sentence(loss, "Μπαρ Μελτέμι", 1127,
                           {"days_early": 72, "purchases_12m_eur": 30567.4})
    assert "σταμάτησε 72 ημέρες νωρίτερα" in text and "30.567 €" in text and "5/4/2027" in text
    assert h.home_sentence(make(3, "ΚΙΝΔΥΝΟΣ_ΑΠΩΛΕΙΑΣ"), "Ταβέρνα Η Αυλή", 1002).startswith("Η ")
    assert "<b>Ταβέρνα &lt;b&gt;" in h.home_sentence(SAMPLE[3], "Ταβέρνα <b>", 1)  # escaped


def test_warehouse_shortage_mark():
    items = h.item_lookup(ITEMS)
    assert h.warehouse_shortage(make(), items) is None
    s = make(flags=["ΕΛΛΕΙΨΗ_ΑΠΟΘΗΚΗΣ"])
    assert h.warehouse_shortage(s, items) == []
    s = make()
    s["payload"]["LINES"][1]["FLAGS"] = ["ΕΛΛΕΙΨΗ_ΑΠΟΘΗΚΗΣ"]
    assert h.warehouse_shortage(s, items) == ["Νερό 1,5L"]
    assert "_" not in h.label("ΕΛΛΕΙΨΗ_ΑΠΟΘΗΚΗΣ")


def test_message_after_approval(monkeypatch):
    assert h.notify_after_approval(1, "draft_only") is None
    monkeypatch.setitem(sys.modules, "rafi.telegram_bot", None)  # import fails
    assert h.notify_after_approval(1, "draft_and_message") == ("warning", h.MESSAGE_NOT_READY)
    calls = []
    bot = types.ModuleType("rafi.telegram_bot")
    bot.notify_customer = lambda sid: calls.append(sid) or True
    monkeypatch.setitem(sys.modules, "rafi.telegram_bot", bot)
    assert h.notify_after_approval(7, "draft_and_message") == ("success", h.MESSAGE_SENT)
    assert calls == [7]
    bot.notify_customer = lambda sid: False
    assert h.notify_after_approval(7, "draft_and_message") == ("warning", h.MESSAGE_NOT_SENT)
    bot.notify_customer = lambda sid: 1 / 0
    assert h.notify_after_approval(7, "draft_and_message") == ("warning", h.MESSAGE_NOT_SENT)


def test_morning_texts(monkeypatch):
    monkeypatch.setitem(sys.modules, "rafi.agents.orchestrator", None)
    assert h.morning_texts() == {}
    orch = types.ModuleType("rafi.agents.orchestrator")
    orch.morning_summary = lambda db=None: {"customers": "3 επείγοντα", "warehouse": {"n": 2}}
    monkeypatch.setitem(sys.modules, "rafi.agents.orchestrator", orch)
    assert h.morning_texts() == {"customers": "3 επείγοντα"}  # no numbers: no empty line


def test_morning_texts_from_the_summary_numbers(monkeypatch):
    # the shape rafi.agents.orchestrator.morning_summary really returns
    summary = {
        "customers": {"counts": {"pending": 179, "extra_routes": 2, "loss_risks": 5,
                                 "marginal": 49, "orders": 109, "offers": 65},
                      "value_at_risk_eur": 42589.2, "offers_value_eur": 26520.0,
                      "needs_you": [{"suggestion_id": i} for i in range(7)]},
        "warehouse": {"orders_count": 9, "short_count": 0, "late_count": 1,
                      "orders": [{"total_value_eur": 14000.0}, {"total_value_eur": 685.3}]}}
    orch = types.ModuleType("rafi.agents.orchestrator")
    orch.morning_summary = lambda db=None: summary
    monkeypatch.setitem(sys.modules, "rafi.agents.orchestrator", orch)
    t = h.morning_texts()
    assert t["customers"] == ("Περιμένουν έγκριση 179 προτάσεις, από τις οποίες 7 χρειάζονται "
                              "εσένα. 2 έκτακτα δρομολόγια και 5 κίνδυνοι απώλειας. "
                              "Αξία σε κίνδυνο 42.589,20 €.")
    assert t["warehouse"] == ("9 παραγγελίες σε προμηθευτές μέσα στην εβδομάδα, 14.685,30 € σε "
                              "τιμή καταλόγου. Αργήσαμε: 1, οι πελάτες καλύπτονται.")
    summary["warehouse"] = {"orders_count": 0, "short_count": 0, "orders": []}
    assert h.morning_texts()["warehouse"] == "Καμία παραγγελία σε προμηθευτή μέσα στην εβδομάδα."


def test_supplier_message_after_purchase_approval(monkeypatch):
    assert h.notify_supplier_after_approval("S07", "2026-11-02", "draft_only") is None
    bot = types.ModuleType("rafi.telegram_bot")
    bot.supplier_notice = lambda sup, order_by: (True, "")
    monkeypatch.setitem(sys.modules, "rafi.telegram_bot", bot)
    assert h.notify_supplier_after_approval("S07", "2026-11-02", "draft_and_message") == \
        ("success", "Στάλθηκε και μήνυμα στον προμηθευτή.")
    bot.supplier_notice = lambda sup, order_by: (False, "λείπει το TELEGRAM_SUPPLIER_CHAT_ID στο .env")
    assert h.notify_supplier_after_approval("S07", "2026-11-02", "draft_and_message") == \
        ("warning", "Το μήνυμα στον προμηθευτή δεν στάλθηκε: λείπει το "
                    "TELEGRAM_SUPPLIER_CHAT_ID στο .env.")
    bot.supplier_notice = lambda sup, order_by: 1 / 0
    assert h.notify_supplier_after_approval("S07", "2026-11-02", "draft_and_message")[0] == "warning"
    assert h.AFTER_APPROVAL_OPTIONS["draft_and_message"] == \
        "Πρόχειρη και μήνυμα σε πελάτη/προμηθευτή"


def test_extra_settings(tmp_path):
    path = tmp_path / "settings.json"
    h.write_settings(60, "draft_only", path, extra={"stock_cover_weeks": 5, "safety_weeks": 1.5,
                                                     "ai_texts_top_n": 20})
    v = h.read_settings(path)
    assert v["stock_cover_weeks"] == 5 and v["safety_weeks"] == 1.5 and v["ai_texts_top_n"] == 20
    import pytest
    with pytest.raises(ValueError):
        h.write_settings(60, "draft_only", path, extra={"ai_texts_top_n": 2.5})
    with pytest.raises(ValueError):
        h.write_settings(60, "draft_only", path, extra={"safety_weeks": -1})
    with pytest.raises(ValueError):
        h.write_settings(60, "draft_only", path, extra={"κάτι": 1})
