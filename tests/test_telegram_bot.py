"""Telegram bot with a fake Telegram API and a fake ERP: no real network calls."""
import sys
import types

import pytest

from rafi import erp_client, settings, store
from rafi import telegram_bot as tb
from rafi.suggestions import EXTRA_ROUTE, LOSS, MARGINAL, ORDER

ADMIN, CUSTOMER = "111", "222"
CFG = {"TELEGRAM_BOT_TOKEN": "SECRET-TOKEN", "TELEGRAM_ADMIN_CHAT_ID": ADMIN,
       "TELEGRAM_CUSTOMER_CHAT_ID": CUSTOMER}


class FakeApi:
    def __init__(self, updates=None):
        self.calls, self.updates = [], list(updates or [])

    def call(self, method, http_timeout=20, **params):
        self.calls.append((method, params))
        if method == "getUpdates":
            out, self.updates = self.updates, []
            return out
        return {"message_id": len(self.calls)}

    def sent(self, chat=None):
        return [p for m, p in self.calls if m == "sendMessage"
                and (chat is None or str(p["chat_id"]) == chat)]

    def texts(self, chat=None):
        return [p["text"] for p in self.sent(chat)]

    def buttons(self, i=-1):
        kb = self.sent()[i].get("reply_markup", {}).get("inline_keyboard", [])
        return [(b["text"], b["callback_data"]) for row in kb for b in row]


def suggestion(trdr, mtrl, kind=ORDER, flags=(), value=100.0, qty=4):
    return {"VALUE_EUR": value, "PRIORITY": 1, "payload": {
        "TRDR": trdr, "CUSTOMER_CODE": f"30.00.{trdr - 1000:04d}", "SALESMAN": 103,
        "SUGGESTED_DELIVERY_DATE": "2026-11-04", "ACTION_TYPE": kind, "FLAGS": list(flags),
        "REASON": "Το νερό τελειώνει περίπου 6/11.", "SALES_NOTE": "Καλημέρα σας, προτείνουμε παράδοση 4/11.",
        "TEXT_SOURCE": "ΠΡΟΤΥΠΟ",
        "LINES": [{"MTRL": mtrl, "ITEM_CODE": "x", "QTY1": qty, "PRICER": 4.2,
                   "PREDICTED_DEPLETION_DATE": "2026-11-06", "CONFIDENCE": 0.8}]}}


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "rafi.db"
    monkeypatch.setenv("RAFI_DB", str(path))
    # no orchestrator / warehouse functions unless a test adds them
    monkeypatch.setitem(sys.modules, "rafi.agents.orchestrator", None)
    ids = store.save_new([
        suggestion(1001, 5001, flags=[EXTRA_ROUTE], value=250.0),   # needs you
        suggestion(1002, 5002, kind=LOSS, value=80.0),              # needs you
        suggestion(1003, 5003, value=100.0),                        # regular order
        suggestion(1004, 5004, value=50.5),                         # regular order
        suggestion(1005, 5005, flags=[MARGINAL], value=30.0),       # regular, marginal
    ], db=path)["saved"]
    return path, ids


@pytest.fixture
def erp(monkeypatch):
    sent = []

    def post(payload):
        sent.append(payload["RAFI_SUGGESTION_ID"])
        return f"DRAFT-{payload['RAFI_SUGGESTION_ID']}"
    monkeypatch.setattr(erp_client, "post_draft", post)
    return sent


def press(bot, data, chat=ADMIN):
    return bot.handle({"update_id": 1, "callback_query": {
        "id": "cb", "data": data, "message": {"chat": {"id": int(chat)}}}})


def write(bot, text, chat=ADMIN):
    return bot.handle({"update_id": 1, "message": {"chat": {"id": int(chat)}, "text": text}})


def message_mode(on):
    settings.save_settings({"after_approval_mode":
                            settings.DRAFT_AND_MESSAGE if on else settings.DRAFT_ONLY})


# --- settings and API ---------------------------------------------------------

def test_config_reads_env_and_never_shows_the_token(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("TELEGRAM_BOT_TOKEN=abc123\nTELEGRAM_ADMIN_CHAT_ID=42\n", encoding="utf-8")
    for k in tb.KEYS:
        monkeypatch.delenv(k, raising=False)
    cfg = tb.config(env)
    assert cfg["TELEGRAM_BOT_TOKEN"] == "abc123" and cfg["TELEGRAM_ADMIN_CHAT_ID"] == "42"
    assert cfg["TELEGRAM_CUSTOMER_CHAT_ID"] == ""

    def boom(url, **kw):
        raise tb.requests.ConnectionError(f"cannot reach {url}")
    monkeypatch.setattr(tb.requests, "post", boom)
    with pytest.raises(tb.TelegramError) as e:
        tb.TelegramApi("abc123").call("getMe")
    assert "abc123" not in str(e.value) and "***" in str(e.value)


def test_env_example_has_the_telegram_keys_without_values():
    lines = (tb.ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    for k in tb.KEYS:
        assert f"{k}=" in lines


def test_whoami_lists_people_who_pressed_start():
    api = FakeApi([{"update_id": 1, "message": {"text": "/start", "chat": {"id": 5},
                                                "from": {"first_name": "Νίκος", "username": "nikos"}}},
                   {"update_id": 2, "message": {"text": "/start", "chat": {"id": 5},
                                                "from": {"first_name": "Νίκος"}}}])
    assert tb.whoami(api) == [(5, "Νίκος")]


# --- brief --------------------------------------------------------------------

def test_brief_without_orchestrator_uses_the_store(db):
    path, ids = db
    msgs = tb.brief_messages(path)
    assert len(msgs) == 3  # no warehouse message while the orchestrator is missing
    summary = msgs[0][0]
    assert "Προς έγκριση: 5" in summary and "Έκτακτα δρομολόγια: 1" in summary
    assert "Κίνδυνοι απώλειας: 1" in summary and "Οριακές (τηλεφώνημα πρώτα): 1" in summary
    assert "Αξία σε κίνδυνο: 510,50 €" in summary
    assert "αποθήκης δεν είναι διαθέσιμη ακόμα" in summary
    text, kb = msgs[1]
    assert "Χρειάζονται εσένα σήμερα (2)" in text
    assert [b["callback_data"] for row in kb for b in row] == [f"s:{ids[0]}", f"s:{ids[1]}"]
    assert kb[0][0]["text"] == "30.00.0001 Hotel Ιόνιο"
    text, kb = msgs[2]
    assert "Κανονικές παραγγελίες: 3 · 180,50 €" in text
    assert "1 οριακή θέλει τηλεφώνημα πρώτα" in text
    assert [b["callback_data"] for b in kb[0]] == ["ol", "oa"]
    for text, _ in msgs:
        assert "_" not in text  # no store codes


def test_brief_with_orchestrator_and_warehouse(db, monkeypatch):
    path, ids = db
    mod = types.ModuleType("rafi.agents.orchestrator")
    mod.morning_summary = lambda db=None: {
        "customers": {"counts": {"pending": 9}, "value_at_risk_eur": 1234.5,
                      "offers_value_eur": 0, "needs_you": [ids[1]]},
        "warehouse": {"orders_count": 2, "short_count": 1, "orders": [
            {"supplier_id": "SUP01", "supplier_name": "Προμηθευτής Νερών Α.Ε.",
             "order_by": "2026-11-03", "value_eur": 300.0, "flag": "ΔΕΝ_ΦΤΑΝΕΙ"},
            {"supplier_id": "SUP01", "supplier_name": "Προμηθευτής Νερών Α.Ε.",
             "order_by": "2026-11-03", "value_eur": 200.0, "flag": None},
            {"supplier_id": "SUP02", "supplier_name": "Αναψυκτικά", "order_by": "2026-11-20",
             "value_eur": 50.0}]}}
    monkeypatch.setitem(sys.modules, "rafi.agents.orchestrator", mod)
    msgs = tb.brief_messages(path)
    assert len(msgs) == 4
    assert "Προς έγκριση: 9" in msgs[0][0] and "Αξία σε κίνδυνο: 1.234,50 €" in msgs[0][0]
    assert "Ελλείψεις: 1" in msgs[0][0]
    assert "(1)" in msgs[1][0]
    text, kb = msgs[3]
    assert "Προμηθευτής Νερών Α.Ε.: έως Τρ. 03/11/2026 · 500,00 € · έλλειψη" in text
    assert "Αναψυκτικά" not in text  # outside the 7 days
    assert kb[0][0]["callback_data"] == "p:SUP01:2026-11-03"


def test_send_brief_goes_to_the_admin_only(db):
    path, _ = db
    api = FakeApi()
    assert tb.send_brief(api, CFG, path) == 3
    assert len(api.sent(ADMIN)) == 3 and not api.sent(CUSTOMER)


# --- customer suggestions -------------------------------------------------------

def test_open_shows_reason_items_and_buttons(db):
    path, ids = db
    api = FakeApi()
    press(tb.Bot(api, CFG, path), f"s:{ids[0]}")
    text = api.texts()[-1]
    assert "Hotel Ιόνιο" in text and "Το νερό τελειώνει περίπου 6/11." in text
    assert "Τετάρτη 04/11/2026" in text and "250,00 €" in text and "Νερό 0,5L" in text
    assert "Έκτακτο δρομολόγιο" in text
    assert [t for t, _ in api.buttons()] == ["Εγκρίνω", "Απόρριψη", "Ρώτα"]
    assert ("answerCallbackQuery", {"callback_query_id": "cb"}) in api.calls


def test_approve_sends_once(db, erp):
    path, ids = db
    api = FakeApi()
    bot = tb.Bot(api, CFG, path)
    press(bot, f"a:{ids[0]}")
    assert api.texts()[-1] == f"Στάλθηκε στο ERP ως πρόχειρη: DRAFT-{ids[0]}"
    press(bot, f"a:{ids[0]}")
    assert erp == [ids[0]]
    assert "δεν στάλθηκε τίποτα" in api.texts()[-1]
    assert "_" not in api.texts()[-1]
    assert store.get(ids[0], db=path)["status"] == store.SENT
    assert any(h["user"] == "telegram" and h["action"] == store.APPROVE
               for h in store.history(suggestion_id=ids[0], db=path))


def test_approve_when_erp_is_down(db, monkeypatch):
    path, ids = db

    def down(payload):
        try:
            raise tb.requests.ConnectionError("refused")
        except tb.requests.ConnectionError as e:
            raise erp_client.ErpError("Το ERP δεν απαντά") from e
    monkeypatch.setattr(erp_client, "post_draft", down)
    api = FakeApi()
    press(tb.Bot(api, CFG, path), f"a:{ids[0]}")
    assert api.texts()[-1].startswith("Ο ERP δεν απαντά αυτή τη στιγμή")
    assert store.get(ids[0], db=path)["status"] == store.PROPOSED


def test_reject_with_a_listed_reason(db):
    path, ids = db
    api = FakeApi()
    bot = tb.Bot(api, CFG, path)
    press(bot, f"r:{ids[1]}")
    assert [t for t, _ in api.buttons()] == ["Ο πελάτης έχει ακόμα απόθεμα", "Ο πελάτης έκλεισε",
                                            "Το χειρίζομαι τηλεφωνικά", "Άλλο"]
    press(bot, f"rr:{ids[1]}:1")
    assert store.get(ids[1], db=path)["status"] == store.REJECTED
    assert store.history(suggestion_id=ids[1], db=path)[-1]["details"] == {
        "reason": "Ο πελάτης έκλεισε"}


def test_reject_other_takes_the_next_text(db):
    path, ids = db
    bot = tb.Bot(FakeApi(), CFG, path)
    press(bot, f"rr:{ids[1]}:3")
    write(bot, "Αλλάζει προμηθευτή")
    assert store.history(suggestion_id=ids[1], db=path)[-1]["details"] == {
        "reason": "Άλλο — Αλλάζει προμηθευτή"}


def test_ask_answers_the_next_text_with_the_suggestion(db, monkeypatch):
    path, ids = db
    seen = {}

    def answer(q, ctx):
        seen.update(q=q, ctx=ctx)
        return {"text": "Γιατί τελειώνει το νερό.", "source": "AI"}
    import rafi.llm
    monkeypatch.setattr(rafi.llm, "answer", answer)
    api = FakeApi()
    bot = tb.Bot(api, CFG, path)
    press(bot, f"q:{ids[0]}")
    write(bot, "Γιατί;")
    assert seen["q"] == "Γιατί;" and seen["ctx"]["id"] == ids[0]
    assert seen["ctx"]["customer_name"] == "Hotel Ιόνιο Μ.Ε.Π.Ε."
    assert "Γιατί τελειώνει το νερό." in api.texts()[-1] and "Βοηθός AI" in api.texts()[-1]


# --- bulk approval --------------------------------------------------------------

def test_bulk_approval_skips_marginal_and_asks_first(db, erp):
    path, ids = db
    api = FakeApi()
    bot = tb.Bot(api, CFG, path)
    press(bot, "oa")
    assert erp == []  # nothing before the second confirmation
    text = api.texts()[-1]
    assert "2 παραγγελίες, συνολικά 150,50 €" in text and "1 οριακή δεν περιλαμβάνεται." in text
    assert api.buttons() == [("Ναι, έγκριση 2", "ob"), ("Άκυρο", "x")]
    press(bot, "ob")
    assert sorted(erp) == [ids[2], ids[3]]
    assert "2 παραγγελίες · 150,50 €" in api.texts()[-1]
    assert store.get(ids[4], db=path)["status"] == store.PROPOSED
    assert store.get(ids[0], db=path)["status"] == store.PROPOSED  # needs-you is not bulk
    press(bot, "ob")  # a second tap sends nothing
    assert len(erp) == 2


def test_bulk_cancel_sends_nothing(db, erp):
    path, _ = db
    bot = tb.Bot(FakeApi(), CFG, path)
    press(bot, "oa")
    press(bot, "x")
    press(bot, "ob")
    assert erp == []


def test_order_list_marks_marginal(db):
    path, ids = db
    api = FakeApi()
    press(tb.Bot(api, CFG, path), "ol")
    assert "οριακή, τηλεφώνημα πρώτα" in api.texts()[-1]
    assert [d for _, d in api.buttons()] == [f"s:{i}" for i in ids[2:]]


# --- chats ----------------------------------------------------------------------

def test_unknown_chat_is_ignored(db, erp):
    path, ids = db
    api = FakeApi()
    bot = tb.Bot(api, CFG, path)
    assert press(bot, f"a:{ids[0]}", chat="999") is False
    assert write(bot, "/brief", chat="999") is False
    assert api.calls == [] and erp == []


def test_customer_reply_is_shown_to_the_admin(db):
    path, _ = db
    api = FakeApi()
    write(tb.Bot(api, CFG, path), "Εντάξει, ευχαριστώ", chat=CUSTOMER)
    assert api.sent(ADMIN) and "Εντάξει, ευχαριστώ" in api.texts(ADMIN)[-1]


def test_poll_once_moves_the_offset_and_survives_a_bad_update(db):
    path, _ = db
    api = FakeApi([{"update_id": 7, "callback_query": {
        "id": "c", "data": "s:9999", "message": {"chat": {"id": int(ADMIN)}}}}])
    bot = tb.Bot(api, CFG, path)
    assert bot.poll_once() == 1 and bot.offset == 8
    assert "Κάτι πήγε στραβά" in api.texts()[-1]


# --- message to the customer ----------------------------------------------------

def test_notify_customer_only_in_message_mode(db, erp, monkeypatch):
    path, ids = db
    api = FakeApi()
    monkeypatch.setattr(tb, "config", lambda: CFG)
    monkeypatch.setattr(tb, "_default_api", lambda cfg: api)
    store.approve_and_send(ids[0], "test", db=path)

    message_mode(False)
    assert tb.notify_customer(ids[0]) is False and api.calls == []

    message_mode(True)
    assert tb.notify_customer(ids[0]) is True
    text = api.texts(CUSTOMER)[-1]
    assert "Καλημέρα σας, προτείνουμε παράδοση 4/11." in text
    assert "Τετάρτη 04/11/2026" in text and "Νερό 0,5L" in text and ": 4 κιβ." in text
    last = store.history(suggestion_id=ids[0], db=path)[-1]
    assert last["action"] == tb.CUSTOMER_MESSAGE and last["details"]["Κανάλι"] == "Telegram"
    assert tb.notify_customer(ids[0]) is False  # once per suggestion
    assert len(api.sent(CUSTOMER)) == 1


def test_notify_customer_not_before_approval(db, monkeypatch):
    path, ids = db
    api = FakeApi()
    monkeypatch.setattr(tb, "config", lambda: CFG)
    monkeypatch.setattr(tb, "_default_api", lambda cfg: api)
    message_mode(True)
    assert tb.notify_customer(ids[2]) is False and api.calls == []


def test_approval_from_the_bot_messages_the_customer_in_message_mode(db, erp):
    path, ids = db
    message_mode(True)
    api = FakeApi()
    press(tb.Bot(api, CFG, path), f"a:{ids[0]}")
    assert len(api.sent(CUSTOMER)) == 1
    assert "Ο πελάτης ενημερώθηκε" in api.texts(ADMIN)[-1]


def test_approval_from_the_bot_without_message_mode(db, erp):
    path, ids = db
    api = FakeApi()
    press(tb.Bot(api, CFG, path), f"a:{ids[0]}")
    assert api.sent(CUSTOMER) == []


# --- supplier orders ------------------------------------------------------------

def purchases(path):
    return store.save_purchase_new([
        {"MTRL": m, "SUPPLIER_ID": "SUP01", "SUPPLIER_NAME": "Προμηθευτής Νερών Α.Ε.",
         "ORDER_BY": "2026-11-03", "ARRIVAL": "2026-11-06", "QTY": 60, "VALUE_EUR": 252.0,
         "ITEM_NAME": f"Νερό {m}", "UNIT": "κιβ.", "REASON": "Τελειώνει περίπου 10/11."}
        for m in (5001, 5002)], db=path)["saved"]


def test_purchase_open_and_approve_fallback(db, tmp_path, monkeypatch):
    path, _ = db
    pids = purchases(path)
    monkeypatch.setattr(store, "PURCHASE_DIR", tmp_path / "po")
    api = FakeApi()
    bot = tb.Bot(api, CFG, path)
    press(bot, "p:SUP01:2026-11-03")
    text = api.texts()[-1]
    assert "Προμηθευτής Νερών Α.Ε." in text and "504,00 €" in text and "Τελειώνει περίπου 10/11." in text
    assert [d for _, d in api.buttons()] == ["pa:SUP01:2026-11-03", "pr:SUP01:2026-11-03"]
    press(bot, "pa:SUP01:2026-11-03")
    assert api.texts()[-1] == "Η παραγγελία προμηθευτή εγκρίθηκε."
    assert {r["status"] for r in store.list_purchase_suggestions(db=path)} == {store.APPROVED}
    press(bot, "pa:SUP01:2026-11-03")
    assert "δεν είναι πια προς έγκριση" in api.texts()[-1]
    assert len(pids) == 2


def test_purchase_uses_the_warehouse_functions_when_present(db, monkeypatch):
    path, _ = db
    calls = []
    import rafi.agents.warehouse_agent as wa
    monkeypatch.setattr(wa, "approve_purchase_order",
                        lambda s, o, u: calls.append(("ok", s, o, u)), raising=False)
    monkeypatch.setattr(wa, "reject_purchase_order",
                        lambda s, o, r, u: calls.append(("no", s, o, r, u)), raising=False)
    bot = tb.Bot(FakeApi(), CFG, path)
    press(bot, "pa:SUP01:2026-11-03")
    press(bot, "prr:SUP01:2026-11-03:0")
    assert calls == [("ok", "SUP01", "2026-11-03", "telegram"),
                     ("no", "SUP01", "2026-11-03", "Υπάρχει ακόμα απόθεμα", "telegram")]


def test_purchase_reject_fallback_with_other_reason(db):
    path, _ = db
    purchases(path)
    bot = tb.Bot(FakeApi(), CFG, path)
    press(bot, "prr:SUP01:2026-11-03:2")
    write(bot, "Αλλάζουμε προμηθευτή")
    rows = store.list_purchase_suggestions(db=path)
    assert {r["status"] for r in rows} == {store.REJECTED}
    assert store.purchase_history(rows[0]["id"], db=path)[-1]["details"] == {
        "reason": "Άλλο — Αλλάζουμε προμηθευτή"}
