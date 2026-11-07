"""Telegram bot of Ράφι: morning brief and approvals for the Administrator, and the
message to the customer of the demo after an approval.

Only the chat ids of .env are served (TELEGRAM_ADMIN_CHAT_ID, TELEGRAM_CUSTOMER_CHAT_ID);
updates from any other chat are ignored. Nothing reaches the ERP without a button press,
and approvals go through rafi.store, which never sends a suggestion twice.

Settings (environment or .env in the repository root):
  TELEGRAM_BOT_TOKEN, TELEGRAM_ADMIN_CHAT_ID, TELEGRAM_CUSTOMER_CHAT_ID

CLI: python -m rafi.telegram_bot whoami   chat ids of whoever pressed Start
     python -m rafi.telegram_bot brief    morning brief to the Administrator
     python -m rafi.telegram_bot run      answer the buttons (long polling)
"""
import argparse
import html
import logging
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
import requests

from app import helpers as h
from rafi import settings as rafi_settings
from rafi import store
from rafi.llm import _read_dotenv
from rafi.suggestions import MARGINAL, ORDER

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
log = logging.getLogger(__name__)

API_URL = "https://api.telegram.org"
POLL_SECONDS = 30
BOT_USER = "telegram"            # user recorded in the store for actions from the bot
CUSTOMER_MESSAGE = "ΜΗΝΥΜΑ_ΠΕΛΑΤΗ"  # history action written by notify_customer
WAREHOUSE_DAYS = 7
MAX_BUTTONS = 30
NOT_AVAILABLE = "δεν είναι διαθέσιμη ακόμα"
PURCHASE_REASONS = ["Υπάρχει ακόμα απόθεμα", "Θα παραγγείλω αργότερα", "Άλλο"]
OTHER = "Άλλο"
LEGAL_SUFFIXES = ("Μ.Ε.Π.Ε.", "Ι.Κ.Ε.", "Α.Ε.", "Ο.Ε.", "Ε.Ε.")


class TelegramError(Exception):
    """A call to the Telegram Bot API failed. The message never holds the token."""


# --- settings -----------------------------------------------------------------

KEYS = ("TELEGRAM_BOT_TOKEN", "TELEGRAM_ADMIN_CHAT_ID", "TELEGRAM_CUSTOMER_CHAT_ID")


def config(dotenv=ROOT / ".env"):
    """The three Telegram settings: .env, overridden by real environment variables."""
    cfg = {k: "" for k in KEYS}
    cfg.update({k: v for k, v in _read_dotenv(dotenv).items() if k in KEYS and v})
    cfg.update({k: os.environ[k] for k in KEYS if os.environ.get(k)})
    return {k: str(v).strip() for k, v in cfg.items()}


# --- Telegram Bot API ---------------------------------------------------------

class TelegramApi:
    """Plain HTTPS calls with requests. Errors are re-raised without the token."""

    def __init__(self, token, base_url=API_URL):
        if not token:
            raise TelegramError("Λείπει το TELEGRAM_BOT_TOKEN στο .env.")
        self._token = token
        self._base = base_url.rstrip("/")

    def _safe(self, text):
        return str(text).replace(self._token, "***")

    def call(self, method, http_timeout=20, **params):
        url = f"{self._base}/bot{self._token}/{method}"
        try:
            r = requests.post(url, json=params, timeout=http_timeout)
            body = r.json()
        except (requests.RequestException, ValueError) as e:
            raise TelegramError(self._safe(f"{method}: {type(e).__name__}: {e}")) from None
        if not body.get("ok"):
            raise TelegramError(self._safe(f"{method}: {body.get('description', r.status_code)}"))
        return body.get("result")


def send(api, chat_id, text, keyboard=None):
    params = {"chat_id": chat_id, "text": text, "parse_mode": "HTML",
              "disable_web_page_preview": True}
    if keyboard:
        params["reply_markup"] = {"inline_keyboard": keyboard}
    return api.call("sendMessage", **params)


def button(text, data):
    return {"text": text, "callback_data": data}


# --- reference data -----------------------------------------------------------

_cache = {}


def customers_map():
    """{TRDR: (name, code)}"""
    if "customers" not in _cache:
        df = pd.read_csv(DATA / "TRDR.csv", encoding="utf-8-sig")
        _cache["customers"] = {int(r.TRDR): (str(r.NAME), str(r.CODE))
                               for r in df.itertuples()}
    return _cache["customers"]


def items_map():
    """{MTRL: (name, unit)}"""
    if "items" not in _cache:
        _cache["items"] = h.item_lookup(pd.read_csv(DATA / "MTRL.csv", encoding="utf-8-sig"))
    return _cache["items"]


def esc(x):
    return html.escape(str(x), quote=False)


def short_name(name, limit=24):
    """'Ταβέρνα Η Αυλή Α.Ε.' -> 'Ταβέρνα Η Αυλή'."""
    name = str(name).strip()
    for suffix in LEGAL_SUFFIXES:
        if name.endswith(" " + suffix):
            name = name[:-len(suffix) - 1].strip()
            break
    return name if len(name) <= limit else name[:limit - 1].rstrip() + "…"


def customer_of(s):
    name, code = customers_map().get(int(s["trdr"]), (str(s["trdr"]), ""))
    return name, s["payload"].get("CUSTOMER_CODE") or code


def kind_text(s):
    """'Παραγγελία · Έκτακτο δρομολόγιο' (no codes)."""
    return " · ".join([h.label(s["action_type"])] + h.flag_labels(s))


def lines_text(s):
    rows = h.lines_rows(s, items_map())
    return "\n".join(f"• {esc(r['Είδος'])}: {r['Ποσότητα']} {esc(r['Μονάδα'])}".rstrip()
                     for r in rows)


def ai_mark(payload):
    return f"\n<i>({h.AI_BADGE})</i>" if payload.get("TEXT_SOURCE") == "AI" else ""


# --- morning summary ----------------------------------------------------------

def _first(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and d.get(k) is not None:
            return d[k]
    return default


def local_customers(suggestions):
    """Customer part of the summary from the store, with app/helpers (fallback)."""
    m = h.home_metrics(suggestions)
    pending = [s for s in suggestions if s["status"] == h.PROPOSED]
    return {
        "counts": {"pending": m["pending"], "extra_routes": m["extra_routes"],
                   "loss_risks": m["loss_risks"], "offers": m["offers"],
                   "marginal": sum(1 for s in pending
                                   if MARGINAL in s["payload"].get("FLAGS", []))},
        "value_at_risk_eur": m["at_risk_eur"],
        "offers_value_eur": m["offers_eur"],
        "needs_you": [s["id"] for s in h.split_tabs(suggestions)[h.TAB_NEEDS_YOU]],
    }


def morning_summary(db=None):
    """The orchestrator's summary, filled in from the store where it is missing.

    The warehouse part is None when the orchestrator is not there yet.
    """
    suggestions = store.list_suggestions(db=db)
    local = local_customers(suggestions)
    try:
        from rafi.agents.orchestrator import morning_summary as orchestrator_summary
        data = orchestrator_summary(db=db) or {}
    except Exception as e:  # missing module or a failing run: keep the local numbers
        log.info("morning summary without the orchestrator (%s)", type(e).__name__)
        data = {}
    c = data.get("customers") or {}
    counts = {**local["counts"], **(c.get("counts") or {})}
    needs = c.get("needs_you")
    ids = [int(_first(n, "id", "suggestion_id", "ID") if isinstance(n, dict) else n)
           for n in needs] if isinstance(needs, list) else local["needs_you"]
    customers = {
        "counts": counts,
        "value_at_risk_eur": _first(c, "value_at_risk_eur", default=local["value_at_risk_eur"]),
        "offers_value_eur": _first(c, "offers_value_eur", default=local["offers_value_eur"]),
        "needs_you": ids,
    }
    return {"customers": customers, "warehouse": data.get("warehouse"),
            "suggestions": suggestions}


def supplier_orders(warehouse, today=h.SCENARIO_TODAY, days=WAREHOUSE_DAYS):
    """Supplier orders to approve within `days`, one per (supplier, order-by date)."""
    last = (pd.Timestamp(today) + timedelta(days=days - 1)).strftime("%Y-%m-%d")
    groups = {}
    for o in (warehouse or {}).get("orders") or []:
        sup = str(_first(o, "supplier_id", "SUPPLIER_ID", default=""))
        order_by = str(_first(o, "order_by", "ORDER_BY", default=""))[:10]
        if not sup or not order_by or order_by > last:
            continue
        g = groups.setdefault((sup, order_by), {
            "supplier_id": sup, "order_by": order_by, "value_eur": 0.0, "lines": 0,
            "supplier_name": _first(o, "supplier_name", "SUPPLIER_NAME", default=sup),
            "short": False})
        g["value_eur"] += float(_first(o, "value_eur", "VALUE_EUR", default=0) or 0)
        lines = _first(o, "lines", "LINES", "lines_count")
        g["lines"] += len(lines) if isinstance(lines, list) else int(lines or 1)
        flag = _first(o, "flag", "FLAG", default="")
        g["short"] = g["short"] or bool(flag) or bool(_first(o, "short", default=False))
    return sorted(groups.values(), key=lambda g: (g["order_by"], g["supplier_name"]))


def regular_orders(suggestions):
    """Proposals of the Orders tab, and those of them that bulk approval may send."""
    orders = h.split_tabs(suggestions)[h.TAB_ORDERS]
    bulk = [s for s in orders if MARGINAL not in s["payload"].get("FLAGS", [])]
    return orders, bulk


def marginal_text(n, verb_one, verb_many):
    """'1 οριακή δεν μπαίνει' / '3 οριακές δεν μπαίνουν'."""
    return f"{n} οριακή {verb_one}" if n == 1 else f"{n} οριακές {verb_many}"


def _value(rows):
    return sum(s["value_eur"] or 0.0 for s in rows)


def brief_messages(db=None):
    """The morning brief: up to 4 (text, keyboard) pairs."""
    summary = morning_summary(db)
    c, w, suggestions = summary["customers"], summary["warehouse"], summary["suggestions"]
    n = c["counts"]
    out = []

    # (a) summary
    text = [f"<b>Καλημέρα. Πρωινή ενημέρωση · {h.fmt_date_long(h.SCENARIO_TODAY)}</b>", "",
            "<b>Πελάτες</b>",
            f"Προς έγκριση: {h.fmt_num(n.get('pending'))}",
            f"Έκτακτα δρομολόγια: {h.fmt_num(n.get('extra_routes'))}",
            f"Κίνδυνοι απώλειας: {h.fmt_num(n.get('loss_risks'))}",
            f"Οριακές (τηλεφώνημα πρώτα): {h.fmt_num(n.get('marginal'))}",
            f"Προσφορές: {h.fmt_num(n.get('offers'))} · {h.fmt_eur(c['offers_value_eur'])}",
            f"Αξία σε κίνδυνο: {h.fmt_eur(c['value_at_risk_eur'])}", "",
            "<b>Αποθήκη</b>"]
    if w is None:
        text.append(f"Η σύνοψη της αποθήκης {NOT_AVAILABLE}.")
    else:
        text += [f"Παραγγελίες προμηθευτών προς έγκριση: "
                 f"{h.fmt_num(_first(w, 'orders_count', default=0))}",
                 f"Ελλείψεις: {h.fmt_num(_first(w, 'short_count', default=0))}"]
    out.append(("\n".join(text), None))

    # (b) needs you
    by_id = {s["id"]: s for s in suggestions}
    needs = [by_id[i] for i in c["needs_you"] if i in by_id and by_id[i]["status"] == h.PROPOSED]
    text = [f"<b>Χρειάζονται εσένα σήμερα ({len(needs)})</b>"]
    keyboard = []
    for s in needs[:MAX_BUTTONS]:
        name, code = customer_of(s)
        text.append(f"• {esc(short_name(name))}: {esc(kind_text(s))} · "
                    f"{h.fmt_date(s['delivery_date'])} · {h.fmt_eur(s['value_eur'])}")
        keyboard.append([button(f"{code} {short_name(name)}", f"s:{s['id']}")])
    if len(needs) > MAX_BUTTONS:
        text.append(f"…και άλλες {len(needs) - MAX_BUTTONS} στο dashboard.")
    if not needs:
        text.append("Τίποτα επείγον σήμερα.")
    out.append(("\n".join(text), keyboard or None))

    # (c) regular orders
    orders, bulk = regular_orders(suggestions)
    text = [f"<b>Κανονικές παραγγελίες: {len(orders)} · {h.fmt_eur(_value(orders))}</b>"]
    marginal = len(orders) - len(bulk)
    if marginal:
        text.append(marginal_text(marginal, "θέλει τηλεφώνημα πρώτα και δεν μπαίνει",
                                  "θέλουν τηλεφώνημα πρώτα και δεν μπαίνουν")
                    + " στην έγκριση όλων.")
    keyboard = [[button("Δες λίστα", "ol"), button("Έγκριση όλων", "oa")]] if orders else None
    out.append(("\n".join(text), keyboard))

    # (d) warehouse
    if w is not None:
        groups = supplier_orders(w)
        text = [f"<b>Αποθήκη · παραγγελίες προμηθευτών {WAREHOUSE_DAYS} ημερών ({len(groups)})</b>"]
        keyboard = []
        for g in groups[:MAX_BUTTONS]:
            short = " · έλλειψη" if g["short"] else ""
            text.append(f"• {esc(g['supplier_name'])}: έως {h.fmt_date(g['order_by'])} · "
                        f"{h.fmt_eur(g['value_eur'])}{short}")
            keyboard.append([button(f"{short_name(g['supplier_name'], 28)} · "
                                    f"{h.fmt_date(g['order_by'])[:-5]}",
                                    f"p:{g['supplier_id']}:{g['order_by']}")])
        if not groups:
            text.append("Καμία παραγγελία προς έγκριση αυτή την εβδομάδα.")
        out.append(("\n".join(text), keyboard or None))
    return out


def send_brief(api, cfg, db=None):
    msgs = brief_messages(db)
    for text, keyboard in msgs:
        send(api, cfg["TELEGRAM_ADMIN_CHAT_ID"], text, keyboard)
    return len(msgs)


# --- customer message ---------------------------------------------------------

def customer_text(s):
    p = s["payload"]
    text = []
    if p.get("SALES_NOTE"):
        text += [esc(p["SALES_NOTE"]), ""]
    text += [f"Προτεινόμενη παράδοση: {h.fmt_date_long(s['delivery_date'])}", "Είδη:",
             lines_text(s)]
    if p.get("TEXT_SOURCE") == "AI":
        text += ["", "<i>Το κείμενο γράφτηκε με τη βοήθεια AI.</i>"]
    return "\n".join(text)


def _notify_customer(suggestion_id, api, cfg, db=None):
    if rafi_settings.load_settings().get("after_approval_mode") != rafi_settings.DRAFT_AND_MESSAGE:
        return False
    chat = cfg.get("TELEGRAM_CUSTOMER_CHAT_ID")
    if not chat:
        log.warning("notify_customer: λείπει το TELEGRAM_CUSTOMER_CHAT_ID")
        return False
    s = store.get(suggestion_id, db=db)
    if s["status"] != store.SENT:
        return False
    if any(e["action"] == CUSTOMER_MESSAGE for e in store.history(suggestion_id=suggestion_id,
                                                                  db=db)):
        return False  # one message per suggestion
    try:
        send(api, chat, customer_text(s))
    except TelegramError as e:
        log.warning("notify_customer: %s", e)
        return False
    with store._transaction(db) as conn:
        store._log(conn, suggestion_id, BOT_USER, CUSTOMER_MESSAGE,
                   {"Κανάλι": "Telegram",
                    "Παράδοση": h.fmt_date(s["delivery_date"]),
                    "Είδη": len(s["payload"].get("LINES", []))})
    return True


def _default_api(cfg):
    return TelegramApi(cfg["TELEGRAM_BOT_TOKEN"])


def notify_customer(suggestion_id) -> bool:
    """Message the demo customer about an approved suggestion. True if it was sent.

    Sends only when after_approval_mode is "draft_and_message", the suggestion is
    ΣΤΑΛΘΗΚΕ_ΣΤΟ_ERP and the customer was not messaged for it before. Writes
    ΜΗΝΥΜΑ_ΠΕΛΑΤΗ in the history.
    """
    cfg = config()
    try:
        api = _default_api(cfg)
        return _notify_customer(suggestion_id, api, cfg)
    except (TelegramError, store.StoreError) as e:
        log.warning("notify_customer: %s", e)
        return False


# --- bot ----------------------------------------------------------------------

def _erp_message(e):
    if h.erp_error_kind(e) == "down":
        return "Ο ERP δεν απαντά αυτή τη στιγμή. Η πρόταση έμεινε προς έγκριση· δοκίμασε ξανά."
    return f"Ο ERP δεν δέχτηκε την πρόχειρη: {esc(h.erp_rejection_message(e))}"


class Bot:
    """Answers the buttons of the brief. State (what text is expected next) is in memory."""

    def __init__(self, api, cfg, db=None):
        self.api, self.cfg, self.db = api, cfg, db
        self.admin = str(cfg.get("TELEGRAM_ADMIN_CHAT_ID") or "")
        self.customer = str(cfg.get("TELEGRAM_CUSTOMER_CHAT_ID") or "")
        self.waiting = {}   # chat id -> ("ask", id) | ("reject", id) | ("preject", sup, date)
        self.bulk = {}      # chat id -> ids shown in the bulk confirmation
        self.offset = None

    def say(self, text, keyboard=None):
        return send(self.api, self.admin, text, keyboard)

    # -- updates --

    def handle(self, update):
        if "callback_query" in update:
            q = update["callback_query"]
            chat = str(((q.get("message") or {}).get("chat") or {}).get("id", ""))
            if not self.admin or chat != self.admin:
                return False
            self.api.call("answerCallbackQuery", callback_query_id=q["id"])
            self.on_button(q.get("data") or "")
            return True
        msg = update.get("message") or {}
        chat = str((msg.get("chat") or {}).get("id", ""))
        text = (msg.get("text") or "").strip()
        if self.admin and chat == self.admin:
            self.on_text(text)
            return True
        if self.customer and chat == self.customer:
            if text and not text.startswith("/"):
                self.say(f"Ο πελάτης του demo απάντησε:\n«{esc(text)}»")
            return True
        return False  # unknown chat: ignored

    def poll_once(self):
        params = {"timeout": POLL_SECONDS, "allowed_updates": ["message", "callback_query"]}
        if self.offset is not None:
            params["offset"] = self.offset
        updates = self.api.call("getUpdates", http_timeout=POLL_SECONDS + 10, **params) or []
        for u in updates:
            self.offset = u["update_id"] + 1
            try:
                self.handle(u)
            except Exception as e:  # one bad update must not stop the bot
                log.exception("update %s", u.get("update_id"))
                if self.admin:
                    try:
                        self.say(f"Κάτι πήγε στραβά: {esc(h.humanize(e))}")
                    except TelegramError:
                        pass
        return len(updates)

    def run(self):
        print("Το bot τρέχει. Ctrl+C για τερματισμό.")
        while True:
            try:
                self.poll_once()
            except TelegramError as e:
                log.warning("%s", e)
                time.sleep(5)

    # -- text --

    def on_text(self, text):
        waiting = self.waiting.pop(self.admin, None)
        if text in ("/start", "/help"):
            self.say("Ράφι · Telegram. Στείλε /brief για την πρωινή ενημέρωση.")
        elif text == "/brief":
            send_brief(self.api, self.cfg, self.db)
        elif waiting and waiting[0] == "ask":
            self.ask(waiting[1], text)
        elif waiting and waiting[0] == "reject":
            self.reject(waiting[1], h.rejection_text(OTHER, text))
        elif waiting and waiting[0] == "preject":
            self.reject_purchase(waiting[1], waiting[2], h.rejection_text(OTHER, text))
        else:
            self.say("Στείλε /brief για την πρωινή ενημέρωση ή διάλεξε από τα κουμπιά.")

    # -- buttons --

    def on_button(self, data):
        cmd, _, arg = data.partition(":")
        if cmd == "s":
            self.open(int(arg))
        elif cmd == "a":
            self.approve(int(arg))
        elif cmd == "r":
            sid = int(arg)
            self.say("Γιατί απορρίπτεις;", [[button(r, f"rr:{sid}:{i}")]
                                            for i, r in enumerate(h.REJECT_REASONS)])
        elif cmd == "rr":
            sid, i = (int(x) for x in arg.split(":"))
            reason = h.REJECT_REASONS[i]
            if reason == OTHER:
                self.waiting[self.admin] = ("reject", sid)
                self.say("Γράψε τον λόγο σε ένα μήνυμα.")
            else:
                self.reject(sid, reason)
        elif cmd == "q":
            self.waiting[self.admin] = ("ask", int(arg))
            self.say("Γράψε την ερώτησή σου για αυτή την πρόταση.")
        elif cmd == "ol":
            self.order_list()
        elif cmd == "oa":
            self.bulk_confirm()
        elif cmd == "ob":
            self.bulk_approve()
        elif cmd == "x":
            self.bulk.pop(self.admin, None)
            self.say("Ακυρώθηκε. Δεν στάλθηκε τίποτα.")
        elif cmd == "p":
            self.open_purchase(*arg.split(":", 1))
        elif cmd == "pa":
            self.approve_purchase(*arg.split(":", 1))
        elif cmd == "pr":
            sup, order_by = arg.split(":", 1)
            self.say("Γιατί απορρίπτεις την παραγγελία;",
                     [[button(r, f"prr:{sup}:{order_by}:{i}")]
                      for i, r in enumerate(PURCHASE_REASONS)])
        elif cmd == "prr":
            sup, order_by, i = arg.split(":")
            reason = PURCHASE_REASONS[int(i)]
            if reason == OTHER:
                self.waiting[self.admin] = ("preject", sup, order_by)
                self.say("Γράψε τον λόγο σε ένα μήνυμα.")
            else:
                self.reject_purchase(sup, order_by, reason)

    # -- customer suggestions --

    def open(self, sid):
        s = store.get(sid, db=self.db)
        p = s["payload"]
        name, code = customer_of(s)
        text = [f"<b>{esc(name)}</b> ({esc(code)})", esc(kind_text(s)),
                f"Παράδοση: {h.fmt_date_long(s['delivery_date'])}",
                f"Αξία: {h.fmt_eur(s['value_eur'])}", "",
                esc(p.get("REASON") or "") + ai_mark(p), "", "<b>Είδη</b>", lines_text(s)]
        if s["status"] != h.PROPOSED:
            text += ["", f"Κατάσταση: {h.label(s['status'])}"
                     + (f" · ERP {esc(s['erp_id'])}" if s.get("erp_id") else "")]
            keyboard = [[button("Ρώτα", f"q:{sid}")]]
        else:
            keyboard = [[button("Εγκρίνω", f"a:{sid}"), button("Απόρριψη", f"r:{sid}"),
                         button("Ρώτα", f"q:{sid}")]]
        self.say("\n".join(text), keyboard)

    def _approve(self, sid):
        """(ok, message). Never sends twice: the store refuses anything but ΠΡΟΤΑΣΗ."""
        from rafi.erp_client import ErpError
        try:
            erp_id = store.approve_and_send(sid, BOT_USER, db=self.db)
        except store.StoreError as e:
            return False, esc(h.humanize(e))
        except ErpError as e:
            return False, _erp_message(e)
        if _notify_customer(sid, self.api, self.cfg, self.db):
            return True, f"Στάλθηκε στο ERP ως πρόχειρη: {esc(erp_id)}. Ο πελάτης ενημερώθηκε."
        return True, f"Στάλθηκε στο ERP ως πρόχειρη: {esc(erp_id)}"

    def approve(self, sid):
        self.say(self._approve(sid)[1])

    def reject(self, sid, reason):
        try:
            store.reject(sid, reason, BOT_USER, db=self.db)
        except store.StoreError as e:
            self.say(esc(h.humanize(e)))
            return
        self.say(f"Η πρόταση απορρίφθηκε. Λόγος: {esc(reason)}")

    def ask(self, sid, question):
        s = store.get(sid, db=self.db)
        out = h.ask_assistant(question, h.suggestion_context(s, customers_map(), items_map()))
        mark = f"\n<i>({h.AI_BADGE})</i>" if out["source"] == "AI" else ""
        self.say(esc(out["text"]) + mark, [[button("Ρώτα ξανά", f"q:{sid}")]])

    # -- regular orders --

    def order_list(self):
        orders, _ = regular_orders(store.list_suggestions(db=self.db))
        text = [f"<b>Κανονικές παραγγελίες ({len(orders)})</b>"]
        keyboard = []
        for s in orders[:MAX_BUTTONS]:
            name, code = customer_of(s)
            call = " · οριακή, τηλεφώνημα πρώτα" if MARGINAL in s["payload"].get("FLAGS", []) else ""
            text.append(f"• {esc(short_name(name))}: {h.fmt_date(s['delivery_date'])} · "
                        f"{h.fmt_eur(s['value_eur'])}{call}")
            keyboard.append([button(f"{code} {short_name(name)}", f"s:{s['id']}")])
        if len(orders) > MAX_BUTTONS:
            text.append(f"…και άλλες {len(orders) - MAX_BUTTONS} στο dashboard.")
        self.say("\n".join(text), keyboard or None)

    def bulk_confirm(self):
        orders, bulk = regular_orders(store.list_suggestions(db=self.db))
        if not bulk:
            self.say("Δεν υπάρχουν κανονικές παραγγελίες για έγκριση.")
            return
        self.bulk[self.admin] = [s["id"] for s in bulk]
        text = (f"Θα σταλούν στο ERP ως πρόχειρες {len(bulk)} παραγγελίες, συνολικά "
                f"{h.fmt_eur(_value(bulk))}.")
        if len(orders) > len(bulk):
            text += " " + marginal_text(len(orders) - len(bulk), "δεν περιλαμβάνεται.",
                                        "δεν περιλαμβάνονται.")
        self.say(text + "\nΕπιβεβαιώνεις;",
                 [[button(f"Ναι, έγκριση {len(bulk)}", "ob"), button("Άκυρο", "x")]])

    def bulk_approve(self):
        ids = self.bulk.pop(self.admin, None)
        if not ids:
            self.say("Η επιβεβαίωση έληξε. Πάτα ξανά «Έγκριση όλων».")
            return
        from rafi.erp_client import ErpError
        sent, failed, value, stopped = 0, 0, 0.0, False
        for sid in ids:
            s = store.get(sid, db=self.db)
            if s["status"] != h.PROPOSED or MARGINAL in s["payload"].get("FLAGS", []):
                continue
            try:
                store.approve_and_send(sid, BOT_USER, db=self.db)
            except store.StoreError:
                continue
            except ErpError as e:
                failed += 1
                if h.erp_error_kind(e) == "down":
                    stopped = True
                    break
                continue
            sent += 1
            value += s["value_eur"] or 0.0
            _notify_customer(sid, self.api, self.cfg, self.db)
        text = f"Στάλθηκαν στο ERP ως πρόχειρες {sent} παραγγελίες · {h.fmt_eur(value)}."
        if failed:
            text += f" Δεν στάλθηκαν {failed}."
        if stopped:
            text += " Ο ERP δεν απαντά· οι υπόλοιπες έμειναν προς έγκριση."
        self.say(text)

    # -- supplier orders --

    def _purchase_rows(self, sup, order_by):
        return [r for r in store.list_purchase_suggestions(db=self.db)
                if r["supplier_id"] == sup and r["order_by"] == order_by]

    def open_purchase(self, sup, order_by):
        rows = self._purchase_rows(sup, order_by)
        if not rows:
            self.say("Δεν βρέθηκε αυτή η παραγγελία προμηθευτή.")
            return
        p0 = rows[0]["payload"]
        text = [f"<b>{esc(p0.get('SUPPLIER_NAME') or sup)}</b>",
                f"Παραγγελία έως {h.fmt_date_long(order_by)} · άφιξη "
                f"{h.fmt_date(rows[0]['arrival'])}",
                f"Αξία: {h.fmt_eur(sum(r['value_eur'] or 0 for r in rows))}", "", "<b>Είδη</b>"]
        for r in rows:
            p = r["payload"]
            state = "" if r["status"] == h.PROPOSED else f" ({h.label(r['status']).lower()})"
            text.append(f"• {esc(p.get('ITEM_NAME') or r['mtrl'])}: {h.fmt_qty(r['qty'])} "
                        f"{esc(p.get('UNIT') or '')}{state}")
            if p.get("REASON"):
                text.append(f"  {esc(p['REASON'])}")
        keyboard = None
        if any(r["status"] == h.PROPOSED for r in rows):
            keyboard = [[button("Εγκρίνω", f"pa:{sup}:{order_by}"),
                         button("Απόρριψη", f"pr:{sup}:{order_by}")]]
        self.say("\n".join(text), keyboard)

    def approve_purchase(self, sup, order_by):
        try:
            try:
                from rafi.agents.warehouse_agent import approve_purchase_order
            except ImportError:
                approve_purchase_order = self._approve_purchase_fallback
            approve_purchase_order(sup, order_by, BOT_USER)
        except store.StoreError as e:
            self.say(esc(h.humanize(e)))
            return
        self.say("Η παραγγελία προμηθευτή εγκρίθηκε.")

    def reject_purchase(self, sup, order_by, reason):
        try:
            try:
                from rafi.agents.warehouse_agent import reject_purchase_order
            except ImportError:
                reject_purchase_order = self._reject_purchase_fallback
            reject_purchase_order(sup, order_by, reason, BOT_USER)
        except store.StoreError as e:
            self.say(esc(h.humanize(e)))
            return
        self.say(f"Η παραγγελία προμηθευτή απορρίφθηκε. Λόγος: {esc(reason)}")

    def _open_rows(self, sup, order_by):
        rows = [r for r in self._purchase_rows(sup, order_by) if r["status"] == h.PROPOSED]
        if not rows:
            raise store.StoreError("Η παραγγελία δεν είναι πια προς έγκριση.")
        return rows

    def _approve_purchase_fallback(self, sup, order_by, user):
        for r in self._open_rows(sup, order_by):
            store.approve_purchase(r["id"], user, db=self.db)

    def _reject_purchase_fallback(self, sup, order_by, reason, user):
        for r in self._open_rows(sup, order_by):
            store.reject_purchase(r["id"], reason, user, db=self.db)


# --- CLI ----------------------------------------------------------------------

def whoami(api):
    """[(chat id, name)] of everyone in the bot's recent updates."""
    seen = {}
    for u in api.call("getUpdates") or []:
        msg = u.get("message") or (u.get("callback_query") or {}).get("message") or {}
        chat = msg.get("chat") or {}
        if "id" not in chat:
            continue
        who = u.get("message", {}).get("from") or chat
        name = " ".join(x for x in (who.get("first_name"), who.get("last_name")) if x)
        if who.get("username"):
            name += f" (@{who['username']})"
        seen[chat["id"]] = name or chat.get("title") or "—"
    return list(seen.items())


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m rafi.telegram_bot",
                                 description="Telegram bot του Ράφι.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("whoami", help="chat id όσων πάτησαν Start")
    sub.add_parser("brief", help="πρωινή ενημέρωση στον Διαχειριστή")
    sub.add_parser("run", help="απαντά στα κουμπιά (long polling)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    cfg = config()
    try:
        api = TelegramApi(cfg["TELEGRAM_BOT_TOKEN"])
        if args.cmd == "whoami":
            people = whoami(api)
            if not people:
                print("Κανένα μήνυμα. Άνοιξε το bot στο Telegram, πάτα Start και ξανατρέξε.")
            for chat_id, name in people:
                print(f"{chat_id}\t{name}")
            return 0
        if not cfg["TELEGRAM_ADMIN_CHAT_ID"]:
            print("Λείπει το TELEGRAM_ADMIN_CHAT_ID στο .env (δες: whoami).", file=sys.stderr)
            return 1
        if args.cmd == "brief":
            print(f"Στάλθηκαν {send_brief(api, cfg)} μηνύματα στον Διαχειριστή.")
            return 0
        try:
            Bot(api, cfg).run()
        except KeyboardInterrupt:
            print("\nΤο bot σταμάτησε.")
        return 0
    except TelegramError as e:
        print(f"Σφάλμα Telegram: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
