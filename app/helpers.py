"""Plain functions behind the Never Empty dashboard (no Streamlit here, so they are testable).

Formatting (Greek numbers and dates), labels for the store's codes, the split of
suggestions into tabs, the home page numbers, history rows and the assistant call.
"""
import re
from datetime import date

import pandas as pd

from rafi import settings as _settings
from rafi.suggestions import (EXTRA_ROUTE, LOSS, MARGINAL, OFFER, ORDER, URGENT,
                              pending_totals)

PROPOSED = "ΠΡΟΤΑΣΗ"  # same value as rafi.store.PROPOSED

# Scenario day of the challenge (same as TODAY in scripts/daily_run.py).
SCENARIO_TODAY = date(2026, 11, 1)
COMPANY_LINE = "DEMO ΔΙΑΝΟΜΕΣ Α.Ε. · SoftOne ERP"

WEEKDAYS = ["Δευτέρα", "Τρίτη", "Τετάρτη", "Πέμπτη", "Παρασκευή", "Σάββατο", "Κυριακή"]
WEEKDAYS_SHORT = ["Δευτ.", "Τρ.", "Τετ.", "Πέμ.", "Παρ.", "Σάβ.", "Κυρ."]
UNITS = {"ΚΙΒ": "κιβ.", "ΤΕΜ": "τεμ."}

LABELS = {
    # suggestion types
    ORDER: "Παραγγελία",
    LOSS: "Κίνδυνος απώλειας",
    OFFER: "Προσφορά",
    # flags
    URGENT: "Επείγον",
    EXTRA_ROUTE: "Έκτακτο δρομολόγιο",
    MARGINAL: "Οριακό",
    "ΚΛΕΙΣΤΟ_ΧΕΙΜΩΝΑ": "Κλειστό τον χειμώνα",
    "ΗΔΗ_ΕΞΑΝΤΛΗΜΕΝΟ": "Ήδη εξαντλημένο",
    # statuses
    PROPOSED: "Πρόταση",
    "ΑΠΟΣΤΕΛΛΕΤΑΙ": "Αποστέλλεται",
    "ΣΤΑΛΘΗΚΕ_ΣΤΟ_ERP": "Στάλθηκε στο ERP",
    "ΑΠΟΡΡΙΦΘΗΚΕ": "Απορρίφθηκε",
    # history actions
    "ΔΗΜΙΟΥΡΓΙΑ": "Δημιουργία",
    "ΕΓΚΡΙΣΗ": "Έγκριση",
    "ΔΙΟΡΘΩΣΗ": "Διόρθωση",
    "ΑΠΟΡΡΙΨΗ": "Απόρριψη",
    "ΑΠΟΣΤΟΛΗ": "Αποστολή στο ERP",
    "ΣΦΑΛΜΑ_ΑΠΟΣΤΟΛΗΣ": "Σφάλμα αποστολής",
}

TAB_NEEDS_YOU, TAB_ORDERS, TAB_OFFERS = "Χρειάζονται εσένα", "Παραγγελίες", "Προσφορές"
TABS = [TAB_NEEDS_YOU, TAB_ORDERS, TAB_OFFERS]
CALL_FIRST = "τηλεφώνημα πρώτα"

REJECT_REASONS = ["Ο πελάτης έχει ακόμα απόθεμα", "Ο πελάτης έκλεισε",
                  "Το χειρίζομαι τηλεφωνικά", "Άλλο"]

AFTER_APPROVAL_OPTIONS = {
    _settings.DRAFT_ONLY: "Μόνο πρόχειρη στο ERP",
    _settings.DRAFT_AND_MESSAGE: "Πρόχειρη στο ERP και μήνυμα στον πελάτη",
}

ASSISTANT_UNAVAILABLE = "Ο βοηθός AI δεν είναι ακόμα διαθέσιμος"
AI_BADGE = "Βοηθός AI"


# --- numbers and dates -----------------------------------------------------

def fmt_num(x, decimals=0):
    """1234.5 -> '1.234,5' style (Greek separators). None -> '—'."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    s = f"{float(x):,.{decimals}f}"
    return s.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def fmt_eur(x):
    """42589.2 -> '42.589,20 €'."""
    return "—" if x is None else f"{fmt_num(x, 2)} €"


def fmt_qty(x):
    """Whole quantities without decimals, others with one."""
    if x is None:
        return "—"
    return fmt_num(x, 0) if float(x).is_integer() else fmt_num(x, 1)


def _to_date(d):
    if d is None or d == "" or (not isinstance(d, str) and pd.isna(d)):
        return None
    return pd.Timestamp(d)


def fmt_date(d):
    """'2026-11-02' -> 'Δευτ. 02/11/2026'. Empty -> '—'."""
    t = _to_date(d)
    return "—" if t is None else f"{WEEKDAYS_SHORT[t.weekday()]} {t:%d/%m/%Y}"


def fmt_date_long(d):
    """'2026-11-01' -> 'Κυριακή 01/11/2026'."""
    t = _to_date(d)
    return "—" if t is None else f"{WEEKDAYS[t.weekday()]} {t:%d/%m/%Y}"


def fmt_datetime(d):
    """'2026-11-07T16:20:05' -> 'Σάβ. 07/11/2026 16:20'."""
    t = _to_date(d)
    return "—" if t is None else f"{fmt_date(t)} {t:%H:%M}"


# --- codes -> words ----------------------------------------------------------

def label(code):
    """Greek label for a store code; unknown codes lose their underscores."""
    if code is None:
        return ""
    code = str(code)
    if code in LABELS:
        return LABELS[code]
    text = code.replace("_", " ").strip()
    return text[:1].upper() + text[1:].lower() if text.isupper() else text


def humanize(text):
    """Replace any known code inside a message (e.g. a store error) by its label."""
    out = str(text)
    for code in sorted(LABELS, key=len, reverse=True):
        if "_" in code:
            out = out.replace(code, LABELS[code])
    return out


def flag_labels(s):
    """Visible marks of a suggestion, e.g. ['Επείγον', 'Έκτακτο δρομολόγιο']."""
    flags = s["payload"].get("FLAGS", [])
    out = [label(f) for f in flags if f != MARGINAL]
    if MARGINAL in flags:
        out.append(f"{label(MARGINAL)} · {CALL_FIRST}")
    return out


def user_label(user, salesmen=None):
    """Who did it, in words: 'Σύστημα', 'Διαχειριστής' or the salesman's name."""
    u = str(user)
    if u == "system":
        return "Σύστημα"
    if u == "admin":
        return "Διαχειριστής"
    if salesmen is not None and u.isdigit() and int(u) in salesmen:
        return str(salesmen[int(u)])
    return u


ADMIN = "admin"  # the dashboard has one user for now; shown as «Διαχειριστής»


def store_user(s=None):
    """User recorded in the store for every action from the dashboard."""
    return ADMIN


def is_ai(source):
    """True only for text written by the assistant (source or TEXT_SOURCE equal to "AI")."""
    return source == "AI"


# --- tabs and home page ------------------------------------------------------

def needs_you(s):
    """Loss risks, and orders marked urgent or needing an extra route."""
    if s["action_type"] == LOSS:
        return True
    flags = s["payload"].get("FLAGS", [])
    return s["action_type"] == ORDER and (URGENT in flags or EXTRA_ROUTE in flags)


def tab_of(s):
    if needs_you(s):
        return TAB_NEEDS_YOU
    if s["action_type"] == ORDER:
        return TAB_ORDERS
    if s["action_type"] == OFFER:
        return TAB_OFFERS
    return None


def split_tabs(suggestions):
    """{tab: [suggestions]} for proposals only, no overlap, keeping the store's order."""
    tabs = {t: [] for t in TABS}
    for s in suggestions:
        if s["status"] != PROPOSED:
            continue
        tab = tab_of(s)
        if tab:
            tabs[tab].append(s)
    return tabs


def home_metrics(suggestions):
    """Numbers of the home page, from proposals only.

    The two values come from rafi.suggestions.pending_totals, the same function the
    morning briefing of scripts/daily_run.py uses.
    """
    pending = [s for s in suggestions if s["status"] == PROPOSED]
    totals = pending_totals(pending)
    extra = sum(1 for s in pending if s["action_type"] == ORDER
                and ({URGENT, EXTRA_ROUTE} & set(s["payload"].get("FLAGS", []))))
    return {
        "pending": len(pending),
        "extra_routes": extra,
        "loss_risks": sum(1 for s in pending if s["action_type"] == LOSS),
        "offers": sum(1 for s in pending if s["action_type"] == OFFER),
        "at_risk_eur": totals["at_risk_eur"],
        "offers_eur": totals["offers_eur"],
    }


def summary(s, limit=120):
    """One line about the suggestion: the first sentence of its reason, shortened."""
    reason = str(s["payload"].get("REASON") or "").strip()
    # first clause: up to "· " or the end of a sentence (". " before a capital letter,
    # so that "κιβ. 12τεμ" inside an item name does not cut it)
    first = re.split(r"\.\s+(?=[«A-ZΑ-ΩΆΈΉΊΌΎΏ])", reason.split("· ")[0])[0].rstrip(".")
    if len(first) > limit:
        first = first[:limit - 1].rstrip() + "…"
    return first


# --- lines, corrections, rejection --------------------------------------------

def item_lookup(items):
    """{MTRL: (name, unit)} from MTRL.csv."""
    return {int(r.MTRL): (str(r.NAME), UNITS.get(str(r.MTRUNIT1), str(r.MTRUNIT1)))
            for r in items.itertuples(index=False)}


def lines_rows(s, items_map):
    """Rows of the items table of one suggestion, with real values (int, Timestamp or None).

    Earliest depletion first; lines without a date at the end.
    """
    rows = []
    for l in s["payload"]["LINES"]:
        name, unit = items_map.get(int(l["MTRL"]), (l.get("ITEM_NAME") or str(l["MTRL"]), ""))
        rows.append({"Είδος": name, "Ποσότητα": int(l["QTY1"]), "Μονάδα": unit,
                     "Τελειώνει": _to_date(l.get("PREDICTED_DEPLETION_DATE"))})
    return sorted(rows, key=lambda r: (r["Τελειώνει"] is None, r["Τελειώνει"] or 0))


def lines_frame(rows):
    """DataFrame of lines_rows: the quantity keeps its unit in the display only."""
    df = pd.DataFrame(rows, columns=["Είδος", "Ποσότητα", "Μονάδα", "Τελειώνει"])
    df["Ποσότητα"] = df["Ποσότητα"].astype("int64")
    df["Τελειώνει"] = pd.to_datetime(df["Τελειώνει"])
    return df


def fmt_day(d):
    """'2026-11-02' -> 'Δευτ. 02/11' (short, for tables and cards). Empty -> '—'."""
    t = _to_date(d)
    return "—" if t is None else f"{WEEKDAYS_SHORT[t.weekday()]} {t:%d/%m}"


# Display formats of the tables: the data stay numbers and dates (so sorting is right),
# only what is shown is Greek.
TABLE_FORMATS = {
    "Ποσότητα": lambda x: fmt_qty(x),
    "Αξία": lambda x: fmt_eur(x),
    "Τελειώνει": lambda x: fmt_day(x),
    "Ημερομηνία": lambda x: fmt_datetime(x),
}


def table_formats(df):
    """{column: function} for the columns of df that have a Greek display format."""
    return {c: f for c, f in TABLE_FORMATS.items() if c in df.columns}


def changed_quantities(lines, new_values):
    """[(MTRL, new qty)] only for the lines whose quantity changed.

    new_values: {MTRL: qty}. Raises ValueError for anything that is not a whole number > 0.
    """
    out = []
    for l in lines:
        mtrl = int(l["MTRL"])
        if mtrl not in new_values:
            continue
        v = new_values[mtrl]
        if isinstance(v, bool) or float(v) != int(float(v)) or int(float(v)) <= 0:
            raise ValueError("Οι ποσότητες πρέπει να είναι ακέραιοι αριθμοί μεγαλύτεροι από 0.")
        v = int(float(v))
        if v != int(l["QTY1"]):
            out.append((mtrl, v))
    return out


def rejection_text(reason, comment=""):
    """'Ο πελάτης έκλεισε — σχόλιο'. The reason is required."""
    if not reason or not str(reason).strip():
        raise ValueError("Διάλεξε λόγο απόρριψης.")
    comment = (comment or "").strip()
    return f"{reason.strip()} — {comment}" if comment else reason.strip()


def erp_error_kind(error):
    """'down' if the ERP did not answer, 'rejected' if it refused the draft, else 'other'."""
    import requests
    cause = error.__cause__
    if isinstance(cause, (requests.ConnectionError, requests.Timeout)):
        return "down"
    if getattr(error, "errors", None) or str(error).startswith("Το ERP απέρριψε"):
        return "rejected"
    return "other"


def erp_rejection_message(error):
    errors = getattr(error, "errors", None)
    return "; ".join(map(str, errors)) if errors else str(error)


# --- history -----------------------------------------------------------------

def history_details(h, items_map=None):
    """Details of one history entry as plain text (never raw JSON)."""
    d = h.get("details") or {}
    action = h.get("action")
    items_map = items_map or {}
    if action == "ΔΗΜΙΟΥΡΓΙΑ":
        parts = []
        if d.get("value_eur") is not None:
            parts.append(f"Αξία {fmt_eur(d['value_eur'])}")
        if d.get("priority") is not None:
            parts.append(f"προτεραιότητα {d['priority']}")
        if d.get("flags"):
            parts.append("σημάνσεις: " + ", ".join(label(f) for f in d["flags"]))
        return " · ".join(parts)
    if action == "ΔΙΟΡΘΩΣΗ":
        name = items_map.get(int(d.get("MTRL", 0)), (str(d.get("MTRL", "")), ""))[0]
        return f"{name}: ποσότητα {d.get('old_qty')} → {d.get('new_qty')}"
    if action == "ΑΠΟΡΡΙΨΗ":
        return f"Λόγος: {d.get('reason', '')}"
    if action == "ΑΠΟΣΤΟΛΗ":
        return f"Πρόχειρη στο ERP με αριθμό {h.get('erp_id')}" if h.get("erp_id") else ""
    if action == "ΣΦΑΛΜΑ_ΑΠΟΣΤΟΛΗΣ":
        if d.get("errors"):
            return "Το ERP απέρριψε την πρόχειρη: " + "; ".join(map(str, d["errors"]))
        return "Το ERP δεν απάντησε· η πρόταση έμεινε προς έγκριση"
    if action == "ΕΓΚΡΙΣΗ":
        return "Εγκρίθηκε για αποστολή στο ERP"
    if isinstance(d, dict):
        return " · ".join(f"{label(k)}: {v}" for k, v in d.items() if v not in (None, "", []))
    return str(d)


def history_rows(entries, customers_map=None, items_map=None, salesmen=None):
    """Rows for the history table, newest first."""
    customers_map = customers_map or {}
    rows = []
    for h in sorted(entries, key=lambda h: h["id"], reverse=True):
        name, code = customers_map.get(int(h["trdr"]), (str(h["trdr"]), ""))
        rows.append({"Ημερομηνία": pd.Timestamp(h["at"]), "Πελάτης": name, "Κωδικός": code,
                     "Τύπος": label(h["action_type"]), "Κίνηση": label(h["action"]),
                     "Λεπτομέρειες": history_details(h, items_map),
                     "Χρήστης": user_label(h["user"], salesmen), "_trdr": int(h["trdr"])})
    return rows


# --- assistant ---------------------------------------------------------------

def ask_assistant(question, context):
    """{"text", "source"} from rafi.llm.answer, or the 'not available yet' message.

    source is None when the text did not come from the assistant.
    """
    try:
        import importlib
        llm = importlib.import_module("rafi.llm")
        out = llm.answer(question, context)
        text = out.get("text") if isinstance(out, dict) else None
        if not text or not str(text).strip():
            raise ValueError("empty answer")
        return {"text": str(text).strip(), "source": str(out.get("source") or "AI")}
    except Exception:
        return {"text": ASSISTANT_UNAVAILABLE, "source": None}


def _brief(s, customers_map):
    p = s["payload"]
    name = customers_map.get(int(s["trdr"]), ("", ""))[0]
    return {"id": s["id"], "customer": name, "customer_code": p.get("CUSTOMER_CODE"),
            "type": label(s["action_type"]), "flags": [label(f) for f in p.get("FLAGS", [])],
            "delivery_date": s["delivery_date"], "value_eur": s["value_eur"],
            "priority": s["priority"], "reason": p.get("REASON")}


def general_context(suggestions, customers_map=None):
    """Context for the general chat: home numbers and the first 10 proposals."""
    customers_map = customers_map or {}
    pending = [s for s in suggestions if s["status"] == PROPOSED]
    return {"today": SCENARIO_TODAY.isoformat(), "metrics": home_metrics(suggestions),
            "top_suggestions": [_brief(s, customers_map) for s in pending[:10]]}


def suggestion_context(s, customers_map=None, items_map=None):
    """Context for the chat of one suggestion: all its fields and its lines."""
    customers_map, items_map = customers_map or {}, items_map or {}
    p = s["payload"]
    name = customers_map.get(int(s["trdr"]), ("", ""))[0]
    lines = []
    for l in p.get("LINES", []):
        item, unit = items_map.get(int(l["MTRL"]), (l.get("ITEM_NAME"), ""))
        lines.append({**l, "ITEM_NAME": item, "UNIT": unit})
    fields = {k: v for k, v in s.items() if k != "payload"}
    return {"today": SCENARIO_TODAY.isoformat(), "customer_name": name, **fields,
            "type": label(s["action_type"]), "payload": {**p, "LINES": lines}}


# --- settings ----------------------------------------------------------------

def read_settings(path=None):
    return _settings.load_settings(path)


# Extra numbers of the settings page, shown only when the settings file has them:
# {key: (label, step, whole number)}
EXTRA_SETTINGS = {
    "stock_cover_weeks": ("Εβδομάδες κάλυψης αποθήκης", 1.0, False),
    "safety_weeks": ("Απόθεμα ασφαλείας, εβδομάδες", 0.5, False),
    "ai_texts_top_n": ("Κείμενα AI για τις πρώτες N προτάσεις", 1, True),
}


def write_settings(extra_route_cost_eur, after_approval_mode, path=None, extra=None):
    """Save the settings of the page; the other keys keep their values.

    extra: {key: number} for the keys of EXTRA_SETTINGS (never negative; whole where asked).
    """
    if after_approval_mode not in AFTER_APPROVAL_OPTIONS:
        raise ValueError(f"unknown mode {after_approval_mode}")
    cost = float(extra_route_cost_eur)
    if cost < 0:
        raise ValueError("Το κόστος δεν μπορεί να είναι αρνητικό.")
    current = _settings.load_settings(path)
    current.update(extra_route_cost_eur=int(cost) if cost.is_integer() else cost,
                   after_approval_mode=after_approval_mode)
    for key, value in (extra or {}).items():
        if key not in EXTRA_SETTINGS:
            raise ValueError(f"unknown setting {key}")
        label_, _, whole = EXTRA_SETTINGS[key]
        v = float(value)
        if v < 0 or (whole and not v.is_integer()):
            raise ValueError(f"Μη έγκυρη τιμή για «{label_}».")
        current[key] = int(v) if v.is_integer() else v
    return _settings.save_settings(current, path)


# --- cards (Πελάτες and Αρχική) ------------------------------------------------

WAREHOUSE_SHORTAGE = "ΕΛΛΕΙΨΗ_ΑΠΟΘΗΚΗΣ"
LABELS[WAREHOUSE_SHORTAGE] = "Έλλειψη αποθήκης"
DAY_ARTICLE = ["τη", "την", "την", "την", "την", "το", "την"]


def caps(text):
    """Greek capitals without accents, as on signs: 'Κίνδυνος απώλειας' -> 'ΚΙΝΔΥΝΟΣ ΑΠΩΛΕΙΑΣ'."""
    import unicodedata
    plain = "".join(c for c in unicodedata.normalize("NFD", str(text))
                    if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", plain).upper()


def dm(d):
    """'2026-11-02' -> '2/11'."""
    t = _to_date(d)
    return "—" if t is None else f"{t.day}/{t.month}"


def day_phrase(d):
    """'2026-11-02' -> 'τη Δευτέρα 2/11'; '2026-11-07' -> 'το Σάββατο 7/11'."""
    t = _to_date(d)
    return "—" if t is None else f"{DAY_ARTICLE[t.weekday()]} {WEEKDAYS[t.weekday()]} {dm(t)}"


def items_count(n):
    return f"{n} είδος" if n == 1 else f"{n} είδη"


def card_badge(s):
    """(text, kind) of the one mark a card shows; kind picks the colour."""
    flags = set(s["payload"].get("FLAGS", []))
    kind = s["action_type"]
    if kind == LOSS:
        return "Κίνδυνος απώλειας", "loss"
    if kind == OFFER:
        return "Προσφορά", "offer"
    if URGENT in flags and EXTRA_ROUTE in flags:
        return "Επείγον · Έκτακτο", "urgent"
    if EXTRA_ROUTE in flags:
        return "Έκτακτο", "urgent"
    if URGENT in flags:
        return "Επείγον", "urgent"
    if MARGINAL in flags:
        return "Οριακό · τηλεφώνημα", "call"
    return "Παραγγελία", "order"


def facts_by_suggestion(entries):
    """{suggestion id: facts} from the ΔΗΜΙΟΥΡΓΙΑ entries of the history."""
    out = {}
    for e in entries:
        d = e.get("details") or {}
        if e.get("action") == "ΔΗΜΙΟΥΡΓΙΑ" and isinstance(d, dict) and d.get("facts"):
            out[int(e["suggestion_id"])] = d["facts"]
    return out


def card_line(s, facts=None):
    """One short human line under the customer's name on a card."""
    facts = facts or {}
    flags = set(s["payload"].get("FLAGS", []))
    n = items_count(len(s["payload"].get("LINES", [])))
    value = fmt_eur(s["value_eur"])
    when = day_phrase(s["delivery_date"])
    kind = s["action_type"]
    if kind == LOSS:
        if facts.get("days_early"):
            return f"Σταμάτησε {facts['days_early']} ημέρες νωρίς · τηλεφώνημα σήμερα"
        if facts.get("days_silent"):
            return (f"Καμία παραγγελία εδώ και {facts['days_silent']} ημέρες, πέρα από το "
                    f"συνηθισμένο · τηλεφώνημα")
        return "Σιωπή πέρα από το συνηθισμένο · τηλεφώνημα"
    if kind == OFFER:
        return f"Εποχικά είδη · παράδοση {when} · {n} · {value}"
    if EXTRA_ROUTE in flags:
        return f"Έκτακτο δρομολόγιο {when} · {n} · {value}"
    if URGENT in flags:
        return f"Επείγουσα παράδοση {when} · {n} · {value}"
    if MARGINAL in flags:
        return f"Κανονική επίσκεψη {when} · τηλεφώνημα πρώτα · {n} · {value}"
    return f"Παράδοση {when} · {n} · {value}"


def name_with_article(name):
    """'Ταβέρνα Η Αυλή' -> 'Η Ταβέρνα Η Αυλή'; most shop names take 'Το'."""
    feminine = ("Ταβέρνα", "Καφετέρια", "Πιτσαρία", "Ψαροταβέρνα")
    return f"{'Η' if str(name).startswith(feminine) else 'Το'} {name}"


def home_sentence(s, name, code, facts=None, route_cost=60):
    """HTML sentence of the home list 'Χρειάζονται εσένα σήμερα' (customers)."""
    import html as _html
    facts = facts or {}
    who = f"<b>{_html.escape(name)} ({_html.escape(str(code))})</b>"
    flags = set(s["payload"].get("FLAGS", []))
    n = items_count(len(s["payload"].get("LINES", [])))
    value = fmt_eur(s["value_eur"])
    when = day_phrase(s["delivery_date"])
    article = name_with_article(name).split(" ")[0]
    if s["action_type"] == LOSS:
        if facts.get("days_early"):
            yearly = (f", με {fmt_num(facts['purchases_12m_eur'])} € αγορές τον χρόνο"
                      if facts.get("purchases_12m_eur") else "")
            t = _to_date(s["delivery_date"])
            return (f"{article} {who} σταμάτησε {facts['days_early']} ημέρες νωρίτερα από το "
                    f"συνηθισμένο του κλείσιμο{yearly}. Τηλεφώνημα σήμερα και παραγγελία "
                    f"επαναλειτουργίας για {t.day}/{t.month}/{t.year}.")
        return (f"{article} {who} σταμάτησε να παραγγέλνει απρόσμενα. Τηλεφώνημα και "
                f"παραγγελία {n} ({value}) για {WEEKDAYS[_to_date(s['delivery_date']).weekday()]} "
                f"{dm(s['delivery_date'])}.")
    if EXTRA_ROUTE in flags:
        return (f"Προτείνουμε <b>έκτακτο δρομολόγιο {when}</b> για {article.lower()} {who}: "
                f"{n}, {value}. Μέχρι την κανονική επίσκεψη θα χάνονταν πωλήσεις πάνω από τα "
                f"{fmt_num(route_cost)} € του δρομολογίου.")
    return (f"{article} {who} χρειάζεται παράδοση <b>{when}</b>: {n}, {value}.")


def warehouse_shortage(s, items_map=None):
    """Names of the items of a suggestion that the warehouse may not have in time.

    The mark is ΕΛΛΕΙΨΗ_ΑΠΟΘΗΚΗΣ, on the suggestion (FLAGS) or on single lines (FLAGS of the
    line). Returns None when there is no such mark, else a list of item names (may be empty).
    """
    items_map = items_map or {}
    p = s["payload"]
    lines = [l for l in p.get("LINES", []) if WAREHOUSE_SHORTAGE in (l.get("FLAGS") or [])]
    if not lines and WAREHOUSE_SHORTAGE not in p.get("FLAGS", []):
        return None
    return [items_map.get(int(l["MTRL"]), (l.get("ITEM_NAME") or str(l["MTRL"]), ""))[0]
            for l in lines]


MONTHS_GEN = ["Ιανουαρίου", "Φεβρουαρίου", "Μαρτίου", "Απριλίου", "Μαΐου", "Ιουνίου", "Ιουλίου",
              "Αυγούστου", "Σεπτεμβρίου", "Οκτωβρίου", "Νοεμβρίου", "Δεκεμβρίου"]


def fmt_sync(d):
    """'2026-11-01T06:00:12' -> '01/11 06:00'."""
    t = _to_date(d)
    return "—" if t is None else f"{t:%d/%m %H:%M}"


def fmt_date_words(d):
    """'2026-11-01' -> 'Κυριακή 1 Νοεμβρίου 2026'."""
    t = _to_date(d)
    return "—" if t is None else f"{WEEKDAYS[t.weekday()]} {t.day} {MONTHS_GEN[t.month - 1]} {t.year}"


def morning_texts():
    """{"customers": text, "warehouse": text} from rafi.agents.orchestrator.morning_summary,
    keeping only plain texts. {} while the coordinator is not there or fails."""
    import importlib
    try:
        out = importlib.import_module("rafi.agents.orchestrator").morning_summary()
    except Exception:
        return {}
    if not isinstance(out, dict):
        return {}
    texts = {}
    for k in ("customers", "warehouse"):
        v = out.get(k)
        if isinstance(v, dict):
            v = v.get("text") or v.get("summary")
        if isinstance(v, str) and v.strip():
            texts[k] = v.strip()
    return texts


MESSAGE_SENT = "Στάλθηκε και μήνυμα στον πελάτη."
MESSAGE_NOT_SENT = "Το μήνυμα στον πελάτη δεν στάλθηκε."
MESSAGE_NOT_READY = "Το μήνυμα στον πελάτη δεν στάλθηκε: η αποστολή μηνυμάτων δεν είναι ακόμα ενεργή."


def notify_after_approval(suggestion_id, mode):
    """After an approval: message the customer when the setting asks for it.

    Returns None when no message is wanted, else (streamlit kind, text) for the page.
    """
    if mode != _settings.DRAFT_AND_MESSAGE:
        return None
    import importlib
    try:
        notify = importlib.import_module("rafi.telegram_bot").notify_customer
    except Exception:
        return ("warning", MESSAGE_NOT_READY)
    try:
        ok = bool(notify(int(suggestion_id)))
    except Exception:
        ok = False
    return ("success", MESSAGE_SENT) if ok else ("warning", MESSAGE_NOT_SENT)
