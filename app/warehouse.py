"""Plain functions behind the Αποθήκη page (no Streamlit here, so they are testable).

The orders come from rafi.agents.warehouse_agent.open_purchase_orders(): one dict per
supplier and order day, with its lines and a status:
  ΕΛΛΕΙΨΗ   ordered too late and some customer deliveries before the arrival are not
            covered (affected_deliveries lists only those)
  ΑΡΓΗΣΑΜΕ  ordered too late, the customers are covered, but we drop under the safety stock
  ""        a normal order
"""
import html
import importlib

import pandas as pd

from app import helpers as h

SOON = "Ο έλεγχος αποθήκης ενεργοποιείται σύντομα"
WEEK_DAYS = 7
ALL_DAYS = 90
SHORT, LATE = "ΕΛΛΕΙΨΗ", "ΑΡΓΗΣΑΜΕ"
OLD_TOO_LATE = "ΔΕΝ_ΦΤΑΝΕΙ"  # older stores, read as ΕΛΛΕΙΨΗ
TAGS = {SHORT: ("Έλλειψη", "short"), LATE: ("Αργήσαμε", "late"), "": ("Αγορά", "buy")}


def agent():
    """The agreed warehouse functions, or None while the warehouse agent does not have them."""
    try:
        mod = importlib.import_module("rafi.agents.warehouse_agent")
    except Exception:
        return None
    names = ("open_purchase_orders", "approve_purchase_order", "reject_purchase_order")
    if not all(callable(getattr(mod, n, None)) for n in names):
        return None
    return {n: getattr(mod, n) for n in names}


def order_key(o):
    """Text key of an order: supplier and order day ('S01|2026-11-02')."""
    t = h._to_date(o.get("order_by"))
    return f"{o.get('supplier_id')}|{t:%Y-%m-%d}" if t is not None else str(o.get("supplier_id"))


def status(o):
    """ΕΛΛΕΙΨΗ, ΑΡΓΗΣΑΜΕ or "" (an order without "status" falls back to "short")."""
    s = o.get("status")
    if s in (SHORT, LATE, ""):
        return s
    if s == OLD_TOO_LATE or o.get("short"):
        return SHORT
    return LATE if o.get("late") else ""


def tag_of(o):
    """(text, css kind) of the order's tag: Έλλειψη / Αργήσαμε / Αγορά."""
    return TAGS[status(o)]


def sort_orders(orders):
    """Shortages first, then late orders, then by order day and the most valuable."""
    rank = {SHORT: 0, LATE: 1, "": 2}

    def key(o):
        t = h._to_date(o.get("order_by"))
        return (rank[status(o)], t if t is not None else pd.Timestamp.max,
                -float(o.get("total_value_eur") or 0))
    return sorted(orders, key=key)


def _marked(o, state):
    if status(o) != state:
        return []
    lines = o.get("lines") or []
    flags = {state, OLD_TOO_LATE} if state == SHORT else {state}
    marked = [l for l in lines if flags & set(l.get("flags") or [])]
    return marked or lines


def short_lines(o):
    """Lines that leave customer deliveries uncovered (ΕΛΛΕΙΨΗ)."""
    return _marked(o, SHORT)


def late_lines(o):
    """Lines ordered too late while the customers are still covered (ΑΡΓΗΣΑΜΕ)."""
    return _marked(o, LATE)


def metrics(orders):
    """Numbers at the top of the page (for the orders of the week).

    short_deliveries counts only the customer deliveries that are not covered.
    """
    short = [o for o in orders if status(o) == SHORT]
    return {
        "orders": len(orders),
        "short_items": len({l.get("mtrl") for o in short for l in short_lines(o)}),
        "short_deliveries": sum(len(o.get("affected_deliveries") or []) for o in short),
        "late_items": len({l.get("mtrl") for o in orders for l in late_lines(o)}),
        "value_eur": round(sum(float(o.get("total_value_eur") or 0) for o in orders), 2),
    }


def affected_phrase(n):
    """'Επηρεάζεται 1 παράδοση σε πελάτη' / 'Επηρεάζονται 3 παραδόσεις σε πελάτες'."""
    return ("Επηρεάζεται 1 παράδοση σε πελάτη" if n == 1
            else f"Επηρεάζονται {n} παραδόσεις σε πελάτες")


def late_text(o):
    """Panel text of an ΑΡΓΗΣΑΜΕ order: customers covered, under the safety stock."""
    n = int(o.get("deliveries_before_arrival") or 0)
    if n == 0:
        cover = "Μέχρι την άφιξη δεν έχουμε παραδόσεις σε πελάτες"
    elif n == 1:
        cover = "Η 1 παράδοση σε πελάτη μέχρι την άφιξη καλύπτεται από το απόθεμα"
    else:
        cover = f"Οι {n} παραδόσεις σε πελάτες μέχρι την άφιξη καλύπτονται από το απόθεμα"
    return f"{cover}, αλλά θα πέσουμε κάτω από το απόθεμα ασφαλείας. Παράγγειλε σήμερα."


def _qty_unit(line):
    unit = line.get("unit") or ""
    return f"{h.fmt_qty(line.get('qty'))} {unit}".strip()


def card_line(o):
    """'Λαμπρυντικό πλυντηρίου 5L · 36 τεμ. · έως Δευτ. 2/11 · 460,80 €' or, with several
    items, '2 είδη (Χαρτί υγείας, Μεμβράνη) · έως Παρ. 6/11 · 1.551,50 €'."""
    lines = o.get("lines") or []
    until = f"έως {h.WEEKDAYS_SHORT[h._to_date(o['order_by']).weekday()]} {h.dm(o['order_by'])}"
    value = h.fmt_eur(o.get("total_value_eur"))
    if len(lines) == 1:
        return f"{lines[0].get('item_name')} · {_qty_unit(lines[0])} · {until} · {value}"
    names = ", ".join(str(l.get("item_name")) for l in lines[:3])
    more = f" και {len(lines) - 3} ακόμη" if len(lines) > 3 else ""
    return f"{h.items_count(len(lines))} ({names}{more}) · {until} · {value}"


def lines_frame(o):
    """Items table of an order, with real values (sorting stays right)."""
    rows = [{"Είδος": str(l.get("item_name")), "Ποσότητα": int(l.get("qty") or 0),
             "Μονάδα": l.get("unit") or "", "Αξία": float(l.get("value_eur") or 0)}
            for l in o.get("lines") or []]
    df = pd.DataFrame(rows, columns=["Είδος", "Ποσότητα", "Μονάδα", "Αξία"])
    df["Ποσότητα"] = df["Ποσότητα"].astype("int64")
    df["Αξία"] = df["Αξία"].astype(float)
    return df


def affected_text(o, limit=2):
    """'Mini Market Ήλιος · 2/11 · 3 — Μπαρ Πέτρα · 3/11 · 2 — + 9 ακόμη'."""
    rows = sorted(o.get("affected_deliveries") or [],
                  key=lambda a: (h._to_date(a.get("date")) or pd.Timestamp.max))
    parts = [f"{a.get('customer_name') or a.get('trdr')} · {h.dm(a.get('date'))} · "
             f"{h.fmt_qty(a.get('qty'))}" for a in rows[:limit]]
    if len(rows) > limit:
        parts.append(f"+ {len(rows) - limit} ακόμη")
    return " — ".join(parts)


def home_sentence(o):
    """HTML sentence of the home list 'Χρειάζονται εσένα σήμερα' (warehouse)."""
    lines = o.get("lines") or []
    supplier = html.escape(str(o.get("supplier_name") or o.get("supplier_id")))
    when = h.day_phrase(o.get("order_by"))
    arrival = h.dm(o.get("arrival"))
    state = status(o)
    qty = (f"{_qty_unit(lines[0])} " if len(lines) == 1 else "")
    if state == SHORT:
        names = ", ".join(f"<b>{html.escape(str(l.get('item_name')))}</b>" for l in short_lines(o))
        n = len(o.get("affected_deliveries") or [])
        return (f"Δεν φτάνει το απόθεμα για: {names}. Παράγγειλε {qty}από τον {supplier} "
                f"{when}, για να φτάσουν {arrival}· {affected_phrase(n).lower()}.")
    if state == LATE:
        names = ", ".join(f"<b>{html.escape(str(l.get('item_name')))}</b>" for l in late_lines(o))
        return (f"Αργήσαμε για: {names}. Οι πελάτες καλύπτονται, αλλά πέφτουμε κάτω από το "
                f"απόθεμα ασφαλείας. Παράγγειλε {qty}από τον {supplier} {when}, για να φτάσουν "
                f"{arrival}.")
    what = (f"{_qty_unit(lines[0])} {html.escape(str(lines[0].get('item_name')))}"
            if len(lines) == 1 else h.items_count(len(lines)))
    return f"<b>{supplier}:</b> {what}, παραγγελία έως {when.split(' ', 1)[1]}, φτάνει {arrival}."


def later_count(all_orders, week_orders):
    """Orders planned after this week."""
    week = {order_key(o) for o in week_orders}
    return sum(1 for o in all_orders if order_key(o) not in week)


def _plain(v):
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if hasattr(v, "isoformat") and not isinstance(v, str):
        return v.isoformat()
    return v


def order_context(o):
    """Context for the chat of one purchase order (JSON-ready)."""
    return {"today": h.SCENARIO_TODAY.isoformat(), "purchase_order": _plain(o)}


def week_context(orders, limit=15):
    """The week's orders for the general chat, or the 'coming soon' text."""
    if orders is None:
        return SOON
    return [_plain(o) for o in sort_orders(orders)[:limit]]
