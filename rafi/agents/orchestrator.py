"""Orchestrator of the council: runs the agents in order and joins their results.

  1. forecast        submissions/forecast_details.csv, computed if missing (customer agent)
  2. customer agent  suggestions for customers, AI texts for the important ones, store
  3. warehouse agent purchase suggestions from suppliers (store)
  4. supplier notes  message to each supplier for the orders of the next 7 days (AI or template)
  5. coordination    an item that will not arrive in time (ΔΕΝ_ΦΤΑΝΕΙ): the stock we have goes
                     to customer deliveries in priority order (urgent and loss risks first,
                     then by value). Suggestions left without stock get ΕΛΛΕΙΨΗ_ΑΠΟΘΗΚΗΣ.
  6. summary         morning_summary(): the same numbers for the terminal, Telegram, dashboard

Every decision is plain code; the AI only writes texts. Nothing goes to the ERP or to a
supplier here: approvals stay with people.

CLI: python -m rafi.agents.orchestrator [--no-ai] [--today 2026-11-01] [--db rafi.db]
"""
import argparse
import logging
import math
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from rafi import store
from rafi.agents import customer_agent
from rafi.agents import warehouse_agent as wh
from rafi.suggestions import (EXTRA_ROUTE, LOSS, MARGINAL, OFFER, ORDER, URGENT, WEEKDAYS,
                              pending_totals)

ROOT = Path(__file__).resolve().parents[2]
TODAY = customer_agent.TODAY
TOP = 10
WAREHOUSE_DAYS = 7
USER = "orchestrator"


# --- warehouse --------------------------------------------------------------

def run_warehouse(data, details, today=TODAY, db=None):
    """The warehouse agent's run, with the data and forecast already in memory."""
    from rafi.settings import settings_path

    suggestions = store.list_suggestions(db=db)
    orders, stock = wh.plan(data["deliveries"], data["items"], data["customers"], details,
                            suggestions, data["promos"], wh.load_suppliers(data["items"]),
                            today, wh.load_settings(settings_path()), wh.STOCK_PATH,
                            wh.incoming_from_store(db, today))
    result = store.save_purchase_new(wh.to_records(orders), user="warehouse_agent", db=db)
    return orders, result, stock


# --- coordination of shortages ----------------------------------------------

def _fmt_qty(x):
    return f"{x:,.0f}".replace(",", ".")


def _dm(d):
    d = pd.Timestamp(d)
    return f"{d.day}/{d.month}"


def coordinate_shortages(db=None, today=TODAY, user=USER):
    """Share the stock of every ΔΕΝ_ΦΤΑΝΕΙ item among the customer deliveries before it arrives.

    Stock = the warehouse agent's stock today + approved purchases arriving before then.
    Order: suggestions that need you (urgent, extra route, loss risk), then other suggestions
    by value, then forecast deliveries without a suggestion; same rank -> earlier date first.
    A suggestion (ΠΡΟΤΑΣΗ) not fully covered gets ΕΛΛΕΙΨΗ_ΑΠΟΘΗΚΗΣ with a note: which item,
    how much is missing and when the supplier's goods arrive.
    Returns one dict per delivery not fully covered.
    """
    today = pd.Timestamp(today)
    by_id = {s["id"]: s for s in store.list_suggestions(db=db)}
    incoming = wh.incoming_from_store(db, today)
    short_rows = [r for r in store.list_purchase_suggestions(status=store.PROPOSED, db=db)
                  if r["flag"] == wh.TOO_LATE]
    out = []
    for r in short_rows:
        p = r["payload"]
        arrival = pd.Timestamp(r["arrival"])
        stock = float(p.get("STOCK_NOW") or 0.0) + sum(
            q for d, q in incoming.get(int(r["mtrl"]), {}).items() if d < arrival)

        def rank(a):
            s = by_id.get(a.get("SUGGESTION_ID"))
            if s is None:
                return (2, 0.0, a["DELIVERY_DATE"], a["TRDR"])
            return (0 if customer_agent.needs_you(s) else 1, -(s["value_eur"] or 0.0),
                    a["DELIVERY_DATE"], a["TRDR"])

        for a in sorted(p.get("AFFECTED") or [], key=rank):
            given = min(stock, float(a["QTY"]))
            stock -= given
            missing = float(a["QTY"]) - given
            if missing < 0.5:  # shown as a whole number: nothing to say
                continue
            sid = a.get("SUGGESTION_ID")
            s = by_id.get(sid)
            entry = {"suggestion_id": sid, "trdr": int(a["TRDR"]), "mtrl": int(r["mtrl"]),
                     "item_name": p.get("ITEM_NAME"), "date": a["DELIVERY_DATE"],
                     "qty": float(a["QTY"]), "missing": round(missing, 2),
                     "arrival": r["arrival"], "flagged": False}
            if s is not None and s["status"] == store.PROPOSED:
                note = (f"Από το «{p.get('ITEM_NAME')}» λείπουν {_fmt_qty(math.ceil(missing))} "
                        f"{p.get('UNIT', '')} για την παράδοση της {_dm(a['DELIVERY_DATE'])}: "
                        f"η αποθήκη δεν έχει αρκετό απόθεμα και η παραγγελία του προμηθευτή "
                        f"φτάνει {_dm(r['arrival'])}.")
                store.add_flag(sid, store.WAREHOUSE_SHORT, note, user, db=db)
                entry["flagged"] = True
            out.append(entry)
    return out


# --- summary ----------------------------------------------------------------

def one_line(reason, limit=140):
    """First sentence of a REASON (a dot before a capital letter ends it, so "κιβ. 6τεμ" does not)."""
    first = re.split(r"\.\s+(?=[«A-ZΑ-ΩΆΈΉΊΌΎΏ])", str(reason or "").strip())[0].rstrip(".")
    return first if len(first) <= limit else first[:limit - 1].rstrip() + "…"


def _kind(s):
    flags = s["payload"].get("FLAGS", [])
    if s["action_type"] == LOSS:
        return LOSS
    return URGENT if URGENT in flags else EXTRA_ROUTE


def morning_summary(db=None) -> dict:
    """What is waiting for people this morning, for the terminal, Telegram and the dashboard.

    customers: counts (pending, extra_routes, loss_risks, marginal, orders, offers),
               value_at_risk_eur, offers_value_eur (rafi.suggestions.pending_totals),
               needs_you [suggestion_id, trdr, customer_name, kind, delivery_date, value_eur,
               one_line] by priority
    warehouse: orders_count (supplier orders of the next 7 days), short_count, orders
               (open_purchase_orders(7))
    """
    pending = store.list_suggestions(status=store.PROPOSED, db=db)
    names = wh.customer_names()
    totals = pending_totals(pending)

    def has(s, flag):
        return flag in s["payload"].get("FLAGS", [])

    counts = {
        "pending": len(pending),
        "extra_routes": sum(1 for s in pending if s["action_type"] == ORDER
                            and (has(s, URGENT) or has(s, EXTRA_ROUTE))),
        "loss_risks": sum(1 for s in pending if s["action_type"] == LOSS),
        "marginal": sum(1 for s in pending if has(s, MARGINAL)),
        "orders": sum(1 for s in pending if s["action_type"] == ORDER),
        "offers": sum(1 for s in pending if s["action_type"] == OFFER),
    }
    needs = [{"suggestion_id": s["id"], "trdr": s["trdr"],
              "customer_name": names.get(int(s["trdr"]), ""), "kind": _kind(s),
              "delivery_date": s["delivery_date"], "value_eur": s["value_eur"],
              "one_line": one_line(s["payload"].get("REASON"))}
             for s in pending if customer_agent.needs_you(s)]
    orders = wh.open_purchase_orders(WAREHOUSE_DAYS, db)
    return {
        "customers": {"counts": counts, "value_at_risk_eur": round(totals["at_risk_eur"], 2),
                      "offers_value_eur": round(totals["offers_eur"], 2), "needs_you": needs},
        "warehouse": {"orders_count": len(orders),
                      "short_count": sum(1 for o in orders if o["short"]),
                      "orders": orders},
    }


# --- whole run --------------------------------------------------------------

def run(today=TODAY, db=None, ai=True, data_dir=customer_agent.DATA_DIR,
        details_path=customer_agent.DETAILS_PATH, method=customer_agent.DEFAULT_METHOD,
        top_n=None, route_cost=None, text_writer=None):
    """All steps in order. Returns a dict with the result of every agent and the summary."""
    today = pd.Timestamp(today)
    data = customer_agent.load_data(data_dir)
    customers = customer_agent.run(today, db, data, data_dir, details_path, method, ai, top_n,
                                   route_cost, text_writer)
    orders, purchases, stock = run_warehouse(data, customers["details"], today, db)
    notes = wh.write_group_notes(WAREHOUSE_DAYS, db, ai, today)
    shortages = coordinate_shortages(db, today)
    return {"customer": customers, "warehouse": {"orders": orders, "result": purchases,
                                                 "stock": stock, "notes": notes},
            "shortages": shortages, "summary": morning_summary(db), "today": today}


# --- texts for the terminal -------------------------------------------------

def customer_briefing(result, customers, salesmen, today=TODAY, db=None):
    """Morning briefing for the customer side, from what is waiting for approval in the store."""
    today = pd.Timestamp(today)
    pending = store.list_suggestions(status=store.PROPOSED, db=db)
    names = customers.set_index("TRDR")["NAME"]
    people = salesmen.set_index("PRSN")["NAME"]
    kinds = Counter(s["action_type"] for s in pending)
    flags = Counter(f for s in pending for f in s["payload"].get("FLAGS", []))
    totals = pending_totals(pending)
    at_risk, offers = totals["at_risk_eur"], totals["offers_eur"]

    out = [f"Ράφι · πρωινή ενημέρωση · {WEEKDAYS[today.weekday()]} {today:%d/%m/%Y}",
           f"Νέες προτάσεις σήμερα: {len(result['saved'])} "
           f"(παραλείφθηκαν {result['skipped']} που υπάρχουν ήδη)",
           f"Προς έγκριση: {len(pending)}"]
    for kind in (ORDER, LOSS, OFFER):
        out.append(f"  {kind:<19} {kinds.get(kind, 0):>4}")
    for flag in (URGENT, EXTRA_ROUTE, MARGINAL, store.WAREHOUSE_SHORT):
        out.append(f"  {flag:<19} {flags.get(flag, 0):>4}")
    out.append(f"Αξία σε κίνδυνο (παραγγελίες + κίνδυνοι απώλειας): {at_risk:>12,.2f} €")
    out.append(f"Αξία προτεινόμενων προσφορών (εποχικά):          {offers:>12,.2f} €")
    sources = Counter(s["payload"].get("TEXT_SOURCE", "ΠΡΟΤΥΠΟ") for s in pending)
    out.append(f"Κείμενα: AI {sources.get('AI', 0)} · πρότυπο {sources.get('ΠΡΟΤΥΠΟ', 0)}")
    out.append("")
    out.append(f"Με ποιον ξεκινάς σήμερα (οι {TOP} πρώτες κατά προτεραιότητα):")
    out.append(f"{'#':>3} {'id':>4}  {'πελάτης':<38} {'πωλητής':<20} {'παράδοση':<10} "
               f"{'είδη':>4} {'€':>10}  τύπος")
    for i, s in enumerate(pending[:TOP], start=1):
        p = s["payload"]
        who = f"{s['trdr']} {names.get(s['trdr'], '')}"[:38]
        kind = s["action_type"] + "".join(f" · {f}" for f in p.get("FLAGS", []))
        out.append(f"{i:>3} {s['id']:>4}  {who:<38} {people.get(s['salesman'], s['salesman']):<20} "
                   f"{s['delivery_date']:<10} {len(p['LINES']):>4} {s['value_eur']:>10,.2f}  {kind}")
    return "\n".join(out)


def warehouse_briefing(summary, shortages=None):
    """The warehouse part of the morning briefing (orders to suppliers, next 7 days)."""
    w = summary["warehouse"]
    value = sum(o["total_value_eur"] for o in w["orders"])
    out = [f"Αποθήκη · παραγγελίες σε προμηθευτές για τις επόμενες {WAREHOUSE_DAYS} ημέρες: "
           f"{w['orders_count']} (δεν φτάνουν εγκαίρως: {w['short_count']}) · "
           f"αξία {value:,.2f} € σε τιμή καταλόγου"]
    if shortages is not None:
        flagged = sum(1 for s in shortages if s["flagged"])
        out.append(f"Συντονισμός ελλείψεων: {len(shortages)} παραδόσεις σε πελάτες δεν "
                   f"καλύπτονται πλήρως · {flagged} προτάσεις πήραν {store.WAREHOUSE_SHORT}")
    if w["orders"]:
        out.append(f"{'έως':<10} {'άφιξη':<10} {'προμηθευτής':<34} {'είδη':>4} {'€':>10}  "
                   f"σημείωμα")
    for o in w["orders"]:
        mark = "ΔΕΝ_ΦΤΑΝΕΙ " if o["short"] else ""
        out.append(f"{o['order_by']:<10} {o['arrival']:<10} {str(o['supplier_name'])[:34]:<34} "
                   f"{len(o['lines']):>4} {o['total_value_eur']:>10,.2f}  {mark}"
                   f"[{o['note_source']}]")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m rafi.agents.orchestrator",
                                 description="Συντονιστής του Ράφι: πελάτες, αποθήκη, ελλείψεις.")
    ap.add_argument("--today", default=str(TODAY.date()))
    ap.add_argument("--db", default=None, help="αρχείο βάσης (προεπιλογή rafi.db)")
    ap.add_argument("--method", choices=["hybrid", "rafi"], default=customer_agent.DEFAULT_METHOD)
    ap.add_argument("--no-ai", action="store_true", help="μόνο κείμενα από πρότυπα")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(message)s")

    out = run(args.today, db=args.db, ai=not args.no_ai, method=args.method)
    c = out["customer"]
    print(f"Πρόβλεψη: {c['details_source']} · κείμενα AI: {c['ai_status']} · "
          f"επιλέχθηκαν {c['ai_chosen']}, γράφτηκαν {c['ai_written']}")
    print()
    print(customer_briefing(c["result"], c["data"]["customers"], c["data"]["salesmen"],
                            out["today"], db=args.db))
    print()
    print(warehouse_briefing(out["summary"], out["shortages"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
