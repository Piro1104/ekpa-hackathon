"""Daily run: forecast -> suggestions -> store (status ΠΡΟΤΑΣΗ) -> morning briefing.

Nothing is sent to the ERP here; that happens only after approval (rafi.store approve).
Running it again the same day creates no duplicates.

Run from the repository root: python scripts/daily_run.py
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rafi import store  # noqa: E402
from rafi.forecast import forecast  # noqa: E402
from rafi.ingest import build_deliveries, daily_qty, load_from_files  # noqa: E402
from rafi.suggestions import (EXTRA_ROUTE, LOSS, OFFER, ORDER, URGENT, WEEKDAYS,  # noqa: E402
                              build_suggestions, load_promos)

TODAY = pd.Timestamp("2026-11-01")
TOP = 10


def run(today=TODAY, db=None, data_dir=ROOT / "data"):
    """Compute and save today's suggestions. Returns (save result, customers, salesmen)."""
    today = pd.Timestamp(today)
    cutoff = today - pd.Timedelta(days=1)
    headers, lines, items, customers = load_from_files(data_dir)
    deliveries = build_deliveries(headers, lines, items)
    past = deliveries[deliveries["TRNDATE"] <= cutoff]
    targets = pd.read_csv(Path(data_dir) / "prediction_targets.csv", encoding="utf-8-sig")
    details = forecast(daily_qty(past), past, customers, items, targets, cutoff)
    suggestions = build_suggestions(details, customers, items, past, load_promos(data_dir), today)
    result = store.save_new(suggestions, user="system", db=db)
    salesmen = pd.read_csv(Path(data_dir) / "PRSN.csv", encoding="utf-8-sig")
    return result, customers, salesmen


def briefing(result, customers, salesmen, today=TODAY, db=None):
    """Morning briefing text, from what is waiting for approval in the store."""
    today = pd.Timestamp(today)
    pending = store.list_suggestions(status=store.PROPOSED, db=db)
    names = customers.set_index("TRDR")["NAME"]
    people = salesmen.set_index("PRSN")["NAME"]
    kinds = Counter(s["action_type"] for s in pending)
    flags = Counter(f for s in pending for f in s["payload"].get("FLAGS", []))
    at_risk = sum(s["value_eur"] for s in pending if s["action_type"] in (ORDER, LOSS))
    offers = sum(s["value_eur"] for s in pending if s["action_type"] == OFFER)

    out = [f"Ράφι · πρωινή ενημέρωση · {WEEKDAYS[today.weekday()]} {today:%d/%m/%Y}",
           f"Νέες προτάσεις σήμερα: {len(result['saved'])} "
           f"(παραλείφθηκαν {result['skipped']} που υπάρχουν ήδη)",
           f"Προς έγκριση: {len(pending)}"]
    for kind in (ORDER, LOSS, OFFER):
        out.append(f"  {kind:<19} {kinds.get(kind, 0):>4}")
    out.append(f"  {URGENT:<19} {flags.get(URGENT, 0):>4}")
    out.append(f"  {EXTRA_ROUTE:<19} {flags.get(EXTRA_ROUTE, 0):>4}")
    out.append(f"Αξία σε κίνδυνο (παραγγελίες + κίνδυνοι απώλειας): {at_risk:>12,.2f} €")
    out.append(f"Αξία προτεινόμενων προσφορών (εποχικά):          {offers:>12,.2f} €")
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


def main(argv=None):
    ap = argparse.ArgumentParser(description="Καθημερινός υπολογισμός προτάσεων του Ράφι.")
    ap.add_argument("--today", default=str(TODAY.date()))
    ap.add_argument("--db", default=None)
    args = ap.parse_args(argv)
    result, customers, salesmen = run(args.today, db=args.db)
    print(briefing(result, customers, salesmen, args.today, db=args.db))
    return result


if __name__ == "__main__":
    main()
