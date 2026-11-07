"""Daily run: everything goes through the orchestrator (rafi.agents.orchestrator).

forecast (if missing) -> customer agent -> warehouse agent -> shortages -> morning briefing.
Nothing is sent to the ERP or to a supplier here; that happens only after approval.
Running it again the same day creates no duplicates.

Run from the repository root: python scripts/daily_run.py [--no-ai]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rafi.agents import orchestrator  # noqa: E402
from rafi.agents.orchestrator import customer_briefing as briefing  # noqa: E402,F401

TODAY = orchestrator.TODAY
DEFAULT_METHOD = "hybrid"


def run(today=TODAY, db=None, data_dir=ROOT / "data", method=DEFAULT_METHOD, ai=True,
        full=False):
    """Run the orchestrator. Returns (save result, customers, salesmen), or everything with full."""
    out = orchestrator.run(pd.Timestamp(today), db=db, ai=ai, data_dir=data_dir, method=method)
    if full:
        return out
    data = out["customer"]["data"]
    return out["customer"]["result"], data["customers"], data["salesmen"]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Καθημερινός υπολογισμός προτάσεων του Ράφι.")
    ap.add_argument("--today", default=str(TODAY.date()))
    ap.add_argument("--db", default=None)
    ap.add_argument("--method", choices=["hybrid", "rafi"], default=DEFAULT_METHOD)
    ap.add_argument("--no-ai", action="store_true", help="μόνο κείμενα από πρότυπα")
    args = ap.parse_args(argv)
    out = run(args.today, db=args.db, method=args.method, ai=not args.no_ai, full=True)
    data = out["customer"]["data"]
    print(briefing(out["customer"]["result"], data["customers"], data["salesmen"], args.today,
                   db=args.db))
    print()
    print(orchestrator.warehouse_briefing(out["summary"], out["shortages"]))
    return out["customer"]["result"]


if __name__ == "__main__":
    main()
