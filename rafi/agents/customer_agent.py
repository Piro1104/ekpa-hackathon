"""Customer sub-agent: forecast -> suggestions -> AI texts for the important ones -> store.

A thin wrapper around what scripts/daily_run.py used to do on its own:
  1. forecast      submissions/forecast_details.csv when it is for today's cutoff, else computed
                   (hybrid by default) and written there, so the warehouse agent reads the same
  2. suggestions   rafi.suggestions.build_suggestions, all texts from templates
  3. AI texts      rafi.llm.text_writer, only for the suggestions that "need you" (loss risks,
                   urgent orders, extra routes) and for the next `top_n` by priority. The rest
                   keep the template, so the run stays fast. If the model does not answer,
                   everything keeps the template.
  4. store         rafi.store.save_new (status ΠΡΟΤΑΣΗ, no duplicates)

The AI only rewrites the texts: numbers, dates, quantities and priorities never change.
"""
from pathlib import Path

import pandas as pd

from rafi import store
from rafi.ingest import build_deliveries, daily_qty, load_from_files
from rafi.suggestions import (EXTRA_ROUTE, LOSS, ORDER, URGENT, build_suggestions, load_promos,
                              write_texts)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DETAILS_PATH = ROOT / "submissions" / "forecast_details.csv"
TODAY = pd.Timestamp("2026-11-01")
DETAILS_CUTOFF = pd.Timestamp("2026-10-31")  # the cutoff of scripts/make_submission.py
DEFAULT_METHOD = "hybrid"


def needs_you(s):
    """Loss risks, and orders marked urgent or needing an extra route.

    Works on build_suggestions dicts and on store rows (same rule as the dashboard).
    """
    kind = s.get("ACTION_TYPE") or s.get("action_type")
    payload = s.get("payload") or {}
    flags = s.get("FLAGS") or payload.get("FLAGS") or []
    return kind == LOSS or (kind == ORDER and (URGENT in flags or EXTRA_ROUTE in flags))


def load_data(data_dir=DATA_DIR):
    """Clean deliveries and the master data, read once for both agents."""
    headers, lines, items, customers = load_from_files(data_dir)
    deliveries = build_deliveries(headers, lines, items)
    return {"deliveries": deliveries, "items": items, "customers": customers,
            "promos": load_promos(data_dir),
            "targets": pd.read_csv(Path(data_dir) / "prediction_targets.csv",
                                   encoding="utf-8-sig"),
            "salesmen": pd.read_csv(Path(data_dir) / "PRSN.csv", encoding="utf-8-sig")}


def compute_details(data, cutoff, method=DEFAULT_METHOD):
    from rafi.forecast import forecast
    from rafi.lgbm_engine import forecast_hybrid
    methods = {"hybrid": forecast_hybrid, "rafi": forecast}
    past = data["deliveries"][data["deliveries"]["TRNDATE"] <= cutoff]
    return methods[method](daily_qty(past), past, data["customers"], data["items"],
                           data["targets"], cutoff)


def forecast_details(data, today=TODAY, details_path=DETAILS_PATH, method=DEFAULT_METHOD):
    """(details, source). The saved file is used only for its own cutoff (2026-10-31) and the
    default method; otherwise the forecast is computed (and saved when the file is missing)."""
    today = pd.Timestamp(today)
    cutoff = today - pd.Timedelta(days=1)
    path = Path(details_path)
    if cutoff == DETAILS_CUTOFF and method == DEFAULT_METHOD and path.exists():
        return pd.read_csv(path, encoding="utf-8-sig"), "αρχείο"
    details = compute_details(data, cutoff, method)
    if cutoff == DETAILS_CUTOFF and method == DEFAULT_METHOD:
        path.parent.mkdir(parents=True, exist_ok=True)
        details.to_csv(path, index=False, date_format="%Y-%m-%d", encoding="utf-8")
    return details, "υπολογισμός"


def pick_for_ai(suggestions, top_n, skip_pairs=frozenset()):
    """Suggestions that get AI texts: every "needs you" one, plus the first `top_n` of the rest
    by PRIORITY. Ones that store.save_new would skip (pair already active) are left out."""
    def new(s):
        trdr = int(s["payload"]["TRDR"])
        return not any((trdr, int(l["MTRL"])) in skip_pairs for l in s["payload"]["LINES"])

    fresh = sorted((s for s in suggestions if new(s)), key=lambda s: s["PRIORITY"])
    urgent = [s for s in fresh if needs_you(s)]
    rest = [s for s in fresh if not needs_you(s)][:max(int(top_n), 0)]
    return urgent + rest


def rewrite_texts(chosen, text_writer):
    """Replace REASON / SALES_NOTE / TEXT_SOURCE of the chosen suggestions in place.

    Returns how many got AI texts. A failing text keeps its template (write_texts does that).
    """
    n = 0
    for s in chosen:
        texts = write_texts(s["FACTS"], text_writer)
        s["payload"].update(REASON=texts["REASON"], SALES_NOTE=texts["SALES_NOTE"],
                            TEXT_SOURCE=texts["TEXT_SOURCE"])
        s["TEXT_SOURCE"] = texts["TEXT_SOURCE"]
        n += texts["TEXT_SOURCE"] == "AI"
    return n


def run(today=TODAY, db=None, data=None, data_dir=DATA_DIR, details_path=DETAILS_PATH,
        method=DEFAULT_METHOD, ai=True, top_n=None, route_cost=None, text_writer=None):
    """One run of the customer agent. Returns a dict:

    result (save_new: saved ids, skipped), suggestions (all built today, by priority),
    details, details_source, ai_chosen, ai_written, ai_status, data.
    `ai=False` keeps every template. `text_writer` replaces rafi.llm.text_writer (tests).
    """
    from rafi.settings import ai_texts_top_n

    today = pd.Timestamp(today)
    cutoff = today - pd.Timedelta(days=1)
    data = data or load_data(data_dir)
    details, source = forecast_details(data, today, details_path, method)
    past = data["deliveries"][data["deliveries"]["TRNDATE"] <= cutoff]
    suggestions = build_suggestions(details, data["customers"], data["items"], past,
                                    data["promos"], today, route_cost=route_cost)

    chosen, written, status = [], 0, "ανενεργό (--no-ai)"
    if ai:
        if text_writer is None:
            from rafi import llm
            ok = llm.available()
            text_writer = llm.text_writer if ok else None
            status = f"LLM_MODE={llm.settings()['LLM_MODE']}" + ("" if ok else " · δεν απαντά")
        else:
            status = "δοκιμαστικό"
        if text_writer is not None:
            top_n = ai_texts_top_n() if top_n is None else top_n
            chosen = pick_for_ai(suggestions, top_n, store.active_pairs(db))
            written = rewrite_texts(chosen, text_writer)

    result = store.save_new(suggestions, user="system", db=db)
    return {"result": result, "suggestions": suggestions, "details": details,
            "details_source": source, "ai_chosen": len(chosen), "ai_written": written,
            "ai_status": status, "data": data}
