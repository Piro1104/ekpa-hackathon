"""Warehouse sub-agent: when and how much DEMO ΔΙΑΝΟΜΕΣ should order from each supplier.

Step by step, all in plain code (no AI):
  1. suppliers     config/suppliers.csv, one invented supplier per item category
  2. stock today   config/stock.csv if it exists (ΑΡΧΕΙΟ), else last year's net sales
                   over the same weeks (ΕΚΤΙΜΗΣΗ)
  3. demand        per item and day for 90 days:
                   - target pairs: the deliveries we will make to customers (first one from the
                     store's suggestion or the forecast, then every usual interval), none while
                     the customer is closed for the winter
                   - other pairs: mean daily sales of the last 180 days
                   - seasonal items outside the targets: last year's sales on the same weekdays
                   - our own promotions in the horizon: demand x the item's past promo lift
  4. simulation    day by day: stock - demand + arrivals. When the stock at the end of the
                   supplier's lead time would drop under the safety stock, we order.
                   An order that should have gone out before today is either
                   ΑΡΓΗΣΑΜΕ   the stock covers every customer delivery until the goods arrive,
                              but we drop under the safety stock, or
                   ΕΛΛΕΙΨΗ    some customer deliveries before the arrival are not covered
                              (the orchestrator flags them ΕΛΛΕΙΨΗ_ΑΠΟΘΗΚΗΣ)
  5. store         purchase suggestions ΠΡΟΤΑΣΗ -> ΕΓΚΡΙΘΗΚΕ / ΑΠΟΡΡΙΦΘΗΚΕ (rafi.store);
                   an approved one is written as JSON to purchase_orders/

CLI: python -m rafi.agents.warehouse_agent [run] | list | approve <id> | reject <id> "λόγος"
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rafi import store
from rafi.features import is_closed_on, likely_closer, winter_history, working_day
from rafi.forecast import DEFAULT_INTERVAL_DAYS, K_DELIVERIES

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CONFIG_DIR = ROOT / "config"
SUPPLIERS_PATH = CONFIG_DIR / "suppliers.csv"
STOCK_PATH = CONFIG_DIR / "stock.csv"
SETTINGS_PATH = CONFIG_DIR / "settings.json"
DETAILS_PATH = ROOT / "submissions" / "forecast_details.csv"

TODAY = pd.Timestamp("2026-11-01")
HORIZON_DAYS = 90            # orders are planned for days today .. today + 89
DEMAND_PAD_DAYS = 40         # demand is also built past the horizon, to size the last orders
RECENT_DAYS = 180            # rate window for pairs outside the targets
REVIEW_WEEKS = 2             # an order covers the lead time + 2 weeks + safety stock
PROMO_BEFORE_DAYS = 28       # promo lift = sales/day in the promo / sales/day 4 weeks before
SEASONAL_CATEGORY = "Εποχικά"
DEFAULTS = {"stock_cover_weeks": 4, "safety_weeks": 1}

FILE, ESTIMATE = "ΑΡΧΕΙΟ", "ΕΚΤΙΜΗΣΗ"
LATE = "ΑΡΓΗΣΑΜΕ"           # ordered too late; customers covered, under the safety stock
SHORT = "ΕΛΛΕΙΨΗ"          # ordered too late; some customer deliveries are not covered
OLD_TOO_LATE = "ΔΕΝ_ΦΤΑΝΕΙ"  # older stores: read as ΕΛΛΕΙΨΗ
LATE_FLAGS = (LATE, SHORT, OLD_TOO_LATE)
MISSING_MIN = 0.5          # less than this is shown as 0, so it is not a shortage
UNITS = {"ΚΙΒ": "κιβ.", "ΤΕΜ": "τεμ."}


# --- settings, suppliers, stock --------------------------------------------

def load_settings(path=SETTINGS_PATH):
    """stock_cover_weeks and safety_weeks from config/settings.json, else the defaults.

    The file belongs to the dashboard; we only read it. Other keys are ignored.
    """
    out = dict(DEFAULTS)
    path = Path(path)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for key in DEFAULTS:
            if data.get(key) is not None:
                out[key] = float(data[key])
    return out


def load_suppliers(items, path=SUPPLIERS_PATH):
    """Suppliers per category, checked: every goods category (MTRTYPE = 0) has exactly one."""
    sup = pd.read_csv(path, encoding="utf-8-sig", dtype={"CONTACT": "string"})
    goods = set(items.loc[items["MTRTYPE"] == 0, "MTRCATEGORY"])
    missing = goods - set(sup["MTRCATEGORY"])
    if missing:
        raise ValueError(f"Λείπει προμηθευτής για: {', '.join(sorted(missing))}")
    if sup["MTRCATEGORY"].duplicated().any():
        raise ValueError("Δύο προμηθευτές για την ίδια κατηγορία")
    return sup


def goods(items):
    return items[items["MTRTYPE"] == 0]


def estimate_stock(deliveries, items, today=TODAY, weeks=DEFAULTS["stock_cover_weeks"]):
    """Our stock today ~ net sales of each item on the same days last year, for `weeks` weeks.

    Net = invoices minus credit notes (credits already carry a negative QTY in `deliveries`;
    cancelled invoices and shipping are already out). Never below 0.
    """
    start = pd.Timestamp(today) - pd.DateOffset(years=1)
    end = start + pd.Timedelta(days=int(round(7 * weeks)))
    d = deliveries[(deliveries["TRNDATE"] >= start) & (deliveries["TRNDATE"] < end)]
    sold = d.groupby("MTRL")["QTY"].sum()
    g = goods(items)
    qty = sold.reindex(g["MTRL"]).fillna(0).clip(lower=0)
    return pd.DataFrame({"MTRL": g["MTRL"].to_numpy(), "QTY": qty.to_numpy(dtype=float),
                         "STOCK_SOURCE": ESTIMATE})


def current_stock(deliveries, items, today=TODAY, weeks=DEFAULTS["stock_cover_weeks"],
                  stock_path=STOCK_PATH):
    """Stock today per item: from config/stock.csv (MTRL, QTY) when it exists, else estimated.

    An item missing from the file keeps the estimate (and says so in STOCK_SOURCE).
    """
    est = estimate_stock(deliveries, items, today, weeks)
    path = Path(stock_path)
    if not path.exists():
        return est
    given = pd.read_csv(path, encoding="utf-8-sig").groupby("MTRL")["QTY"].sum()
    has = est["MTRL"].isin(given.index)
    est.loc[has, "QTY"] = est.loc[has, "MTRL"].map(given).astype(float)
    est.loc[has, "STOCK_SOURCE"] = FILE
    return est


# --- working days -----------------------------------------------------------

def add_working_days(date, n):
    """`n` working days after `date` (no Sundays, no holidays)."""
    d = pd.Timestamp(date).normalize()
    while n > 0:
        d += pd.Timedelta(days=1)
        if working_day(d):
            n -= 1
    return d


def working_on_or_before(date):
    d = pd.Timestamp(date).normalize()
    while not working_day(d):
        d -= pd.Timedelta(days=1)
    return d


def working_on_or_after(date):
    d = pd.Timestamp(date).normalize()
    while not working_day(d):
        d += pd.Timedelta(days=1)
    return d


def latest_order_day(needed_by, lead_days):
    """Last working day from which an order still arrives on or before `needed_by`."""
    d = working_on_or_before(needed_by)
    for _ in range(int(lead_days)):
        d -= pd.Timedelta(days=1)
        while not working_day(d):
            d -= pd.Timedelta(days=1)
    return d


def delivery_day(date, today):
    """Customer deliveries happen on working days: move back to one, but never before today."""
    d = working_on_or_before(date)
    return d if d >= today else working_on_or_after(today)


# --- demand -----------------------------------------------------------------

def closure_windows(deliveries, customers, cutoff):
    """{TRDR: (last delivery before closing 'MM-DD', first after re-opening 'MM-DD')}
    for winter closers and for new customers whose similar customers close (likely closers)."""
    hist = winter_history(deliveries, customers, cutoff)
    out = {r.TRDR: (r.LAST_BEFORE_CLOSE, r.FIRST_AFTER_OPEN)
           for r in hist[hist["IS_CLOSER"]].itertuples(index=False)}
    seen = set(hist["TRDR"])
    for trdr in customers["TRDR"]:
        if trdr not in seen:
            likely, last, first = likely_closer(trdr, customers, hist)
            if likely:
                out[trdr] = (last, first)
    return out


def is_closed(trdr, date, windows):
    w = windows.get(trdr)
    return bool(w) and is_closed_on(date, w[0], w[1])


def usual_intervals(daily, k=K_DELIVERIES):
    """Median days between the last k deliveries of every pair (as in the forecast)."""
    out = {}
    pos = daily[daily["QTY"] > 0].sort_values("TRNDATE")
    for key, g in pos.groupby(["TRDR", "MTRL"]):
        dates = g["TRNDATE"].to_numpy()[-k:]
        if len(dates) >= 2:
            gaps = np.diff(dates).astype("timedelta64[D]").astype(int)
            out[key] = max(1, int(round(float(np.median(gaps)))))
    return out


def first_deliveries(details, suggestions, today=TODAY):
    """The first delivery of every target pair: (date, qty, suggestion id or None).

    A pair in an active store suggestion takes that suggestion's date and quantity (the earliest
    one, if there were several); only the other pairs fall back to the forecast
    (SUGGESTED_QTY on PREDICTED_DEPLETION_DATE). So a delivery is never counted twice.
    """
    targets = set(zip(details["TRDR"], details["MTRL"]))
    from_store = {}
    for s in suggestions:
        if s.get("status") not in store.ACTIVE:
            continue
        p = s["payload"]
        when = pd.Timestamp(p["SUGGESTED_DELIVERY_DATE"])
        for line in p["LINES"]:
            key = (int(p["TRDR"]), int(line["MTRL"]))
            if key in targets and (key not in from_store or when < from_store[key][0]):
                from_store[key] = (when, float(line["QTY1"]), s.get("id"))

    out = {}
    for r in details.itertuples(index=False):
        key = (int(r.TRDR), int(r.MTRL))
        if key in from_store:
            out[key] = from_store[key]
        elif pd.notna(r.PREDICTED_DEPLETION_DATE) and pd.notna(r.SUGGESTED_QTY):
            out[key] = (pd.Timestamp(r.PREDICTED_DEPLETION_DATE), float(r.SUGGESTED_QTY), None)
    return out


def target_deliveries(details, suggestions, intervals, windows, today=TODAY, last_day=None):
    """Planned deliveries to customers for the target pairs, one row each.

    First delivery from `first_deliveries`, then every usual interval of the pair with the
    forecast's SUGGESTED_QTY. Dates move to a working day; none while the customer is closed.
    Columns: TRDR, MTRL, DATE, QTY, SUGGESTION_ID, SOURCE.
    """
    today = pd.Timestamp(today)
    last_day = pd.Timestamp(last_day) if last_day is not None else \
        today + pd.Timedelta(days=HORIZON_DAYS - 1)
    usual_qty = {(int(t), int(m)): q for t, m, q in
                 zip(details["TRDR"], details["MTRL"], details["SUGGESTED_QTY"]) if pd.notna(q)}
    rows = []
    for (trdr, mtrl), (first, qty, sid) in first_deliveries(details, suggestions, today).items():
        step = pd.Timedelta(days=intervals.get((trdr, mtrl), DEFAULT_INTERVAL_DAYS))
        repeat_qty = usual_qty.get((trdr, mtrl), qty)
        when, n = max(pd.Timestamp(first), today), 0
        while when <= last_day:
            day = delivery_day(when, today)
            if not is_closed(trdr, day, windows):
                rows.append({"TRDR": trdr, "MTRL": mtrl, "DATE": day,
                             "QTY": qty if n == 0 else repeat_qty,
                             "SUGGESTION_ID": sid if n == 0 else None,
                             "SOURCE": ("ΠΡΟΤΑΣΗ_STORE" if sid is not None else "ΠΡΟΒΛΕΨΗ")
                             if n == 0 else "ΕΠΑΝΑΛΗΨΗ"})
            when, n = when + step, n + 1
    cols = ["TRDR", "MTRL", "DATE", "QTY", "SUGGESTION_ID", "SOURCE"]
    return pd.DataFrame(rows, columns=cols)


def background_demand(deliveries, items, targets, windows, today=TODAY, days=HORIZON_DAYS):
    """Daily demand of the pairs outside the targets, per (DATE, MTRL).

    - mean daily net sales of the last 180 days, per pair
    - seasonal items: net sales of the same weekday 52 weeks earlier (their windows are short,
      a 180-day mean would miss Christmas)
    Nothing for a customer while closed for the winter.
    """
    today = pd.Timestamp(today)
    cutoff = today - pd.Timedelta(days=1)
    dates = pd.date_range(today, periods=days, freq="D")
    target_keys = set(zip(targets["TRDR"], targets["MTRL"]))
    seasonal = set(items.loc[items["MTRCATEGORY"] == SEASONAL_CATEGORY, "MTRL"])
    past = deliveries[deliveries["TRNDATE"] <= cutoff]
    keep = [(t, m) not in target_keys for t, m in zip(past["TRDR"], past["MTRL"])]
    past = past[keep]

    open_days = {}  # TRDR -> 1.0 on days the customer is open, 0.0 while closed

    def is_open(trdr):
        if trdr not in open_days:
            open_days[trdr] = np.array([0.0 if is_closed(trdr, d, windows) else 1.0
                                        for d in dates]) if trdr in windows else None
        return open_days[trdr]

    per_item = {}
    recent = past[(past["TRNDATE"] > cutoff - pd.Timedelta(days=RECENT_DAYS))
                  & ~past["MTRL"].isin(seasonal)]
    rate = recent.groupby(["TRDR", "MTRL"])["QTY"].sum() / RECENT_DAYS
    for (trdr, mtrl), r in rate.items():
        if r <= 0:
            continue
        mask = is_open(trdr)
        add = r * (mask if mask is not None else np.ones(len(dates)))
        per_item[mtrl] = per_item.get(mtrl, 0.0) + add

    rows = [(d, m, q) for m, arr in per_item.items() for d, q in zip(dates, arr) if q > 0]
    year = past[past["MTRL"].isin(seasonal)]
    shifted = year.assign(DATE=year["TRNDATE"] + pd.Timedelta(days=364))
    shifted = shifted[shifted["DATE"].between(dates[0], dates[-1])]
    for r in shifted.itertuples(index=False):
        if not is_closed(r.TRDR, r.DATE, windows):
            rows.append((r.DATE, r.MTRL, float(r.QTY)))

    out = pd.DataFrame(rows, columns=["DATE", "MTRL", "QTY"])
    return out.groupby(["DATE", "MTRL"], as_index=False)["QTY"].sum()


def promo_lift(deliveries, items, promos, mtrl, today=TODAY):
    """Past demand lift of our own promotions for an item (or, failing that, its category).

    For every past promo covering the item: sales/day inside the promo / sales/day in the
    4 weeks before it, over all customers. Mean over those promos. If no promo gives a number,
    the same for the whole category (all its promos, all its items). Else 1.0.
    """
    today = pd.Timestamp(today)
    cat = items.set_index("MTRL").at[mtrl, "MTRCATEGORY"]
    cat_items = set(items.loc[items["MTRCATEGORY"] == cat, "MTRL"])
    past = promos[promos["TODATE"] < today]

    def lifts(promo_rows, sold_items):
        sales = deliveries[deliveries["MTRL"].isin(sold_items)]
        out = []
        for p in promo_rows.itertuples(index=False):
            inside = sales[sales["TRNDATE"].between(p.FROMDATE, p.TODATE)]["QTY"].sum()
            before = sales[(sales["TRNDATE"] >= p.FROMDATE - pd.Timedelta(days=PROMO_BEFORE_DAYS))
                           & (sales["TRNDATE"] < p.FROMDATE)]["QTY"].sum()
            days_in = (p.TODATE - p.FROMDATE).days + 1
            if before > 0:
                out.append((inside / days_in) / (before / PROMO_BEFORE_DAYS))
        return out

    own = past[(past["MTRL"] == mtrl) | (past["MTRCATEGORY"] == cat)]
    values = lifts(own, {mtrl})
    if not values:
        in_cat = past[(past["MTRCATEGORY"] == cat) | past["MTRL"].isin(cat_items)]
        values = lifts(in_cat, cat_items)
    return round(float(np.mean(values)), 3) if values else 1.0


def apply_promos(demand, deliveries, items, promos, today=TODAY):
    """Multiply the demand on days inside a promotion of the horizon by the item's lift.

    `demand` has DATE, MTRL, QTY (one row per demand, never a total). Applied once per row;
    returns a copy with PROMO_FACTOR.
    """
    today = pd.Timestamp(today)
    out = demand.copy()
    out["PROMO_FACTOR"] = 1.0
    cats = items.set_index("MTRL")["MTRCATEGORY"]
    future = promos[promos["TODATE"] >= today]
    cache = {}
    for p in future.itertuples(index=False):
        covered = (out["MTRL"] == p.MTRL) if pd.notna(p.MTRL) else \
            (out["MTRL"].map(cats) == p.MTRCATEGORY)
        hit = covered & out["DATE"].between(p.FROMDATE, p.TODATE) & (out["PROMO_FACTOR"] == 1.0)
        for mtrl in out.loc[hit, "MTRL"].unique():
            if mtrl not in cache:
                cache[mtrl] = promo_lift(deliveries, items, promos, mtrl, today)
            rows = hit & (out["MTRL"] == mtrl)
            out.loc[rows, "PROMO_FACTOR"] = cache[mtrl]
    out["QTY"] = out["QTY"] * out["PROMO_FACTOR"]
    return out


def daily_demand(target_rows, background, deliveries, items, promos, today=TODAY,
                 days=HORIZON_DAYS + DEMAND_PAD_DAYS):
    """Demand table: one row per day (`days` from today), one column per goods item."""
    parts = [target_rows[["DATE", "MTRL", "QTY"]], background[["DATE", "MTRL", "QTY"]]]
    long = pd.concat([p for p in parts if len(p)], ignore_index=True) if any(
        len(p) for p in parts) else pd.DataFrame(columns=["DATE", "MTRL", "QTY"])
    long = apply_promos(long, deliveries, items, promos, today)
    dates = pd.date_range(pd.Timestamp(today), periods=days, freq="D")
    table = long.pivot_table(index="DATE", columns="MTRL", values="QTY", aggfunc="sum")
    return table.reindex(index=dates, columns=goods(items)["MTRL"]).fillna(0.0)


# --- simulation -------------------------------------------------------------

def simulate_item(demand, stock, lead_days, min_qty, safety_weeks=DEFAULTS["safety_weeks"],
                  today=TODAY, horizon_days=HORIZON_DAYS, incoming=None):
    """Day-by-day stock of one item and the supplier orders it needs.

    demand: Series of daily demand indexed by date, from today (may run past the horizon)
    stock: stock today; incoming: {date: qty} already ordered (approved) from the supplier

    Breach = first day the end-of-day stock (stock - demand + arrivals) is under the safety
    stock (safety_weeks x mean weekly demand over the horizon). The order must arrive by then:
      ORDER_BY = last working day from which LEAD_TIME working days still arrive by the breach
      ARRIVAL  = ORDER_BY + LEAD_TIME working days
      QTY      = demand for 2 weeks after arrival + safety stock - stock just before arrival,
                 rounded up, at least MIN_ORDER_QTY. (Same as "demand over lead time + 2 weeks
                 + safety - stock when ordering": the lead-time demand is counted once.)
    If ORDER_BY is before today, the order goes out on the first working day from today and is
    flagged ΑΡΓΗΣΑΜΕ (plan() makes it ΕΛΛΕΙΨΗ when customer deliveries are not covered). The simulation goes on after every arrival. Missing stock is carried
    as negative (customers wait for it). Orders are only planned with a breach in the horizon.
    """
    today = pd.Timestamp(today)
    dates = demand.index
    dem = demand.to_numpy(dtype=float)
    horizon = dem[:horizon_days]
    safety = safety_weeks * horizon.sum() / (len(horizon) / 7) if len(horizon) else 0.0
    arrivals = np.zeros(len(dates))
    for when, qty in (incoming or {}).items():
        i = (pd.Timestamp(when) - today).days
        if 0 <= i < len(dates):
            arrivals[i] += qty

    def levels():
        return stock + np.cumsum(arrivals - dem)

    orders, start = [], 0
    while True:
        level = levels()
        below = np.nonzero(level[start:horizon_days] < safety - 1e-9)[0]
        if not len(below):
            break
        b = start + int(below[0])
        breach = dates[b]
        ideal = latest_order_day(breach, lead_days)
        order_by = ideal if ideal >= today else working_on_or_after(today)
        arrival = add_working_days(order_by, lead_days)
        a = (arrival - today).days
        if a >= len(dates):
            break
        before = level[a]  # end of the arrival day without this order
        cover = dem[a + 1:a + 1 + 7 * REVIEW_WEEKS].sum()
        need = cover + safety - before
        qty = max(int(min_qty), int(math.ceil(need - 1e-9)), 1)
        out_day = np.nonzero(level[b:] < 1e-9)[0]
        orders.append({
            "ORDER_BY": order_by, "ARRIVAL": arrival, "IDEAL_ORDER_BY": ideal,
            "BREACH_DATE": breach,
            "STOCKOUT_DATE": dates[b + out_day[0]] if len(out_day) else None,
            "QTY": qty, "NEEDED_QTY": round(max(need, 0.0), 2),
            "SAFETY_STOCK": round(safety, 2),
            "STOCK_BEFORE_ARRIVAL": round(float(before), 2),
            "FLAG": LATE if ideal < today else "",
        })
        arrivals[a] += qty
        start = a + 1
        if start >= horizon_days:
            break
    return orders, levels()


def affected_deliveries(target_rows, mtrl, today, arrival, level=None):
    """Customer deliveries of the item before the supplier's goods arrive (late orders).

    SHORT says whether the projected stock is already gone on that day.
    """
    t = target_rows[(target_rows["MTRL"] == mtrl) & (target_rows["DATE"] >= today)
                    & (target_rows["DATE"] < arrival)].sort_values(["DATE", "TRDR"])
    out = []
    for r in t.itertuples(index=False):
        short = None
        if level is not None:
            short = bool(level[(r.DATE - today).days] < 0)
        out.append({"TRDR": int(r.TRDR), "MTRL": int(r.MTRL),
                    "DELIVERY_DATE": r.DATE.strftime("%Y-%m-%d"), "QTY": float(r.QTY),
                    "SUGGESTION_ID": None if pd.isna(r.SUGGESTION_ID) else int(r.SUGGESTION_ID),
                    "SHORT": short})
    return out


def delivery_rank(suggestions):
    """Sort key for sharing a short stock among customer deliveries.

    Suggestions that need you (urgent, extra route, loss risk) first, then the other
    suggestions by value, then forecast deliveries without a suggestion; same rank ->
    earlier date first. The orchestrator uses the same order.
    """
    from rafi.agents.customer_agent import needs_you

    by_id = {s.get("id"): s for s in suggestions or []}

    def rank(a):
        s = by_id.get(a.get("SUGGESTION_ID"))
        if s is None:
            return (2, 0.0, a["DELIVERY_DATE"], a["TRDR"])
        return (0 if needs_you(s) else 1, -(s.get("value_eur") or 0.0),
                a["DELIVERY_DATE"], a["TRDR"])
    return rank


def cover_deliveries(stock, deliveries, rank=None):
    """Give `stock` to the deliveries in `rank` order. Returns copies, in the input order,
    with MISSING = quantity not covered (0 when covered)."""
    order = sorted(range(len(deliveries)),
                   key=lambda i: (rank or (lambda a: (a["DELIVERY_DATE"], a["TRDR"])))(
                       deliveries[i]))
    left, missing = max(float(stock), 0.0), {}
    for i in order:
        qty = float(deliveries[i]["QTY"])
        given = min(left, qty)
        left -= given
        miss = qty - given
        missing[i] = round(miss, 2) if miss >= MISSING_MIN else 0.0
    return [{**a, "MISSING": missing[i]} for i, a in enumerate(deliveries)]


def stock_until(stock_now, incoming, arrival):
    """Stock today + approved purchases that arrive before `arrival`."""
    arrival = pd.Timestamp(arrival)
    return float(stock_now or 0.0) + sum(q for d, q in (incoming or {}).items()
                                         if pd.Timestamp(d) < arrival)


def flag_state(flag):
    """ΕΛΛΕΙΨΗ, ΑΡΓΗΣΑΜΕ or "" (the old ΔΕΝ_ΦΤΑΝΕΙ reads as ΕΛΛΕΙΨΗ)."""
    if flag in (SHORT, OLD_TOO_LATE):
        return SHORT
    return LATE if flag == LATE else ""


def uncovered(affected, flag=SHORT):
    """Customer deliveries not covered by the stock. Older rows without MISSING: all of them."""
    if flag_state(flag) != SHORT:
        return []
    affected = affected or []
    if any("MISSING" in a for a in affected):
        return [a for a in affected if (a.get("MISSING") or 0) > 0]
    return list(affected)


# --- texts ------------------------------------------------------------------

def _dm(d):
    d = pd.Timestamp(d)
    return f"{d.day}/{d.month}"


def _qty(x):
    return f"{x:,.0f}".replace(",", ".")


def affected_phrase(n):
    """'επηρεάζεται 1 παράδοση' / 'επηρεάζονται 3 παραδόσεις' (+ ' σε πελάτες')."""
    return ("επηρεάζεται 1 παράδοση σε πελάτη" if n == 1
            else f"επηρεάζονται {n} παραδόσεις σε πελάτες")


def _reason_head(o):
    """First sentence of the REASON: when the item runs out (or under the safety stock).

    ΑΡΓΗΣΑΜΕ always says "under the safety stock": the customers are covered until the
    goods arrive, so "runs out" would contradict the rest of the text.
    """
    if (o.get("FLAG") != LATE and o.get("STOCKOUT_DATE") is not None
            and pd.notna(o["STOCKOUT_DATE"])):
        return (f"Το «{o['ITEM_NAME']}» τελειώνει περίπου {_dm(o['STOCKOUT_DATE'])} "
                f"με τη ζήτηση που προβλέπουμε.")
    return (f"Το «{o['ITEM_NAME']}» πέφτει κάτω από το απόθεμα ασφαλείας "
            f"({_qty(o['SAFETY_STOCK'])} {o['UNIT']}) περίπου {_dm(o['BREACH_DATE'])} "
            f"με τη ζήτηση που προβλέπουμε.")


def reason_text(o):
    """REASON from a template, with the numbers of the order `o`."""
    unit = o["UNIT"]
    head = _reason_head(o)
    lead = f" Ο προμηθευτής θέλει {o['LEAD_TIME_DAYS']} εργάσιμες· "
    state = flag_state(o["FLAG"])
    if state:
        affected = o.get("AFFECTED") or []
        n_all = len(affected)
        if state == LATE:
            if n_all == 0:
                cover = "Μέχρι την άφιξη δεν έχουμε παραδόσεις σε πελάτες"
            elif n_all == 1:
                cover = "Η 1 παράδοση σε πελάτη μέχρι την άφιξη καλύπτεται από το απόθεμα"
            else:
                cover = (f"Οι {n_all} παραδόσεις σε πελάτες μέχρι την άφιξη καλύπτονται "
                         f"από το απόθεμα")
            cover += ", αλλά θα πέσουμε κάτω από το απόθεμα ασφαλείας."
        else:
            short = uncovered(affected, o["FLAG"])
            missing = sum(a.get("MISSING") or a["QTY"] for a in short)
            cover = (f"Το απόθεμα δεν φτάνει για όλες τις παραδόσεις μέχρι την άφιξη: "
                     f"{affected_phrase(len(short))} (λείπουν "
                     f"{_qty(math.ceil(missing - 1e-9))} {unit}).")
        tail = (f"έπρεπε να είχε παραγγελθεί έως {_dm(o['IDEAL_ORDER_BY'])}. {cover} "
                f"Παράγγειλε {_qty(o['QTY'])} {unit} το συντομότερο (έως {_dm(o['ORDER_BY'])}), "
                f"φτάνουν {_dm(o['ARRIVAL'])}.")
    else:
        tail = f"παράγγειλε {_qty(o['QTY'])} {unit} έως {_dm(o['ORDER_BY'])}."
    if o["QTY"] > math.ceil(o["NEEDED_QTY"] - 1e-9) and o["QTY"] == o["MIN_ORDER_QTY"]:
        tail += (f" Χρειάζονται {_qty(math.ceil(o['NEEDED_QTY'] - 1e-9))} {unit}, αλλά η "
                 f"ελάχιστη ποσότητα του προμηθευτή είναι {_qty(o['MIN_ORDER_QTY'])} {unit}.")
    return head + lead + tail


# --- whole run --------------------------------------------------------------

def load_details(path=DETAILS_PATH):
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"Δεν βρέθηκε το {path}. Τρέξε πρώτα: python scripts/make_submission.py")
    d = pd.read_csv(path, encoding="utf-8-sig")
    d["PREDICTED_DEPLETION_DATE"] = pd.to_datetime(d["PREDICTED_DEPLETION_DATE"])
    return d


def incoming_from_store(db=None, today=TODAY):
    """{MTRL: {arrival date: qty}} of approved purchase orders still to arrive."""
    out = {}
    for p in store.list_purchase_suggestions(status=store.APPROVED, db=db):
        arrival = pd.Timestamp(p["arrival"])
        if arrival >= pd.Timestamp(today):
            per = out.setdefault(int(p["mtrl"]), {})
            per[arrival] = per.get(arrival, 0) + int(p["qty"])
    return out


def plan(deliveries, items, customers, details, suggestions, promos, suppliers,
         today=TODAY, settings=None, stock_path=STOCK_PATH, incoming=None):
    """All purchase suggestions, most urgent first. Returns (orders DataFrame, stock DataFrame)."""
    today = pd.Timestamp(today)
    cutoff = today - pd.Timedelta(days=1)
    settings = settings or dict(DEFAULTS)
    past = deliveries[deliveries["TRNDATE"] <= cutoff]
    incoming = incoming or {}

    stock = current_stock(past, items, today, settings["stock_cover_weeks"], stock_path)
    windows = closure_windows(past, customers, cutoff)
    days = HORIZON_DAYS + DEMAND_PAD_DAYS
    last_day = today + pd.Timedelta(days=days - 1)
    daily = past.groupby(["TRDR", "MTRL", "TRNDATE"], as_index=False)["QTY"].sum()
    intervals = usual_intervals(daily)
    targets = target_deliveries(details, suggestions, intervals, windows, today, last_day)
    background = background_demand(past, items, details[["TRDR", "MTRL"]], windows, today, days)
    demand = daily_demand(targets, background, past, items, promos, today, days)

    it = items.set_index("MTRL")
    sup = suppliers.set_index("MTRCATEGORY")
    rank = delivery_rank(suggestions)
    rows = []
    for s in stock.itertuples(index=False):
        cat = it.at[s.MTRL, "MTRCATEGORY"]
        supplier = sup.loc[cat]
        lead, moq = int(supplier["LEAD_TIME_DAYS"]), int(supplier["MIN_ORDER_QTY"])
        orders, level = simulate_item(demand[s.MTRL], s.QTY, lead, moq,
                                      settings["safety_weeks"], today,
                                      incoming=incoming.get(int(s.MTRL)))
        for o in orders:
            o.update({
                "MTRL": int(s.MTRL), "ITEM_CODE": str(it.at[s.MTRL, "CODE"]),
                "ITEM_NAME": str(it.at[s.MTRL, "NAME"]), "MTRCATEGORY": cat,
                "UNIT": UNITS.get(it.at[s.MTRL, "MTRUNIT1"], it.at[s.MTRL, "MTRUNIT1"]),
                "SUPPLIER_ID": supplier["SUPPLIER_ID"], "SUPPLIER_NAME": supplier["NAME"],
                "LEAD_TIME_DAYS": lead, "MIN_ORDER_QTY": moq,
                "STOCK_NOW": round(float(s.QTY), 2), "STOCK_SOURCE": s.STOCK_SOURCE,
                "DEMAND_90D": round(float(demand[s.MTRL].iloc[:HORIZON_DAYS].sum()), 2),
                "PRICER": float(it.at[s.MTRL, "PRICER"]),
            })
            o["VALUE_EUR"] = round(o["QTY"] * o["PRICER"], 2)
            o["AFFECTED"], o["UNCOVERED"] = [], 0
            if o["FLAG"] == LATE:
                have = stock_until(s.QTY, incoming.get(int(s.MTRL)), o["ARRIVAL"])
                o["AFFECTED"] = cover_deliveries(
                    have, affected_deliveries(targets, s.MTRL, today, o["ARRIVAL"], level), rank)
                o["UNCOVERED"] = len(uncovered(o["AFFECTED"]))
                if o["UNCOVERED"]:
                    o["FLAG"] = SHORT
            o["REASON"] = reason_text(o)
            rows.append(o)
    out = pd.DataFrame(rows)
    if len(out):
        out["URGENT"] = out["FLAG"].map({SHORT: 2, LATE: 1}).fillna(0)
        out = out.sort_values(["ORDER_BY", "URGENT", "VALUE_EUR"], ascending=[True, False, False])
        out = out.drop(columns="URGENT").reset_index(drop=True)
    return out, stock


def to_records(orders):
    """Plain dicts (dates as text) for rafi.store."""
    recs = []
    for o in orders.to_dict("records"):
        r = {}
        for k, v in o.items():
            if isinstance(v, pd.Timestamp):
                v = v.strftime("%Y-%m-%d")
            elif v is None or (not isinstance(v, (list, dict, str)) and pd.isna(v)):
                v = None
            elif isinstance(v, np.generic):
                v = v.item()
            r[k] = v
        recs.append(r)
    return recs


def run(today=TODAY, db=None, data_dir=DATA_DIR, details_path=DETAILS_PATH,
        settings_path=SETTINGS_PATH, stock_path=STOCK_PATH, suppliers_path=SUPPLIERS_PATH):
    """Compute, save the new purchase suggestions, return (orders, save result, stock)."""
    from rafi.ingest import build_deliveries, load_from_files
    from rafi.suggestions import load_promos

    headers, lines, items, customers = load_from_files(data_dir)
    deliveries = build_deliveries(headers, lines, items)
    details = load_details(details_path)
    suggestions = store.list_suggestions(db=db)
    orders, stock = plan(deliveries, items, customers, details, suggestions,
                         load_promos(data_dir), load_suppliers(items, suppliers_path),
                         today, load_settings(settings_path), stock_path,
                         incoming_from_store(db, today))
    result = store.save_purchase_new(to_records(orders), user="warehouse_agent", db=db)
    return orders, result, stock


# --- purchase orders per supplier (dashboard, Telegram, orchestrator) --------

_CUSTOMER_NAMES = {}


def customer_names(data_dir=DATA_DIR):
    """{TRDR: NAME} from TRDR.csv (read once)."""
    key = str(data_dir)
    if key not in _CUSTOMER_NAMES:
        c = pd.read_csv(Path(data_dir) / "TRDR.csv", encoding="utf-8-sig")
        _CUSTOMER_NAMES[key] = dict(zip(c["TRDR"].astype(int), c["NAME"].astype(str)))
    return _CUSTOMER_NAMES[key]


def _items_text(n):
    return "1 είδος" if n == 1 else f"{n} είδη"


def group_reason(supplier_name, order_by, arrival, rows):
    """One REASON for a supplier's order (all its lines with the same ORDER_BY)."""
    if len(rows) == 1:
        return rows[0]["payload"]["REASON"]
    text = (f"Από τον προμηθευτή «{supplier_name}» χρειάζονται {_items_text(len(rows))} με "
            f"παραγγελία έως {_dm(order_by)}, ώστε να φτάσουν {_dm(arrival)}. "
            + " ".join(_reason_head(r["payload"]) for r in rows))
    def which(rs):
        names = ", ".join(f"«{r['payload']['ITEM_NAME']}»" for r in rs)
        return f"{'το' if len(rs) == 1 else 'τα'} {names}"

    late = [r for r in rows if flag_state(r["flag"]) == LATE]
    short = [r for r in rows if flag_state(r["flag"]) == SHORT]
    if late:
        text += (f" Αργήσαμε για {which(late)}: οι παραδόσεις σε πελάτες μέχρι την άφιξη "
                 f"καλύπτονται, αλλά θα πέσουμε κάτω από το απόθεμα ασφαλείας.")
    if short:
        n = sum(len(uncovered(r["payload"].get("AFFECTED"), r["flag"])) for r in short)
        text += (f" Δεν φτάνει το απόθεμα για {which(short)}: {affected_phrase(n)} "
                 f"πριν την άφιξη.")
    return text


def supplier_payload(group):
    """The input of rafi.llm.write_texts(kind="supplier") for an order of open_purchase_orders."""
    return {"SUPPLIER_ID": group["supplier_id"], "SUPPLIER_NAME": group["supplier_name"],
            "ORDER_BY": group["order_by"], "NEEDED_BY": group["arrival"],
            "LINES": [{"ITEM_NAME": l["item_name"], "QTY1": l["qty"], "UNIT": l["unit"]}
                      for l in group["lines"]],
            "REASON": group["reason"]}


def _groups(rows):
    """{(supplier_id, order_by): [rows]} keeping the store's order."""
    out = {}
    for r in rows:
        out.setdefault((r["supplier_id"], r["order_by"]), []).append(r)
    return out


def open_purchase_orders(days: int = 7, db=None, today=None) -> list[dict]:
    """Purchase suggestions still ΠΡΟΤΑΣΗ with ORDER_BY up to today + `days`, one dict per
    supplier and ORDER_BY, earliest first.

    Keys: supplier_id, supplier_name, order_by, arrival,
    status       "ΕΛΛΕΙΨΗ" (a line leaves customer deliveries uncovered), "ΑΡΓΗΣΑΜΕ" (a line
                 was ordered too late, customers covered, under the safety stock) or "",
    short        status == ΕΛΛΕΙΨΗ;  late: status == ΑΡΓΗΣΑΜΕ,
    total_value_eur (list price), reason, note (message to the supplier), note_source,
    lines [purchase_id, mtrl, item_name, qty, unit, value_eur, flags],
    affected_deliveries  ONLY the customer deliveries not covered (ΕΛΛΕΙΨΗ):
                 [trdr, customer_name, date, qty, missing, suggestion_id, mtrl, item_name],
    deliveries_before_arrival  all customer deliveries of the late lines until the arrival.
    The note is the one written by the last run (AI or template) while it still covers
    exactly the same lines, else the template.
    """
    from rafi import llm

    today = pd.Timestamp(today if today is not None else TODAY)
    last = (today + pd.Timedelta(days=int(days))).strftime("%Y-%m-%d")
    rows = [r for r in store.list_purchase_suggestions(status=store.PROPOSED, db=db)
            if r["order_by"] <= last]
    names = customer_names()
    out = []
    for (supplier_id, order_by), g in _groups(rows).items():
        first = g[0]["payload"]
        arrival = max(r["arrival"] for r in g)
        lines, affected, before = [], [], 0
        for r in g:
            p = r["payload"]
            state = flag_state(r["flag"])
            lines.append({"purchase_id": r["id"], "mtrl": int(r["mtrl"]),
                          "item_name": p.get("ITEM_NAME"), "qty": int(r["qty"]),
                          "unit": p.get("UNIT"), "value_eur": r["value_eur"],
                          "flags": [state] if state else []})
            before += len(p.get("AFFECTED") or []) if state else 0
            for a in uncovered(p.get("AFFECTED"), r["flag"]):
                affected.append({"trdr": int(a["TRDR"]),
                                 "customer_name": names.get(int(a["TRDR"]), ""),
                                 "date": a["DELIVERY_DATE"], "qty": a["QTY"],
                                 "missing": a.get("MISSING", a["QTY"]),
                                 "suggestion_id": a.get("SUGGESTION_ID"),
                                 "mtrl": int(r["mtrl"]), "item_name": p.get("ITEM_NAME")})
        states = {flag_state(r["flag"]) for r in g}
        status = SHORT if SHORT in states else LATE if LATE in states else ""
        group = {"supplier_id": supplier_id, "supplier_name": first.get("SUPPLIER_NAME"),
                 "order_by": order_by, "arrival": arrival, "status": status,
                 "short": status == SHORT, "late": status == LATE,
                 "total_value_eur": round(sum(r["value_eur"] or 0.0 for r in g), 2),
                 "reason": group_reason(first.get("SUPPLIER_NAME"), order_by, arrival, g),
                 "lines": lines, "affected_deliveries": affected,
                 "deliveries_before_arrival": before}
        ids = sorted(r["id"] for r in g)
        stored = [r["payload"] for r in g if r["payload"].get("GROUP_IDS") == ids]
        if len(stored) == len(g) and stored[0].get("GROUP_NOTE"):
            group["note"] = stored[0]["GROUP_NOTE"]
            group["note_source"] = stored[0].get("GROUP_NOTE_SOURCE", llm.TEMPLATE)
        else:
            group["note"] = llm.template_for(supplier_payload(group), "supplier")["note"]
            group["note_source"] = llm.TEMPLATE
        out.append(group)
    rank = {SHORT: 0, LATE: 1, "": 2}
    return sorted(out, key=lambda o: (o["order_by"], rank[o["status"]], -o["total_value_eur"]))


def _group_rows(supplier_id, order_by, db):
    rows = [r for r in store.list_purchase_suggestions(status=store.PROPOSED, db=db)
            if r["supplier_id"] == str(supplier_id) and r["order_by"] == str(order_by)]
    if not rows:
        raise store.StoreError(f"Δεν υπάρχει παραγγελία σε {store.PROPOSED} για τον προμηθευτή "
                               f"{supplier_id} με ημερομηνία {order_by}.")
    return rows


def approve_purchase_order(supplier_id, order_by, user, db=None, out_dir=None):
    """Approve every line of the supplier's order (store.approve_purchase). Returns the files."""
    return [store.approve_purchase(r["id"], user, db=db, out_dir=out_dir)
            for r in _group_rows(supplier_id, order_by, db)]


def reject_purchase_order(supplier_id, order_by, reason, user, db=None):
    """Reject every line of the supplier's order with the same reason. Returns the ids."""
    if not reason or not str(reason).strip():
        raise store.StoreError("Η απόρριψη χρειάζεται λόγο.")
    rows = _group_rows(supplier_id, order_by, db)
    for r in rows:
        store.reject_purchase(r["id"], reason, user, db=db)
    return [r["id"] for r in rows]


def write_group_notes(days=7, db=None, ai=True, today=None):
    """Write the supplier note of every order in the next `days` days and keep it in the store.

    With `ai` and a model that answers: rafi.llm.write_texts(kind="supplier"), which falls back
    to the template by itself. A note already written by AI for the same lines is kept.
    Returns {"orders", "ai"}.
    """
    from rafi import llm

    groups = open_purchase_orders(days, db, today)
    use_ai = ai and bool(groups) and llm.available()
    n_ai = 0
    for g in groups:
        if g["note_source"] == llm.AI:
            n_ai += 1
            continue
        texts = (llm.write_texts(supplier_payload(g), kind="supplier") if use_ai
                 else {"note": g["note"], "source": llm.TEMPLATE})
        n_ai += texts["source"] == llm.AI
        store.set_purchase_fields([l["purchase_id"] for l in g["lines"]],
                                  {"GROUP_NOTE": texts["note"], "GROUP_NOTE_SOURCE": texts["source"],
                                   "GROUP_IDS": sorted(l["purchase_id"] for l in g["lines"])},
                                  db=db)
    return {"orders": len(groups), "ai": n_ai}


# --- CLI --------------------------------------------------------------------

def summary(orders, result, stock, top=10):
    late = orders[orders["FLAG"] == LATE] if len(orders) else orders
    short = orders[orders["FLAG"] == SHORT] if len(orders) else orders
    value = float(orders["VALUE_EUR"].sum()) if len(orders) else 0.0
    src = stock["STOCK_SOURCE"].value_counts().to_dict()
    out = ["Ράφι · agent αποθήκης · προτάσεις αγοράς από προμηθευτές",
           f"Απόθεμα σήμερα: {src.get(FILE, 0)} είδη από αρχείο, {src.get(ESTIMATE, 0)} εκτίμηση",
           f"Προτάσεις αγοράς (90 ημέρες): {len(orders)}",
           f"  {SHORT}: {len(short)} (δεν καλύπτονται "
           f"{int(short['UNCOVERED'].sum()) if len(short) else 0} παραδόσεις σε πελάτες)",
           f"  {LATE}: {len(late)} (οι πελάτες καλύπτονται, κάτω από το απόθεμα ασφαλείας)",
           f"  Αξία (τιμή καταλόγου PRICER): {value:,.2f} €",
           f"  Αποθηκεύτηκαν νέες: {len(result['saved'])} · υπήρχαν ήδη: {result['skipped']}",
           "", f"Οι {top} πιο βιαστικές:",
           f"{'#':>3} {'είδος':<38} {'προμηθευτής':<32} {'έως':<10} {'άφιξη':<10} "
           f"{'ποσ.':>6} {'€':>9}  σήμανση"]
    for i, o in enumerate(orders.head(top).itertuples(index=False), start=1):
        out.append(f"{i:>3} {o.MTRL} {o.ITEM_NAME[:33]:<33} {o.SUPPLIER_NAME[:32]:<32} "
                   f"{o.ORDER_BY:%Y-%m-%d} {o.ARRIVAL:%Y-%m-%d} {o.QTY:>6} "
                   f"{o.VALUE_EUR:>9,.2f}  {o.FLAG}")
    return "\n".join(out)


def _print_list(rows):
    print(f"{'id':>4}  {'κατάσταση':<12} {'είδος':<40} {'προμηθ.':<6} {'έως':<10} "
          f"{'άφιξη':<10} {'ποσ.':>6} {'€':>9}  σήμανση")
    for r in rows:
        p = r["payload"]
        print(f"{r['id']:>4}  {r['status']:<12} {r['mtrl']} {p['ITEM_NAME'][:35]:<35} "
              f"{r['supplier_id']:<6} {r['order_by']:<10} {r['arrival']:<10} {r['qty']:>6} "
              f"{r['value_eur']:>9,.2f}  {r['flag'] or ''}")
    print(f"Σύνολο: {len(rows)}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m rafi.agents.warehouse_agent",
                                 description="Agent αποθήκης: προτάσεις αγοράς από προμηθευτές.")
    ap.add_argument("--db", default=None, help="αρχείο βάσης (προεπιλογή rafi.db)")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("run", help="υπολογισμός και αποθήκευση (προεπιλογή)")
    p.add_argument("--today", default=str(TODAY.date()))
    p = sub.add_parser("list")
    p.add_argument("--status")
    p = sub.add_parser("approve")
    p.add_argument("id", type=int)
    p.add_argument("--user", default="αποθήκη")
    p = sub.add_parser("reject")
    p.add_argument("id", type=int)
    p.add_argument("reason")
    p.add_argument("--user", default="αποθήκη")
    args = ap.parse_args(argv)

    try:
        if args.cmd in (None, "run"):
            today = pd.Timestamp(getattr(args, "today", None) or TODAY)
            orders, result, stock = run(today, db=args.db)
            print(summary(orders, result, stock))
        elif args.cmd == "list":
            _print_list(store.list_purchase_suggestions(args.status, db=args.db))
        elif args.cmd == "approve":
            path = store.approve_purchase(args.id, args.user, db=args.db)
            print(f"Η πρόταση αγοράς {args.id} εγκρίθηκε· γράφτηκε {path}")
        elif args.cmd == "reject":
            store.reject_purchase(args.id, args.reason, args.user, db=args.db)
            print(f"Η πρόταση αγοράς {args.id} απορρίφθηκε: {args.reason}")
    except store.StoreError as e:
        print(f"Σφάλμα: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
