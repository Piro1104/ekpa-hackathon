"""Depletion forecast per (customer, item): explainable, step by step.

For every target pair:
  a) base rate: consumption per day from the last K deliveries, with the seasonality removed
  b) depletion: start from the last delivery and take away rate x seasonal factor day by day
  c) winter closures: a closed customer consumes nothing -> no depletion in the horizon
  d) cold start: new customers borrow the rate of similar customers, scaled to their size
  e) confidence from how steady the rate is and how much history there is
  f) suggested quantity: what the customer uses between two usual deliveries

Everything uses data up to and including the cutoff only.
"""
import math

import numpy as np
import pandas as pd

from rafi.features import (FACTOR_FLOOR, is_closed_on, likely_closer, seasonal_factors,
                           winter_history)

K_DELIVERIES = 3             # (a) recent deliveries for the base rate (k - 1 intervals); 3 won the backtest over 4 and 6
HORIZON_DAYS = 90            # (b) a depletion after cutoff + 90 days is left blank
COLD_START_MIN_DELIVERIES = 2    # (d) fewer deliveries than this -> borrow the rate from peers
NEW_CUSTOMER_MIN_DELIVERIES = 4  # (d) a new customer's pair needs this many to use its own rate
# Seasonal factors without the winter closers. Tried and rejected: better on the 2025-10-31
# backtest, worse on 2026-06-30 (closers then lose their own summer peak). Kept as a switch.
FACTORS_WITHOUT_CLOSERS = False
PEER_RECENT_DAYS = 180       # (d) peers' rates only from pairs bought in the last 180 days
CONF_MAX = 0.95              # (e) best possible confidence for a normal forecast
CONF_COLD_START = 0.30       # (e) confidence for a borrowed (cold start) rate
CONF_CLOSED = {2: 0.90, 1: 0.75}  # (e) by number of closed winters seen; likely closer -> 0.5
CONF_LIKELY_CLOSED = 0.50
DEFAULT_INTERVAL_DAYS = 14   # (f) order spacing when a pair has no interval of its own

REASONS = {
    "ΚΑΝΟΝΙΚΗ": "εξάντληση μέσα στον ορίζοντα από τον ρυθμό του ζεύγους",
    "ΗΔΗ_ΕΞΑΝΤΛΗΜΕΝΟ": "το απόθεμα έχει ήδη τελειώσει στο cutoff -> cutoff + 1",
    "ΕΚΤΟΣ_ΟΡΙΖΟΝΤΑ": "το απόθεμα φτάνει πέρα από τις 90 ημέρες",
    "ΚΛΕΙΣΤΟ_ΧΕΙΜΩΝΑ": "ο πελάτης κλείνει τον χειμώνα, δεν καταναλώνει",
    "ΝΕΟΣ_ΠΕΛΑΤΗΣ": "λίγο ιστορικό: ρυθμός από παρόμοιους πελάτες",
    "ΧΩΡΙΣ_ΡΥΘΜΟ": "δεν υπάρχει αρκετό ιστορικό για ρυθμό",
}

OUTPUT_COLUMNS = ["TRDR", "MTRL", "PREDICTED_DEPLETION_DATE", "SUGGESTED_QTY", "CONFIDENCE",
                  "REASON_CODE", "BASE_RATE", "SEASONAL_FACTOR_NOW", "STOCK_AT_CUTOFF",
                  "LAST_DELIVERY_DATE", "LAST_DELIVERY_QTY", "N_DELIVERIES"]


class SeasonCalendar:
    """Daily seasonal factor per (AREA, MTRCATEGORY), with running sums.

    sum of factors over the days (a, b] = cum[b] - cum[a], so both the "seasonally weighted
    days" of an interval and the day-by-day depletion are simple look-ups.
    """

    def __init__(self, factors, start, end, use_seasonality=True):
        self.start = pd.Timestamp(start)
        self.days = pd.date_range(self.start, pd.Timestamp(end), freq="D")
        self.month_idx = self.days.month.values - 1
        self.table = {}
        if use_seasonality:
            for (area, cat), g in factors.groupby(["AREA", "MTRCATEGORY"]):
                f = g.sort_values("MONTH")["FACTOR"].to_numpy()
                self.table[(area, cat)] = np.maximum(f, FACTOR_FLOOR)
        self._cum = {}

    def monthly(self, combo):
        return self.table.get(combo, np.ones(12))

    def factor(self, combo, date):
        return float(self.monthly(combo)[pd.Timestamp(date).month - 1])

    def cum(self, combo):
        if combo not in self._cum:
            daily = self.monthly(combo)[self.month_idx]
            self._cum[combo] = np.cumsum(daily)
        return self._cum[combo]

    def idx(self, date):
        return (pd.Timestamp(date) - self.start).days


def _pair_series(daily):
    """{(TRDR, MTRL): (delivery dates, quantities, credit dates, credit quantities)}."""
    out = {}
    for key, g in daily.sort_values("TRNDATE").groupby(["TRDR", "MTRL"]):
        pos, neg = g[g["QTY"] > 0], g[g["QTY"] < 0]
        out[key] = (pos["TRNDATE"].to_numpy(), pos["QTY"].to_numpy(dtype=float),
                    neg["TRNDATE"].to_numpy(), neg["QTY"].to_numpy(dtype=float))
    return out


def pair_rate(dates, qtys, cdates, cqtys, cal, combo, k=K_DELIVERIES):
    """Step (a): base rate from the last k deliveries.

    For each interval between two consecutive deliveries:
      consumed  = quantity of the first delivery (minus credits inside the interval)
      weighted  = sum of the daily seasonal factors over the interval's days
    BASE_RATE = sum(consumed) / sum(weighted). Also returns each interval's own rate
    (for the confidence) and the interval lengths in days.
    """
    dates, qtys = dates[-k:], qtys[-k:]
    if len(dates) < 2:
        return None, [], []
    cum = cal.cum(combo)
    consumed, weighted, rates, lengths = [], [], [], []
    for j in range(len(dates) - 1):
        a, b = dates[j], dates[j + 1]
        credit = cqtys[(cdates > a) & (cdates < b)].sum()  # credits are negative
        c = max(0.0, qtys[j] + credit)
        w = cum[cal.idx(b)] - cum[cal.idx(a)]
        consumed.append(c)
        weighted.append(w)
        rates.append(c / w if w > 0 else 0.0)
        lengths.append((pd.Timestamp(b) - pd.Timestamp(a)).days)
    total_w = sum(weighted)
    rate = sum(consumed) / total_w if total_w > 0 else None
    return rate, rates, lengths


def confidence(rates, k=K_DELIVERIES):
    """Step (e): CONFIDENCE = 0.95 x stability x history, between 0 and 1.

    stability = 1 / (1 + CV), CV = std / mean of the interval rates (1 if fewer than 2 intervals)
    history   = min(1, number of intervals / (k - 1))
    """
    rates = np.asarray(rates, dtype=float)
    if len(rates) >= 2 and rates.mean() > 0:
        cv = rates.std() / rates.mean()
    else:
        cv = 1.0
    history = min(1.0, len(rates) / max(1, k - 1))
    return round(CONF_MAX * history / (1 + cv), 2)


def deplete(stock, start_date, rate, cal, combo, last_day):
    """Step (b): first day after `start_date` on which stock - rate x factors <= 0.

    Returns None if that does not happen by `last_day`.
    """
    if rate is None or rate <= 0:
        return None
    if stock <= 0:
        return pd.Timestamp(start_date)
    cum = cal.cum(combo)
    i0, i_last = cal.idx(start_date), cal.idx(last_day)
    target = cum[i0] + stock / rate - 1e-9
    i = int(np.searchsorted(cum, target, side="left"))
    if i > i_last:
        return None
    return cal.days[i]


def _closure_status(customers, history, cutoff, use_closers):
    """{TRDR: (closed_now, confidence)} for closers and likely closers."""
    out = {}
    if not use_closers:
        return out
    seen = set(history["TRDR"])
    for _, r in history[history["IS_CLOSER"]].iterrows():
        closed = is_closed_on(cutoff, r["LAST_BEFORE_CLOSE"], r["FIRST_AFTER_OPEN"])
        out[r["TRDR"]] = (closed, CONF_CLOSED.get(min(int(r["N_CLOSED"]), 2), 0.75))
    known = customers[pd.to_datetime(customers["INSDATE"]) <= cutoff]
    for trdr in known["TRDR"]:
        if trdr in seen:
            continue
        likely, last, first = likely_closer(trdr, customers, history)
        if likely:
            out[trdr] = (is_closed_on(cutoff, last, first), CONF_LIKELY_CLOSED)
    return out


def forecast(daily, deliveries, customers, items, targets, cutoff, k=K_DELIVERIES,
             use_seasonality=True, use_closers=True, use_cold_start=True,
             factors_without_closers=FACTORS_WITHOUT_CLOSERS, factors=None):
    """One row per target pair with the date, the quantity and the numbers that explain them.

    daily:      net quantity per TRDR, MTRL, TRNDATE (rafi.ingest.daily_qty)
    deliveries: clean deliveries (rafi.ingest.build_deliveries), for seasonality and closures
    The use_* switches turn steps off, for the ablation table in rafi.evaluate;
    factors_without_closers leaves winter closers out of the seasonal factors.
    `factors` may replace the AREA x MTRCATEGORY seasonal factors (same columns).
    """
    cutoff = pd.Timestamp(cutoff)
    horizon_last = cutoff + pd.Timedelta(days=HORIZON_DAYS)
    daily = daily[daily["TRNDATE"] <= cutoff]
    deliveries = deliveries[deliveries["TRNDATE"] <= cutoff]

    history = winter_history(deliveries, customers, cutoff)
    if factors is None:
        exclude = history.loc[history["IS_CLOSER"], "TRDR"] if factors_without_closers else ()
        factors = seasonal_factors(deliveries, customers, items, cutoff, exclude=exclude)
    cal = SeasonCalendar(factors, daily["TRNDATE"].min() - pd.Timedelta(days=1),
                         horizon_last + pd.Timedelta(days=1), use_seasonality)
    area = customers.set_index("TRDR")["AREA"]
    tcat = customers.set_index("TRDR")["TRDCATEGORY"]
    mcat = items.set_index("MTRL")["MTRCATEGORY"]
    combo_of = lambda trdr, mtrl: (area.get(trdr), mcat.get(mtrl))
    new_customers = set(customers.loc[pd.to_datetime(customers["INSDATE"]).dt.year == cutoff.year,
                                      "TRDR"])
    series = _pair_series(daily)

    closure = _closure_status(customers, history, cutoff, use_closers)

    peers = _peer_tables(series, daily, cal, combo_of, customers, new_customers, cutoff, k) \
        if use_cold_start else None

    rows = []
    for trdr, mtrl in zip(targets["TRDR"], targets["MTRL"]):
        combo = combo_of(trdr, mtrl)
        empty = np.array([], dtype="datetime64[ns]"), np.array([]), \
            np.array([], dtype="datetime64[ns]"), np.array([])
        dates, qtys, cdates, cqtys = series.get((trdr, mtrl), empty)
        n = len(dates)
        rate, rates, lengths = pair_rate(dates, qtys, cdates, cqtys, cal, combo, k)

        cold = n < COLD_START_MIN_DELIVERIES or (trdr in new_customers
                                                 and n < NEW_CUSTOMER_MIN_DELIVERIES)
        reason = None
        if use_cold_start and cold:
            peer_rate, peer_interval = _peer_rate(trdr, mtrl, peers, tcat, area)
            if peer_rate is not None:
                rate, lengths = peer_rate, [peer_interval]
            conf = CONF_COLD_START
            reason = "ΝΕΟΣ_ΠΕΛΑΤΗΣ"
        else:
            conf = confidence(rates, k) if rate else 0.0

        # stock after the last delivery: its quantity minus credits that came after it
        last_date = pd.Timestamp(dates[-1]) if n else pd.NaT
        last_qty = float(qtys[-1]) if n else 0.0
        stock = max(0.0, last_qty + cqtys[cdates > dates[-1]].sum()) if n else 0.0
        cum = cal.cum(combo)
        stock_now = None
        if rate and n:
            used = rate * (cum[cal.idx(cutoff)] - cum[cal.idx(last_date)])
            stock_now = max(0.0, stock - used)

        date = deplete(stock, last_date, rate, cal, combo, horizon_last) if n else None
        if rate is None or not n:
            reason = reason or "ΧΩΡΙΣ_ΡΥΘΜΟ"
            date = None
        elif date is None:
            reason = reason or "ΕΚΤΟΣ_ΟΡΙΖΟΝΤΑ"
        elif date <= cutoff:
            date = cutoff + pd.Timedelta(days=1)
            reason = reason or "ΗΔΗ_ΕΞΑΝΤΛΗΜΕΝΟ"
        else:
            reason = reason or "ΚΑΝΟΝΙΚΗ"

        closed, closed_conf = closure.get(trdr, (False, None))
        if closed:
            date, reason, conf = None, "ΚΛΕΙΣΤΟ_ΧΕΙΜΩΝΑ", closed_conf

        qty = None
        if rate and not closed:
            when = date if date is not None else cutoff + pd.Timedelta(days=1)
            interval = float(np.median(lengths)) if lengths else DEFAULT_INTERVAL_DAYS
            qty = max(1, math.ceil(rate * cal.factor(combo, when) * interval - 1e-9))

        rows.append({
            "TRDR": trdr, "MTRL": mtrl,
            "PREDICTED_DEPLETION_DATE": date if date is not None else pd.NaT,
            "SUGGESTED_QTY": qty, "CONFIDENCE": conf, "REASON_CODE": reason,
            "BASE_RATE": round(rate, 4) if rate else None,
            "SEASONAL_FACTOR_NOW": round(cal.factor(combo, cutoff + pd.Timedelta(days=1)), 3),
            "STOCK_AT_CUTOFF": round(stock_now, 2) if stock_now is not None else None,
            "LAST_DELIVERY_DATE": last_date, "LAST_DELIVERY_QTY": last_qty if n else None,
            "N_DELIVERIES": n,
        })

    out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    out["PREDICTED_DEPLETION_DATE"] = pd.to_datetime(out["PREDICTED_DEPLETION_DATE"])
    out["SUGGESTED_QTY"] = out["SUGGESTED_QTY"].astype("Int64")
    for col in ["BASE_RATE", "STOCK_AT_CUTOFF", "LAST_DELIVERY_QTY", "CONFIDENCE"]:
        out[col] = out[col].astype(float)
    return out


# --- cold start ------------------------------------------------------------

def _peer_tables(series, daily, cal, combo_of, customers, new_customers, cutoff, k):
    """Own rates of old customers' recent pairs, and each customer's first delivery date."""
    recent = cutoff - pd.Timedelta(days=PEER_RECENT_DAYS)
    rates, firsts = [], {}
    for (trdr, mtrl), (dates, qtys, cdates, cqtys) in series.items():
        if len(dates):
            firsts[trdr] = min(firsts.get(trdr, dates[0]), dates[0])
        if trdr in new_customers or len(dates) < COLD_START_MIN_DELIVERIES \
                or pd.Timestamp(dates[-1]) < recent:
            continue
        rate, _, lengths = pair_rate(dates, qtys, cdates, cqtys, cal, combo_of(trdr, mtrl), k)
        if rate:
            rates.append({"TRDR": trdr, "MTRL": mtrl, "RATE": rate,
                          "INTERVAL": float(np.median(lengths))})
    rates = pd.DataFrame(rates, columns=["TRDR", "MTRL", "RATE", "INTERVAL"])
    rates = rates.merge(customers[["TRDR", "AREA", "TRDCATEGORY"]], on="TRDR")
    bought = daily[daily["QTY"] > 0].groupby(["TRDR", "TRNDATE"])["QTY"].sum()
    by_customer = {t: s.droplevel(0) for t, s in bought.groupby(level=0)}
    return {"rates": rates, "bought": by_customer, "first": firsts}


def _total_qty(peers, trdr, since):
    """All positive quantity a customer bought from `since` up to the cutoff."""
    s = peers["bought"].get(trdr)
    return 0.0 if s is None else float(s[s.index >= since].sum())


def _peer_rate(trdr, mtrl, peers, tcat, area):
    """Step (d): median peer rate for the same item, scaled to the customer's size.

    Peers: old customers with the same AREA and TRDCATEGORY; if none buys this item,
    the same TRDCATEGORY in any area; then any customer.
    size ratio = customer's total quantity since its first delivery
                 / median total quantity of the peers over the same days.
    Returns (rate, typical interval in days) or (None, None).
    """
    r = peers["rates"]
    item = r[r["MTRL"] == mtrl]
    for group in (item[(item["AREA"] == area.get(trdr)) & (item["TRDCATEGORY"] == tcat.get(trdr))],
                  item[item["TRDCATEGORY"] == tcat.get(trdr)],
                  item):
        if len(group):
            break
    else:
        return None, None

    since = peers["first"].get(trdr)
    if since is None:
        return None, None
    mine = _total_qty(peers, trdr, since)
    peer_totals = [_total_qty(peers, p, since) for p in group["TRDR"].unique()]
    peer_median = float(np.median(peer_totals)) if peer_totals else 0.0
    ratio = mine / peer_median if peer_median > 0 else 1.0
    return float(group["RATE"].median()) * ratio, float(group["INTERVAL"].median())
