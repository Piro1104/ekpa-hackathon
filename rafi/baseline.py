"""Naive baselines, only for comparison with the real forecast.

Both functions use data up to and including the cutoff, nothing after it.
They return one row per target pair: TRDR, MTRL, PREDICTED_DEPLETION_DATE
(a Timestamp, or NaT for "no depletion inside the horizon").
"""
import math

import pandas as pd

HORIZON_DAYS = 90
WINDOW_DAYS = 180


def plus30(targets, cutoff):
    """Every pair runs out 30 days after the cutoff."""
    cutoff = pd.Timestamp(cutoff)
    out = targets[["TRDR", "MTRL"]].copy().reset_index(drop=True)
    out["PREDICTED_DEPLETION_DATE"] = cutoff + pd.Timedelta(days=30)
    return out


def rolling180(daily, targets, cutoff, window=WINDOW_DAYS, horizon_days=HORIZON_DAYS):
    """Average rate of the last 180 days, applied to the last delivery.

    rate  = net quantity of the 180 days up to and including the cutoff / 180
    stock = last delivery quantity - rate * days since that delivery (never below 0)
    date  = cutoff + ceil(stock / rate), at least cutoff + 1
    Blank (NaT) when the rate is 0 or the date falls after horizon_end.
    """
    cutoff = pd.Timestamp(cutoff)
    horizon_end = cutoff + pd.Timedelta(days=horizon_days)
    past = daily[daily["TRNDATE"] <= cutoff]  # never look past the cutoff

    recent = past[past["TRNDATE"] > cutoff - pd.Timedelta(days=window)]
    rate = recent.groupby(["TRDR", "MTRL"])["QTY"].sum() / window

    # Last day with a positive net delivery (a credit-only day is not a delivery).
    last = (past[past["QTY"] > 0].sort_values("TRNDATE")
            .groupby(["TRDR", "MTRL"]).last()[["TRNDATE", "QTY"]])

    dates = []
    for trdr, mtrl in zip(targets["TRDR"], targets["MTRL"]):
        r = rate.get((trdr, mtrl), 0.0)
        if r <= 0 or (trdr, mtrl) not in last.index:
            dates.append(pd.NaT)
            continue
        last_date, last_qty = last.loc[(trdr, mtrl)]
        stock = max(0.0, last_qty - r * (cutoff - last_date).days)
        days = max(1, math.ceil(stock / r))
        date = cutoff + pd.Timedelta(days=days)
        dates.append(pd.NaT if date > horizon_end else date)

    out = targets[["TRDR", "MTRL"]].copy().reset_index(drop=True)
    out["PREDICTED_DEPLETION_DATE"] = pd.to_datetime(pd.Series(dates, dtype="datetime64[ns]"))
    return out
