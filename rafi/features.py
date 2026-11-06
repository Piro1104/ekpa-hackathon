"""Features built from the invoices: working days, route days, seasonality, winter closures.

Every function that takes `deliveries` uses only rows up to and including the cutoff.
`deliveries` is the clean table from rafi.ingest.build_deliveries (credits have negative QTY).
"""
from functools import lru_cache
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

SUNDAY = 6
ROUTE_DAY_MIN_SHARE = 0.20   # a weekday is a route day if it has >= 20% of the customer's deliveries
SUMMER_MONTHS = (6, 7, 8)    # a closer must have deliveries in these months...
WINTER_MONTHS = (12, 1, 2)   # ...and none in these (December of year y to February of y+1)
FACTOR_FLOOR = 0.05          # never divide by a (near) zero seasonal factor
REFERENCE_YEAR = 2025        # non-leap year used to turn day-of-year back into a month-day


# --- working days ----------------------------------------------------------

@lru_cache(maxsize=4)
def load_holidays(path=DATA_DIR / "greek_holidays.csv"):
    """Set of holiday dates (Timestamps) from greek_holidays.csv."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    return frozenset(pd.to_datetime(df["DATE"], format="%Y-%m-%d"))


def working_day(date, holidays=None):
    """True if deliveries can happen that day: not a Sunday and not a holiday."""
    holidays = load_holidays() if holidays is None else holidays
    date = pd.Timestamp(date).normalize()
    return date.dayofweek != SUNDAY and date not in holidays


def next_working_day(date, holidays=None):
    """The first working day strictly after `date`."""
    date = pd.Timestamp(date).normalize() + pd.Timedelta(days=1)
    while not working_day(date, holidays):
        date += pd.Timedelta(days=1)
    return date


# --- helpers ---------------------------------------------------------------

def _past_purchases(deliveries, cutoff):
    """Real deliveries (positive QTY, not credits) up to and including the cutoff."""
    cutoff = pd.Timestamp(cutoff)
    d = deliveries[deliveries["TRNDATE"] <= cutoff]
    if "IS_CREDIT" in d:
        d = d[~d["IS_CREDIT"]]
    return d[d["QTY"] > 0]


def _month_day(day_of_year):
    """Day of year (1..365) -> 'MM-DD' in a non-leap year."""
    date = pd.Timestamp(REFERENCE_YEAR, 1, 1) + pd.Timedelta(days=int(round(day_of_year)) - 1)
    return date.strftime("%m-%d")


def _day_of_year(date):
    """Day of year in a non-leap calendar, so 'MM-DD' compares the same every year."""
    date = pd.Timestamp(date)
    day = min(date.day, 28) if date.month == 2 else date.day  # 29 Feb -> 28 Feb
    return pd.Timestamp(REFERENCE_YEAR, date.month, day).dayofyear


# --- route days ------------------------------------------------------------

def route_days(deliveries, cutoff, min_share=ROUTE_DAY_MIN_SHARE):
    """Main route weekdays per customer (0 = Monday ... 5 = Saturday).

    A weekday counts if at least `min_share` of the customer's delivery dates fall on it.
    Returns {TRDR: [weekday, ...]} sorted, never empty for a customer with deliveries
    (falls back to the single most common weekday).
    """
    d = _past_purchases(deliveries, cutoff)[["TRDR", "TRNDATE"]].drop_duplicates()
    d = d.assign(WD=d["TRNDATE"].dt.day_of_week)
    share = d.groupby("TRDR")["WD"].value_counts(normalize=True)
    out = {}
    for trdr, s in share.groupby(level=0):
        s = s.droplevel(0)
        days = sorted(int(wd) for wd, v in s.items() if v >= min_share)
        out[int(trdr)] = days or [int(s.idxmax())]
    return out


# --- seasonality -----------------------------------------------------------

def seasonal_factors(deliveries, customers, items, cutoff):
    """Monthly factor per AREA x MTRCATEGORY.

    1. net quantity per AREA, MTRCATEGORY and calendar month (only complete months <= cutoff)
    2. average per month of the year (1..12) over the years we have
    3. divide by the average of the 12 months, so the 12 factors average to 1

    Returns a DataFrame: AREA, MTRCATEGORY, MONTH (1..12), FACTOR.
    A month with no data at all for a combination gets factor 1 before normalising.
    """
    cutoff = pd.Timestamp(cutoff)
    last_full_month = (cutoff + pd.Timedelta(days=1)).to_period("M") - 1
    d = deliveries[deliveries["TRNDATE"] <= cutoff]
    d = d.merge(customers[["TRDR", "AREA"]], on="TRDR").merge(items[["MTRL", "MTRCATEGORY"]], on="MTRL")
    d = d.assign(PERIOD=d["TRNDATE"].dt.to_period("M"))
    d = d[d["PERIOD"] <= last_full_month]

    combos = d[["AREA", "MTRCATEGORY"]].drop_duplicates()
    first = d["PERIOD"].min()
    periods = pd.period_range(first, last_full_month, freq="M")
    grid = combos.merge(pd.DataFrame({"PERIOD": periods}), how="cross")
    monthly = d.groupby(["AREA", "MTRCATEGORY", "PERIOD"], as_index=False)["QTY"].sum()
    monthly = grid.merge(monthly, how="left").fillna({"QTY": 0})
    monthly["MONTH"] = monthly["PERIOD"].dt.month

    out = monthly.groupby(["AREA", "MTRCATEGORY", "MONTH"], as_index=False)["QTY"].mean()
    out = combos.merge(pd.DataFrame({"MONTH": range(1, 13)}), how="cross").merge(out, how="left")
    out["QTY"] = out["QTY"].fillna(out.groupby(["AREA", "MTRCATEGORY"])["QTY"].transform("mean"))
    mean12 = out.groupby(["AREA", "MTRCATEGORY"])["QTY"].transform("mean")
    out["FACTOR"] = (out["QTY"] / mean12).where(mean12 > 0, 1.0)
    return out[["AREA", "MTRCATEGORY", "MONTH", "FACTOR"]].reset_index(drop=True)


# --- winter closures -------------------------------------------------------

def winter_history(deliveries, customers, cutoff):
    """Per customer, what happened in each complete winter (Dec y .. Feb y+1) before the cutoff.

    A winter is "observed" for a customer if they had deliveries in the summer before it
    (June-August of y). It is "closed" if they had no delivery from December to February.
    For closed winters we keep the last delivery before closing (Jun..Nov of y) and
    the first one after re-opening (from March of y+1, if it happened before the cutoff).

    Returns one row per customer with at least one observed winter:
    TRDR, AREA, TRDCATEGORY, N_WINTERS, N_CLOSED, IS_CLOSER, LAST_BEFORE_CLOSE, FIRST_AFTER_OPEN
    (the last two as typical 'MM-DD', median over the closed winters).
    """
    cutoff = pd.Timestamp(cutoff)
    d = _past_purchases(deliveries, cutoff)[["TRDR", "TRNDATE"]].drop_duplicates()
    rows = []
    first_year = d["TRNDATE"].dt.year.min()
    for y in range(first_year, cutoff.year):
        winter_start, winter_end = pd.Timestamp(y, 12, 1), pd.Timestamp(y + 1, 3, 1)
        if winter_end - pd.Timedelta(days=1) > cutoff:
            continue  # winter not complete before the cutoff
        summer = d[(d["TRNDATE"].dt.year == y) & d["TRNDATE"].dt.month.isin(SUMMER_MONTHS)]
        winter = d[(d["TRNDATE"] >= winter_start) & (d["TRNDATE"] < winter_end)]
        before = d[(d["TRNDATE"] >= pd.Timestamp(y, 6, 1)) & (d["TRNDATE"] < winter_start)]
        after = d[d["TRNDATE"] >= winter_end]
        last_before = before.groupby("TRDR")["TRNDATE"].max()
        first_after = after.groupby("TRDR")["TRNDATE"].min()
        winter_buyers = set(winter["TRDR"])
        for trdr in summer["TRDR"].unique():
            closed = trdr not in winter_buyers
            rows.append({
                "TRDR": trdr, "WINTER": y, "CLOSED": closed,
                "LAST_DOY": _day_of_year(last_before[trdr]) if closed else None,
                "FIRST_DOY": (_day_of_year(first_after[trdr])
                              if closed and trdr in first_after.index else None),
            })
    cols = ["TRDR", "AREA", "TRDCATEGORY", "N_WINTERS", "N_CLOSED", "IS_CLOSER",
            "LAST_BEFORE_CLOSE", "FIRST_AFTER_OPEN"]
    if not rows:
        return pd.DataFrame(columns=cols)
    w = pd.DataFrame(rows)
    h = w.groupby("TRDR").agg(N_WINTERS=("WINTER", "size"), N_CLOSED=("CLOSED", "sum"),
                              LAST_DOY=("LAST_DOY", "median"), FIRST_DOY=("FIRST_DOY", "median"))
    h = h.reset_index()
    # Closer: closed in every winter we observed (at least one).
    h["IS_CLOSER"] = (h["N_CLOSED"] == h["N_WINTERS"]) & (h["N_WINTERS"] >= 1)
    h["LAST_BEFORE_CLOSE"] = [(_month_day(v) if c and pd.notna(v) else None)
                              for v, c in zip(h["LAST_DOY"], h["IS_CLOSER"])]
    h["FIRST_AFTER_OPEN"] = [(_month_day(v) if c and pd.notna(v) else None)
                             for v, c in zip(h["FIRST_DOY"], h["IS_CLOSER"])]
    h = h.merge(customers[["TRDR", "AREA", "TRDCATEGORY"]], on="TRDR", how="left")
    return h[cols].sort_values("TRDR").reset_index(drop=True)


def winter_closers(deliveries, customers, cutoff):
    """Customers that close every winter: summer deliveries, none from December to February.

    Returns TRDR, AREA, TRDCATEGORY, N_WINTERS, LAST_BEFORE_CLOSE ('MM-DD'),
    FIRST_AFTER_OPEN ('MM-DD').
    """
    h = winter_history(deliveries, customers, cutoff)
    h = h[h["IS_CLOSER"]]
    return h[["TRDR", "AREA", "TRDCATEGORY", "N_WINTERS", "LAST_BEFORE_CLOSE",
              "FIRST_AFTER_OPEN"]].reset_index(drop=True)


def likely_closer(trdr, customers, history):
    """For a customer with no winter history: do most similar old customers close in winter?

    Similar = same AREA and TRDCATEGORY, among customers with at least one observed winter
    (`history` from winter_history). True only if more than half of them are closers.
    Returns (is_likely_closer, typical LAST_BEFORE_CLOSE, typical FIRST_AFTER_OPEN).
    """
    me = customers.loc[customers["TRDR"] == trdr].iloc[0]
    peers = history[(history["AREA"] == me["AREA"]) & (history["TRDCATEGORY"] == me["TRDCATEGORY"])
                    & (history["TRDR"] != trdr)]
    if peers.empty or peers["IS_CLOSER"].mean() <= 0.5:
        return False, None, None
    closers = peers[peers["IS_CLOSER"]]
    last = closers["LAST_BEFORE_CLOSE"].dropna().map(lambda s: _day_of_year(f"{REFERENCE_YEAR}-{s}"))
    first = closers["FIRST_AFTER_OPEN"].dropna().map(lambda s: _day_of_year(f"{REFERENCE_YEAR}-{s}"))
    return (True,
            _month_day(last.median()) if len(last) else None,
            _month_day(first.median()) if len(first) else None)


def is_closed_on(date, last_before_close, first_after_open):
    """True if `date` falls in the closed season: on/after the typical last delivery
    before closing, or before the typical first delivery after re-opening.

    The cutoff day itself counts as closed when it equals the typical last delivery day:
    that delivery is the stock for the closing, not for normal consumption.
    """
    if last_before_close is None:
        return False
    doy = _day_of_year(date)
    last = _day_of_year(f"{REFERENCE_YEAR}-{last_before_close}")
    first = (_day_of_year(f"{REFERENCE_YEAR}-{first_after_open}")
             if first_after_open else _day_of_year(f"{REFERENCE_YEAR}-04-01"))
    return doy >= last or doy < first
