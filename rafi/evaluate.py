"""Official score, target rule, proxy truth and time-travel backtest.

Run `python -m rafi.evaluate` to compare the baselines on two past cutoffs.
"""
from functools import lru_cache

import pandas as pd

from rafi.baseline import plus30, rolling180
from rafi.forecast import K_DELIVERIES, forecast
from rafi.ingest import build_deliveries, daily_qty, load_from_files

HORIZON_DAYS = 90
AT_RISK_DAYS = 14
MISSING_PENALTY = 60
KEYS = ["TRDR", "MTRL"]


def _rafi(k=K_DELIVERIES, **switches):
    """The rafi forecast as a method; `switches` turn steps off for the ablation table."""
    def method(daily, targets, cutoff, deliveries, customers, items, **_):
        return forecast(daily, deliveries, customers, items, targets, cutoff, k=k, **switches)
    return method


# Every method has the same signature: (daily, targets, cutoff, **ctx) -> predictions.
# ctx holds deliveries, customers and items (all cut at the cutoff); baselines ignore it.
METHODS = {
    "plus30": lambda daily, targets, cutoff, **_: plus30(targets, cutoff),
    "rolling180": lambda daily, targets, cutoff, **_: rolling180(daily, targets, cutoff),
    "rafi": _rafi(),
}

# Ablation: each row adds one step to the one before; the last row is a rejected experiment.
ABLATION = {
    "rolling180": METHODS["rolling180"],
    "rafi: μόνο ρυθμός": _rafi(use_seasonality=False, use_closers=False, use_cold_start=False),
    "+ εποχικότητα": _rafi(use_closers=False, use_cold_start=False,
                           factors_without_closers=False),
    "+ κλεισίματα": _rafi(use_cold_start=False, factors_without_closers=False),
    "+ cold start": _rafi(factors_without_closers=False),
    # One extra idea, kept only if it lowers MAE on both cutoffs (it does not).
    "(δοκιμή) συντελ. χωρίς κλεισίματα": _rafi(factors_without_closers=True),
}
K_CANDIDATES = (3, 4, 6)
CUTOFFS = ("2025-10-31", "2026-06-30")


def _parse_one(value):
    """Return (date, readable). Blank is readable (NaT); text must be YYYY-MM-DD."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return pd.NaT, True
    if isinstance(value, str):
        if not value.strip():
            return pd.NaT, True
        try:
            return pd.to_datetime(value.strip(), format="%Y-%m-%d"), True
        except ValueError:
            return pd.NaT, False
    try:
        return pd.Timestamp(value), True
    except (TypeError, ValueError):
        return pd.NaT, False


def _parse(values):
    parsed = [_parse_one(v) for v in values]
    dates = pd.Series([d for d, _ in parsed], dtype="datetime64[ns]")
    readable = pd.Series([ok for _, ok in parsed], dtype=bool)
    return dates, readable


def score(predictions, truth, cutoff, horizon_days=HORIZON_DAYS):
    """Official per-pair error, plus summary metrics.

    predictions: TRDR, MTRL, PREDICTED_DEPLETION_DATE (Timestamp, 'YYYY-MM-DD' or blank)
    truth:       TRDR, MTRL, TRUE_DEPLETION_DATE (Timestamp or blank)
    horizon_end = cutoff + 90. A date >= horizon_end counts as blank (both sides).

    Per pair:
      missing pair or unreadable date          -> 60
      both are dates                           -> |pred - true|
      true blank, pred blank                   -> 0
      true blank, pred date                    -> horizon_end - pred
      true date,  pred blank                   -> horizon_end - true

    Returns (table, metrics). Precision/recall are for "At Risk": a date <= cutoff + 14.
    """
    cutoff = pd.Timestamp(cutoff)
    horizon_end = cutoff + pd.Timedelta(days=horizon_days)
    at_risk_end = cutoff + pd.Timedelta(days=AT_RISK_DAYS)

    pred = predictions[KEYS + ["PREDICTED_DEPLETION_DATE"]].drop_duplicates(KEYS)
    df = truth[KEYS + ["TRUE_DEPLETION_DATE"]].reset_index(drop=True)
    df = df.merge(pred, on=KEYS, how="left", indicator=True)

    true_date, _ = _parse(df["TRUE_DEPLETION_DATE"])
    pred_date, readable = _parse(df["PREDICTED_DEPLETION_DATE"])
    true_date = true_date.where(true_date < horizon_end)
    pred_date = pred_date.where(pred_date < horizon_end)
    missing = (df["_merge"] == "left_only") | ~readable

    days = lambda x: x.dt.days.astype("float")
    error = pd.Series(0.0, index=df.index)
    both = true_date.notna() & pred_date.notna()
    error[both] = days((pred_date - true_date).abs())[both]
    only_pred = true_date.isna() & pred_date.notna()
    error[only_pred] = days(horizon_end - pred_date)[only_pred]
    only_true = true_date.notna() & pred_date.isna()
    error[only_true] = days(horizon_end - true_date)[only_true]
    error[missing] = MISSING_PENALTY

    both &= ~missing
    table = df[KEYS].copy()
    table["TRUE_DATE"] = true_date
    table["PRED_DATE"] = pred_date.where(~missing)
    table["MISSING"] = missing
    table["ERROR"] = error

    pred_risk = (table["PRED_DATE"] <= at_risk_end).fillna(False)
    true_risk = (table["TRUE_DATE"] <= at_risk_end).fillna(False)
    tp = int((pred_risk & true_risk).sum())
    metrics = {
        "n": len(table),
        "MAE": float(error.mean()) if len(table) else float("nan"),
        "n_both_dates": int(both.sum()),
        "within_3": float((error[both] <= 3).mean()) if both.any() else float("nan"),
        "within_7": float((error[both] <= 7).mean()) if both.any() else float("nan"),
        "precision": tp / int(pred_risk.sum()) if pred_risk.any() else float("nan"),
        "recall": tp / int(true_risk.sum()) if true_risk.any() else float("nan"),
    }
    return table, metrics


def make_targets(deliveries, customers, cutoff):
    """Pairs to forecast at a cutoff (the rule behind prediction_targets.csv).

    - at least 3 purchases and last purchase within the 120 days before the cutoff, or
    - customer registered in the cutoff's calendar year (up to the cutoff), at least 1 purchase.
    A purchase is a line of a non-cancelled invoice (cancelled ones are already gone
    from `deliveries`); credit notes do not count.
    """
    cutoff = pd.Timestamp(cutoff)
    buys = deliveries[(deliveries["TRNDATE"] <= cutoff) & ~deliveries["IS_CREDIT"]
                      & (deliveries["QTY"] > 0)]
    g = buys.groupby(KEYS).agg(n=("FINDOC", "nunique"), last=("TRNDATE", "max")).reset_index()

    ins = pd.to_datetime(customers["INSDATE"])
    new = customers.loc[(ins.dt.year == cutoff.year) & (ins <= cutoff), "TRDR"]

    regular = (g["n"] >= 3) & (g["last"] >= cutoff - pd.Timedelta(days=120))
    keep = regular | g["TRDR"].isin(new)
    return g.loc[keep, KEYS].sort_values(KEYS).reset_index(drop=True)


def proxy_truth(deliveries, targets, cutoff, horizon_days=HORIZON_DAYS):
    """Backtest "truth": the first real delivery (positive QTY, not a credit) after the cutoff.

    Blank (NaT) if there is none before horizon_end = cutoff + 90.
    This is an approximation for comparing methods, NOT the official truth:
    a customer reorders some time before running out, not on the exact day.
    """
    cutoff = pd.Timestamp(cutoff)
    horizon_end = cutoff + pd.Timedelta(days=horizon_days)
    future = deliveries[(deliveries["TRNDATE"] > cutoff) & (deliveries["TRNDATE"] < horizon_end)
                        & ~deliveries["IS_CREDIT"] & (deliveries["QTY"] > 0)]
    first = future.groupby(KEYS)["TRNDATE"].min().rename("TRUE_DEPLETION_DATE").reset_index()
    out = targets[KEYS].merge(first, on=KEYS, how="left")
    return out


@lru_cache(maxsize=1)
def _load(data_dir="data"):
    headers, lines, items, customers = load_from_files(data_dir)
    return build_deliveries(headers, lines, items), customers, items


def _metrics_by(table, customers, column):
    df = table.merge(customers[["TRDR", column]], on="TRDR", how="left")
    return (df.groupby(column)["ERROR"].agg(n="size", MAE="mean")
              .reset_index().sort_values(column).reset_index(drop=True))


def backtest(method, cutoff, data_dir="data"):
    """Cut the data at `cutoff`, build targets, run the method, score against proxy_truth.

    `method` is a name from METHODS or a function (daily, targets, cutoff, **ctx) -> predictions,
    where ctx = deliveries, customers, items (deliveries cut at the cutoff).
    Returns a dict: overall metrics, per-pair table, and MAE by AREA, TRDCATEGORY, new/old.
    """
    fn = METHODS[method] if isinstance(method, str) else method
    cutoff = pd.Timestamp(cutoff)
    deliveries, customers, items = _load(data_dir)

    past = deliveries[deliveries["TRNDATE"] <= cutoff]
    targets = make_targets(past, customers, cutoff)
    predictions = fn(daily_qty(past), targets, cutoff,
                     deliveries=past, customers=customers, items=items)
    truth = proxy_truth(deliveries, targets, cutoff)
    table, metrics = score(predictions, truth, cutoff)

    cust = customers.copy()
    ins = pd.to_datetime(cust["INSDATE"])
    cust["NEW_CUSTOMER"] = (ins.dt.year == cutoff.year).map({True: "νέοι", False: "παλιοί"})
    return {
        "metrics": metrics,
        "table": table,
        "by_area": _metrics_by(table, cust, "AREA"),
        "by_category": _metrics_by(table, cust, "TRDCATEGORY"),
        "by_new": _metrics_by(table, cust, "NEW_CUSTOMER"),
    }


def _pct(x):
    return "   -" if pd.isna(x) else f"{100 * x:4.0f}%"


def _row(r):
    """One line of the ablation table: overall, by area, new/old, and the other metrics."""
    k = r["metrics"]
    row = {"MAE": k["MAE"]}
    row.update(r["by_area"].set_index("AREA")["MAE"].to_dict())
    row.update(r["by_new"].set_index("NEW_CUSTOMER")["MAE"].to_dict())
    row.update({"±3": _pct(k["within_3"]), "±7": _pct(k["within_7"]),
                "Precision": _pct(k["precision"]), "Recall": _pct(k["recall"])})
    return row


def _table(rows):
    df = pd.DataFrame(rows).T
    num = [c for c in df.columns if c not in ("±3", "±7", "Precision", "Recall")]
    df[num] = df[num].astype(float).round(1)
    return df.fillna("-").to_string()


def main():
    print("Ράφι · backtest με «αλήθεια» = επόμενη πραγματική παράδοση (προσέγγιση)")
    print(f"{'Μέθοδος':<11} {'Cutoff':<10} {'n':>5} {'MAE':>6} {'±3':>5} {'±7':>5} "
          f"{'Precision':>9} {'Recall':>6}")
    for m in METHODS:
        for c in CUTOFFS:
            k = backtest(m, c)["metrics"]
            print(f"{m:<11} {c:<10} {k['n']:>5} {k['MAE']:>6.1f} {_pct(k['within_3']):>5} "
                  f"{_pct(k['within_7']):>5} {_pct(k['precision']):>9} {_pct(k['recall']):>6}")

    for c in CUTOFFS:
        print(f"\nABLATION · cutoff {c} (κάθε γραμμή προσθέτει ένα βήμα· MAE σε ημέρες)")
        print(_table({name: _row(backtest(fn, c)) for name, fn in ABLATION.items()}))

    print("\nΔοκιμή k (πλήθος πρόσφατων παραδόσεων για τον ρυθμό) · MAE")
    rows = {}
    for k in K_CANDIDATES:
        maes = [backtest(_rafi(k=k), c)["metrics"]["MAE"] for c in CUTOFFS]
        rows[f"k = {k}"] = dict(zip(CUTOFFS, maes), **{"μέσος όρος": sum(maes) / len(maes)})
    df = pd.DataFrame(rows).T.round(2)
    print(df.to_string())
    print(f"Καλύτερο στον μέσο όρο: {df['μέσος όρος'].idxmin()} (σε χρήση: k = {K_DELIVERIES})")

    print("\nrafi · MAE ανά περιοχή και τύπο πελάτη")
    cols = {}
    for c in CUTOFFS:
        r = backtest("rafi", c)
        _, customers, _ = _load()
        t = r["table"].merge(customers[["TRDR", "AREA", "TRDCATEGORY"]], on="TRDR")
        cols[c] = t.groupby(["AREA", "TRDCATEGORY"])["ERROR"].mean()
    print(pd.DataFrame(cols).round(1).to_string())


if __name__ == "__main__":
    main()
