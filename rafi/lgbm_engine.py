"""Hybrid forecast: the rafi forecast plus a LightGBM correction learned from the past.

1. Run the rafi forecast at the cutoff (the explainable "physical" estimate).
2. Go back in time: at earlier virtual cutoffs (all at least 90 days before the real one)
   run the same forecast and note when each pair was really delivered next.
   Only data up to the real cutoff is used, so nothing from the future leaks in.
3. Train LightGBM with an L1 (MAE) loss to predict the days to the next delivery from
   the rafi numbers plus a few order-rhythm features (days since the last delivery,
   usual interval, route days, promo, customer and item category).
4. Final days = w x LightGBM + (1 - w) x rafi, with w = 0.55 (less for new customers).
   Winter closures and pairs without a rate keep the rafi answer (blank).

The output has the same columns as rafi.forecast.forecast plus RAFI_DEPLETION_DATE
(the date before the correction) and MODEL_CORRECTION_DAYS (how many days the model
moved it), so every date can still be explained.
If lightgbm is not installed, the plain rafi forecast is returned with a warning.
"""
import math
import warnings

import numpy as np
import pandas as pd

from rafi.evaluate import make_targets
from rafi.features import next_working_day, route_days, working_day
from rafi.forecast import HORIZON_DAYS, K_DELIVERIES, forecast
from rafi.ingest import daily_qty

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:  # optional dependency: fall back to plain rafi
    lgb = None
    HAS_LGBM = False

BLEND_WEIGHT = 0.55          # share of LightGBM in the final number of days
COLD_START_SHARE = 0.6       # new customers: w x 0.6, trust rafi's peer-based rate more
CENSORED_DAYS = HORIZON_DAYS + 12   # training label for "no delivery within 90 days"
CLOSED_DAYS = HORIZON_DAYS + 25     # feature value for a customer closed for winter
SNAPSHOT_STEP_DAYS = 28
MAX_SNAPSHOTS = 8
MIN_HISTORY_DAYS = 300       # a virtual cutoff needs this much history before it

LGBM_PARAMS = {
    "objective": "regression_l1",
    "learning_rate": 0.035,
    "num_leaves": 23,
    "max_depth": 5,
    "min_data_in_leaf": 25,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "feature_fraction": 0.85,
    "lambda_l1": 1.5,
    "lambda_l2": 2.0,
    "seed": 42,
    "verbose": -1,
}
NUM_ROUNDS = 220

CAT_COLS = ["AREA_CAT", "TRDCATEGORY_CAT", "MTRCATEGORY_CAT", "MTRUNIT1_CAT", "REASON_CAT"]
FEATURE_COLS = [
    "rafi_days", "base_rate", "seasonal_factor_now", "stock_at_cutoff", "last_delivery_qty",
    "elapsed_days", "n_deliveries", "confidence", "median_interval", "last_interval",
    "cv_interval", "elapsed_to_median_ratio", "expected_total_interval", "last_free_ratio",
    "last_had_promo", "route_days_per_week", "days_to_route_day", "cutoff_month_sin",
    "cutoff_month_cos", "is_new_customer",
] + CAT_COLS

# Reason codes whose date the model may move (closures and "no rate" stay blank).
MOVABLE = ("ΚΑΝΟΝΙΚΗ", "ΗΔΗ_ΕΞΑΝΤΛΗΜΕΝΟ", "ΕΚΤΟΣ_ΟΡΙΖΟΝΤΑ", "ΝΕΟΣ_ΠΕΛΑΤΗΣ")


# --- features ----------------------------------------------------------------

def days_to_next_delivery(deliveries, targets, cutoff):
    """Training label: days from the cutoff to the first real delivery of each pair.

    CENSORED_DAYS if there is none within the 90-day horizon.
    """
    cutoff = pd.Timestamp(cutoff)
    end = cutoff + pd.Timedelta(days=HORIZON_DAYS)
    future = deliveries[(deliveries["TRNDATE"] > cutoff) & (deliveries["TRNDATE"] < end)
                        & ~deliveries["IS_CREDIT"] & (deliveries["QTY"] > 0)]
    first = future.groupby(["TRDR", "MTRL"])["TRNDATE"].min().rename("NEXT").reset_index()
    df = targets[["TRDR", "MTRL"]].merge(first, on=["TRDR", "MTRL"], how="left")
    return (df["NEXT"] - cutoff).dt.days.astype(float).fillna(float(CENSORED_DAYS)).to_numpy()


def _interval_stats(deliveries, cutoff, k=K_DELIVERIES):
    """{(TRDR, MTRL): (median interval, last interval, CV of intervals,
    free share of the last delivery, last delivery had a promo discount)}."""
    d = deliveries[(deliveries["TRNDATE"] <= cutoff) & ~deliveries["IS_CREDIT"]
                   & (deliveries["QTY"] > 0)]
    d = d.groupby(["TRDR", "MTRL", "TRNDATE"], as_index=False).agg(
        QTY=("QTY", "sum"), QTY_FREE=("QTY_FREE", "sum"), PROMO_DISC=("PROMO_DISC", "any"))
    stats = {}
    for key, g in d.sort_values("TRNDATE").groupby(["TRDR", "MTRL"]):
        qty, free = g["QTY"].to_numpy(float), g["QTY_FREE"].to_numpy(float)
        free_share = float(free[-1] / qty[-1]) if qty[-1] > 0 else 0.0
        promo = float(g["PROMO_DISC"].iloc[-1])
        if len(g) >= 2:
            gaps = np.diff(g["TRNDATE"].to_numpy()[-k:]) / np.timedelta64(1, "D")
            mean = gaps.mean()
            stats[key] = (float(np.median(gaps)), float(gaps[-1]),
                          float(gaps.std() / mean) if mean > 0 else 1.0, free_share, promo)
        else:
            stats[key] = (14.0, 14.0, 1.0, free_share, promo)
    return stats


def _days_to_route_day(date, weekdays):
    """Days from `date` to the next working day that is one of the customer's route days."""
    if pd.isna(date) or not weekdays:
        return 0.0
    d = pd.Timestamp(date).normalize()
    for offset in range(8):
        cand = d + pd.Timedelta(days=offset)
        if cand.weekday() in weekdays and working_day(cand):
            return float(offset)
    return 0.0


def build_features(base, deliveries, customers, items, cutoff):
    """One feature row per row of `base` (the output of rafi.forecast.forecast)."""
    cutoff = pd.Timestamp(cutoff)
    routes = route_days(deliveries, cutoff)
    stats = _interval_stats(deliveries, cutoff)
    cust = customers.set_index("TRDR")
    item = items.set_index("MTRL")
    ins = pd.to_datetime(customers["INSDATE"])
    new = set(customers.loc[(ins.dt.year == cutoff.year) & (ins <= cutoff), "TRDR"])

    pred = pd.to_datetime(base["PREDICTED_DEPLETION_DATE"])
    rafi_days = (pred - cutoff).dt.days.astype(float).fillna(float(CENSORED_DAYS))
    rafi_days = rafi_days.where(base["REASON_CODE"] != "ΚΛΕΙΣΤΟ_ΧΕΙΜΩΝΑ", float(CLOSED_DAYS))
    elapsed = (cutoff - pd.to_datetime(base["LAST_DELIVERY_DATE"])).dt.days.astype(float)
    elapsed = elapsed.fillna(60.0).to_numpy()

    s = np.array([stats.get((t, m), (14.0, 14.0, 1.0, 0.0, 0.0))
                  for t, m in zip(base["TRDR"], base["MTRL"])], dtype=float).reshape(-1, 5)
    rd = [routes.get(int(t), []) for t in base["TRDR"]]

    rate = (base["BASE_RATE"].fillna(0.0) * base["SEASONAL_FACTOR_NOW"].fillna(1.0)).to_numpy()
    last_qty = base["LAST_DELIVERY_QTY"].fillna(0.0).to_numpy()
    safe_rate = np.where(rate > 1e-6, rate, 1.0)
    angle = 2.0 * math.pi * (cutoff.month - 1) / 12.0

    X = pd.DataFrame({
        "rafi_days": rafi_days.to_numpy(),
        "base_rate": base["BASE_RATE"].fillna(0.0).to_numpy(float),
        "seasonal_factor_now": base["SEASONAL_FACTOR_NOW"].fillna(1.0).to_numpy(float),
        "stock_at_cutoff": base["STOCK_AT_CUTOFF"].fillna(0.0).to_numpy(float),
        "last_delivery_qty": last_qty,
        "elapsed_days": elapsed,
        "n_deliveries": base["N_DELIVERIES"].fillna(0).to_numpy(float),
        "confidence": base["CONFIDENCE"].fillna(0.0).to_numpy(float),
        "median_interval": s[:, 0],
        "last_interval": s[:, 1],
        "cv_interval": s[:, 2],
        "elapsed_to_median_ratio": np.where(s[:, 0] > 0, elapsed / np.where(s[:, 0] > 0, s[:, 0], 1), 1.0),
        "expected_total_interval": np.where(rate > 1e-6, np.minimum(150.0, last_qty / safe_rate), 100.0),
        "last_free_ratio": s[:, 3],
        "last_had_promo": s[:, 4],
        "route_days_per_week": [float(len(r)) for r in rd],
        "days_to_route_day": [_days_to_route_day(p, r) for p, r in zip(pred, rd)],
        "cutoff_month_sin": math.sin(angle),
        "cutoff_month_cos": math.cos(angle),
        "is_new_customer": base["TRDR"].isin(new).to_numpy(float),
        "AREA_CAT": base["TRDR"].map(cust["AREA"]).to_numpy(),
        "TRDCATEGORY_CAT": base["TRDR"].map(cust["TRDCATEGORY"]).to_numpy(),
        "MTRCATEGORY_CAT": base["MTRL"].map(item["MTRCATEGORY"]).to_numpy(),
        "MTRUNIT1_CAT": base["MTRL"].map(item["MTRUNIT1"]).to_numpy(),
        "REASON_CAT": base["REASON_CODE"].fillna("ΑΓΝΩΣΤΟ").to_numpy(),
    })
    return X[FEATURE_COLS]


# --- training ----------------------------------------------------------------

def virtual_cutoffs(first_date, cutoff, step_days=SNAPSHOT_STEP_DAYS, max_snapshots=MAX_SNAPSHOTS):
    """Past cutoffs for training, each with its whole 90-day horizon before `cutoff`."""
    latest = pd.Timestamp(cutoff) - pd.Timedelta(days=HORIZON_DAYS)
    earliest = pd.Timestamp(first_date) + pd.Timedelta(days=MIN_HISTORY_DAYS)
    cuts = []
    while latest >= earliest and len(cuts) < max_snapshots:
        cuts.append(latest)
        latest -= pd.Timedelta(days=step_days)
    return sorted(cuts)


def training_set(deliveries, customers, items, cutoff, k=K_DELIVERIES):
    """Features, labels and weights from all virtual cutoffs (data up to `cutoff` only)."""
    deliveries = deliveries[deliveries["TRNDATE"] <= pd.Timestamp(cutoff)]
    cuts = virtual_cutoffs(deliveries["TRNDATE"].min(), cutoff)
    X, y, w = [], [], []
    for i, vc in enumerate(cuts):
        past = deliveries[deliveries["TRNDATE"] <= vc]
        targets = make_targets(past, customers, vc)
        if targets.empty:
            continue
        base = forecast(daily_qty(past), past, customers, items, targets, vc, k=k)
        keep = (base["REASON_CODE"] != "ΚΛΕΙΣΤΟ_ΧΕΙΜΩΝΑ").to_numpy()
        if not keep.any():
            continue
        base, targets = base[keep].reset_index(drop=True), targets[keep].reset_index(drop=True)
        X.append(build_features(base, past, customers, items, vc))
        y.append(days_to_next_delivery(deliveries, targets, vc))
        # recent snapshots count a bit more (0.7 for the oldest, 1.0 for the newest)
        w.append(np.full(len(base), 0.7 + 0.3 * i / max(1, len(cuts) - 1)))
    if not X:
        return None, None, None
    return pd.concat(X, ignore_index=True), np.concatenate(y), np.concatenate(w)


def _as_categories(train, test):
    """Same category list for train and test, so LightGBM codes them the same way."""
    train, test = train.copy(), test.copy()
    for col in CAT_COLS:
        cats = sorted(set(train[col].dropna()) | set(test[col].dropna()), key=str)
        train[col] = pd.Categorical(train[col], categories=cats)
        test[col] = pd.Categorical(test[col], categories=cats)
    return train, test


# --- forecast ----------------------------------------------------------------

def _to_date(days, cutoff):
    """Days after the cutoff -> a working day inside the horizon, or NaT."""
    days = max(1, int(round(days)))
    if days >= HORIZON_DAYS:
        return pd.NaT
    d = cutoff + pd.Timedelta(days=days)
    if not working_day(d):
        d = next_working_day(d)
    return d if (d - cutoff).days < HORIZON_DAYS else pd.NaT


def _reason(old, date, cutoff):
    """Keep REASON_CODE consistent with the corrected date."""
    if old not in ("ΚΑΝΟΝΙΚΗ", "ΗΔΗ_ΕΞΑΝΤΛΗΜΕΝΟ", "ΕΚΤΟΣ_ΟΡΙΖΟΝΤΑ"):
        return old
    if pd.isna(date):
        return "ΕΚΤΟΣ_ΟΡΙΖΟΝΤΑ"
    if old == "ΗΔΗ_ΕΞΑΝΤΛΗΜΕΝΟ" and date == cutoff + pd.Timedelta(days=1):
        return old
    return "ΚΑΝΟΝΙΚΗ"


def forecast_hybrid(daily, deliveries, customers, items, targets, cutoff, k=K_DELIVERIES,
                    blend_weight=BLEND_WEIGHT):
    """rafi forecast corrected by LightGBM. Same columns as forecast() plus
    RAFI_DEPLETION_DATE and MODEL_CORRECTION_DAYS. blend_weight=1 gives pure LightGBM."""
    cutoff = pd.Timestamp(cutoff)
    deliveries = deliveries[deliveries["TRNDATE"] <= cutoff]
    daily = daily[daily["TRNDATE"] <= cutoff]
    base = forecast(daily, deliveries, customers, items, targets, cutoff, k=k)
    out = base.copy()
    out["RAFI_DEPLETION_DATE"] = base["PREDICTED_DEPLETION_DATE"]
    out["MODEL_CORRECTION_DAYS"] = 0.0

    if not HAS_LGBM:
        warnings.warn("Το lightgbm δεν είναι εγκατεστημένο· χρησιμοποιείται μόνο το rafi.")
        return out
    X_train, y, w = training_set(deliveries, customers, items, cutoff, k=k)
    if X_train is None:
        warnings.warn("Λίγο ιστορικό για εκπαίδευση LightGBM· χρησιμοποιείται μόνο το rafi.")
        return out

    X_test = build_features(base, deliveries, customers, items, cutoff)
    X_train, X_test = _as_categories(X_train, X_test)
    data = lgb.Dataset(X_train, y, weight=w, categorical_feature=CAT_COLS, free_raw_data=False)
    model = lgb.train(LGBM_PARAMS, data, num_boost_round=NUM_ROUNDS)

    rafi_days = X_test["rafi_days"].to_numpy()
    cold = (base["REASON_CODE"] == "ΝΕΟΣ_ΠΕΛΑΤΗΣ").to_numpy()
    weight = np.where(cold, blend_weight * COLD_START_SHARE, blend_weight)
    days = weight * model.predict(X_test) + (1 - weight) * rafi_days

    movable = base["REASON_CODE"].isin(MOVABLE).to_numpy()
    dates = [(_to_date(d, cutoff) if m else old)
             for d, m, old in zip(days, movable, base["PREDICTED_DEPLETION_DATE"])]
    out["PREDICTED_DEPLETION_DATE"] = pd.to_datetime(pd.Series(dates, dtype="datetime64[ns]"))
    out["REASON_CODE"] = [_reason(r, d, cutoff) for r, d in
                          zip(base["REASON_CODE"], out["PREDICTED_DEPLETION_DATE"])]
    out["MODEL_CORRECTION_DAYS"] = ((out["PREDICTED_DEPLETION_DATE"] - out["RAFI_DEPLETION_DATE"])
                                    .dt.days.astype(float))
    return out
