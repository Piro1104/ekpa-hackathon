"""Load and clean the sales invoices (SoftOne schema), from CSV files or the mock ERP API.

Both sources return the same DataFrames, so everything after loading is shared.
Run `python -m rafi.ingest` (or `python -m rafi.ingest --api`) for a short summary.
"""
import argparse
from pathlib import Path

import pandas as pd
import requests

CREDIT_SERIES = 7071  # credit notes (returns): positive QTY1 in the file, but goods came back
SHIPPING_MTRTYPE = 1  # service items (shipping, MTRL 5081)

# Column order and types, taken from the CSV files. The API output is cast to the same schema.
HEADER_SCHEMA = {
    "FINDOC": "int64", "COMPANY": "int64", "SOSOURCE": "int64", "SERIES": "int64",
    "FINCODE": "str", "TRNDATE": "str", "TRDR": "int64", "SALESMAN": "int64",
    "ISCANCEL": "int64", "NETAMNT": "float64", "VATAMNT": "float64", "SUMAMNT": "float64",
    "COMMENTS": "str",
}
LINE_SCHEMA = {
    "FINDOC": "int64", "MTRLINES": "int64", "MTRL": "int64", "QTY1": "int64",
    "PRICE": "float64", "DISC1PRC": "int64", "LINEVAL": "float64", "VAT": "int64",
}
ITEM_SCHEMA = {
    "MTRL": "int64", "CODE": "str", "NAME": "str", "MTRCATEGORY": "str",
    "MTRTYPE": "int64", "MTRUNIT1": "str", "PRICER": "float64",
}
CUSTOMER_SCHEMA = {
    "TRDR": "int64", "CODE": "str", "NAME": "str", "TRDCATEGORY": "str", "AREA": "str",
    "CITY": "str", "SALESMAN": "int64", "INSDATE": "str", "ISACTIVE": "int64",
}

# Fields that come only from the invoice header, never from the lines.
HEADER_FIELDS = ["FINDOC", "TRNDATE", "TRDR", "SERIES", "ISCANCEL", "SALESMAN"]

DELIVERY_COLUMNS = ["FINDOC", "TRNDATE", "TRDR", "SALESMAN", "MTRL",
                    "QTY", "QTY_FREE", "IS_CREDIT", "LINEVAL", "PROMO_DISC"]


def _conform(df, schema, sort_by):
    """Same columns, order, types and row order, whatever the source."""
    df = df[list(schema)].copy()
    for col, dtype in schema.items():
        if dtype == "str":
            # Empty text (API) and missing value (CSV) mean the same thing.
            df[col] = df[col].astype("str").replace({"": None, "nan": None, "None": None})
        else:
            df[col] = df[col].astype(dtype)
    return df.sort_values(sort_by).reset_index(drop=True)


def load_from_files(data_dir="data"):
    """Read the CSV exports. They start with a UTF-8 BOM, hence utf-8-sig."""
    data_dir = Path(data_dir)

    def read(name):
        return pd.read_csv(data_dir / name, encoding="utf-8-sig")

    headers = _conform(read("SALDOC.csv"), HEADER_SCHEMA, ["FINDOC"])
    lines = _conform(read("MTRLINES.csv"), LINE_SCHEMA, ["FINDOC", "MTRLINES"])
    items = _conform(read("MTRL.csv"), ITEM_SCHEMA, ["MTRL"])
    customers = _conform(read("TRDR.csv"), CUSTOMER_SCHEMA, ["TRDR"])
    return headers, lines, items, customers


def load_from_api(base_url="http://localhost:8080", page_size=1000, timeout=30):
    """Read the same data from the ERP API.

    GET /api/saldoc is paged; we keep asking until we have `total` invoices.
    Each invoice carries its lines in LINES; we unfold them into a separate
    table with a FINDOC column, exactly like MTRLINES.csv.
    """
    base_url = base_url.rstrip("/")
    session = requests.Session()

    def get(path, **params):
        r = session.get(f"{base_url}{path}", params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()

    header_rows, line_rows = [], []
    page, total = 1, None
    while total is None or len(header_rows) < total:
        body = get("/api/saldoc", page=page, pageSize=page_size)
        total = body["total"]
        if not body["items"]:
            break  # defensive: never loop forever if total and pages disagree
        for doc in body["items"]:
            doc = dict(doc)
            for line in doc.pop("LINES", []):
                line_rows.append({"FINDOC": doc["FINDOC"], **line})
            header_rows.append(doc)
        page += 1
    if len(header_rows) != total:
        raise RuntimeError(f"API returned {len(header_rows)} invoices, expected {total}")

    headers = _conform(pd.DataFrame(header_rows), HEADER_SCHEMA, ["FINDOC"])
    lines = _conform(pd.DataFrame(line_rows), LINE_SCHEMA, ["FINDOC", "MTRLINES"])
    items = _conform(pd.DataFrame(get("/api/items")), ITEM_SCHEMA, ["MTRL"])
    customers = _conform(pd.DataFrame(get("/api/customers")), CUSTOMER_SCHEMA, ["TRDR"])
    return headers, lines, items, customers


def build_deliveries(headers, lines, items):
    """One clean row per (FINDOC, MTRL).

    - join only on FINDOC; date, customer, series, cancel flag and salesman come from the header
    - cancelled invoices (ISCANCEL = 1) are dropped with all their lines
    - shipping lines (MTRTYPE = 1) are dropped
    - free goods (PRICE = 0 and DISC1PRC = 100) count in QTY and are also shown in QTY_FREE
    - same item twice in an invoice: quantities and values are summed
    - credit notes (SERIES = 7071, never inferred from COMMENTS): QTY and LINEVAL become negative
    """
    hdr = headers.loc[headers["ISCANCEL"] != 1, HEADER_FIELDS]
    shipping = items.loc[items["MTRTYPE"] == SHIPPING_MTRTYPE, "MTRL"]

    ln = lines.loc[~lines["MTRL"].isin(shipping),
                   ["FINDOC", "MTRLINES", "MTRL", "QTY1", "PRICE", "DISC1PRC", "LINEVAL"]]
    df = ln.merge(hdr, on="FINDOC", how="inner", validate="many_to_one")

    is_free = (df["PRICE"] == 0) & (df["DISC1PRC"] == 100)
    df["QTY_FREE"] = df["QTY1"].where(is_free, 0)
    df["PROMO_DISC"] = (df["DISC1PRC"] > 0) & (df["DISC1PRC"] < 100)

    out = (df.groupby(["FINDOC", "MTRL"], as_index=False)
             .agg(TRNDATE=("TRNDATE", "first"), TRDR=("TRDR", "first"),
                  SALESMAN=("SALESMAN", "first"), SERIES=("SERIES", "first"),
                  QTY=("QTY1", "sum"), QTY_FREE=("QTY_FREE", "sum"),
                  LINEVAL=("LINEVAL", "sum"), PROMO_DISC=("PROMO_DISC", "any")))

    out["IS_CREDIT"] = out["SERIES"] == CREDIT_SERIES
    sign = out["IS_CREDIT"].map({True: -1, False: 1})
    out["QTY"] = out["QTY"] * sign
    out["QTY_FREE"] = out["QTY_FREE"] * sign
    out["LINEVAL"] = (out["LINEVAL"] * sign).round(2)
    out["TRNDATE"] = pd.to_datetime(out["TRNDATE"], format="%Y-%m-%d")

    return out[DELIVERY_COLUMNS].sort_values(["FINDOC", "MTRL"]).reset_index(drop=True)


def daily_qty(deliveries):
    """Net quantity per (TRDR, MTRL, TRNDATE). Credits already carry a negative QTY."""
    return (deliveries.groupby(["TRDR", "MTRL", "TRNDATE"], as_index=False)
                      .agg(QTY=("QTY", "sum"), QTY_FREE=("QTY_FREE", "sum")))


def summarize(headers, lines, items, deliveries):
    """Counts per cleaning rule, for the console summary and for checks."""
    cancelled = set(headers.loc[headers["ISCANCEL"] == 1, "FINDOC"])
    credits = set(headers.loc[headers["SERIES"] == CREDIT_SERIES, "FINDOC"])
    shipping = set(items.loc[items["MTRTYPE"] == SHIPPING_MTRTYPE, "MTRL"])
    in_cancelled = lines["FINDOC"].isin(cancelled)
    kept = lines[~in_cancelled & ~lines["MTRL"].isin(shipping)]
    return {
        "invoices": len(headers),
        "lines": len(lines),
        "cancelled_invoices": len(cancelled),
        "cancelled_lines": int(in_cancelled.sum()),
        "credit_invoices": len(credits),
        "credit_rows": int(deliveries["IS_CREDIT"].sum()),
        "free_lines": int(((lines["PRICE"] == 0) & (lines["DISC1PRC"] == 100)).sum()),
        "shipping_lines": int(lines["MTRL"].isin(shipping).sum()),
        "merged_lines": len(kept) - len(deliveries),
        "delivery_rows": len(deliveries),
        "orphan_lines": int((~lines["FINDOC"].isin(headers["FINDOC"])).sum()),
    }


def _print_summary(s):
    print("Ράφι · φόρτωση και καθαρισμός τιμολογίων")
    print(f"  Παραστατικά που διαβάστηκαν:          {s['invoices']:>8,}")
    print(f"  Γραμμές που διαβάστηκαν:              {s['lines']:>8,}")
    print(f"  Γραμμές χωρίς παραστατικό:            {s['orphan_lines']:>8,}")
    print(f"  Ακυρωμένα (εξαιρούνται):              {s['cancelled_invoices']:>8,} παραστατικά, {s['cancelled_lines']:,} γραμμές")
    print(f"  Πιστωτικά (ποσότητα με αρνητικό):     {s['credit_invoices']:>8,} παραστατικά, {s['credit_rows']:,} γραμμές στον καθαρό πίνακα")
    print(f"  Δωρεάν γραμμές (μετράνε στην ποσότ.): {s['free_lines']:>8,}")
    print(f"  Μεταφορικά (εξαιρούνται):             {s['shipping_lines']:>8,} γραμμές")
    print(f"  Διπλό είδος στο ίδιο παραστατικό:     {s['merged_lines']:>8,} γραμμές ενώθηκαν")
    print(f"  Καθαρός πίνακας παραδόσεων:           {s['delivery_rows']:>8,} γραμμές")


def main():
    ap = argparse.ArgumentParser(description="Load and clean the invoices.")
    ap.add_argument("--api", nargs="?", const="http://localhost:8080", default=None,
                    help="read from the ERP API instead of the files (default URL http://localhost:8080)")
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()

    if args.api:
        headers, lines, items, _ = load_from_api(args.api)
    else:
        headers, lines, items, _ = load_from_files(args.data_dir)
    deliveries = build_deliveries(headers, lines, items)
    _print_summary(summarize(headers, lines, items, deliveries))


if __name__ == "__main__":
    main()
