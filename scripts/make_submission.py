"""Write submissions/team_rafi.csv for all pairs of data/prediction_targets.csv,
and submissions/forecast_details.csv with the numbers behind every forecast (for the dashboard).

Run from the repository root: python scripts/make_submission.py
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rafi.forecast import forecast  # noqa: E402
from rafi.ingest import build_deliveries, daily_qty, load_from_files  # noqa: E402

CUTOFF = pd.Timestamp("2026-10-31")
AT_RISK_END = CUTOFF + pd.Timedelta(days=14)
EXPECTED_ROWS = 1975
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def build_submission(data_dir=ROOT / "data"):
    """Run the rafi forecast. Returns (submission in the template's format and order,
    template, full forecast with the explanation columns)."""
    data_dir = Path(data_dir)
    headers, lines, items, customers = load_from_files(data_dir)
    deliveries = build_deliveries(headers, lines, items)
    past = deliveries[deliveries["TRNDATE"] <= CUTOFF]

    template = pd.read_csv(data_dir / "submission_template.csv", encoding="utf-8-sig")
    targets = pd.read_csv(data_dir / "prediction_targets.csv", encoding="utf-8-sig")
    details = forecast(daily_qty(past), past, customers, items, targets, CUTOFF)

    out = template[["TRDR", "MTRL"]].merge(details, on=["TRDR", "MTRL"], how="left")
    has_date = out["PREDICTED_DEPLETION_DATE"].notna()
    out["PREDICTED_DEPLETION_DATE"] = (out["PREDICTED_DEPLETION_DATE"]
                                       .dt.strftime("%Y-%m-%d").fillna(""))
    # A quantity only where we expect the stock to run out inside the horizon.
    out["SUGGESTED_QTY"] = out["SUGGESTED_QTY"].where(has_date).astype("Int64")
    return out[list(template.columns)], template, details


def check(sub, template):
    """Stop with a clear message if the file would not match the template."""
    errors = []
    if len(sub) != EXPECTED_ROWS:
        errors.append(f"{len(sub)} γραμμές αντί για {EXPECTED_ROWS}")
    if list(sub.columns) != list(template.columns):
        errors.append(f"στήλες {list(sub.columns)} αντί για {list(template.columns)}")
    if not sub[["TRDR", "MTRL"]].reset_index(drop=True).equals(
            template[["TRDR", "MTRL"]].reset_index(drop=True)):
        errors.append("τα ζεύγη ή η σειρά τους διαφέρουν από το template")
    dates = sub["PREDICTED_DEPLETION_DATE"]
    bad = dates[(dates != "") & ~dates.str.match(DATE_RE)]
    if len(bad):
        errors.append(f"{len(bad)} ημερομηνίες όχι σε μορφή YYYY-MM-DD")
    filled = pd.to_datetime(dates[dates != ""], format="%Y-%m-%d", errors="coerce")
    if filled.isna().any() or (filled <= CUTOFF).any():
        errors.append("υπάρχουν ημερομηνίες που δεν διαβάζονται ή δεν είναι μετά το 2026-10-31")
    if errors:
        raise SystemExit("Η υποβολή ΔΕΝ γράφτηκε:\n  " + "\n  ".join(errors))
    return filled


def main(out_path=ROOT / "submissions" / "team_rafi.csv"):
    sub, template, details = build_submission()
    filled = check(sub, template)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    details_path = out_path.parent / "forecast_details.csv"
    details.to_csv(details_path, index=False, date_format="%Y-%m-%d")

    print("Ράφι · υποβολή (μέθοδος rafi, cutoff 2026-10-31)")
    print(f"  Ζεύγη:                     {len(sub):>5,}")
    print(f"  Με ημερομηνία εξάντλησης:  {len(filled):>5,}")
    print(f"  Κενά:                      {len(sub) - len(filled):>5,}")
    blanks = details.loc[details["PREDICTED_DEPLETION_DATE"].isna(), "REASON_CODE"]
    for reason, n in blanks.value_counts().items():
        print(f"    {reason:<23} {n:>5,}")
    print(f"  At Risk (≤ 2026-11-14):    {int((filled <= AT_RISK_END).sum()):>5,}")
    print("  Όλα τα ζεύγη ανά REASON_CODE:")
    for reason, n in details["REASON_CODE"].value_counts().items():
        print(f"    {reason:<23} {n:>5,}")
    for path in (out_path, details_path):
        shown = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        print(f"  Γράφτηκε: {shown}")


if __name__ == "__main__":
    main()
