"""Write submissions/team_rafi.csv for all pairs of data/prediction_targets.csv.

Run from the repository root: python scripts/make_submission.py
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rafi.baseline import rolling180  # noqa: E402
from rafi.ingest import build_deliveries, daily_qty, load_from_files  # noqa: E402

CUTOFF = pd.Timestamp("2026-10-31")
AT_RISK_END = CUTOFF + pd.Timedelta(days=14)
EXPECTED_ROWS = 1975
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def build_submission(data_dir=ROOT / "data", method=rolling180):
    """Run the method and return a DataFrame in the template's format and order."""
    data_dir = Path(data_dir)
    headers, lines, items, _ = load_from_files(data_dir)
    deliveries = build_deliveries(headers, lines, items)
    daily = daily_qty(deliveries[deliveries["TRNDATE"] <= CUTOFF])

    template = pd.read_csv(data_dir / "submission_template.csv", encoding="utf-8-sig")
    targets = pd.read_csv(data_dir / "prediction_targets.csv", encoding="utf-8-sig")
    pred = method(daily, targets, CUTOFF)

    out = template[["TRDR", "MTRL"]].merge(pred, on=["TRDR", "MTRL"], how="left")
    out["PREDICTED_DEPLETION_DATE"] = (pd.to_datetime(out["PREDICTED_DEPLETION_DATE"])
                                       .dt.strftime("%Y-%m-%d").fillna(""))
    out["SUGGESTED_QTY"] = ""  # not filled yet
    return out[list(template.columns)], template


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
    sub, template = build_submission()
    filled = check(sub, template)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)

    print("Ράφι · υποβολή (μέθοδος rolling180, cutoff 2026-10-31)")
    print(f"  Ζεύγη:                     {len(sub):>5,}")
    print(f"  Με ημερομηνία εξάντλησης:  {len(filled):>5,}")
    print(f"  Κενά (όχι μέσα στον ορίζ.):{len(sub) - len(filled):>5,}")
    print(f"  At Risk (≤ 2026-11-14):    {int((filled <= AT_RISK_END).sum()):>5,}")
    print(f"  Γράφτηκε: {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
