"""The four fixes of the second dashboard version, with fake data."""
import re
from pathlib import Path

import pandas as pd

from app import helpers as h
from tests.test_app_helpers import ITEMS, make

ROOT = Path(__file__).resolve().parents[1]


def test_store_user_is_always_admin():
    for s in (make(), {"salesman": 101}, {"salesman": None}, None):
        assert h.store_user(s) == "admin"
    assert h.user_label("admin") == "Διαχειριστής"


def test_ai_badge_only_for_ai_source():
    assert h.is_ai("AI")
    for other in ("ΠΡΟΤΥΠΟ", None, "", "local", "ai"):
        assert not h.is_ai(other)


def test_no_view_calls_the_store_with_another_user():
    """Every store call in the pages passes the one dashboard user."""
    for path in (ROOT / "app").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "s.get(\"salesman\") or" not in text, path


def test_tables_keep_real_values_and_show_greek():
    rows = h.lines_rows(make(), h.item_lookup(ITEMS))
    df = h.lines_frame(rows)
    assert str(df["Ποσότητα"].dtype) == "int64"
    assert str(df["Τελειώνει"].dtype).startswith("datetime64")
    # sorting by the real date, not by the text ('Παρ.' < 'Τρ.' alphabetically would differ)
    df2 = pd.DataFrame({"Τελειώνει": pd.to_datetime(["2026-11-10", "2026-11-03"])})
    assert list(df2.sort_values("Τελειώνει")["Τελειώνει"].dt.day) == [3, 10]
    f = h.table_formats(df)
    assert f["Τελειώνει"](df["Τελειώνει"].iloc[0]) == "Παρ. 06/11"
    assert f["Τελειώνει"](pd.NaT) == "—"
    assert h.TABLE_FORMATS["Αξία"](1173.5) == "1.173,50 €"
    assert h.TABLE_FORMATS["Ποσότητα"](1500) == "1.500"
    assert re.match(r"^Σάβ\. 07/11/2026 16:20$",
                    h.TABLE_FORMATS["Ημερομηνία"](pd.Timestamp("2026-11-07T16:20:05")))


def test_requirements_ask_for_streamlit_1_67():
    lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    assert "streamlit>=1.67" in lines
