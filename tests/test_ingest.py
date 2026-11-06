"""One test per data rule, using the real numbers of the organisers' data pack."""
import socket
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pytest
import requests

from rafi.ingest import build_deliveries, daily_qty, load_from_api, load_from_files

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


@pytest.fixture(scope="module")
def raw():
    return load_from_files(DATA)


@pytest.fixture(scope="module")
def deliveries(raw):
    headers, lines, items, _ = raw
    return build_deliveries(headers, lines, items)


def _free_mask(lines):
    return (lines["PRICE"] == 0) & (lines["DISC1PRC"] == 100)


# --- BOM ---------------------------------------------------------------

def test_bom_is_stripped_from_first_column(raw):
    headers, lines, items, customers = raw
    assert headers.columns[0] == "FINDOC"
    assert lines.columns[0] == "FINDOC"
    assert items.columns[0] == "MTRL"
    assert customers.columns[0] == "TRDR"
    for df in raw:
        assert not any(c.startswith("﻿") for c in df.columns)


# --- Join header/lines ---------------------------------------------------

def test_join_counts_and_header_fields(raw, deliveries):
    headers, lines, _, _ = raw
    assert len(headers) == 25_184
    assert len(lines) == 105_757
    assert headers["FINDOC"].is_unique
    assert lines["FINDOC"].isin(headers["FINDOC"]).all(), "line without header"
    assert deliveries["TRNDATE"].notna().all()
    assert deliveries["TRDR"].notna().all()
    assert pd.api.types.is_datetime64_any_dtype(deliveries["TRNDATE"])

    # date, customer and salesman must match the header of the same FINDOC
    h = headers.set_index("FINDOC")
    assert (deliveries["TRDR"].values == h.loc[deliveries["FINDOC"], "TRDR"].values).all()
    assert (deliveries["SALESMAN"].values == h.loc[deliveries["FINDOC"], "SALESMAN"].values).all()
    assert (deliveries["TRNDATE"].dt.strftime("%Y-%m-%d").values
            == h.loc[deliveries["FINDOC"], "TRNDATE"].values).all()


# --- ISCANCEL = 1 --------------------------------------------------------

def test_cancelled_invoices_are_dropped(raw, deliveries):
    headers, lines, _, _ = raw
    cancelled = headers.loc[headers["ISCANCEL"] == 1, "FINDOC"]
    assert len(cancelled) == 362
    assert lines["FINDOC"].isin(cancelled).sum() == 1_573
    assert not deliveries["FINDOC"].isin(cancelled).any()


# --- SERIES = 7071 credit notes -----------------------------------------

def test_credit_notes_have_negative_qty(raw, deliveries):
    headers, _, _, _ = raw
    credits = headers.loc[(headers["SERIES"] == 7071) & (headers["ISCANCEL"] != 1), "FINDOC"]
    assert (headers["SERIES"] == 7071).sum() == 457
    rows = deliveries[deliveries["FINDOC"].isin(credits)]
    assert rows["FINDOC"].nunique() == 457
    assert rows["IS_CREDIT"].all()
    assert (rows["QTY"] < 0).all()
    assert (rows["LINEVAL"] <= 0).all()
    # everything else is a normal sale
    others = deliveries[~deliveries["FINDOC"].isin(credits)]
    assert not others["IS_CREDIT"].any()
    assert (others["QTY"] > 0).all()


def test_credit_note_without_comment_is_still_a_credit(raw, deliveries):
    headers, lines, _, _ = raw
    no_comment = headers[(headers["SERIES"] == 7071) & headers["COMMENTS"].isna()]
    assert len(no_comment) == 215

    # FINDOC 100223: credit note with an empty COMMENTS, 2 x MTRL 5044 returned
    doc = headers.set_index("FINDOC").loc[100223]
    assert doc["SERIES"] == 7071 and pd.isna(doc["COMMENTS"])
    assert lines.loc[lines["FINDOC"] == 100223, "QTY1"].tolist() == [2]
    row = deliveries[deliveries["FINDOC"] == 100223]
    assert row["IS_CREDIT"].tolist() == [True]
    assert row["QTY"].tolist() == [-2]
    assert row["LINEVAL"].tolist() == [-28.28]

    # all 215 credits without comment are negative in the clean table
    rows = deliveries[deliveries["FINDOC"].isin(no_comment["FINDOC"])]
    assert rows["FINDOC"].nunique() == 215
    assert (rows["QTY"] < 0).all()


# --- Free goods (PRICE = 0, DISC1PRC = 100) -----------------------------

def test_free_goods_count_in_quantity(raw, deliveries):
    headers, lines, _, _ = raw
    free = lines[_free_mask(lines)]
    assert len(free) == 6_552
    # 88 of them sit in cancelled invoices and are dropped with them
    cancelled = headers.loc[headers["ISCANCEL"] == 1, "FINDOC"]
    kept_free = free[~free["FINDOC"].isin(cancelled)]
    assert deliveries["QTY_FREE"].sum() == kept_free["QTY1"].sum()


def test_free_line_adds_to_paid_line_of_same_item():
    # FINDOC 100002, MTRL 5040: 12 paid + 1 free -> QTY 13, QTY_FREE 1, value only from the paid line
    headers, lines, items, _ = load_from_files(DATA)
    src = lines[(lines["FINDOC"] == 100002) & (lines["MTRL"] == 5040)]
    assert sorted(src["QTY1"]) == [1, 12]
    row = build_deliveries(headers, lines, items).query("FINDOC == 100002 and MTRL == 5040")
    assert row["QTY"].tolist() == [13]
    assert row["QTY_FREE"].tolist() == [1]
    assert row["LINEVAL"].tolist() == [198.0]


# --- Shipping (MTRTYPE = 1, MTRL 5081) ----------------------------------

def test_shipping_lines_are_dropped(raw, deliveries):
    _, lines, items, _ = raw
    assert items.loc[items["MTRTYPE"] == 1, "MTRL"].tolist() == [5081]
    assert (lines["MTRL"] == 5081).sum() == 2_498
    assert not (deliveries["MTRL"] == 5081).any()


# --- Same item twice in one invoice -------------------------------------

def test_one_row_per_invoice_and_item(raw, deliveries):
    _, lines, _, _ = raw
    assert lines.duplicated(["FINDOC", "MTRL"]).sum() > 0  # the raw data does have duplicates
    assert not deliveries.duplicated(["FINDOC", "MTRL"]).any()


def test_promo_discount_flag(deliveries):
    # partial discounts exist; free goods (100%) are not a promo discount
    assert deliveries["PROMO_DISC"].any()
    assert not deliveries["PROMO_DISC"].all()


def test_daily_qty_is_net_and_unique(deliveries):
    daily = daily_qty(deliveries)
    assert not daily.duplicated(["TRDR", "MTRL", "TRNDATE"]).any()
    assert daily["QTY"].sum() == deliveries["QTY"].sum()


# --- Deliveries only on working days (data check) ------------------------

def test_no_invoice_on_sunday_or_holiday(raw):
    headers, _, _, _ = raw
    dates = pd.to_datetime(headers["TRNDATE"])
    holidays = pd.to_datetime(pd.read_csv(DATA / "greek_holidays.csv", encoding="utf-8-sig")["DATE"])
    assert len(holidays) > 0
    assert not (dates.dt.dayofweek == 6).any(), "invoice on a Sunday"
    assert not dates.isin(holidays).any(), "invoice on a public holiday"


# --- Files and API give the same result ---------------------------------

def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def erp_url():
    port = _free_port()
    proc = subprocess.Popen([sys.executable, str(ROOT / "mock_erp_server.py"), "--port", str(port)],
                            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 30
        while True:
            try:
                requests.get(f"{url}/api/items", timeout=1).raise_for_status()
                break
            except requests.RequestException:
                if proc.poll() is not None or time.time() > deadline:
                    raise RuntimeError("mock ERP server did not start")
                time.sleep(0.2)
        yield url
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_api_matches_files(raw, deliveries, erp_url):
    api = load_from_api(erp_url, page_size=1000)
    for from_file, from_api in zip(raw, api):
        pd.testing.assert_frame_equal(from_file, from_api)
    api_deliveries = build_deliveries(api[0], api[1], api[2])
    pd.testing.assert_frame_equal(deliveries, api_deliveries)
