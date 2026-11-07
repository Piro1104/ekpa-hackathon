"""Store, approval flow and ERP client. Every test uses its own database in tmp_path."""
import json
import sys
from pathlib import Path

import pytest
import requests

from rafi import erp_client, store
from rafi.erp_client import ErpError
from rafi.store import StoreError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import daily_run  # noqa: E402


class FakeErp:
    """Counts POSTs; fails while `fail` is set."""

    def __init__(self, fail=None):
        self.calls, self.fail = [], fail

    def post_draft(self, payload):
        self.calls.append(json.loads(json.dumps(payload)))
        if self.fail:
            raise ErpError(self.fail, ["LINES[0]: QTY1 must be a number > 0"])
        return f"DRAFT-TEST-{len(self.calls):04d}"


def make(trdr=1001, mtrls=(5001, 5002), qty=3, kind="ΠΑΡΑΓΓΕΛΙΑ"):
    lines = [{"MTRL": m, "ITEM_CODE": f"X{m}", "QTY1": qty, "PRICER": 2.5,
              "PREDICTED_DEPLETION_DATE": "2026-11-06", "CONFIDENCE": 0.7} for m in mtrls]
    payload = {"TRDR": trdr, "CUSTOMER_CODE": "30.00.0001", "SALESMAN": 103,
               "SUGGESTED_DELIVERY_DATE": "2026-11-04", "ACTION_TYPE": kind, "FLAGS": [],
               "REASON": "Το νερό τελειώνει 6/11.", "SALES_NOTE": "Καλημέρα σας ...",
               "TEXT_SOURCE": "ΠΡΟΤΥΠΟ", "RAFI_SUGGESTION_ID": None, "LINES": lines}
    return {"ACTION_TYPE": kind, "TRDR": trdr, "SALESMAN": 103, "FLAGS": [],
            "VALUE_EUR": round(qty * 2.5 * len(mtrls), 2), "PRIORITY": 1,
            "FACTS": {"note": "test"}, "payload": payload}


@pytest.fixture
def db(tmp_path):
    return tmp_path / "test.db"


def _actions(db, sid):
    return [h["action"] for h in store.history(suggestion_id=sid, db=db)]


# --- creation --------------------------------------------------------------

def test_save_new_creates_proposals_with_history(db):
    res = store.save_new([make(1001), make(1002)], db=db)
    assert len(res["saved"]) == 2 and res["skipped"] == 0
    for sid in res["saved"]:
        s = store.get(sid, db=db)
        assert s["status"] == store.PROPOSED
        assert s["payload"]["RAFI_SUGGESTION_ID"] == sid
        assert _actions(db, sid) == [store.CREATE]


def test_no_new_suggestion_for_a_pair_already_active(db):
    first = store.save_new([make(1001, (5001, 5002))], db=db)["saved"][0]
    again = store.save_new([make(1001, (5002, 5003)), make(1001, (5009,))], db=db)
    assert again["skipped"] == 1 and len(again["saved"]) == 1
    # still blocked once it is sent
    store.approve_and_send(first, "101", erp=FakeErp(), db=db)
    assert store.save_new([make(1001, (5001,))], db=db)["skipped"] == 1
    # a rejected suggestion does not block the pair
    other = store.save_new([make(1003, (5001,))], db=db)["saved"][0]
    store.reject(other, "έχει ακόμα απόθεμα", "101", db=db)
    assert len(store.save_new([make(1003, (5001,))], db=db)["saved"]) == 1


# --- approval --------------------------------------------------------------

def test_nothing_is_sent_without_approval(db, monkeypatch):
    erp = FakeErp()
    monkeypatch.setattr(erp_client, "post_draft", erp.post_draft)
    store.save_new([make(1001), make(1002)], db=db)
    store.list_suggestions(db=db)
    assert erp.calls == []
    assert {s["status"] for s in store.list_suggestions(db=db)} == {store.PROPOSED}


def test_approve_twice_posts_once(db):
    sid = store.save_new([make()], db=db)["saved"][0]
    erp = FakeErp()
    erp_id = store.approve_and_send(sid, "101", erp=erp, db=db)
    with pytest.raises(StoreError):
        store.approve_and_send(sid, "101", erp=erp, db=db)
    assert len(erp.calls) == 1
    assert erp.calls[0]["RAFI_SUGGESTION_ID"] == sid
    s = store.get(sid, db=db)
    assert s["status"] == store.SENT and s["erp_id"] == erp_id
    assert _actions(db, sid) == [store.CREATE, store.APPROVE, store.SEND]
    assert store.history(suggestion_id=sid, db=db)[-1]["erp_id"] == erp_id


def test_rejected_suggestion_is_never_sent(db):
    sid = store.save_new([make()], db=db)["saved"][0]
    for reason in ("", "   ", None):
        with pytest.raises(StoreError):
            store.reject(sid, reason, "101", db=db)
    assert store.get(sid, db=db)["status"] == store.PROPOSED

    store.reject(sid, "ο πελάτης έχει ακόμα απόθεμα", "101", db=db)
    erp = FakeErp()
    with pytest.raises(StoreError):
        store.approve_and_send(sid, "101", erp=erp, db=db)
    assert erp.calls == []
    assert store.get(sid, db=db)["status"] == store.REJECTED
    last = store.history(suggestion_id=sid, db=db)[-1]
    assert last["action"] == store.REJECT
    assert last["details"]["reason"] == "ο πελάτης έχει ακόμα απόθεμα"


def test_correction_is_logged_and_sent(db):
    sid = store.save_new([make(qty=3)], db=db)["saved"][0]
    store.correct(sid, 5002, 7, "101", db=db)
    h = store.history(suggestion_id=sid, db=db)[-1]
    assert h["action"] == store.CORRECT
    assert h["details"] == {"MTRL": 5002, "old_qty": 3, "new_qty": 7}
    assert store.get(sid, db=db)["value_eur"] == round((3 + 7) * 2.5, 2)

    for bad in (0, -1, 2.5, "4"):
        with pytest.raises(StoreError):
            store.correct(sid, 5002, bad, "101", db=db)
    with pytest.raises(StoreError):
        store.correct(sid, 5999, 4, "101", db=db)

    erp = FakeErp()
    store.approve_and_send(sid, "101", erp=erp, db=db)
    sent = {l["MTRL"]: l["QTY1"] for l in erp.calls[0]["LINES"]}
    assert sent == {5001: 3, 5002: 7}
    with pytest.raises(StoreError):
        store.correct(sid, 5002, 9, "101", db=db)  # no corrections after sending


def test_failed_post_goes_back_to_proposal(db):
    sid = store.save_new([make()], db=db)["saved"][0]
    with pytest.raises(ErpError):
        store.approve_and_send(sid, "101", erp=FakeErp(fail="Το ERP απέρριψε το draft"), db=db)
    assert store.get(sid, db=db)["status"] == store.PROPOSED
    h = store.history(suggestion_id=sid, db=db)
    assert [x["action"] for x in h] == [store.CREATE, store.APPROVE, store.SEND_ERROR]
    assert "απέρριψε" in h[-1]["details"]["message"]
    assert h[-1]["details"]["errors"]
    # it can be approved again once the problem is fixed
    assert store.approve_and_send(sid, "101", erp=FakeErp(), db=db).startswith("DRAFT-")


def test_lists_and_history_filters(db):
    a, b = store.save_new([make(1001), make(1002)], db=db)["saved"]
    store.reject(b, "κλειστό κατάστημα", "102", db=db)
    assert [s["id"] for s in store.list_suggestions(status=store.PROPOSED, db=db)] == [a]
    assert len(store.list_suggestions(salesman=103, db=db)) == 2
    assert {h["trdr"] for h in store.history(trdr=1002, db=db)} == {1002}


def test_cli_refuses_second_approve(db, monkeypatch, capsys):
    sid = store.save_new([make()], db=db)["saved"][0]
    fake = FakeErp()
    monkeypatch.setattr(erp_client, "post_draft", fake.post_draft)
    assert store.main(["--db", str(db), "approve", str(sid), "--user", "101"]) == 0
    assert store.main(["--db", str(db), "approve", str(sid), "--user", "101"]) == 1
    assert len(fake.calls) == 1
    assert "δεν στάλθηκε τίποτα" in capsys.readouterr().err


# --- daily run -------------------------------------------------------------

def test_daily_run_twice_no_duplicates_and_no_post(db, monkeypatch):
    def no_post(*a, **k):
        raise AssertionError("POST to the ERP without approval")
    monkeypatch.setattr(erp_client, "post_draft", no_post)
    monkeypatch.setattr(requests, "post", no_post)

    first, customers, salesmen = daily_run.run(db=db)
    assert len(first["saved"]) > 50
    second, _, _ = daily_run.run(db=db)
    assert second["saved"] == [] and second["skipped"] == len(first["saved"])
    rows = store.list_suggestions(db=db)
    assert len(rows) == len(first["saved"])
    assert {r["status"] for r in rows} == {store.PROPOSED}
    text = daily_run.briefing(second, customers, salesmen, db=db)
    assert "πρωινή ενημέρωση" in text and "ΚΙΝΔΥΝΟΣ_ΑΠΩΛΕΙΑΣ" in text


# --- ERP client and the real mock ERP --------------------------------------

def test_erp_down_gives_a_clear_message():
    with pytest.raises(ErpError, match="δεν απαντά"):
        erp_client.post_draft(make()["payload"], base_url="http://127.0.0.1:9", timeout=2)


def test_erp_400_returns_the_erp_errors(erp_url):
    bad = make(mtrls=(9999,))["payload"]
    with pytest.raises(ErpError) as e:
        erp_client.post_draft(bad, base_url=erp_url)
    assert any("9999" in err for err in e.value.errors)


def test_approve_sends_a_draft_to_the_real_mock_erp(db, erp_url, monkeypatch):
    monkeypatch.setenv("ERP_URL", erp_url)
    sid = store.save_new([make(1001, (5001, 5009), qty=4)], db=db)["saved"][0]
    drafts = ROOT / "draft_orders"
    existed = drafts.exists()
    erp_id = None
    try:
        erp_id = store.approve_and_send(sid, "101", db=db)
        assert erp_id.startswith("DRAFT-")
        saved = json.loads((drafts / f"{erp_id}.json").read_text(encoding="utf-8"))
        assert saved["RAFI_SUGGESTION_ID"] == sid
        assert saved["ACTION_TYPE"] == "ΠΑΡΑΓΓΕΛΙΑ"
        assert saved["REASON"] == "Το νερό τελειώνει 6/11."
        assert saved["SALES_NOTE"]
        assert store.get(sid, db=db)["erp_id"] == erp_id
    finally:
        if erp_id:
            (drafts / f"{erp_id}.json").unlink(missing_ok=True)
        if not existed and drafts.exists() and not any(drafts.iterdir()):
            drafts.rmdir()
