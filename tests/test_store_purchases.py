"""Purchase suggestions in rafi.store: dedupe, approve once, reject with a reason, history."""
import json

import pytest

from rafi import store


def order(mtrl=5001, order_by="2026-11-10", qty=60, flag=""):
    return {"MTRL": mtrl, "ITEM_CODE": "10.01.001", "ITEM_NAME": "Νερό 0,5L", "UNIT": "κιβ.",
            "SUPPLIER_ID": "SUP01", "SUPPLIER_NAME": "Προμηθευτής Νερών Α.Ε.",
            "ORDER_BY": order_by, "ARRIVAL": "2026-11-13", "QTY": qty, "VALUE_EUR": qty * 4.2,
            "FLAG": flag, "REASON": "Το «Νερό 0,5L» τελειώνει περίπου 15/11.", "AFFECTED": []}


@pytest.fixture
def db(tmp_path):
    return tmp_path / "t.db"


def test_same_item_and_order_by_saved_once(db):
    r = store.save_purchase_new([order(), order(qty=70), order(order_by="2026-11-20")], db=db)
    assert len(r["saved"]) == 2 and r["skipped"] == 1
    assert store.save_purchase_new([order()], db=db) == {"saved": [], "skipped": 1}
    rows = store.list_purchase_suggestions(db=db)
    assert [x["status"] for x in rows] == [store.PROPOSED] * 2
    assert rows[0]["qty"] == 60 and rows[0]["payload"]["SUPPLIER_ID"] == "SUP01"


def test_rejected_item_and_order_by_is_not_proposed_again(db):
    pid = store.save_purchase_new([order()], db=db)["saved"][0]
    store.reject_purchase(pid, "έχουμε απόθεμα", "αποθήκη", db=db)
    assert store.save_purchase_new([order()], db=db)["skipped"] == 1


def test_approve_writes_json_once(db, tmp_path):
    pid = store.save_purchase_new([order()], db=db)["saved"][0]
    path = store.approve_purchase(pid, "αποθήκη", db=db, out_dir=tmp_path / "po")
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["SUPPLIER"] == {"SUPPLIER_ID": "SUP01", "NAME": "Προμηθευτής Νερών Α.Ε."}
    assert doc["LINES"] == [{"MTRL": 5001, "ITEM_CODE": "10.01.001", "ITEM_NAME": "Νερό 0,5L",
                             "QTY": 60, "UNIT": "κιβ."}]
    assert doc["ORDER_BY"] == "2026-11-10" and doc["ARRIVAL"] == "2026-11-13"
    assert doc["REASON"].startswith("Το «Νερό 0,5L»")
    with pytest.raises(store.StoreError):
        store.approve_purchase(pid, "αποθήκη", db=db, out_dir=tmp_path / "po")
    with pytest.raises(store.StoreError):
        store.reject_purchase(pid, "άλλαξα γνώμη", "αποθήκη", db=db)
    assert store.list_purchase_suggestions(store.APPROVED, db=db)[0]["file_path"] == str(path)
    actions = [h["action"] for h in store.purchase_history(pid, db=db)]
    assert actions == [store.CREATE, store.APPROVE]


def test_reject_needs_a_reason_and_is_final(db, tmp_path):
    pid = store.save_purchase_new([order()], db=db)["saved"][0]
    for reason in ("", "   ", None):
        with pytest.raises(store.StoreError):
            store.reject_purchase(pid, reason, "αποθήκη", db=db)
    store.reject_purchase(pid, "έχουμε απόθεμα", "αποθήκη", db=db)
    with pytest.raises(store.StoreError):
        store.approve_purchase(pid, "αποθήκη", db=db, out_dir=tmp_path / "po")
    assert not (tmp_path / "po").exists()
    h = store.purchase_history(pid, db=db)
    assert h[-1]["action"] == store.REJECT and h[-1]["details"] == {"reason": "έχουμε απόθεμα"}


def test_failed_file_write_keeps_the_proposal(db, tmp_path):
    pid = store.save_purchase_new([order()], db=db)["saved"][0]
    blocker = tmp_path / "po"
    blocker.write_text("not a directory")
    with pytest.raises(OSError):
        store.approve_purchase(pid, "αποθήκη", db=db, out_dir=blocker)
    assert store.list_purchase_suggestions(db=db)[0]["status"] == store.PROPOSED


def test_unknown_id(db):
    with pytest.raises(store.StoreError):
        store.approve_purchase(999, "αποθήκη", db=db)


def test_customer_suggestions_untouched(db):
    store.save_purchase_new([order()], db=db)
    assert store.list_suggestions(db=db) == []
