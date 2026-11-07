"""SQLite store for suggestions and the decision history.

Status flow:
  ΠΡΟΤΑΣΗ -> ΑΠΟΣΤΕΛΛΕΤΑΙ -> ΣΤΑΛΘΗΚΕ_ΣΤΟ_ERP
  ΠΡΟΤΑΣΗ -> ΑΠΟΡΡΙΦΘΗΚΕ
  ΑΠΟΣΤΕΛΛΕΤΑΙ -> ΠΡΟΤΑΣΗ when the POST fails (with a ΣΦΑΛΜΑ_ΑΠΟΣΤΟΛΗΣ entry)

Nothing reaches the ERP unless approve_and_send is called for that suggestion, and the
switch to ΑΠΟΣΤΕΛΛΕΤΑΙ is a single conditional UPDATE, so a suggestion is sent at most once.

The database file is rafi.db in the repository root (or $RAFI_DB, or the `db` argument).
CLI: python -m rafi.store list | show <id> | approve <id> | correct <id> <mtrl> <qty>
     | reject <id> "λόγος" | history   (add --user 101)
"""
import argparse
import json
import os
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "rafi.db"

PROPOSED, SENDING, SENT, REJECTED = "ΠΡΟΤΑΣΗ", "ΑΠΟΣΤΕΛΛΕΤΑΙ", "ΣΤΑΛΘΗΚΕ_ΣΤΟ_ERP", "ΑΠΟΡΡΙΦΘΗΚΕ"
ACTIVE = (PROPOSED, SENDING, SENT)  # a pair in one of these gets no new suggestion
CREATE, APPROVE, CORRECT, REJECT, SEND, SEND_ERROR = (
    "ΔΗΜΙΟΥΡΓΙΑ", "ΕΓΚΡΙΣΗ", "ΔΙΟΡΘΩΣΗ", "ΑΠΟΡΡΙΨΗ", "ΑΠΟΣΤΟΛΗ", "ΣΦΑΛΜΑ_ΑΠΟΣΤΟΛΗΣ")

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS suggestions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT NOT NULL,
    trdr          INTEGER NOT NULL,
    salesman      INTEGER,
    action_type   TEXT NOT NULL,
    delivery_date TEXT NOT NULL,
    value_eur     REAL,
    priority      INTEGER,
    payload_json  TEXT NOT NULL,
    status        TEXT NOT NULL CHECK (status IN
                  ('{PROPOSED}', '{SENDING}', '{SENT}', '{REJECTED}')),
    erp_id        TEXT
);
CREATE TABLE IF NOT EXISTS history (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    suggestion_id INTEGER NOT NULL REFERENCES suggestions(id),
    at            TEXT NOT NULL,
    user          TEXT NOT NULL,
    action        TEXT NOT NULL,
    details_json  TEXT,
    erp_id        TEXT
);
CREATE INDEX IF NOT EXISTS ix_suggestions_status ON suggestions(status);
CREATE INDEX IF NOT EXISTS ix_suggestions_trdr ON suggestions(trdr);
CREATE INDEX IF NOT EXISTS ix_history_suggestion ON history(suggestion_id);
"""


class StoreError(Exception):
    """The action is not allowed (wrong status, missing reason, unknown id ...)."""


def _now():
    return datetime.now().isoformat(timespec="seconds")


def _dumps(obj):
    return json.dumps(obj, ensure_ascii=False, default=str)


def connect(db=None):
    path = Path(db or os.environ.get("RAFI_DB") or DEFAULT_DB)
    conn = sqlite3.connect(path, isolation_level=None)  # we open transactions ourselves
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


@contextmanager
def _transaction(db):
    """One write transaction. BEGIN IMMEDIATE locks the file, so two clicks cannot interleave."""
    conn = connect(db)
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")
    finally:
        conn.close()


def _log(conn, suggestion_id, user, action, details=None, erp_id=None):
    conn.execute("INSERT INTO history (suggestion_id, at, user, action, details_json, erp_id) "
                 "VALUES (?, ?, ?, ?, ?, ?)",
                 (suggestion_id, _now(), str(user), action,
                  _dumps(details) if details is not None else None, erp_id))


def _row(conn, suggestion_id):
    r = conn.execute("SELECT * FROM suggestions WHERE id = ?", (suggestion_id,)).fetchone()
    if r is None:
        raise StoreError(f"Δεν υπάρχει πρόταση με id {suggestion_id}.")
    return r


def _as_dict(r):
    d = dict(r)
    d["payload"] = json.loads(d.pop("payload_json"))
    return d


def _pairs(payload):
    return {(int(payload["TRDR"]), int(l["MTRL"])) for l in payload["LINES"]}


# --- create ----------------------------------------------------------------

def save_new(suggestions, user="system", db=None):
    """Save suggestions whose (TRDR, MTRL) pairs have no active suggestion yet.

    Active = ΠΡΟΤΑΣΗ, ΑΠΟΣΤΕΛΛΕΤΑΙ or ΣΤΑΛΘΗΚΕ_ΣΤΟ_ERP. A suggestion that shares any pair
    with an active one is skipped whole. Returns {"saved": [ids], "skipped": count}.
    """
    saved, skipped = [], 0
    with _transaction(db) as conn:
        active = set()
        for r in conn.execute("SELECT payload_json FROM suggestions WHERE status IN (?, ?, ?)",
                              ACTIVE):
            active |= _pairs(json.loads(r["payload_json"]))
        for s in suggestions:
            payload = dict(s["payload"])
            pairs = _pairs(payload)
            if pairs & active:
                skipped += 1
                continue
            cur = conn.execute(
                "INSERT INTO suggestions (created_at, trdr, salesman, action_type, delivery_date, "
                "value_eur, priority, payload_json, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (_now(), payload["TRDR"], payload.get("SALESMAN"), payload["ACTION_TYPE"],
                 payload["SUGGESTED_DELIVERY_DATE"], s.get("VALUE_EUR"), s.get("PRIORITY"),
                 "{}", PROPOSED))
            sid = cur.lastrowid
            payload["RAFI_SUGGESTION_ID"] = sid
            conn.execute("UPDATE suggestions SET payload_json = ? WHERE id = ?",
                         (_dumps(payload), sid))
            _log(conn, sid, user, CREATE, {"flags": s.get("FLAGS", []),
                                           "value_eur": s.get("VALUE_EUR"),
                                           "priority": s.get("PRIORITY"),
                                           "reason": payload.get("REASON"),
                                           "facts": s.get("FACTS")})
            active |= pairs
            saved.append(sid)
    return {"saved": saved, "skipped": skipped}


# --- decisions -------------------------------------------------------------

def correct(suggestion_id, mtrl, new_qty, user, db=None):
    """Change the quantity of one line. Only while the suggestion is still ΠΡΟΤΑΣΗ."""
    if isinstance(new_qty, bool) or not isinstance(new_qty, int) or new_qty <= 0:
        raise StoreError("Η νέα ποσότητα πρέπει να είναι ακέραιος > 0 "
                         "(για να βγει ένα είδος, απόρριψε την πρόταση).")
    with _transaction(db) as conn:
        r = _row(conn, suggestion_id)
        if r["status"] != PROPOSED:
            raise StoreError(f"Η πρόταση {suggestion_id} είναι {r['status']}· "
                             f"διόρθωση γίνεται μόνο σε {PROPOSED}.")
        payload = json.loads(r["payload_json"])
        line = next((l for l in payload["LINES"] if int(l["MTRL"]) == int(mtrl)), None)
        if line is None:
            raise StoreError(f"Η πρόταση {suggestion_id} δεν έχει το είδος {mtrl}.")
        old = line["QTY1"]
        line["QTY1"] = new_qty
        value = round(sum(l["QTY1"] * l.get("PRICER", 0) for l in payload["LINES"]), 2)
        conn.execute("UPDATE suggestions SET payload_json = ?, value_eur = ? WHERE id = ?",
                     (_dumps(payload), value, suggestion_id))
        _log(conn, suggestion_id, user, CORRECT,
             {"MTRL": int(mtrl), "old_qty": old, "new_qty": new_qty})
    return payload


def reject(suggestion_id, reason, user, db=None):
    """Reject with a reason (required). Only while the suggestion is still ΠΡΟΤΑΣΗ."""
    if not reason or not str(reason).strip():
        raise StoreError("Η απόρριψη χρειάζεται λόγο.")
    with _transaction(db) as conn:
        r = _row(conn, suggestion_id)
        cur = conn.execute("UPDATE suggestions SET status = ? WHERE id = ? AND status = ?",
                           (REJECTED, suggestion_id, PROPOSED))
        if cur.rowcount != 1:
            raise StoreError(f"Η πρόταση {suggestion_id} είναι {r['status']}· "
                             f"απόρριψη γίνεται μόνο σε {PROPOSED}.")
        _log(conn, suggestion_id, user, REJECT, {"reason": str(reason).strip()})


def approve_and_send(suggestion_id, user, erp=None, db=None):
    """Approve and POST the draft to the ERP. Returns the ERP id.

    1. ΠΡΟΤΑΣΗ -> ΑΠΟΣΤΕΛΛΕΤΑΙ in one conditional UPDATE; if no row changed, stop (nothing sent)
    2. POST through `erp.post_draft` (rafi.erp_client by default)
    3. success: ΣΤΑΛΘΗΚΕ_ΣΤΟ_ERP + erp id; failure: back to ΠΡΟΤΑΣΗ and the error is raised
    """
    if erp is None:
        from rafi import erp_client as erp

    with _transaction(db) as conn:
        r = _row(conn, suggestion_id)
        cur = conn.execute("UPDATE suggestions SET status = ? WHERE id = ? AND status = ?",
                           (SENDING, suggestion_id, PROPOSED))
        if cur.rowcount != 1:
            raise StoreError(f"Η πρόταση {suggestion_id} είναι {r['status']}· "
                             f"δεν στάλθηκε τίποτα στο ERP.")
        _log(conn, suggestion_id, user, APPROVE)
        payload = json.loads(r["payload_json"])

    try:
        erp_id = erp.post_draft(payload)
    except Exception as e:
        with _transaction(db) as conn:
            conn.execute("UPDATE suggestions SET status = ? WHERE id = ? AND status = ?",
                         (PROPOSED, suggestion_id, SENDING))
            _log(conn, suggestion_id, user, SEND_ERROR,
                 {"message": str(e), "errors": getattr(e, "errors", [])})
        raise

    with _transaction(db) as conn:
        conn.execute("UPDATE suggestions SET status = ?, erp_id = ? WHERE id = ?",
                     (SENT, erp_id, suggestion_id))
        _log(conn, suggestion_id, user, SEND, erp_id=erp_id)
    return erp_id


# --- read (dashboard) ------------------------------------------------------

def get(suggestion_id, db=None):
    conn = connect(db)
    try:
        return _as_dict(_row(conn, suggestion_id))
    finally:
        conn.close()


def list_suggestions(status=None, salesman=None, db=None):
    """Suggestions with their payload, by priority."""
    sql, args = "SELECT * FROM suggestions WHERE 1 = 1", []
    if status:
        sql += " AND status = ?"
        args.append(status)
    if salesman is not None:
        sql += " AND salesman = ?"
        args.append(int(salesman))
    conn = connect(db)
    try:
        rows = conn.execute(sql + " ORDER BY priority, id", args).fetchall()
        return [_as_dict(r) for r in rows]
    finally:
        conn.close()


def history(trdr=None, suggestion_id=None, db=None):
    """Decision history, oldest first, with the customer and type of each suggestion."""
    sql = ("SELECT h.*, s.trdr, s.action_type FROM history h "
           "JOIN suggestions s ON s.id = h.suggestion_id WHERE 1 = 1")
    args = []
    if trdr is not None:
        sql += " AND s.trdr = ?"
        args.append(int(trdr))
    if suggestion_id is not None:
        sql += " AND h.suggestion_id = ?"
        args.append(int(suggestion_id))
    conn = connect(db)
    try:
        rows = conn.execute(sql + " ORDER BY h.id", args).fetchall()
    finally:
        conn.close()
    out = []
    for r in rows:
        d = dict(r)
        d["details"] = json.loads(d.pop("details_json")) if d["details_json"] else None
        out.append(d)
    return out


# --- purchase suggestions (warehouse agent) ------------------------------------
# Our own orders to suppliers. They never go to the mock ERP (it takes customers only):
# ΠΡΟΤΑΣΗ -> ΕΓΚΡΙΘΗΚΕ (JSON written to purchase_orders/) or ΠΡΟΤΑΣΗ -> ΑΠΟΡΡΙΦΘΗΚΕ (reason).
# One suggestion per (item, ORDER_BY), whatever its status.

APPROVED = "ΕΓΚΡΙΘΗΚΕ"
PURCHASE_DIR = ROOT / "purchase_orders"

PURCHASE_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS purchase_suggestions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT NOT NULL,
    mtrl          INTEGER NOT NULL,
    supplier_id   TEXT NOT NULL,
    order_by      TEXT NOT NULL,
    arrival       TEXT NOT NULL,
    qty           INTEGER NOT NULL,
    value_eur     REAL,
    flag          TEXT,
    payload_json  TEXT NOT NULL,
    status        TEXT NOT NULL CHECK (status IN ('{PROPOSED}', '{APPROVED}', '{REJECTED}')),
    file_path     TEXT,
    UNIQUE (mtrl, order_by)
);
CREATE TABLE IF NOT EXISTS purchase_history (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    purchase_id   INTEGER NOT NULL REFERENCES purchase_suggestions(id),
    at            TEXT NOT NULL,
    user          TEXT NOT NULL,
    action        TEXT NOT NULL,
    details_json  TEXT
);
CREATE INDEX IF NOT EXISTS ix_purchase_status ON purchase_suggestions(status);
CREATE INDEX IF NOT EXISTS ix_purchase_history ON purchase_history(purchase_id);
"""


def _purchase_connect(db=None):
    conn = connect(db)
    conn.executescript(PURCHASE_SCHEMA)
    return conn


@contextmanager
def _purchase_transaction(db):
    conn = _purchase_connect(db)
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")
    finally:
        conn.close()


def _purchase_log(conn, purchase_id, user, action, details=None):
    conn.execute("INSERT INTO purchase_history (purchase_id, at, user, action, details_json) "
                 "VALUES (?, ?, ?, ?, ?)",
                 (purchase_id, _now(), str(user), action,
                  _dumps(details) if details is not None else None))


def _purchase_row(conn, purchase_id):
    r = conn.execute("SELECT * FROM purchase_suggestions WHERE id = ?", (purchase_id,)).fetchone()
    if r is None:
        raise StoreError(f"Δεν υπάρχει πρόταση αγοράς με id {purchase_id}.")
    return r


def save_purchase_new(orders, user="system", db=None):
    """Save purchase suggestions; one with the same MTRL and ORDER_BY as any saved one is skipped.

    `orders`: dicts with at least MTRL, SUPPLIER_ID, ORDER_BY, ARRIVAL, QTY (dates as text).
    Returns {"saved": [ids], "skipped": count}.
    """
    saved, skipped = [], 0
    with _purchase_transaction(db) as conn:
        for o in orders:
            exists = conn.execute("SELECT 1 FROM purchase_suggestions WHERE mtrl = ? "
                                  "AND order_by = ?", (int(o["MTRL"]), o["ORDER_BY"])).fetchone()
            if exists:
                skipped += 1
                continue
            cur = conn.execute(
                "INSERT INTO purchase_suggestions (created_at, mtrl, supplier_id, order_by, "
                "arrival, qty, value_eur, flag, payload_json, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (_now(), int(o["MTRL"]), str(o["SUPPLIER_ID"]), o["ORDER_BY"], o["ARRIVAL"],
                 int(o["QTY"]), o.get("VALUE_EUR"), o.get("FLAG") or None, _dumps(o), PROPOSED))
            _purchase_log(conn, cur.lastrowid, user, CREATE,
                          {"qty": int(o["QTY"]), "order_by": o["ORDER_BY"],
                           "flag": o.get("FLAG") or None, "reason": o.get("REASON")})
            saved.append(cur.lastrowid)
    return {"saved": saved, "skipped": skipped}


def purchase_order_json(row):
    """The purchase order written after approval: SUPPLIER, LINES, ORDER_BY, ARRIVAL, REASON."""
    p = json.loads(row["payload_json"])
    return {
        "PURCHASE_SUGGESTION_ID": row["id"],
        "SUPPLIER": {"SUPPLIER_ID": p["SUPPLIER_ID"], "NAME": p.get("SUPPLIER_NAME")},
        "LINES": [{"MTRL": int(p["MTRL"]), "ITEM_CODE": p.get("ITEM_CODE"),
                   "ITEM_NAME": p.get("ITEM_NAME"), "QTY": int(row["qty"]),
                   "UNIT": p.get("UNIT")}],
        "ORDER_BY": row["order_by"],
        "ARRIVAL": row["arrival"],
        "REASON": p.get("REASON"),
    }


def approve_purchase(purchase_id, user, db=None, out_dir=None):
    """ΠΡΟΤΑΣΗ -> ΕΓΚΡΙΘΗΚΕ and write purchase_orders/<id>.json. Never approves twice.

    The status change and the file are one step: if the file cannot be written, the
    suggestion stays ΠΡΟΤΑΣΗ. Returns the file path.
    """
    out_dir = Path(out_dir or PURCHASE_DIR)
    with _purchase_transaction(db) as conn:
        r = _purchase_row(conn, purchase_id)
        cur = conn.execute("UPDATE purchase_suggestions SET status = ? WHERE id = ? AND status = ?",
                           (APPROVED, purchase_id, PROPOSED))
        if cur.rowcount != 1:
            raise StoreError(f"Η πρόταση αγοράς {purchase_id} είναι {r['status']}· "
                             f"έγκριση γίνεται μόνο σε {PROPOSED}.")
        doc = purchase_order_json(r)
        doc["APPROVED_BY"], doc["APPROVED_AT"] = str(user), _now()
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{purchase_id}.json"
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        conn.execute("UPDATE purchase_suggestions SET file_path = ? WHERE id = ?",
                     (str(path), purchase_id))
        _purchase_log(conn, purchase_id, user, APPROVE, {"file": str(path)})
    return path


def reject_purchase(purchase_id, reason, user, db=None):
    """ΠΡΟΤΑΣΗ -> ΑΠΟΡΡΙΦΘΗΚΕ, with a reason (required)."""
    if not reason or not str(reason).strip():
        raise StoreError("Η απόρριψη χρειάζεται λόγο.")
    with _purchase_transaction(db) as conn:
        r = _purchase_row(conn, purchase_id)
        cur = conn.execute("UPDATE purchase_suggestions SET status = ? WHERE id = ? AND status = ?",
                           (REJECTED, purchase_id, PROPOSED))
        if cur.rowcount != 1:
            raise StoreError(f"Η πρόταση αγοράς {purchase_id} είναι {r['status']}· "
                             f"απόρριψη γίνεται μόνο σε {PROPOSED}.")
        _purchase_log(conn, purchase_id, user, REJECT, {"reason": str(reason).strip()})


def list_purchase_suggestions(status=None, db=None):
    """Purchase suggestions with their payload, most urgent (earliest ORDER_BY) first."""
    sql, args = "SELECT * FROM purchase_suggestions WHERE 1 = 1", []
    if status:
        sql += " AND status = ?"
        args.append(status)
    conn = _purchase_connect(db)
    try:
        rows = conn.execute(sql + " ORDER BY order_by, id", args).fetchall()
        return [_as_dict(r) for r in rows]
    finally:
        conn.close()


def purchase_history(purchase_id=None, db=None):
    """History of the purchase suggestions, oldest first."""
    sql, args = "SELECT * FROM purchase_history WHERE 1 = 1", []
    if purchase_id is not None:
        sql += " AND purchase_id = ?"
        args.append(int(purchase_id))
    conn = _purchase_connect(db)
    try:
        rows = conn.execute(sql + " ORDER BY id", args).fetchall()
    finally:
        conn.close()
    out = []
    for r in rows:
        d = dict(r)
        d["details"] = json.loads(d.pop("details_json")) if d["details_json"] else None
        out.append(d)
    return out


# --- CLI -------------------------------------------------------------------

def _print_list(rows):
    print(f"{'id':>4}  {'κατάσταση':<17} {'τύπος':<18} {'πελάτης':>7} {'πωλ.':>4} "
          f"{'παράδοση':<10} {'είδη':>4} {'€':>9}  σημάνσεις")
    for s in rows:
        p = s["payload"]
        print(f"{s['id']:>4}  {s['status']:<17} {s['action_type']:<18} {s['trdr']:>7} "
              f"{s['salesman']:>4} {s['delivery_date']:<10} {len(p['LINES']):>4} "
              f"{s['value_eur']:>9,.2f}  {', '.join(p.get('FLAGS', []))}")
    print(f"Σύνολο: {len(rows)}")


def _short(details):
    if not details:
        return ""
    if "facts" in details:  # creation: keep it short on screen
        return f"αξία {details.get('value_eur')} € · προτεραιότητα {details.get('priority')}"
    return _dumps(details)


def _print_history(rows):
    for h in rows:
        erp = f" · ERP {h['erp_id']}" if h["erp_id"] else ""
        print(f"{h['at']}  #{h['suggestion_id']:<4} {h['trdr']:>5} {h['action_type']:<18} "
              f"{h['action']:<17} χρήστης {h['user']:<6} {_short(h['details'])}{erp}")
    print(f"Εγγραφές: {len(rows)}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m rafi.store",
                                 description="Προτάσεις και ιστορικό αποφάσεων του Ράφι.")
    ap.add_argument("--db", default=None, help="αρχείο βάσης (προεπιλογή rafi.db)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list")
    p.add_argument("--status")
    p.add_argument("--salesman", type=int)
    p = sub.add_parser("show")
    p.add_argument("id", type=int)
    for name in ("approve", "correct", "reject"):
        p = sub.add_parser(name)
        p.add_argument("id", type=int)
        if name == "correct":
            p.add_argument("mtrl", type=int)
            p.add_argument("qty", type=int)
        if name == "reject":
            p.add_argument("reason")
        p.add_argument("--user", required=True)
    p = sub.add_parser("history")
    p.add_argument("--trdr", type=int)
    p.add_argument("--id", type=int)
    args = ap.parse_args(argv)

    try:
        if args.cmd == "list":
            _print_list(list_suggestions(args.status, args.salesman, db=args.db))
        elif args.cmd == "show":
            s = get(args.id, db=args.db)
            print(f"Πρόταση {s['id']} · {s['status']} · δημιουργήθηκε {s['created_at']}"
                  + (f" · ERP {s['erp_id']}" if s["erp_id"] else ""))
            print(json.dumps(s["payload"], ensure_ascii=False, indent=2))
            _print_history(history(suggestion_id=args.id, db=args.db))
        elif args.cmd == "approve":
            erp_id = approve_and_send(args.id, args.user, db=args.db)
            print(f"Η πρόταση {args.id} εγκρίθηκε και στάλθηκε στο ERP ως draft: {erp_id}")
        elif args.cmd == "correct":
            correct(args.id, args.mtrl, args.qty, args.user, db=args.db)
            print(f"Η πρόταση {args.id} διορθώθηκε: είδος {args.mtrl} → {args.qty}")
        elif args.cmd == "reject":
            reject(args.id, args.reason, args.user, db=args.db)
            print(f"Η πρόταση {args.id} απορρίφθηκε: {args.reason}")
        elif args.cmd == "history":
            _print_history(history(args.trdr, args.id, db=args.db))
    except Exception as e:  # StoreError, ErpError: show the message, exit with an error code
        print(f"Σφάλμα: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
