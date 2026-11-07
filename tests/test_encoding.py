"""Greek text survives every file we write, and every file call names its encoding.

On Windows the default encoding is not UTF-8, so a call without encoding="utf-8" breaks
on Greek text there even when it works on Linux.
"""
import ast
import json
from pathlib import Path

from rafi import settings, store
from rafi.agents import warehouse_agent as wh

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ("rafi", "app", "scripts", "tests")
FILE_CALLS = {"read_text", "write_text", "to_csv", "read_csv"}
GREEK = "Λαμπρυντικό πλυντηρίου 5L · «Προμηθευτής Καθαριστικών Α.Ε.»"


def _kw(call, name):
    return next((k.value for k in call.keywords if k.arg == name), None)


def _binary(call):
    mode = call.args[1] if len(call.args) > 1 else _kw(call, "mode")
    return isinstance(mode, ast.Constant) and "b" in str(mode.value)


def missing_encoding(path):
    """[(line, call)] of file reads/writes in `path` without an encoding keyword."""
    out = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call) or _kw(node, "encoding") is not None:
            continue
        f = node.func
        if isinstance(f, ast.Name) and f.id == "open" and not _binary(node):
            out.append((node.lineno, "open"))
        elif isinstance(f, ast.Attribute) and f.attr in FILE_CALLS:
            out.append((node.lineno, f.attr))
        elif (isinstance(f, ast.Attribute) and f.attr in ("run", "Popen")
              and _kw(node, "text") is not None):
            out.append((node.lineno, f.attr))
    return out


def test_every_file_call_names_its_encoding():
    bad = [f"{p.relative_to(ROOT)}:{line} {call}"
           for folder in FOLDERS for p in sorted((ROOT / folder).rglob("*.py"))
           for line, call in missing_encoding(p)]
    assert bad == []


def test_purchase_order_json_keeps_greek(tmp_path):
    db = tmp_path / "t.db"
    pid = store.save_purchase_new([{
        "MTRL": 5050, "ITEM_CODE": "40.01.001", "ITEM_NAME": GREEK, "UNIT": "τεμ.",
        "SUPPLIER_ID": "S07", "SUPPLIER_NAME": "Καθαριστικά Ελλάς Α.Ε.",
        "ORDER_BY": "2026-11-02", "ARRIVAL": "2026-11-09", "QTY": 36, "VALUE_EUR": 460.8,
        "FLAG": "", "REASON": "Έπρεπε να είχε παραγγελθεί έως 31/10.", "AFFECTED": []}],
        db=db)["saved"][0]
    path = store.approve_purchase(pid, "Διαχειριστής", db=db, out_dir=tmp_path / "purchase_orders")
    raw = path.read_bytes().decode("utf-8")  # strict: fails on any other encoding
    assert GREEK in raw and "Καθαριστικά Ελλάς Α.Ε." in raw  # written as text, not \u escapes
    doc = json.loads(raw)
    assert doc["LINES"][0]["ITEM_NAME"] == GREEK
    assert doc["REASON"] == "Έπρεπε να είχε παραγγελθεί έως 31/10."
    assert doc["APPROVED_BY"] == "Διαχειριστής"


def test_settings_json_keeps_greek(tmp_path):
    path = tmp_path / "config" / "settings.json"
    settings.save_settings({"safety_weeks": 2, "note": GREEK}, path)
    raw = path.read_bytes().decode("utf-8")
    assert GREEK in raw
    assert settings.load_settings(path)["note"] == GREEK
    assert wh.load_settings(path)["safety_weeks"] == 2.0  # the warehouse agent reads it too
