"""User settings, kept in config/settings.json (or $RAFI_SETTINGS).

The file is created with the defaults if it is missing, and missing keys are filled in.
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "config" / "settings.json"

DRAFT_ONLY = "draft_only"
DRAFT_AND_MESSAGE = "draft_and_message"

DEFAULTS = {
    "extra_route_cost_eur": 60,
    "after_approval_mode": DRAFT_ONLY,
    "stock_cover_weeks": 4,
    "safety_weeks": 1,
    "ai_texts_top_n": 20,  # AI texts for "needs you" + this many more suggestions
}


def settings_path(path=None):
    return Path(path or os.environ.get("RAFI_SETTINGS") or DEFAULT_PATH)


def load_settings(path=None):
    """Settings with defaults for anything missing. Writes the file back if it was incomplete."""
    p = settings_path(path)
    try:
        stored = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(stored, dict):
            stored = {}
    except (OSError, ValueError):
        stored = {}
    merged = {**DEFAULTS, **stored}
    if merged != stored:
        try:
            save_settings(merged, p)
        except OSError:
            pass  # read-only place: still return the defaults
    return merged


def save_settings(values, path=None):
    p = settings_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {**DEFAULTS, **values}
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return data


def extra_route_cost_eur(path=None):
    """Cost of one extra delivery run in EUR (60 if missing or not a number)."""
    try:
        value = float(load_settings(path).get("extra_route_cost_eur"))
    except (TypeError, ValueError):
        return float(DEFAULTS["extra_route_cost_eur"])
    return value if value >= 0 else float(DEFAULTS["extra_route_cost_eur"])


def ai_texts_top_n(path=None):
    """How many suggestions outside "needs you" get AI texts (20 if missing or not a number)."""
    try:
        value = int(load_settings(path).get("ai_texts_top_n"))
    except (TypeError, ValueError):
        return DEFAULTS["ai_texts_top_n"]
    return max(value, 0)
