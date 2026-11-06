"""Calls to the ERP API. Only one write: POST /api/draft-orders.

The address comes from the ERP_URL environment variable (default http://localhost:8080).
"""
import os

import requests

DEFAULT_URL = "http://localhost:8080"
TIMEOUT = 15


class ErpError(Exception):
    """The draft was not saved in the ERP. `errors` holds the ERP's own list, if any."""

    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = list(errors or [])


def erp_url():
    return os.environ.get("ERP_URL", DEFAULT_URL).rstrip("/")


def post_draft(payload, base_url=None, timeout=TIMEOUT):
    """Send one suggested order as a draft. Returns the ERP id (e.g. 'DRAFT-20261101...').

    Raises ErpError with a clear Greek message if the ERP is down or rejects the draft.
    """
    url = f"{(base_url or erp_url()).rstrip('/')}/api/draft-orders"
    try:
        r = requests.post(url, json=payload, timeout=timeout)
    except (requests.ConnectionError, requests.Timeout) as e:
        raise ErpError(f"Το ERP δεν απαντά στο {url} (τρέχει το mock_erp_server.py;) "
                       f"— {type(e).__name__}") from e

    try:
        body = r.json()
    except ValueError:
        body = {}
    if r.status_code == 400:
        errors = body.get("errors", []) if isinstance(body, dict) else []
        raise ErpError("Το ERP απέρριψε το draft: " + "; ".join(map(str, errors)), errors)
    if not r.ok:
        raise ErpError(f"Το ERP απάντησε με σφάλμα HTTP {r.status_code}")
    if not isinstance(body, dict) or not body.get("id"):
        raise ErpError("Το ERP δεν επέστρεψε id για το draft")
    return body["id"]
