"""Shared fixtures."""
import socket
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def erp_url():
    """The organisers' mock ERP server, started on a free port for the tests."""
    port = _free_port()
    proc = subprocess.Popen([sys.executable, str(ROOT / "mock_erp_server.py"), "--port", str(port)],
                            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            env={**os.environ, "PYTHONIOENCODING": "utf-8"})
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


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path, monkeypatch):
    """Tests never read or write the real config/settings.json."""
    monkeypatch.setenv("RAFI_SETTINGS", str(tmp_path / "settings.json"))
