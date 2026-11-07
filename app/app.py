"""Never Empty dashboard. Start from the repository root:

    python -m streamlit run app/app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
# Streamlit puts app/ on the path, where "app" would mean this file. Use the repository
# root instead, so that "app" is the package and "rafi" is importable.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != APP_DIR]
sys.path.insert(0, str(ROOT))
if "app" in sys.modules and not hasattr(sys.modules["app"], "__path__"):
    del sys.modules["app"]

import streamlit as st  # noqa: E402

from app import ui  # noqa: E402

st.set_page_config(page_title="Never Empty", page_icon=ui.page_icon(), layout="wide",
                   initial_sidebar_state="expanded")
ui.inject_css()

PAGES = {
    "home": st.Page("views/home.py", title="Αρχική", url_path="arxiki", default=True),
    "customers": st.Page("views/customers.py", title="Πελάτες", url_path="pelates"),
    "history": st.Page("views/history.py", title="Ιστορικό", url_path="istoriko"),
    "settings": st.Page("views/settings_page.py", title="Ρυθμίσεις", url_path="rythmiseis"),
}
current = st.navigation(list(PAGES.values()), position="hidden")

with st.sidebar:
    ui.sidebar_logo()
    for page in PAGES.values():
        st.page_link(page, label=page.title, width="stretch")

current.run()
