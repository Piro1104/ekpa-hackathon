"""Streamlit pieces shared by the pages: look, reference data, empty state, chat."""
import html
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from app import helpers as h
from rafi import store

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LOGO = ROOT / "app" / "assets" / "logo.png"
PRIMARY = "#1E5B43"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
.stApp, .stApp p, .stApp label, .stApp input, .stApp textarea, .stApp button,
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {{
    font-family: 'IBM Plex Sans', 'Helvetica Neue', Arial, sans-serif;
}}
#MainMenu, footer, .stDeployButton, [data-testid="stAppDeployButton"],
[data-testid="stToolbarActions"], [data-testid="stStatusWidget"] {{ display: none !important; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2rem; max-width: 1400px; }}
section[data-testid="stSidebar"] {{ background: #14261E; }}
section[data-testid="stSidebar"] [data-testid="stPageLink"] a {{ border-radius: 2px; }}
section[data-testid="stSidebar"] [data-testid="stPageLink"] p {{ color: #E6EEE9; font-size: 0.95rem; }}
.ne-logo {{ margin: 0.25rem 0 1.5rem 0.25rem; }}
.ne-logo-text {{ color: #FFFFFF; font-size: 1.5rem; font-weight: 600; letter-spacing: 0.02em;
                 margin: 0.25rem 0 1.5rem 0.25rem; }}
.stButton button, .stFormSubmitButton button, [data-baseweb="input"], [data-baseweb="select"] > div,
[data-baseweb="textarea"], [data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stExpander"] details, [data-testid="stAlert"] > div {{ border-radius: 3px !important; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
.ne-sub {{ color: #5B6560; margin-top: -0.6rem; }}
.ne-kpi {{ border-left: 3px solid {PRIMARY}; padding: 0.15rem 0 0.15rem 0.75rem; margin-bottom: 0.6rem; }}
.ne-kpi .v {{ font-size: 1.3rem; font-weight: 600; color: #1C2420; line-height: 1.3; white-space: nowrap; }}
.ne-kpi .l {{ font-size: 0.82rem; color: #5B6560; }}
.ne-badge {{ display: inline-block; font-size: 0.72rem; font-weight: 500; padding: 1px 6px;
             border: 1px solid #C9D3CE; border-radius: 2px; color: #3B4641; margin-right: 4px;
             white-space: nowrap; }}
.ne-badge.ai {{ border-color: {PRIMARY}; color: {PRIMARY}; }}
.ne-badge.warn {{ border-color: #B4651E; color: #8A4A12; }}
.ne-badge.call {{ border-color: #6B5B1E; color: #6B5B1E; }}
.ne-row-title {{ font-weight: 500; }}
.ne-muted {{ color: #5B6560; font-size: 0.86rem; }}
.ne-note {{ background: #F2F4F3; border-left: 3px solid {PRIMARY}; padding: 0.7rem 0.9rem;
            border-radius: 2px; white-space: pre-wrap; }}
.ne-chat-q {{ font-weight: 500; margin-top: 0.6rem; }}
.ne-chat-a {{ white-space: pre-wrap; margin-bottom: 0.2rem; }}
</style>
"""


def page_icon():
    if LOGO.exists():
        try:
            from PIL import Image
            return Image.open(LOGO)
        except Exception:
            return None
    return None


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def sidebar_logo():
    if LOGO.exists():
        st.markdown('<div class="ne-logo">', unsafe_allow_html=True)
        st.image(str(LOGO), width=180)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown('<div class="ne-logo-text">Never Empty</div>', unsafe_allow_html=True)


def badge(text, kind=""):
    return f'<span class="ne-badge {kind}">{html.escape(text)}</span>'


def ai_badge():
    return badge(h.AI_BADGE, "ai")


def flag_badges(s):
    out = []
    for text in h.flag_labels(s):
        kind = "call" if h.CALL_FIRST in text else "warn"
        out.append(badge(text, kind))
    return "".join(out)


# --- reference data ------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _csv(name):
    return pd.read_csv(DATA / name, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def customers_map():
    """{TRDR: (name, code)}"""
    return {int(r.TRDR): (str(r.NAME), str(r.CODE)) for r in _csv("TRDR.csv").itertuples()}


@st.cache_data(show_spinner=False)
def items_map():
    """{MTRL: (name, unit)}"""
    return h.item_lookup(_csv("MTRL.csv"))


@st.cache_data(show_spinner=False)
def salesmen_map():
    return {int(r.PRSN): str(r.NAME) for r in _csv("PRSN.csv").itertuples()}


def customer(trdr):
    return customers_map().get(int(trdr), (str(trdr), ""))


def all_suggestions():
    return store.list_suggestions()


# --- empty store --------------------------------------------------------------

def empty_state(suggestions):
    """Message and a button that runs the daily run. True if there is nothing to show."""
    if suggestions:
        return False
    st.info("Δεν υπάρχουν προτάσεις. Τρέξε python scripts/daily_run.py")
    if st.button("Υπολογισμός προτάσεων τώρα", type="primary"):
        with st.spinner("Υπολογίζονται οι προτάσεις της ημέρας. Χρειάζεται περίπου ένα λεπτό..."):
            done = subprocess.run([sys.executable, str(ROOT / "scripts" / "daily_run.py")],
                                  cwd=ROOT, capture_output=True, text=True)
        if done.returncode == 0:
            st.rerun()
        st.error("Ο υπολογισμός δεν ολοκληρώθηκε. Τρέξε python scripts/daily_run.py "
                 "στο terminal για να δεις τι έφταιξε.")
    return True


# --- navigation and messages ------------------------------------------------

def open_suggestion(sid):
    st.session_state["selected_id"] = int(sid)
    st.session_state["open_from_home"] = True
    st.switch_page("views/customers.py")


def flash(key, kind, text):
    st.session_state.setdefault("flash", {})[key] = (kind, text)


def show_flash(key):
    msg = st.session_state.get("flash", {}).pop(key, None)
    if msg:
        getattr(st, msg[0])(msg[1])


# --- chat ----------------------------------------------------------------------

def chat_box(key, context_fn, placeholder="Γράψε την ερώτησή σου"):
    """Question box with its own history in session_state[key].

    The assistant is called only when the button is pressed (inside a form).
    context_fn builds the context at that moment.
    """
    log = st.session_state.setdefault(key, [])
    for turn in log:
        st.markdown(f'<div class="ne-chat-q">{html.escape(turn["q"])}</div>',
                    unsafe_allow_html=True)
        mark = ai_badge() + "<br>" if turn["source"] else ""
        st.markdown(f'<div class="ne-chat-a">{mark}{html.escape(turn["a"])}</div>',
                    unsafe_allow_html=True)
    with st.form(f"{key}_form", clear_on_submit=True, border=False):
        cols = st.columns([5, 1], vertical_alignment="bottom")
        question = cols[0].text_input("Ερώτηση", placeholder=placeholder,
                                      label_visibility="collapsed")
        asked = cols[1].form_submit_button("Ρώτα", width="stretch")
    if asked and question.strip():
        with st.spinner("Ο βοηθός ετοιμάζει απάντηση..."):
            answer = h.ask_assistant(question.strip(), context_fn())
        log.append({"q": question.strip(), "a": answer["text"], "source": answer["source"]})
        st.rerun()
