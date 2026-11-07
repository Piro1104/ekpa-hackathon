"""Streamlit pieces shared by the pages: look, reference data, cards, tiles, chat."""
import html
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from app import helpers as h
from app import warehouse as wh
from rafi import store

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LOGO = ROOT / "app" / "assets" / "logo.png"
PRIMARY = "#1E5B43"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
.stApp, .stApp p, .stApp label, .stApp input, .stApp textarea, .stApp button,
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {{
    font-family: 'IBM Plex Sans', 'Helvetica Neue', Arial, sans-serif;
}}
#MainMenu, footer, .stDeployButton, [data-testid="stAppDeployButton"],
[data-testid="stToolbarActions"], [data-testid="stStatusWidget"] {{ display: none !important; }}
header[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stMain"] {{ background: #F5F6F5; }}
.block-container {{ padding-top: 2rem; max-width: 1500px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
.stButton button, .stFormSubmitButton button, [data-baseweb="input"], [data-baseweb="select"] > div,
[data-baseweb="textarea"], [data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stExpander"] details, [data-testid="stAlert"] > div {{ border-radius: 3px !important; }}

/* sidebar */
section[data-testid="stSidebar"] {{ background: #14261E; }}
section[data-testid="stSidebar"] [data-testid="stPageLink"] a {{ border-radius: 3px; padding: 0.45rem 0.8rem; }}
section[data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"] {{ background: {PRIMARY}; }}
section[data-testid="stSidebar"] [data-testid="stPageLink"] a > span,
section[data-testid="stSidebar"] [data-testid="stPageLink"] [data-testid="stMarkdownContainer"] {{ width: 100%; }}
section[data-testid="stSidebar"] [data-testid="stPageLink"] p {{
    color: #E6EEE9; font-size: 1rem; display: flex; justify-content: space-between;
    align-items: center; width: 100%; }}
section[data-testid="stSidebar"] [data-testid="stPageLink"] p code {{
    background: transparent; color: #9FB3A8; font-size: 0.82rem; padding: 0; line-height: 1; }}
.ne-logo-text {{ color: #FFFFFF; font-size: 1.5rem; font-weight: 600; margin: 0.25rem 0 1.5rem 0.25rem; }}
.ne-side-foot {{ position: fixed; bottom: 1.1rem; color: #E6EEE9; font-size: 0.9rem; line-height: 1.35;
                 border-top: 1px solid #2B4337; padding-top: 0.8rem; width: 15rem; }}
.ne-side-foot span {{ color: #9FB3A8; font-size: 0.82rem; }}

[data-testid="stMain"] [data-testid="stPageLink"] p {{ color: {PRIMARY}; white-space: nowrap; }}
[class*="st-key-ne_block"] [data-testid="stPageLink"] {{ display: flex; justify-content: flex-end; }}

/* text */
.ne-sub {{ color: #5B6560; margin-top: -0.6rem; }}
.ne-muted {{ color: #5B6560; font-size: 0.86rem; }}
.ne-label {{ color: #5B6560; font-size: 0.86rem; margin: 0.2rem 0 0.25rem 0;
             display: flex; justify-content: space-between; align-items: center; }}
.ne-h {{ font-weight: 600; font-size: 1.02rem; margin: 0.4rem 0 0.1rem 0; padding-bottom: 0.45rem;
         border-bottom: 1px solid #E3E8E5; }}
.ne-title {{ font-size: 1.55rem; font-weight: 700; color: #1C2420; margin: 0.15rem 0 0.6rem 0; line-height: 1.25; }}
.ne-reason {{ line-height: 1.6; margin-bottom: 0.4rem; }}

/* tags */
.ne-tag {{ display: inline-block; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.04em;
           padding: 2px 8px; border-radius: 2px; color: #FFFFFF; background: {PRIMARY};
           white-space: nowrap; vertical-align: 2px; margin-right: 6px; }}
.ne-tag.urgent, .ne-tag.short {{ background: #B4461E; }}
.ne-tag.loss {{ background: #8E2A35; }}
.ne-tag.call {{ background: #7A5B12; }}
.ne-tag.offer {{ background: #2F5D8A; }}
.ne-tag.wh {{ background: #24456B; }}
.ne-tag.buy {{ background: #E3ECF5; color: #24456B; }}
.ne-badge {{ display: inline-block; font-size: 0.72rem; font-weight: 500; padding: 1px 7px;
             border: 1px solid #9AA6A0; border-radius: 2px; color: #3B4641; white-space: nowrap; }}
.ne-badge.ai {{ border-color: #9AA6A0; color: #3B4641; background: #FFFFFF; }}

/* tiles */
.ne-tiles {{ display: grid; gap: 0.75rem; margin-bottom: 0.9rem; }}
.ne-tiles.c2 {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
.ne-tiles.c3 {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
.ne-tile {{ border: 1px solid #DDE3E0; background: #F7F8F7; border-radius: 2px; padding: 0.75rem 1rem; }}
.ne-panel-body .ne-tile, [class*="st-key-ne_page"] .ne-tile {{ background: #FFFFFF; }}
.ne-tile .l {{ color: #5B6560; font-size: 0.86rem; }}
.ne-tile .v {{ font-size: clamp(1.15rem, 1.9vw, 1.8rem); font-weight: 700; color: #1C2420;
               white-space: nowrap; line-height: 1.35; }}
.ne-tile .v.warn {{ color: #B4461E; }}
.ne-tile .v.loss {{ color: #8E2A35; }}
.ne-tile .s {{ color: #5B6560; font-size: 0.82rem; white-space: nowrap; }}
.ne-nums {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.75rem; margin-bottom: 0.9rem; }}
.ne-nums .l {{ color: #5B6560; font-size: 0.86rem; }}
.ne-nums .v {{ font-size: 1.2rem; font-weight: 700; white-space: nowrap; }}

/* blocks and panel */
[class*="st-key-ne_block"], [class*="st-key-ne_panel"] {{
    background: #FFFFFF; border: 1px solid #DDE3E0; border-radius: 2px; padding: 1.1rem 1.4rem; }}
.ne-block-head {{ display: flex; align-items: center; gap: 0.4rem; flex-wrap: wrap; }}
.ne-block-head .t {{ font-size: 1.25rem; font-weight: 700; }}
.ne-block-head .more {{ margin-left: auto; }}

/* rows of the home lists */
.stVerticalBlock[class*="st-key-ne_row"] {{ border-top: 1px solid #E3E8E5; padding-top: 0.7rem; }}
.ne-row {{ display: flex; gap: 0.75rem; align-items: flex-start; line-height: 1.55; }}
.ne-row .ne-tag {{ margin-top: 3px; }}
.st-key-ne_link button, [class*="st-key-ne_open"] button {{ color: {PRIMARY}; padding: 0; min-height: 0; }}
[class*="st-key-ne_open"] button p {{ font-size: 0.95rem; white-space: nowrap; }}
[class*="st-key-ne_open"] {{ display: flex; justify-content: flex-end; }}

/* cards: the whole card is one invisible button laid over it */
[class*="st-key-ne_list"] {{ gap: 0.55rem; }}
.stVerticalBlock[class*="st-key-necard"] {{ position: relative; background: #FFFFFF;
    border: 1px solid #DDE3E0; border-radius: 2px; padding: 0.7rem 1rem; gap: 0; }}
.stVerticalBlock[class*="st-key-necard"]:hover {{ border-color: #9DB5A9; }}
.stVerticalBlock[class*="st-key-necardon"] {{ background: #EEF4F1; border-color: {PRIMARY};
    box-shadow: inset 0 0 0 1px {PRIMARY}; }}
.stVerticalBlock[class*="st-key-necard"] [data-testid="stMarkdownContainer"],
.stVerticalBlock[class*="st-key-ne_row"] [data-testid="stMarkdownContainer"] {{ margin-bottom: 0; }}
.stElementContainer[class*="st-key-necard"][class*="_btn"] {{ position: absolute !important; inset: 0;
    width: 100% !important; height: 100% !important; margin: 0; z-index: 2; opacity: 0; }}
.stElementContainer[class*="st-key-necard"][class*="_btn"] .stButton,
.stElementContainer[class*="st-key-necard"][class*="_btn"] button {{ width: 100%; height: 100%; cursor: pointer; }}
.ne-card-head {{ line-height: 1.5; }}
.ne-card-head .t {{ font-weight: 700; }}
.ne-card-head .c {{ color: #5B6560; margin-left: 4px; }}
.ne-card-line {{ color: #2E3833; line-height: 1.5; }}

/* tab buttons */
[class*="st-key-ne_tabs"] button {{ min-height: 2.6rem; padding: 0 1.1rem; }}

/* notes and boxes */
.ne-note {{ background: #F2F4F3; border-left: 3px solid {PRIMARY}; padding: 0.7rem 0.9rem;
            border-radius: 2px; white-space: pre-wrap; line-height: 1.6; }}
.ne-note.blue {{ background: #EEF3F8; border-left-color: #24456B; }}
.ne-warn {{ background: #FBEFEA; border: 1px solid #E9C2B2; border-radius: 2px; padding: 0.7rem 0.9rem;
            color: #7A2E12; line-height: 1.5; }}
.ne-warn b {{ color: #5E230D; }}
.ne-warn .small {{ font-size: 0.84rem; }}
.st-key-ne_reject_btn button p, [class*="st-key-ne_reject"] button p {{ color: #8E2A35; }}

/* chat */
.ne-q {{ display: flex; justify-content: flex-end; margin: 0.5rem 0; }}
.ne-q div {{ background: #E3EDE8; padding: 0.45rem 0.9rem; border-radius: 3px; max-width: 85%; }}
.ne-a {{ border: 1px solid #DDE3E0; background: #FFFFFF; padding: 0.7rem 0.9rem; border-radius: 3px;
         max-width: 92%; white-space: pre-wrap; line-height: 1.55; margin-bottom: 0.5rem; }}
.ne-a .ne-badge {{ margin-bottom: 0.3rem; }}

/* status chip */
.ne-chip {{ border: 1px solid #DDE3E0; background: #FFFFFF; padding: 0.45rem 0.9rem; border-radius: 3px;
            font-size: 0.9rem; white-space: nowrap; display: inline-block; }}
.ne-chip .dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #2E9E63;
                 margin-right: 6px; }}
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
        st.image(str(LOGO), width=220)
    else:
        st.markdown('<div class="ne-logo-text">Never Empty</div>', unsafe_allow_html=True)


def sidebar_footer():
    st.markdown('<div class="ne-side-foot">DEMO ΔΙΑΝΟΜΕΣ Α.Ε.<br><span>Διαχειριστής</span></div>',
                unsafe_allow_html=True)


def esc(text):
    return html.escape(str(text))


def tag(text, kind=""):
    return f'<span class="ne-tag {kind}">{esc(h.caps(text))}</span>'


def badge(text, kind=""):
    return f'<span class="ne-badge {kind}">{esc(text)}</span>'


def ai_badge():
    return badge(h.AI_BADGE, "ai")


def page_header(kind_text, kind, title, sub):
    st.markdown(f'{tag(kind_text, kind)}<div class="ne-title" style="font-size:2.1rem">'
                f'{esc(title)}</div><p class="ne-sub">{esc(sub)}</p>', unsafe_allow_html=True)


def tiles(items, cols=3):
    """items: [(label, value, sub or '', colour class or '')]."""
    cells = "".join(
        f'<div class="ne-tile"><div class="l">{esc(l)}</div><div class="v {c}">{esc(v)}</div>'
        + (f'<div class="s">{esc(s)}</div>' if s else "") + "</div>"
        for l, v, s, c in items)
    st.markdown(f'<div class="ne-tiles c{cols}">{cells}</div>', unsafe_allow_html=True)


def numbers(items):
    """Three numbers of a detail panel: [(label, value)]."""
    cells = "".join(f'<div><div class="l">{esc(l)}</div><div class="v">{esc(v)}</div></div>'
                    for l, v in items)
    st.markdown(f'<div class="ne-nums">{cells}</div>', unsafe_allow_html=True)


def label_row(text, ai=False):
    mark = ai_badge() if ai else ""
    st.markdown(f'<div class="ne-label"><span>{esc(text)}</span>{mark}</div>',
                unsafe_allow_html=True)


def safe_key(text):
    return re.sub(r"[^0-9A-Za-z_]", "_", str(text))


def card(key, tag_html, title, code, line, selected, on_click, args=()):
    """A clickable alert card. The whole card is one invisible button."""
    k = f"necard{'on' if selected else ''}_{safe_key(key)}"
    with st.container(key=k):
        code_html = f'<span class="c">{esc(code)}</span>' if code else ""
        st.markdown(f'<div class="ne-card-head">{tag_html}<span class="t">{esc(title)}</span>'
                    f'{code_html}</div><div class="ne-card-line">{esc(line)}</div>',
                    unsafe_allow_html=True)
        st.button("Άνοιγμα", key=f"{k}_btn", on_click=on_click, args=args)


def table(df, **kwargs):
    """st.dataframe that sorts by the real values and shows them in Greek format."""
    styled = df.style.format(h.table_formats(df), na_rep="—")
    st.dataframe(styled, hide_index=True, width="stretch", **kwargs)


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


def facts():
    """{suggestion id: facts} (numbers behind the card lines)."""
    return h.facts_by_suggestion(store.history())


# --- warehouse -------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=300)
def _orders(days):
    funcs = wh.agent()
    return None if funcs is None else funcs["open_purchase_orders"](days=days)


def warehouse_orders(days=wh.WEEK_DAYS):
    """Open purchase orders, or None when the warehouse agent is not ready (or fails)."""
    try:
        return _orders(days)
    except Exception:
        return None


def clear_warehouse():
    _orders.clear()


# --- empty store --------------------------------------------------------------

def empty_state(suggestions):
    """Message and a button that runs the daily run. True if there is nothing to show."""
    if suggestions:
        return False
    st.info("Δεν υπάρχουν προτάσεις. Τρέξε python scripts/daily_run.py")
    if st.button("Υπολογισμός προτάσεων τώρα", type="primary"):
        with st.spinner("Υπολογίζονται οι προτάσεις της ημέρας. Χρειάζεται περίπου ένα λεπτό..."):
            done = subprocess.run([sys.executable, str(ROOT / "scripts" / "daily_run.py")],
                                  cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        if done.returncode == 0:
            st.rerun()
        st.error("Ο υπολογισμός δεν ολοκληρώθηκε. Τρέξε python scripts/daily_run.py "
                 "στο terminal για να δεις τι έφταιξε.")
    return True


# --- navigation and messages ------------------------------------------------

def open_suggestion(sid):
    st.session_state["selected_id"] = int(sid)
    st.switch_page("views/customers.py")


def open_purchase(key):
    st.session_state["selected_po"] = key
    st.switch_page("views/warehouse_page.py")


def flash(key, kind, text):
    st.session_state.setdefault("flash", {})[key] = (kind, text)


def show_flash(key):
    msg = st.session_state.get("flash", {}).pop(key, None)
    if msg:
        getattr(st, msg[0])(msg[1])


# --- chat ----------------------------------------------------------------------

def _ask(key, question, context_fn):
    with st.spinner("Ο βοηθός ετοιμάζει απάντηση..."):
        answer = h.ask_assistant(question, context_fn())
    st.session_state.setdefault(key, []).append(
        {"q": question, "a": answer["text"], "source": answer["source"]})


def chat_box(key, context_fn, placeholder="Γράψε την ερώτησή σου", quick=(), footnote=""):
    """Question box with its own history in session_state[key], as chat bubbles.

    The assistant is called only when a button is pressed. context_fn builds the context
    at that moment. `quick`: ready questions shown as buttons under the box.
    """
    for turn in st.session_state.setdefault(key, []):
        st.markdown(f'<div class="ne-q"><div>{esc(turn["q"])}</div></div>', unsafe_allow_html=True)
        mark = ai_badge() + "<br>" if h.is_ai(turn["source"]) else ""
        st.markdown(f'<div class="ne-a">{mark}{esc(turn["a"])}</div>', unsafe_allow_html=True)
    with st.form(f"{key}_form", clear_on_submit=True, border=False):
        cols = st.columns([6, 1], vertical_alignment="bottom")
        question = cols[0].text_input("Ερώτηση", placeholder=placeholder)
        asked = cols[1].form_submit_button("Ρώτα", width="stretch")
    picked = None
    if quick:
        with st.container(horizontal=True):
            for i, q in enumerate(quick):
                if st.button(q, key=f"{key}_quick_{i}"):
                    picked = q
    if footnote:
        st.markdown(f'<p class="ne-muted">{esc(footnote)}</p>', unsafe_allow_html=True)
    question = picked or (question.strip() if asked else "")
    if question:
        _ask(key, question, context_fn)
        st.rerun()
