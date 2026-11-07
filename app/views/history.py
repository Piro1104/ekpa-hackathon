import pandas as pd
import streamlit as st

from app import helpers as h
from app import ui
from rafi import store

ui.page_header("Ιστορικό", "", "Ιστορικό", "Όλες οι κινήσεις: πελάτες (δημιουργία, διορθώσεις, "
               "εγκρίσεις, απορρίψεις, αποστολές στο ERP) και αποθήκη (προτάσεις αγοράς, "
               "εγκρίσεις, απορρίψεις, μηνύματα στον προμηθευτή)")

rows = h.merge_history(
    h.history_rows(store.history(), ui.customers_map(), ui.items_map(), ui.salesmen_map()),
    h.purchase_history_rows(store.purchase_history(), store.list_purchase_suggestions(),
                            ui.salesmen_map()))
if not rows:
    st.info("Δεν υπάρχουν κινήσεις ακόμα. Τρέξε python scripts/daily_run.py")
    st.stop()

kind = st.radio("Τι δείχνω", h.HISTORY_KINDS, horizontal=True, key="history_kind")
of_kind = h.filter_history(rows, kind)
who_options = sorted({(r["Πελάτης"], r["Κωδικός"], r["_key"]) for r in of_kind})
actions = sorted({r["Κίνηση"] for r in of_kind})

c1, c2 = st.columns(2)
who = c1.selectbox("Πελάτης ή προμηθευτής", who_options, index=None,
                   placeholder="Όλοι", format_func=lambda c: f"{c[0]} · {c[1]}")
what = c2.multiselect("Κίνηση", actions, placeholder="Όλες οι κινήσεις")

shown = h.filter_history(rows, kind, who[2] if who else None, what)
st.markdown(f'<p class="ne-muted">{len(shown)} από {len(rows)} κινήσεις</p>',
            unsafe_allow_html=True)
if shown:
    ui.table(pd.DataFrame([h.visible(r) for r in shown]).rename(
                 columns={"Πελάτης": "Πελάτης / προμηθευτής"}),
             height=min(38 + 35 * len(shown), 640),
             column_config={"Λεπτομέρειες": st.column_config.TextColumn(width="large")})
