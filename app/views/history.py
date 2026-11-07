import pandas as pd
import streamlit as st

from app import helpers as h
from app import ui
from rafi import store

st.title("Ιστορικό")
st.markdown('<p class="ne-sub">Όλες οι κινήσεις: δημιουργία, διορθώσεις, εγκρίσεις, '
            'απορρίψεις και αποστολές στο ERP</p>', unsafe_allow_html=True)

rows = h.history_rows(store.history(), ui.customers_map(), ui.items_map(), ui.salesmen_map())
if not rows:
    st.info("Δεν υπάρχουν κινήσεις ακόμα. Τρέξε python scripts/daily_run.py")
    st.stop()

customers = sorted({(r["Πελάτης"], r["Κωδικός"], r["_trdr"]) for r in rows})
actions = sorted({r["Κίνηση"] for r in rows})

c1, c2 = st.columns(2)
who = c1.selectbox("Πελάτης", customers, index=None, placeholder="Όλοι οι πελάτες",
                   format_func=lambda c: f"{c[0]} · {c[1]}")
what = c2.multiselect("Κίνηση", actions, placeholder="Όλες οι κινήσεις")

shown = [r for r in rows
         if (who is None or r["_trdr"] == who[2]) and (not what or r["Κίνηση"] in what)]
st.markdown(f'<p class="ne-muted">{len(shown)} από {len(rows)} κινήσεις</p>',
            unsafe_allow_html=True)
if shown:
    ui.table(pd.DataFrame(shown).drop(columns="_trdr"), height=min(38 + 35 * len(shown), 640),
             column_config={"Λεπτομέρειες": st.column_config.TextColumn(width="large")})
