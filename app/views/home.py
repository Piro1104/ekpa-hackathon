import html

import streamlit as st

from app import helpers as h
from app import ui

st.title("Καλημέρα")
st.markdown(f'<p class="ne-sub">{h.fmt_date_long(h.SCENARIO_TODAY)} · {h.COMPANY_LINE}</p>',
            unsafe_allow_html=True)

suggestions = ui.all_suggestions()
if not ui.empty_state(suggestions):
    m = h.home_metrics(suggestions)
    tabs = h.split_tabs(suggestions)

    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            st.subheader("Πελάτες")
            kpis = [
                (str(m["pending"]), "προς έγκριση"),
                (str(m["extra_routes"]), "έκτακτα δρομολόγια"),
                (str(m["loss_risks"]), "κίνδυνοι απώλειας"),
                (h.fmt_eur(m["at_risk_eur"]), "αξία σε κίνδυνο"),
                (str(m["offers"]), "προσφορές"),
                (h.fmt_eur(m["offers_eur"]), "αξία προσφορών"),
            ]
            cols = st.columns(3)
            for i, (value, text) in enumerate(kpis):
                cols[i % 3].markdown(f'<div class="ne-kpi"><div class="v">{value}</div>'
                                     f'<div class="l">{text}</div></div>', unsafe_allow_html=True)
            if st.button("Όλες οι προτάσεις", key="all_suggestions"):
                st.switch_page("views/customers.py")
    with right:
        with st.container(border=True):
            st.subheader("Αποθήκη")
            st.markdown('<p class="ne-muted">Ο έλεγχος αποθήκης ενεργοποιείται σύντομα</p>',
                        unsafe_allow_html=True)

    st.subheader("Χρειάζονται εσένα σήμερα")
    urgent = tabs[h.TAB_NEEDS_YOU]
    if not urgent:
        st.markdown('<p class="ne-muted">Τίποτα επείγον σήμερα.</p>', unsafe_allow_html=True)
    for s in urgent:
        name, code = ui.customer(s["trdr"])
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([6, 2, 2, 1.3], vertical_alignment="center")
            c1.markdown(
                f'<div class="ne-row-title">{html.escape(name)} '
                f'<span class="ne-muted">{html.escape(code)}</span></div>'
                f'<div>{ui.badge(h.label(s["action_type"]))}{ui.flag_badges(s)}</div>'
                f'<div class="ne-muted">{html.escape(h.summary(s))}</div>',
                unsafe_allow_html=True)
            c2.markdown(f'<div class="ne-muted">Παράδοση</div>{h.fmt_date(s["delivery_date"])}',
                        unsafe_allow_html=True)
            c3.markdown(f'<div class="ne-muted">Αξία</div>{h.fmt_eur(s["value_eur"])}',
                        unsafe_allow_html=True)
            if c4.button("Άνοιγμα", key=f"open_{s['id']}", width="stretch"):
                ui.open_suggestion(s["id"])

    st.subheader("Ρώτα τον βοηθό")
    ui.chat_box("chat_general",
                lambda: h.general_context(suggestions, ui.customers_map()),
                placeholder="π.χ. Με ποιον να ξεκινήσω σήμερα;")
