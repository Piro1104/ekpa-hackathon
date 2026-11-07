import streamlit as st

from app import helpers as h
from app import ui

st.title("Ρυθμίσεις")

values = h.read_settings()
modes = list(h.AFTER_APPROVAL_OPTIONS)

ui.show_flash("settings")
with st.form("settings", border=True):
    cost = st.number_input("Κόστος έκτακτου δρομολογίου (€)", min_value=0, step=5,
                           value=int(float(values["extra_route_cost_eur"])),
                           help="Αν οι χαμένες πωλήσεις μέχρι το κανονικό δρομολόγιο είναι "
                                "λιγότερες από αυτό το ποσό, η παράδοση περιμένει το κανονικό "
                                "δρομολόγιο και η πρόταση σημειώνεται «τηλεφώνημα πρώτα».")
    mode = st.radio("Μετά την έγκριση", modes,
                    index=modes.index(values["after_approval_mode"])
                    if values["after_approval_mode"] in modes else 0,
                    format_func=h.AFTER_APPROVAL_OPTIONS.get)
    saved = st.form_submit_button("Αποθήκευση", type="primary")

if saved:
    try:
        h.write_settings(cost, mode)
        ui.flash("settings", "success", "Οι ρυθμίσεις αποθηκεύτηκαν. Το νέο κόστος ισχύει από "
                                        "τον επόμενο υπολογισμό προτάσεων.")
        st.rerun()
    except ValueError as e:
        st.error(str(e))

st.markdown('<p class="ne-muted">Η αποστολή μηνύματος στον πελάτη μετά την έγκριση δεν είναι '
            'ακόμα ενεργή· προς το παρόν η επιλογή μόνο αποθηκεύεται.</p>',
            unsafe_allow_html=True)
