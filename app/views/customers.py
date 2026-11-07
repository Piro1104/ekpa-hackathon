import html
from functools import partial

import pandas as pd
import streamlit as st

from app import helpers as h
from app import ui
from rafi import store
from rafi.erp_client import ErpError
from rafi.store import StoreError

st.title("Πελάτες")
st.markdown('<p class="ne-sub">Προτάσεις προς έγκριση, με τη σειρά προτεραιότητας</p>',
            unsafe_allow_html=True)

suggestions = ui.all_suggestions()
if ui.empty_state(suggestions):
    st.stop()


def _select(table_key, ids):
    rows = st.session_state[table_key].selection.rows
    if rows:
        st.session_state["selected_id"] = ids[rows[0]]


def _table(tab, rows):
    if not rows:
        st.markdown('<p class="ne-muted">Καμία πρόταση σε αυτή την καρτέλα.</p>',
                    unsafe_allow_html=True)
        return
    data = []
    for s in rows:
        name, code = ui.customer(s["trdr"])
        data.append({"Πελάτης": name, "Κωδικός": code,
                     "Παράδοση": h.fmt_date(s["delivery_date"]),
                     "Αξία": h.fmt_eur(s["value_eur"]),
                     "Σήμανση": " · ".join(h.flag_labels(s)),
                     "Περίληψη": h.summary(s, 110)})
    key = f"table_{tab}"
    ids = [s["id"] for s in rows]
    st.dataframe(
        pd.DataFrame(data), key=key, hide_index=True, width="stretch",
        height=min(38 + 35 * len(data), 330),
        on_select=partial(_select, key, ids), selection_mode="single-row",
        column_config={"Περίληψη": st.column_config.TextColumn(width="large"),
                       "Σήμανση": st.column_config.TextColumn(width="medium")})


tabs = h.split_tabs(suggestions)
selected = None
if st.session_state.get("selected_id") is not None:
    try:
        selected = store.get(st.session_state["selected_id"])
    except StoreError:
        st.session_state.pop("selected_id")

titles = {t: f"{t} ({len(tabs[t])})" for t in h.TABS}
default = titles.get(h.tab_of(selected)) if selected else None
containers = st.tabs([titles[t] for t in h.TABS], default=default)
for tab, box in zip(h.TABS, containers):
    with box:
        _table(tab, tabs[tab])

st.divider()

if selected is None:
    st.markdown('<p class="ne-muted">Επίλεξε μια γραμμή από τη λίστα για να δεις την πρόταση.</p>',
                unsafe_allow_html=True)
    st.stop()

# --- one suggestion -------------------------------------------------------------

s = selected
sid, p = s["id"], s["payload"]
name, code = ui.customer(s["trdr"])
user = h.store_user()
active = s["status"] == store.PROPOSED
ai_text = h.is_ai(p.get("TEXT_SOURCE"))

st.subheader(name)
st.markdown(f'<span class="ne-muted">{html.escape(code)}</span> &nbsp; '
            f'{ui.badge(h.label(s["action_type"]))}{ui.flag_badges(s)}'
            f'{ui.badge(h.label(s["status"]))}', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
for col, value, text in [(c1, h.fmt_date(s["delivery_date"]), "παράδοση"),
                         (c2, h.fmt_eur(s["value_eur"]), "αξία"),
                         (c3, str(len(p["LINES"])), "είδη"),
                         (c4, ui.salesmen_map().get(s["salesman"], "—"), "πωλητής")]:
    col.markdown(f'<div class="ne-kpi"><div class="v">{html.escape(str(value))}</div>'
                 f'<div class="l">{text}</div></div>', unsafe_allow_html=True)

mark = ui.ai_badge() if ai_text else ""
st.markdown(f"**Γιατί** {mark}", unsafe_allow_html=True)
st.write(p.get("REASON") or "—")
st.markdown(f"**Κείμενο για τον πελάτη** {mark}", unsafe_allow_html=True)
st.markdown(f'<div class="ne-note">{html.escape(p.get("SALES_NOTE") or "—")}</div>',
            unsafe_allow_html=True)

st.markdown("**Είδη**")
ui.table(h.lines_frame(h.lines_rows(s, ui.items_map())), height=38 + 35 * len(p["LINES"]))

# --- decisions -------------------------------------------------------------------

st.markdown("**Απόφαση**")
ui.show_flash(f"s{sid}")
if s["status"] == store.SENT:
    st.success(f"Στάλθηκε στο ERP ως πρόχειρη · αριθμός {s['erp_id']}")
elif s["status"] == store.REJECTED:
    reasons = [x["details"]["reason"] for x in store.history(suggestion_id=sid)
               if x["action"] == store.REJECT and x["details"]]
    st.info("Η πρόταση απορρίφθηκε" + (f": {reasons[-1]}" if reasons else "."))
elif s["status"] == store.SENDING:
    st.warning("Η πρόταση στέλνεται αυτή τη στιγμή στο ERP.")

if st.button("Εγκρίνω", type="primary", disabled=not active, key=f"approve_{sid}"):
    try:
        with st.spinner("Αποστολή στο ERP..."):
            store.approve_and_send(sid, user)
        st.rerun()
    except ErpError as e:
        kind = h.erp_error_kind(e)
        if kind == "down":
            st.error("Το ERP δεν απαντά. Ξεκίνα τον mock ERP: python mock_erp_server.py")
        elif kind == "rejected":
            st.error(f"Το ERP απέρριψε την πρόχειρη: {h.erp_rejection_message(e)}")
        else:
            st.error(str(e))
    except StoreError as e:
        st.warning(h.humanize(e))

with st.expander("Διορθώνω ποσότητες", expanded=False):
    items = ui.items_map()
    with st.form(f"correct_{sid}", border=False):
        values = {}
        for line in p["LINES"]:
            mtrl = int(line["MTRL"])
            item, unit = items.get(mtrl, (line.get("ITEM_NAME") or str(mtrl), ""))
            values[mtrl] = st.number_input(f"{item} ({unit})" if unit else item, min_value=1,
                                           step=1, value=int(line["QTY1"]), format="%d",
                                           key=f"qty_{sid}_{mtrl}", disabled=not active)
        save = st.form_submit_button("Αποθήκευση διορθώσεων", disabled=not active)
    if save:
        try:
            changes = h.changed_quantities(p["LINES"], values)
            for mtrl, qty in changes:
                store.correct(sid, mtrl, qty, user)
            if changes:
                text = ("Αποθηκεύτηκε 1 διόρθωση." if len(changes) == 1
                        else f"Αποθηκεύτηκαν {len(changes)} διορθώσεις.")
                ui.flash(f"s{sid}", "success", text)
            else:
                ui.flash(f"s{sid}", "info", "Καμία ποσότητα δεν άλλαξε.")
            st.rerun()
        except (ValueError, StoreError) as e:
            st.error(h.humanize(e))

with st.expander("Απορρίπτω", expanded=False):
    with st.form(f"reject_{sid}", border=False):
        reason = st.selectbox("Λόγος", h.REJECT_REASONS, index=None,
                              placeholder="Διάλεξε λόγο", disabled=not active)
        comment = st.text_input("Σχόλιο (προαιρετικό)", disabled=not active)
        do_reject = st.form_submit_button("Απόρριψη πρότασης", disabled=not active)
    if do_reject:
        try:
            store.reject(sid, h.rejection_text(reason, comment), user)
            st.rerun()
        except (ValueError, StoreError) as e:
            st.error(h.humanize(e))

st.markdown("**Ρώτα τον βοηθό για αυτή την πρόταση**")
ui.chat_box(f"chat_s{sid}",
            lambda: h.suggestion_context(store.get(sid), ui.customers_map(), ui.items_map()),
            placeholder="π.χ. Γιατί προτείνεται παράδοση αυτή τη μέρα;")
