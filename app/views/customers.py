import streamlit as st

from app import helpers as h
from app import ui
from rafi import store
from rafi.erp_client import ErpError
from rafi.store import StoreError

SHOWN_LINES = 5   # rows of the items table before «+N ακόμη»
PAGE_CARDS = 30   # cards shown before «Δείξε περισσότερες»

ui.page_header("Πελάτες", "", "Πελάτες", "Κάθε πρόταση σε απλά λόγια: τι προτείνουμε και γιατί")

suggestions = ui.all_suggestions()
if ui.empty_state(suggestions):
    st.stop()

tabs = h.split_tabs(suggestions)
facts = ui.facts()
state = st.session_state

selected = None
if state.get("selected_id") is not None:
    try:
        selected = store.get(state["selected_id"])
    except StoreError:
        state.pop("selected_id")

# the tab follows the selected suggestion when it comes from the home page
if selected is not None and h.tab_of(selected) and selected["status"] == store.PROPOSED \
        and state.get("cust_tab_for") != selected["id"]:
    state["cust_tab"] = h.tab_of(selected)
    state["cust_tab_for"] = selected["id"]
tab = state.get("cust_tab", h.TAB_NEEDS_YOU)
rows = tabs[tab]
if selected is None and rows:
    selected = rows[0]
    state["selected_id"] = selected["id"]


def pick_tab(t):
    state["cust_tab"] = t
    state["cust_shown"] = PAGE_CARDS
    state["cust_tab_for"] = None
    if tabs[t]:
        state["selected_id"] = tabs[t][0]["id"]


def pick(sid):
    state["selected_id"] = sid
    state["cust_tab_for"] = sid
    state.pop("cust_action", None)


with st.container(horizontal=True, key="ne_tabs"):
    for t in h.TABS:
        st.button(f"{t} ({len(tabs[t])})", key=f"tab_{h.TABS.index(t)}",
                  type="primary" if t == tab else "secondary", on_click=pick_tab, args=(t,))

left, right = st.columns([5, 7], gap="medium")

# --- cards -------------------------------------------------------------------------

with left, st.container(key="ne_list"):
    if not rows:
        st.markdown('<p class="ne-muted">Καμία πρόταση σε αυτή την καρτέλα.</p>',
                    unsafe_allow_html=True)
    shown = state.get("cust_shown", PAGE_CARDS)
    for s in rows[:shown]:
        name, _ = ui.customer(s["trdr"])
        text, kind = h.card_badge(s)
        ui.card(s["id"], ui.tag(text, kind), name, s["trdr"], h.card_line(s, facts.get(s["id"])),
                selected is not None and s["id"] == selected["id"], pick, (s["id"],))
    if len(rows) > shown:
        if st.button(f"Δείξε περισσότερες ({len(rows) - shown} ακόμη)", type="tertiary"):
            state["cust_shown"] = shown + PAGE_CARDS
            st.rerun()
    st.markdown('<p class="ne-muted">Σειρά: επείγοντα → κίνδυνος απώλειας → παραγγελίες → '
                'προσφορές.</p>', unsafe_allow_html=True)

# --- one suggestion -------------------------------------------------------------------

with right:
    if selected is None:
        st.markdown('<p class="ne-muted">Διάλεξε μια κάρτα για να δεις την πρόταση.</p>',
                    unsafe_allow_html=True)
        st.stop()

    s = selected
    sid, p = s["id"], s["payload"]
    name, _ = ui.customer(s["trdr"])
    user = h.store_user()
    active = s["status"] == store.PROPOSED
    ai_text = h.is_ai(p.get("TEXT_SOURCE"))
    items = ui.items_map()
    text, kind = h.card_badge(s)

    with st.container(key="ne_panel_customer"):
        st.markdown(f'{ui.tag(text, kind)}<span class="ne-muted" style="font-size:.95rem">'
                    f'Πελάτης {s["trdr"]}</span><div class="ne-title">{ui.esc(name)}</div>',
                    unsafe_allow_html=True)
        ui.numbers([("Παράδοση", h.fmt_day(s["delivery_date"])),
                    ("Είδη", str(len(p["LINES"]))),
                    ("Αξία", h.fmt_eur(s["value_eur"]))])

        ui.label_row("Τι προτείνουμε και γιατί", ai_text)
        st.markdown(f'<div class="ne-reason">{ui.esc(p.get("REASON") or "—")}</div>',
                    unsafe_allow_html=True)

        short = h.warehouse_shortage(s, items)
        if short is not None:
            what = f": {', '.join(short)}" if short else ""
            st.markdown(f'<div class="ne-warn"><b>Έλλειψη στην αποθήκη{ui.esc(what)}</b><br>'
                        f'<span class="small">Μπορεί να μη φτάσει όλη η ποσότητα την ημέρα '
                        f'της παράδοσης· δες την παραγγελία στον προμηθευτή.</span></div>',
                        unsafe_allow_html=True)
            st.page_link("views/warehouse_page.py", label="Άνοιγμα Αποθήκης →")

        ui.label_row("Είδη")
        lines = h.lines_rows(s, items)
        all_lines = state.get(f"all_lines_{sid}", False)
        visible = lines if all_lines else lines[:SHOWN_LINES]
        ui.table(h.lines_frame(visible), height=38 + 35 * len(visible))
        if len(lines) > SHOWN_LINES:
            more = "Λιγότερα" if all_lines else f"+ {len(lines) - SHOWN_LINES} ακόμη είδη"
            if st.button(more, key=f"more_{sid}", type="tertiary"):
                state[f"all_lines_{sid}"] = not all_lines
                st.rerun()

        ui.label_row("Κείμενο για τον πελάτη", ai_text)
        st.markdown(f'<div class="ne-note">{ui.esc(p.get("SALES_NOTE") or "—")}</div>',
                    unsafe_allow_html=True)
        st.write("")

        # --- decisions ---------------------------------------------------------------
        ui.show_flash(f"s{sid}")
        if s["status"] == store.SENT:
            st.success(f"Στάλθηκε στο ERP ως πρόχειρη · αριθμός {s['erp_id']}")
        elif s["status"] == store.REJECTED:
            reasons = [x["details"]["reason"] for x in store.history(suggestion_id=sid)
                       if x["action"] == store.REJECT and x["details"]]
            st.info("Η πρόταση απορρίφθηκε" + (f": {reasons[-1]}" if reasons else "."))
        elif s["status"] == store.SENDING:
            st.warning("Η πρόταση στέλνεται αυτή τη στιγμή στο ERP.")

        action = state.get("cust_action") if active else None
        with st.container(horizontal=True):
            approve = st.button("Εγκρίνω · πρόχειρη στο ERP", type="primary", disabled=not active,
                                key=f"approve_{sid}")
            if st.button("Διορθώνω", disabled=not active, key=f"correct_btn_{sid}"):
                state["cust_action"] = None if action == "correct" else "correct"
                st.rerun()
            if st.button("Απορρίπτω", disabled=not active, key=f"ne_reject_{sid}"):
                state["cust_action"] = None if action == "reject" else "reject"
                st.rerun()

        if approve:
            try:
                with st.spinner("Αποστολή στο ERP..."):
                    store.approve_and_send(sid, user)
                note = h.notify_after_approval(sid, h.read_settings().get("after_approval_mode"))
                if note:
                    ui.flash(f"m{sid}", *note)
                st.rerun()
            except ErpError as e:
                kind_ = h.erp_error_kind(e)
                if kind_ == "down":
                    st.error("Το ERP δεν απαντά. Ξεκίνα τον mock ERP: python mock_erp_server.py")
                elif kind_ == "rejected":
                    st.error(f"Το ERP απέρριψε την πρόχειρη: {h.erp_rejection_message(e)}")
                else:
                    st.error(str(e))
            except StoreError as e:
                st.warning(h.humanize(e))
        ui.show_flash(f"m{sid}")

        if action == "correct":
            with st.form(f"correct_{sid}", border=True):
                values = {}
                for line in p["LINES"]:
                    mtrl = int(line["MTRL"])
                    item, unit = items.get(mtrl, (line.get("ITEM_NAME") or str(mtrl), ""))
                    values[mtrl] = st.number_input(f"{item} ({unit})" if unit else item,
                                                   min_value=1, step=1, value=int(line["QTY1"]),
                                                   format="%d", key=f"qty_{sid}_{mtrl}")
                save = st.form_submit_button("Αποθήκευση διορθώσεων", type="primary")
            if save:
                try:
                    changes = h.changed_quantities(p["LINES"], values)
                    for mtrl, qty in changes:
                        store.correct(sid, mtrl, qty, user)
                    if changes:
                        msg = ("Αποθηκεύτηκε 1 διόρθωση." if len(changes) == 1
                               else f"Αποθηκεύτηκαν {len(changes)} διορθώσεις.")
                        ui.flash(f"s{sid}", "success", msg)
                    else:
                        ui.flash(f"s{sid}", "info", "Καμία ποσότητα δεν άλλαξε.")
                    state.pop("cust_action", None)
                    st.rerun()
                except (ValueError, StoreError) as e:
                    st.error(h.humanize(e))

        if action == "reject":
            with st.form(f"reject_{sid}", border=True):
                reason = st.selectbox("Λόγος", h.REJECT_REASONS, index=None,
                                      placeholder="Διάλεξε λόγο")
                comment = st.text_input("Σχόλιο (προαιρετικό)")
                do_reject = st.form_submit_button("Απόρριψη πρότασης")
            if do_reject:
                try:
                    store.reject(sid, h.rejection_text(reason, comment), user)
                    state.pop("cust_action", None)
                    st.rerun()
                except (ValueError, StoreError) as e:
                    st.error(h.humanize(e))

        st.markdown('<div class="ne-h" style="margin-top:1rem"></div>', unsafe_allow_html=True)
        ui.label_row("Ρώτα για αυτή την πρόταση")
        ui.chat_box(f"chat_s{sid}",
                    lambda: h.suggestion_context(store.get(sid), ui.customers_map(), items),
                    placeholder="π.χ. Τι γίνεται αν περιμένουμε ως Τετάρτη;",
                    footnote="Ο βοηθός βλέπει μόνο τα νούμερα αυτής της πρότασης.")
